#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/SegmentedConcatConsumerOptimizer.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

struct ConcatInput {
  tensor::ConcatOp concat;
  tensor::ExpandShapeOp expand;
  unsigned operandIndex;
  unsigned consumerAxis;
};

static bool isStaticF32Tensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape() && tensor.getElementType().isF32();
}

static bool hasPureElementwiseBody(linalg::GenericOp generic) {
  if (generic.getRegion().empty())
    return false;
  Block &body = generic.getRegion().front();
  if (!body.getArguments().back().use_empty())
    return false;
  for (Operation &operation : body.without_terminator()) {
    if (isa<linalg::IndexOp>(operation) || !isMemoryEffectFree(&operation))
      return false;
  }
  return isa<linalg::YieldOp>(body.getTerminator());
}

static bool hasIdentityElementwiseMaps(linalg::GenericOp generic,
                                       unsigned rank) {
  if (generic.getNumLoops() != rank ||
      llvm::any_of(generic.getIteratorTypesArray(),
                   [](utils::IteratorType type) {
                     return type != utils::IteratorType::parallel;
                   }))
    return false;
  return llvm::all_of(generic.getIndexingMapsArray(), [rank](AffineMap map) {
    return map.getNumDims() == rank && map.isIdentity();
  });
}

static FailureOr<ConcatInput> matchConcatInput(linalg::GenericOp generic) {
  std::optional<ConcatInput> match;
  for (auto [operandIndex, input] : llvm::enumerate(generic.getDpsInputs())) {
    auto expand = input.getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand || !expand->hasOneUse())
      continue;
    auto concat = expand.getSrc().getDefiningOp<tensor::ConcatOp>();
    if (!concat || !concat->hasOneUse())
      continue;
    if (match)
      return failure();

    auto concatType = concat.getResultType();
    auto expandedType = expand.getResultType();
    if (!isStaticF32Tensor(concatType) || !isStaticF32Tensor(expandedType) ||
        llvm::any_of(concat.getInputs(), [](Value input) {
          return !isStaticF32Tensor(input.getType());
        }))
      return failure();
    int64_t concatAxis = concat.getDim();
    if (concatAxis != concatType.getRank() - 1)
      return failure();

    ArrayRef<int64_t> group = expand.getReassociationIndices()[concatAxis];
    std::optional<unsigned> consumerAxis;
    for (int64_t axis : group) {
      int64_t extent = expandedType.getDimSize(axis);
      if (extent == concatType.getDimSize(concatAxis)) {
        if (consumerAxis)
          return failure();
        consumerAxis = static_cast<unsigned>(axis);
      } else if (extent != 1) {
        return failure();
      }
    }
    if (!consumerAxis)
      return failure();
    match = ConcatInput{concat, expand, static_cast<unsigned>(operandIndex),
                        *consumerAxis};
  }
  if (!match)
    return failure();
  return *match;
}

static SmallVector<OpFoldResult> getIndexAttrs(OpBuilder &builder,
                                               ArrayRef<int64_t> values) {
  return llvm::map_to_vector(values, [&](int64_t value) -> OpFoldResult {
    return builder.getIndexAttr(value);
  });
}

static Value createSlice(IRRewriter &rewriter, Location loc, Value source,
                         unsigned axis, int64_t offset, int64_t size) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<int64_t> offsets(sourceType.getRank(), 0);
  SmallVector<int64_t> sizes(sourceType.getShape());
  SmallVector<int64_t> strides(sourceType.getRank(), 1);
  offsets[axis] = offset;
  sizes[axis] = size;
  auto resultType = RankedTensorType::get(sizes, sourceType.getElementType());
  return rewriter
      .create<tensor::ExtractSliceOp>(
          loc, resultType, source, getIndexAttrs(rewriter, offsets),
          getIndexAttrs(rewriter, sizes), getIndexAttrs(rewriter, strides))
      .getResult();
}

static LogicalResult rewriteConcatConsumer(linalg::GenericOp generic,
                                           IRRewriter &rewriter) {
  if (generic.getNumDpsInputs() == 0 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1)
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!resultType || !isStaticF32Tensor(resultType) ||
      !hasIdentityElementwiseMaps(generic, resultType.getRank()) ||
      !hasPureElementwiseBody(generic))
    return failure();
  func::FuncOp function = generic->getParentOfType<func::FuncOp>();
  if (!function || !function->hasAttr("sculptor.task_kind"))
    return failure();

  FailureOr<ConcatInput> matched = matchConcatInput(generic);
  if (failed(matched))
    return failure();
  ConcatInput concatInput = *matched;

  for (Value input : generic.getDpsInputs()) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || !isStaticF32Tensor(type) || type != resultType)
      return failure();
  }
  if (generic.getDpsInitOperand(0)->get().getType() != resultType)
    return failure();

  Location loc = generic.getLoc();
  rewriter.setInsertionPoint(generic);
  Value destination = rewriter.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());
  int64_t offset = 0;
  Block &sourceBody = generic.getRegion().front();
  auto sourceYield = cast<linalg::YieldOp>(sourceBody.getTerminator());

  for (Value concatSegment : concatInput.concat.getInputs()) {
    auto segmentType = cast<RankedTensorType>(concatSegment.getType());
    int64_t segmentSize = segmentType.getDimSize(concatInput.concat.getDim());
    SmallVector<int64_t> expandedShape(resultType.getShape());
    expandedShape[concatInput.consumerAxis] = segmentSize;
    auto expandedSegmentType =
        RankedTensorType::get(expandedShape, resultType.getElementType());
    Value expandedSegment = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedSegmentType, concatSegment,
        concatInput.expand.getReassociationIndices());

    SmallVector<Value> inputs;
    inputs.reserve(generic.getNumDpsInputs());
    for (auto [operandIndex, input] : llvm::enumerate(generic.getDpsInputs())) {
      if (operandIndex == concatInput.operandIndex)
        inputs.push_back(expandedSegment);
      else
        inputs.push_back(createSlice(rewriter, loc, input,
                                     concatInput.consumerAxis, offset,
                                     segmentSize));
    }

    Value outputSlice =
        createSlice(rewriter, loc, destination, concatInput.consumerAxis,
                    offset, segmentSize);
    auto segmented = rewriter.create<linalg::GenericOp>(
        loc, expandedSegmentType, inputs, ValueRange{outputSlice},
        generic.getIndexingMapsArray(), generic.getIteratorTypesArray(),
        [&](OpBuilder &builder, Location bodyLoc, ValueRange arguments) {
          IRMapping mapping;
          for (auto [sourceArgument, targetArgument] :
               llvm::zip_equal(sourceBody.getArguments(), arguments))
            mapping.map(sourceArgument, targetArgument);
          for (Operation &bodyOp : sourceBody.without_terminator())
            builder.clone(bodyOp, mapping);
          builder.create<linalg::YieldOp>(
              bodyLoc,
              mapping.lookupOrDefault(sourceYield.getValues().front()));
        });
    segmented->setAttr("sculptor.optimization.segmented_concat",
                       rewriter.getI64IntegerAttr(offset));

    SmallVector<int64_t> offsets(resultType.getRank(), 0);
    SmallVector<int64_t> sizes(resultType.getShape());
    SmallVector<int64_t> strides(resultType.getRank(), 1);
    offsets[concatInput.consumerAxis] = offset;
    sizes[concatInput.consumerAxis] = segmentSize;
    destination = rewriter.create<tensor::InsertSliceOp>(
        loc, segmented.getResult(0), destination,
        getIndexAttrs(rewriter, offsets), getIndexAttrs(rewriter, sizes),
        getIndexAttrs(rewriter, strides));
    offset += segmentSize;
  }

  if (offset != resultType.getDimSize(concatInput.consumerAxis))
    return failure();
  rewriter.replaceOp(generic, destination);
  return success();
}

} // namespace

LogicalResult optimizeSegmentedConcatConsumer(ModuleOp module,
                                              func::FuncOp taskGraphFunc,
                                              const TaskGraphDAG &dag,
                                              bool &changed) {
  (void)taskGraphFunc;
  (void)dag;
  changed = false;

  SmallVector<linalg::GenericOp> candidates;
  module.walk([&](linalg::GenericOp generic) {
    if (succeeded(matchConcatInput(generic)))
      candidates.push_back(generic);
  });

  IRRewriter rewriter(module.getContext());
  for (linalg::GenericOp generic : candidates) {
    if (!generic->getBlock())
      continue;
    if (succeeded(rewriteConcatConsumer(generic, rewriter)))
      changed = true;
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
