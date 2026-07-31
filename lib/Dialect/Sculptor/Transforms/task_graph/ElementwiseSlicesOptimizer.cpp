#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/ElementwiseSlicesOptimizer.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

struct SliceInsertion {
  tensor::InsertSliceOp op;
  tensor::ExtractSliceOp extract;
};

static bool hasSameStaticSlice(tensor::InsertSliceOp insert,
                               tensor::ExtractSliceOp extract) {
  return insert.getStaticOffsets() == extract.getStaticOffsets() &&
         insert.getStaticSizes() == extract.getStaticSizes() &&
         insert.getStaticStrides() == extract.getStaticStrides();
}

static bool hasFullyStaticUnitStrideSlice(tensor::InsertSliceOp insert) {
  return llvm::none_of(insert.getStaticOffsets(), ShapedType::isDynamic) &&
         llvm::none_of(insert.getStaticSizes(), ShapedType::isDynamic) &&
         llvm::all_of(insert.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; });
}

static bool areDisjoint(tensor::InsertSliceOp lhs, tensor::InsertSliceOp rhs) {
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhs.getStaticOffsets(), lhs.getStaticSizes(),
                       rhs.getStaticOffsets(), rhs.getStaticSizes())) {
    if (lhsOffset + lhsSize <= rhsOffset || rhsOffset + rhsSize <= lhsOffset)
      return true;
  }
  return false;
}

static bool hasPureElementwiseBody(linalg::GenericOp generic) {
  if (generic.getRegion().empty() ||
      generic.getRegion().front().getNumArguments() != 2)
    return false;

  Block &body = generic.getRegion().front();
  if (!body.getArgument(1).use_empty())
    return false;

  for (Operation &op : body.without_terminator()) {
    if (isa<linalg::IndexOp>(op) || !isMemoryEffectFree(&op))
      return false;
  }
  return isa<linalg::YieldOp>(body.getTerminator());
}

static LogicalResult rewriteElementwiseSlices(linalg::GenericOp generic,
                                              IRRewriter &rewriter,
                                              bool &changed) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1)
    return success();

  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      !linalg_match::hasElementwiseIndexingMaps(generic,
                                                resultType.getRank()) ||
      !hasPureElementwiseBody(generic))
    return success();

  Value combinedInput = generic.getDpsInputs().front();
  SmallVector<tensor::InsertSliceOp, 4> insertions;
  Value insertionRoot = combinedInput;
  while (auto insert = insertionRoot.getDefiningOp<tensor::InsertSliceOp>()) {
    insertions.push_back(insert);
    insertionRoot = insert.getDest();
  }
  if (insertions.size() < 2 || !insertionRoot.getDefiningOp<tensor::EmptyOp>())
    return success();

  SmallVector<tensor::ExtractSliceOp, 4> extracts;
  for (Operation *user : generic.getResult(0).getUsers()) {
    auto extract = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!extract)
      return success();
    extracts.push_back(extract);
  }
  if (extracts.size() != insertions.size())
    return success();

  SmallVector<SliceInsertion, 4> slices;
  slices.reserve(insertions.size());
  for (tensor::InsertSliceOp insert : insertions) {
    if (!hasFullyStaticUnitStrideSlice(insert))
      return success();
    auto sourceType = dyn_cast<RankedTensorType>(insert.getSourceType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        sourceType.getRank() != resultType.getRank())
      return success();

    tensor::ExtractSliceOp matchingExtract;
    for (tensor::ExtractSliceOp extract : extracts) {
      if (extract.getResult().getType() == insert.getSource().getType() &&
          hasSameStaticSlice(insert, extract)) {
        if (matchingExtract)
          return success();
        matchingExtract = extract;
      }
    }
    if (!matchingExtract)
      return success();
    slices.push_back({insert, matchingExtract});
  }

  for (auto [index, lhs] : llvm::enumerate(insertions)) {
    for (tensor::InsertSliceOp rhs : llvm::drop_begin(insertions, index + 1)) {
      if (!areDisjoint(lhs, rhs))
        return success();
    }
  }

  for (tensor::ExtractSliceOp extract : extracts) {
    if (llvm::none_of(slices, [&](const SliceInsertion &slice) {
          return slice.extract == extract;
        }))
      return success();
  }

  rewriter.setInsertionPoint(generic);
  Block &sourceBody = generic.getRegion().front();
  auto sourceYield = cast<linalg::YieldOp>(sourceBody.getTerminator());
  for (auto [sliceIndex, slice] : llvm::enumerate(slices)) {
    auto sourceType = cast<RankedTensorType>(slice.op.getSourceType());
    Value empty = rewriter.create<tensor::EmptyOp>(
        generic.getLoc(), sourceType.getShape(), sourceType.getElementType());
    auto distributed = rewriter.create<linalg::GenericOp>(
        generic.getLoc(), sourceType, ValueRange{slice.op.getSource()},
        ValueRange{empty}, generic.getIndexingMapsArray(),
        generic.getIteratorTypesArray(),
        [&](OpBuilder &builder, Location loc, ValueRange arguments) {
          IRMapping mapping;
          for (auto [sourceArgument, targetArgument] :
               llvm::zip_equal(sourceBody.getArguments(), arguments))
            mapping.map(sourceArgument, targetArgument);
          for (Operation &bodyOp : sourceBody.without_terminator())
            builder.clone(bodyOp, mapping);
          builder.create<linalg::YieldOp>(
              loc, mapping.lookupOrDefault(sourceYield.getValues().front()));
        });
    distributed->setAttr("sculptor.optimization.elementwise_slice",
                         rewriter.getI64IntegerAttr(sliceIndex));
    rewriter.replaceOp(slice.extract, distributed.getResult(0));
  }

  Operation *outputInit = generic.getDpsInits().front().getDefiningOp();
  rewriter.eraseOp(generic);

  for (tensor::InsertSliceOp insert : insertions) {
    if (insert->use_empty())
      rewriter.eraseOp(insert);
  }
  if (Operation *root = insertionRoot.getDefiningOp();
      root && root->use_empty())
    rewriter.eraseOp(root);
  if (outputInit && outputInit->use_empty())
    rewriter.eraseOp(outputInit);

  changed = true;
  return success();
}

} // namespace

LogicalResult optimizeElementwiseSlices(ModuleOp module,
                                        func::FuncOp taskGraphFunc,
                                        const TaskGraphDAG &dag,
                                        bool &changed) {
  (void)taskGraphFunc;
  (void)dag;
  changed = false;

  SmallVector<linalg::GenericOp, 8> candidates;
  module.walk(
      [&](linalg::GenericOp generic) { candidates.push_back(generic); });

  IRRewriter rewriter(module.getContext());
  for (linalg::GenericOp candidate : candidates) {
    if (!candidate->getBlock())
      continue;
    if (failed(rewriteElementwiseSlices(candidate, rewriter, changed)))
      return failure();
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
