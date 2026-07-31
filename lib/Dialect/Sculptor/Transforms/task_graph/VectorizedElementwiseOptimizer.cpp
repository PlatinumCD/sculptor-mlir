#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/VectorizedElementwiseOptimizer.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

constexpr int64_t kVectorWidth = 8;

static bool isResidualAddFunction(linalg::GenericOp generic) {
  auto function = generic->getParentOfType<func::FuncOp>();
  auto taskKind =
      function ? function->getAttrOfType<StringAttr>("sculptor.task_kind")
               : StringAttr{};
  return taskKind && taskKind.getValue() == "digital.residual_add";
}

static bool hasExactAddBody(linalg::GenericOp generic) {
  if (generic.getRegion().empty())
    return false;
  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 3 || !body.getArgument(2).use_empty())
    return false;

  auto operations = body.without_terminator();
  if (!llvm::hasSingleElement(operations))
    return false;
  auto add = dyn_cast<arith::AddFOp>(&*operations.begin());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!add || !yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != add.getResult())
    return false;

  Value lhs = add.getLhs();
  Value rhs = add.getRhs();
  return (lhs == body.getArgument(0) && rhs == body.getArgument(1)) ||
         (lhs == body.getArgument(1) && rhs == body.getArgument(0));
}

static FailureOr<RankedTensorType>
getSupportedResultType(linalg::GenericOp generic) {
  if (!isResidualAddFunction(generic) || generic.getNumDpsInputs() != 2 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
      !hasExactAddBody(generic))
    return failure();

  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      resultType.getRank() != 3 || !resultType.getElementType().isF32() ||
      resultType.getDimSize(2) % kVectorWidth != 0 ||
      !linalg_match::hasParallelIterators(generic, 3))
    return failure();

  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != 3 || !llvm::all_of(maps, [](AffineMap map) {
        return linalg_match::isIdentityMap(map, 3);
      }))
    return failure();

  if (llvm::any_of(generic.getDpsInputs(),
                   [&](Value input) { return input.getType() != resultType; }))
    return failure();
  return resultType;
}

} // namespace

LogicalResult optimizeVectorizedElementwise(ModuleOp module,
                                            func::FuncOp taskGraphFunc,
                                            const TaskGraphDAG &dag,
                                            bool &changed) {
  (void)taskGraphFunc;
  (void)dag;
  changed = false;

  SmallVector<linalg::GenericOp, 8> candidates;
  module.walk(
      [&](linalg::GenericOp generic) { candidates.push_back(generic); });

  Builder builder(module.getContext());
  for (linalg::GenericOp candidate : candidates) {
    if (!candidate->getBlock() ||
        candidate->hasAttr(kVectorizedElementwiseWidthAttrName))
      continue;
    FailureOr<RankedTensorType> resultType = getSupportedResultType(candidate);
    if (failed(resultType))
      continue;
    candidate->setAttr(kVectorizedElementwiseWidthAttrName,
                       builder.getI64IntegerAttr(kVectorWidth));
    changed = true;
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
