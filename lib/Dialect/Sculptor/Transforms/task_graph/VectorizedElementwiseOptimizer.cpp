#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/VectorizedElementwiseOptimizer.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <optional>

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

static void rewriteResidualAdd(linalg::GenericOp generic,
                               RankedTensorType resultType,
                               IRRewriter &rewriter) {
  Location loc = generic.getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value vectorWidth =
      rewriter.create<arith::ConstantIndexOp>(loc, kVectorWidth);
  Value dim0 =
      rewriter.create<arith::ConstantIndexOp>(loc, resultType.getDimSize(0));
  Value dim1 =
      rewriter.create<arith::ConstantIndexOp>(loc, resultType.getDimSize(1));
  Value dim2 =
      rewriter.create<arith::ConstantIndexOp>(loc, resultType.getDimSize(2));
  Value padding = rewriter.create<arith::ConstantFloatOp>(
      loc, rewriter.getF32Type(), APFloat(0.0f));
  auto vectorType =
      mlir::VectorType::get({kVectorWidth}, rewriter.getF32Type());
  Value lhs = generic.getDpsInputs()[0];
  Value rhs = generic.getDpsInputs()[1];
  Value initial = generic.getDpsInits()[0];
  std::array<bool, 1> inBoundsStorage{true};
  ArrayRef<bool> inBounds(inBoundsStorage);

  auto batchLoop = rewriter.create<scf::ForOp>(
      loc, zero, dim0, one, ValueRange{initial},
      [&](OpBuilder &batchBuilder, Location batchLoc, Value batch,
          ValueRange batchArguments) {
        auto rowLoop = batchBuilder.create<scf::ForOp>(
            batchLoc, zero, dim1, one, batchArguments,
            [&](OpBuilder &rowBuilder, Location rowLoc, Value row,
                ValueRange rowArguments) {
              auto columnLoop = rowBuilder.create<scf::ForOp>(
                  rowLoc, zero, dim2, vectorWidth, rowArguments,
                  [&](OpBuilder &columnBuilder, Location columnLoc,
                      Value column, ValueRange columnArguments) {
                    SmallVector<Value, 3> indices{batch, row, column};
                    Value left = columnBuilder.create<vector::TransferReadOp>(
                        columnLoc, vectorType, lhs, indices,
                        std::optional<Value>(padding),
                        std::optional<ArrayRef<bool>>(inBounds));
                    Value right = columnBuilder.create<vector::TransferReadOp>(
                        columnLoc, vectorType, rhs, indices,
                        std::optional<Value>(padding),
                        std::optional<ArrayRef<bool>>(inBounds));
                    Value sum = columnBuilder.create<arith::AddFOp>(
                        columnLoc, left, right);
                    Value next =
                        columnBuilder
                            .create<vector::TransferWriteOp>(
                                columnLoc, sum, columnArguments.front(),
                                indices,
                                std::optional<ArrayRef<bool>>(inBounds))
                            .getResult();
                    columnBuilder.create<scf::YieldOp>(columnLoc, next);
                  });
              columnLoop->setAttr("sculptor.optimization.vector_width",
                                  rewriter.getI64IntegerAttr(kVectorWidth));
              rowBuilder.create<scf::YieldOp>(rowLoc, columnLoop.getResult(0));
            });
        batchBuilder.create<scf::YieldOp>(batchLoc, rowLoop.getResult(0));
      });

  rewriter.replaceOp(generic, batchLoop.getResult(0));
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

  IRRewriter rewriter(module.getContext());
  for (linalg::GenericOp candidate : candidates) {
    if (!candidate->getBlock())
      continue;
    FailureOr<RankedTensorType> resultType = getSupportedResultType(candidate);
    if (failed(resultType))
      continue;
    rewriter.setInsertionPoint(candidate);
    rewriteResidualAdd(candidate, *resultType, rewriter);
    changed = true;
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
