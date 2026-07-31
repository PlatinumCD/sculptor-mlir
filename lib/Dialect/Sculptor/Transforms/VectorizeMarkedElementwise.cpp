#include "sculptor-mlir/Dialect/Sculptor/Transforms/VectorizeMarkedElementwise.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/VectorizedElementwiseOptimizer.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <optional>

namespace {

namespace linalg_match = mlir::sculptor::linalg_match;
namespace task_graph = mlir::sculptor::task_graph;

static bool hasExactAddBody(mlir::linalg::GenericOp generic) {
  if (!generic.getRegion().hasOneBlock())
    return false;
  mlir::Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 3 || !body.getArgument(2).use_empty())
    return false;

  auto operations = body.without_terminator();
  if (!llvm::hasSingleElement(operations))
    return false;
  auto add = mlir::dyn_cast<mlir::arith::AddFOp>(&*operations.begin());
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!add || !yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != add.getResult())
    return false;

  mlir::Value lhs = add.getLhs();
  mlir::Value rhs = add.getRhs();
  return (lhs == body.getArgument(0) && rhs == body.getArgument(1)) ||
         (lhs == body.getArgument(1) && rhs == body.getArgument(0));
}

static mlir::FailureOr<mlir::MemRefType>
getSupportedType(mlir::linalg::GenericOp generic, int64_t vectorWidth) {
  if (vectorWidth <= 0 || generic.getNumDpsInputs() != 2 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 0 ||
      !hasExactAddBody(generic) ||
      !linalg_match::hasParallelIterators(generic, 3))
    return mlir::failure();

  auto outputType =
      mlir::dyn_cast<mlir::MemRefType>(generic.getDpsInits().front().getType());
  if (!outputType || !outputType.hasStaticShape() ||
      outputType.getRank() != 3 || !outputType.getElementType().isF32() ||
      outputType.getDimSize(2) % vectorWidth != 0)
    return mlir::failure();

  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != 3 || !llvm::all_of(maps, [](mlir::AffineMap map) {
        return linalg_match::isIdentityMap(map, 3);
      }))
    return mlir::failure();

  if (llvm::any_of(generic.getDpsInputs(), [&](mlir::Value input) {
        return input.getType() != outputType;
      }))
    return mlir::failure();
  return outputType;
}

static void rewriteAdd(mlir::linalg::GenericOp generic, mlir::MemRefType type,
                       int64_t vectorWidth, mlir::IRRewriter &rewriter) {
  mlir::Location loc = generic.getLoc();
  mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value width =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, vectorWidth);
  mlir::Value dim0 =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, type.getDimSize(0));
  mlir::Value dim1 =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, type.getDimSize(1));
  mlir::Value dim2 =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, type.getDimSize(2));
  mlir::Value padding = rewriter.create<mlir::arith::ConstantFloatOp>(
      loc, rewriter.getF32Type(), llvm::APFloat(0.0f));
  auto vectorType = mlir::VectorType::get({vectorWidth}, rewriter.getF32Type());
  mlir::Value lhs = generic.getDpsInputs()[0];
  mlir::Value rhs = generic.getDpsInputs()[1];
  mlir::Value output = generic.getDpsInits()[0];
  std::array<bool, 1> inBoundsStorage{true};
  llvm::ArrayRef<bool> inBounds(inBoundsStorage);

  auto batchLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, dim0, one);
  rewriter.setInsertionPoint(batchLoop.getBody()->getTerminator());
  auto rowLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, dim1, one);
  rewriter.setInsertionPoint(rowLoop.getBody()->getTerminator());
  auto columnLoop = rewriter.create<mlir::scf::ForOp>(loc, zero, dim2, width);
  rewriter.setInsertionPoint(columnLoop.getBody()->getTerminator());

  llvm::SmallVector<mlir::Value, 3> indices{batchLoop.getInductionVar(),
                                            rowLoop.getInductionVar(),
                                            columnLoop.getInductionVar()};
  mlir::Value left = rewriter.create<mlir::vector::TransferReadOp>(
      loc, vectorType, lhs, indices, std::optional<mlir::Value>(padding),
      std::optional<llvm::ArrayRef<bool>>(inBounds));
  mlir::Value right = rewriter.create<mlir::vector::TransferReadOp>(
      loc, vectorType, rhs, indices, std::optional<mlir::Value>(padding),
      std::optional<llvm::ArrayRef<bool>>(inBounds));
  mlir::Value sum = rewriter.create<mlir::arith::AddFOp>(loc, left, right);
  rewriter.create<mlir::vector::TransferWriteOp>(
      loc, sum, output, indices, std::optional<llvm::ArrayRef<bool>>(inBounds));
  rewriter.eraseOp(generic);
}

} // namespace

namespace mlir {
namespace sculptor {

void VectorizeMarkedElementwisePass::runOnOperation() {
  SmallVector<linalg::GenericOp, 8> candidates;
  getOperation().walk([&](linalg::GenericOp generic) {
    if (generic->hasAttr(task_graph::kVectorizedElementwiseWidthAttrName))
      candidates.push_back(generic);
  });

  bool changed = false;
  IRRewriter rewriter(&getContext());
  for (linalg::GenericOp candidate : candidates) {
    auto widthAttr = candidate->getAttrOfType<IntegerAttr>(
        task_graph::kVectorizedElementwiseWidthAttrName);
    if (!widthAttr) {
      candidate.emitError("expected integer vector-width marker");
      signalPassFailure();
      return;
    }
    int64_t vectorWidth = widthAttr.getInt();
    FailureOr<MemRefType> type = getSupportedType(candidate, vectorWidth);
    if (failed(type)) {
      candidate.emitError(
          "expected a bufferized static rank-3 f32 residual add compatible "
          "with the marked vector width");
      signalPassFailure();
      return;
    }
    rewriter.setInsertionPoint(candidate);
    rewriteAdd(candidate, *type, vectorWidth, rewriter);
    changed = true;
  }

  if (requireChange && !changed) {
    getOperation().emitError("no marked elementwise operation was lowered");
    signalPassFailure();
  }
}

void registerVectorizeMarkedElementwisePass() {
  PassRegistration<VectorizeMarkedElementwisePass>();
}

} // namespace sculptor
} // namespace mlir
