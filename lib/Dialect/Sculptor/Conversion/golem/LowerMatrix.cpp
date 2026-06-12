#include "sculptor-mlir/Dialect/Sculptor/Conversion/golem/GolemUtils.h"

namespace {

// Erases matrix materialization once the type converter carries matrix storage.
class MatrixFromTensorLowering
    : public mlir::OpConversionPattern<mlir::sculptor::MatrixFromTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Replaces the analog wrapper with the already-converted tensor input.
  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::MatrixFromTensorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

// Erases partition views after their grid shape has been captured by the type.
class MatrixPartitionLowering
    : public mlir::OpConversionPattern<mlir::sculptor::MatrixPartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Keeps the lowered matrix value flowing where the grid wrapper was used.
  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::MatrixPartitionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getMatrix());
    return mlir::success();
  }
};

// Lowers one matrix-grid placement into scratch preparation and a set shim.
class ArrayMatrixPlaceLowering
    : public mlir::OpConversionPattern<mlir::sculptor::ArrayMatrixPlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Materializes the selected tile for the runtime and removes the place op.
  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::ArrayMatrixPlaceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Validate both the Analog grid contract and the lowered matrix shape.
    auto gridType =
        llvm::dyn_cast<mlir::sculptor::MatrixGridType>(op.getInput().getType());
    if (!gridType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected sculptor.matrix.grid input type");
    }

    auto matrixType =
        llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!matrixType || matrixType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "expected lowered matrix tensor<mxn>");
    }

    // Build dynamic offsets and clamped copy bounds for the selected array tile.
    mlir::Value fullMemref = mlir::sculptor::golem::materializeTensorMemref(
        rewriter, op.getLoc(), adaptor.getInput());
    auto plan = mlir::sculptor::golem::buildMatrixPlacementPlan(
        rewriter, op, adaptor.getRowIndex(), adaptor.getColIndex(), fullMemref,
        gridType);

    // Prepare a zero-padded scratch tile before copying the live source region.
    auto maybeScratch = mlir::sculptor::golem::allocateZeroedScratchTile(
        rewriter, op, {plan.arrayRows, plan.arrayCols},
        matrixType.getElementType(), plan.cArrayRows, plan.cArrayCols, plan.c0,
        plan.c1);
    if (mlir::failed(maybeScratch))
      return mlir::failure();

    mlir::Value arrayMemref = *maybeScratch;
    mlir::sculptor::golem::copyMatrixTileIntoScratch(
        rewriter, op.getLoc(), fullMemref, arrayMemref, plan.rowOffset,
        plan.colOffset, plan.copyRows, plan.copyCols, plan.c0, plan.c1);

    // Encode the grid coordinate expected by the runtime and emit the set shim.
    mlir::Value arrayId = mlir::sculptor::golem::buildLinearArrayId(
        rewriter, op.getLoc(), adaptor.getRowIndex(), adaptor.getColIndex(),
        plan.gridCols);
    mlir::sculptor::golem::emitShimCall(
        rewriter, op.getLoc(), mlir::sculptor::golem::kSetShimName,
        {arrayMemref, arrayId});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// Lowers logical array setup into the stable Golem set shim.
class ArraySetLowering
    : public mlir::OpConversionPattern<mlir::sculptor::ArraySetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::ArraySetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto matrixType =
        llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getMatrix().getType());
    if (!matrixType || matrixType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "expected matrix tensor<mxn>");
    }

    mlir::FailureOr<mlir::Value> localArrayId =
        mlir::sculptor::golem::buildLocalArrayId(rewriter, op);
    if (mlir::failed(localArrayId))
      return mlir::failure();

    mlir::Value matrixMemref = mlir::sculptor::golem::materializeTensorMemref(
        rewriter, op.getLoc(), adaptor.getMatrix());
    mlir::sculptor::golem::emitShimCall(
        rewriter, op.getLoc(), mlir::sculptor::golem::kSetShimName,
        {matrixMemref, *localArrayId});

    rewriter.replaceOpWithMultiple(op, llvm::ArrayRef<mlir::ValueRange>{
                                           mlir::ValueRange{}});
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace golem {

// Registers matrix wrapper erasure and placement patterns for shim lowering.
void populateLowerMatrixPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext *ctx) {
  patterns.add<MatrixFromTensorLowering, MatrixPartitionLowering,
               ArrayMatrixPlaceLowering, ArraySetLowering>(typeConverter, ctx);
}

} // namespace golem
} // namespace sculptor
} // namespace mlir
