#include "sculptor-mlir/Dialect/Sculptor/Conversion/golem/GolemUtils.h"

namespace {

// Erases vector materialization once the type converter carries vector storage.
class VectorFromTensorLowering
    : public mlir::OpConversionPattern<mlir::sculptor::VectorFromTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Replaces the analog wrapper with the already-converted tensor input.
  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::VectorFromTensorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

// Erases slice partition views after their tiling metadata has been consumed.
class VectorPartitionLowering
    : public mlir::OpConversionPattern<mlir::sculptor::VectorPartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Keeps the lowered vector value flowing where the slice wrapper was used.
  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::VectorPartitionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getVector());
    return mlir::success();
  }
};

// Lowers one vector-slice placement into scratch preparation and a load shim.
class ArrayVectorPlaceLowering
    : public mlir::OpConversionPattern<mlir::sculptor::ArrayVectorPlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Materializes the selected slice for the runtime and removes the place op.
  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::ArrayVectorPlaceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Validate both the Analog slice contract and the lowered vector shape.
    auto sliceType =
        llvm::dyn_cast<mlir::sculptor::VectorSliceType>(op.getInput().getType());
    if (!sliceType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected sculptor.vector.slice input type");
    }

    auto vectorType =
        llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!vectorType || vectorType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "expected lowered vector tensor<1xn>");
    }

    // Build dynamic offsets and clamped copy bounds for the selected slice.
    mlir::Value fullMemref = mlir::sculptor::golem::materializeTensorMemref(
        rewriter, op.getLoc(), adaptor.getInput());
    auto plan = mlir::sculptor::golem::buildVectorPlacementPlan(
        rewriter, op, adaptor.getSliceIndex(), fullMemref, sliceType);

    // Prepare a zero-padded scratch row before copying the live source region.
    auto maybeScratch = mlir::sculptor::golem::allocateZeroedScratchTile(
        rewriter, op, {1, plan.arrayCols}, vectorType.getElementType(), plan.c1,
        plan.cArrayCols, plan.c0, plan.c1);
    if (mlir::failed(maybeScratch))
      return mlir::failure();

    mlir::Value arrayMemref = *maybeScratch;
    mlir::sculptor::golem::copyVectorSliceIntoScratch(
        rewriter, op.getLoc(), fullMemref, arrayMemref, plan.colOffset,
        plan.copyCols, plan.c0, plan.c1);

    // Prefer explicit hardware coordinates, falling back to the slice column.
    mlir::Value row = plan.c0;
    mlir::Value col = adaptor.getSliceIndex();
    if (adaptor.getIndices().size() >= 2) {
      row = adaptor.getIndices()[0];
      col = adaptor.getIndices()[1];
    }

    // Encode the grid coordinate expected by the runtime and emit the load shim.
    mlir::Value arrayId = mlir::sculptor::golem::buildLinearArrayId(
        rewriter, op.getLoc(), row, col, plan.gridCols);
    mlir::sculptor::golem::emitShimCall(
        rewriter, op.getLoc(), mlir::sculptor::golem::kLoadShimName,
        {arrayMemref, arrayId});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// Lowers logical vector loading into the stable Golem load shim.
class ArrayLoadLowering
    : public mlir::OpConversionPattern<mlir::sculptor::ArrayLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::ArrayLoadOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Value vector = adaptor.getVector().front();
    auto vectorType =
        llvm::dyn_cast<mlir::RankedTensorType>(vector.getType());
    if (!vectorType || vectorType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "expected vector tensor<1xn>");
    }

    mlir::FailureOr<mlir::Value> localArrayId =
        mlir::sculptor::golem::buildLocalArrayId(rewriter, op);
    if (mlir::failed(localArrayId))
      return mlir::failure();

    mlir::Value vectorMemref = mlir::sculptor::golem::materializeTensorMemref(
        rewriter, op.getLoc(), vector);
    mlir::sculptor::golem::emitShimCall(
        rewriter, op.getLoc(), mlir::sculptor::golem::kLoadShimName,
        {vectorMemref, *localArrayId});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace golem {

// Registers vector wrapper erasure and placement patterns for shim lowering.
void populateLowerVectorPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext *ctx) {
  patterns.add<VectorFromTensorLowering, VectorPartitionLowering,
               ArrayVectorPlaceLowering, ArrayLoadLowering>(typeConverter, ctx);
}

} // namespace golem
} // namespace sculptor
} // namespace mlir
