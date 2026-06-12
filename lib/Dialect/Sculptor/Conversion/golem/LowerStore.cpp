#include "sculptor-mlir/Dialect/Sculptor/Conversion/golem/GolemUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace {

class ArrayStoreLowering
    : public mlir::OpConversionPattern<mlir::sculptor::ArrayStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::ArrayStoreOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    (void)adaptor;
    auto outputType =
        llvm::dyn_cast<mlir::RankedTensorType>(op.getOutput().getType());
    if (!outputType || outputType.getRank() != 2 ||
        !outputType.hasStaticShape() || outputType.getDimSize(0) != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected static output tensor<1xn>");
    }

    mlir::FailureOr<mlir::Value> localArrayId =
        mlir::sculptor::golem::buildLocalArrayId(rewriter, op);
    if (mlir::failed(localArrayId))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    mlir::Type elementType = outputType.getElementType();
    int64_t lanes = outputType.getDimSize(1);

    auto outputMemrefType =
        mlir::MemRefType::get(outputType.getShape(), elementType);
    mlir::Value outputMemref =
        rewriter.create<mlir::memref::AllocOp>(loc, outputMemrefType);
    mlir::Value scratch = mlir::sculptor::golem::allocateStoreScratchBuffer(
        rewriter, loc, lanes, elementType);

    mlir::sculptor::golem::emitShimCall(
        rewriter, loc, mlir::sculptor::golem::kStoreShimName,
        {scratch, *localArrayId});

    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value cLanes =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, lanes);
    rewriter.create<mlir::scf::ForOp>(
        loc, c0, cLanes, c1, mlir::ValueRange{},
        [&](mlir::OpBuilder &builder, mlir::Location loopLoc,
            mlir::Value laneIndex, mlir::ValueRange) {
          mlir::Value value = builder.create<mlir::memref::LoadOp>(
              loopLoc, scratch, mlir::ValueRange{c0, c0, laneIndex});
          builder.create<mlir::memref::StoreOp>(
              loopLoc, value, outputMemref, mlir::ValueRange{c0, laneIndex});
          builder.create<mlir::scf::YieldOp>(loopLoc);
        });
    rewriter.create<mlir::memref::DeallocOp>(loc, scratch);

    auto tensor = rewriter.create<mlir::bufferization::ToTensorOp>(
        loc, outputType, outputMemref, /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, tensor.getResult());
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace golem {

// Registers store lowering for Golem-to-shim conversion.
void populateLowerStorePatterns(RewritePatternSet &patterns,
                                TypeConverter &typeConverter,
                                MLIRContext *ctx) {
  patterns.add<ArrayStoreLowering>(typeConverter, ctx);
}

} // namespace golem
} // namespace sculptor
} // namespace mlir
