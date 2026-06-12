#include "sculptor-mlir/Dialect/Sculptor/Conversion/golem/GolemUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace {

class ArrayExecuteLowering
    : public mlir::OpConversionPattern<mlir::sculptor::ArrayExecuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::sculptor::ArrayExecuteOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    (void)adaptor;
    mlir::FailureOr<mlir::Value> localArrayId =
        mlir::sculptor::golem::buildLocalArrayId(rewriter, op);
    if (mlir::failed(localArrayId))
      return mlir::failure();

    mlir::sculptor::golem::emitShimCall(
        rewriter, op.getLoc(), mlir::sculptor::golem::kComputeShimName,
        {*localArrayId});
    rewriter.replaceOpWithMultiple(op, llvm::ArrayRef<mlir::ValueRange>{
                                           mlir::ValueRange{}});
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace golem {

// Registers the array execution pattern for Golem-to-shim conversion.
void populateLowerExecutePatterns(RewritePatternSet &patterns,
                                  TypeConverter &typeConverter,
                                  MLIRContext *ctx) {
  patterns.add<ArrayExecuteLowering>(typeConverter, ctx);
}

} // namespace golem
} // namespace sculptor
} // namespace mlir
