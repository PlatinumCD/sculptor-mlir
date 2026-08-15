#include "sculptor-mlir/Dialect/Sculptor/Conversion/golem/GolemUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/GolemTilingAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace {

mlir::FailureOr<int64_t> getPhysicalArrayRows(mlir::sculptor::ArrayStoreOp op) {
  auto shape = op->getAttrOfType<mlir::ArrayAttr>(
      mlir::sculptor::golem_tiling_attrs::kTilePhysicalShapeAttrName);
  if (!shape) {
    return op.emitOpError("expected physical array shape attribute '")
               << mlir::sculptor::golem_tiling_attrs::kTilePhysicalShapeAttrName
               << "'",
           mlir::failure();
  }
  if (shape.size() != 2) {
    return op.emitOpError("expected physical array shape to contain two "
                          "integer dimensions"),
           mlir::failure();
  }

  auto rows = mlir::dyn_cast<mlir::IntegerAttr>(shape[0]);
  auto cols = mlir::dyn_cast<mlir::IntegerAttr>(shape[1]);
  if (!rows || !cols) {
    return op.emitOpError("expected physical array shape to contain two "
                          "integer dimensions"),
           mlir::failure();
  }
  return rows.getInt();
}

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
      return rewriter.notifyMatchFailure(op,
                                         "expected static output tensor<1xn>");
    }

    mlir::FailureOr<mlir::Value> localArrayId =
        mlir::sculptor::golem::buildLocalArrayId(rewriter, op);
    if (mlir::failed(localArrayId))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    mlir::Type elementType = outputType.getElementType();
    int64_t validRows = outputType.getDimSize(1);
    mlir::FailureOr<int64_t> physicalRows = getPhysicalArrayRows(op);
    if (mlir::failed(physicalRows))
      return mlir::failure();
    if (*physicalRows <= 0)
      return op.emitOpError("expected positive physical array row count");
    if (validRows <= 0 || validRows > *physicalRows) {
      return op.emitOpError(
          "logical store width must be between one and physical array rows");
    }

    mlir::Value scratch = mlir::sculptor::golem::allocateStoreScratchBuffer(
        rewriter, loc, *physicalRows, elementType);
    // Mark the backing allocation as well as the logical prefix below. A
    // full-width subview can be folded during canonicalization, while the
    // allocation remains the authoritative origin of the ISA store result.
    scratch.getDefiningOp()->setAttr(
        "sculptor.memory.analog_store_valid_prefix", rewriter.getUnitAttr());

    auto shimScratchType = mlir::MemRefType::get({mlir::ShapedType::kDynamic,
                                                  mlir::ShapedType::kDynamic,
                                                  mlir::ShapedType::kDynamic},
                                                 elementType);
    mlir::Value shimScratch =
        rewriter.create<mlir::memref::CastOp>(loc, shimScratchType, scratch);

    mlir::sculptor::golem::emitShimCall(rewriter, loc,
                                        mlir::sculptor::golem::kStoreShimName,
                                        {shimScratch, *localArrayId});

    // The ISA store has no length operand and must retain its full physical
    // scratch buffer.  The logical result is its valid prefix, however, so
    // expose that prefix as an alias instead of allocating a second buffer and
    // copying every valid lane in a scalar loop.
    auto index = [&](int64_t value) -> mlir::OpFoldResult {
      return rewriter.getIndexAttr(value);
    };
    mlir::Value validPhysicalRows = rewriter.create<mlir::memref::SubViewOp>(
        loc, scratch,
        mlir::SmallVector<mlir::OpFoldResult>{index(0), index(0), index(0)},
        mlir::SmallVector<mlir::OpFoldResult>{index(1), index(1),
                                              index(validRows)},
        mlir::SmallVector<mlir::OpFoldResult>{index(1), index(1), index(1)});
    auto outputMemrefType = mlir::MemRefType::get(
        outputType.getShape(), elementType,
        mlir::StridedLayoutAttr::get(rewriter.getContext(), 0,
                                     {*physicalRows, 1}));
    mlir::Value outputMemref = rewriter.create<mlir::memref::CollapseShapeOp>(
        loc, outputMemrefType, validPhysicalRows,
        mlir::ArrayRef<mlir::ReassociationIndices>{{0, 1}, {2}});
    outputMemref.getDefiningOp()->setAttr(
        "sculptor.memory.analog_store_valid_prefix", rewriter.getUnitAttr());
    auto tensor = rewriter.create<mlir::bufferization::ToTensorOp>(
        loc, outputType, outputMemref, /*restrict=*/true, /*writable=*/true);
    tensor->setAttr("sculptor.memory.direct_physical_store_view",
                    rewriter.getUnitAttr());
    if (auto function = op->getParentOfType<mlir::func::FuncOp>()) {
      constexpr llvm::StringLiteral marker =
          "sculptor.memory.direct_physical_store_view_count";
      int64_t count = 0;
      if (auto current = function->getAttrOfType<mlir::IntegerAttr>(marker))
        count = current.getInt();
      function->setAttr(marker, rewriter.getI64IntegerAttr(count + 1));
    }
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
