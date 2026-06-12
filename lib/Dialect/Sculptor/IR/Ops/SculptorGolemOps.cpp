#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::sculptor;

namespace {

LogicalResult verifyStaticRank2F32Tensor(Operation *op, RankedTensorType type,
                                        StringRef tensorName) {
  if (type.getRank() != 2) {
    return op->emitOpError("expected ")
           << tensorName << " tensor rank to be 2";
  }

  if (!type.hasStaticShape()) {
    return op->emitOpError("expected ")
           << tensorName << " tensor shape to be static";
  }

  if (!llvm::isa<Float32Type>(type.getElementType())) {
    return op->emitOpError("expected ")
           << tensorName << " tensor element type to be f32";
  }

  return success();
}

LogicalResult verifyLeadingDimEquals(Operation *op, RankedTensorType type,
                                     int64_t expectedDim,
                                     StringRef tensorName) {
  int64_t leadingDim = type.getDimSize(0);
  if (leadingDim == expectedDim)
    return success();

  return op->emitOpError("expected ")
         << tensorName << " leading dimension (" << leadingDim << ") to be "
         << expectedDim;
}

} // namespace

mlir::LogicalResult mlir::sculptor::ArraySetOp::verify() {
  RankedTensorType matrixType = getMatrix().getType();
  return verifyStaticRank2F32Tensor(*this, matrixType, "matrix");
}

mlir::LogicalResult mlir::sculptor::ArrayLoadOp::verify() {
  RankedTensorType vectorType = getVector().getType();

  if (failed(verifyStaticRank2F32Tensor(*this, vectorType, "vector")))
    return failure();

  return verifyLeadingDimEquals(*this, vectorType, 1, "vector");
}

mlir::LogicalResult mlir::sculptor::ArrayStoreOp::verify() {
  RankedTensorType outputType = getOutput().getType();

  if (failed(verifyStaticRank2F32Tensor(*this, outputType, "output")))
    return failure();

  return verifyLeadingDimEquals(*this, outputType, 1, "output");
}
