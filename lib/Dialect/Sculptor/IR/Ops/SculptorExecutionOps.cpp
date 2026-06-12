#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::sculptor;

namespace {

bool haveMismatchedStaticDims(int64_t lhsDim, int64_t rhsDim) {
  return !ShapedType::isDynamic(lhsDim) && !ShapedType::isDynamic(rhsDim) &&
         lhsDim != rhsDim;
}

LogicalResult verifyRank2F32Tensor(Operation *op, RankedTensorType type,
                                   StringRef tensorName) {
  if (type.getRank() != 2) {
    return op->emitOpError("expected ")
           << tensorName << " tensor rank to be 2";
  }

  if (!llvm::isa<Float32Type>(type.getElementType())) {
    return op->emitOpError("expected ")
           << tensorName << " tensor element type to be f32";
  }

  return success();
}

LogicalResult verifyStaticDimEquals(Operation *op, int64_t dim,
                                    int64_t expectedDim, StringRef dimName) {
  if (ShapedType::isDynamic(dim))
    return success();

  if (dim == expectedDim)
    return success();

  return op->emitOpError("expected ")
         << dimName << " (" << dim << ") to be " << expectedDim;
}

LogicalResult verifyMatchingStaticDims(Operation *op, int64_t lhsDim,
                                       int64_t rhsDim, StringRef lhsName,
                                       StringRef rhsName) {
  if (!haveMismatchedStaticDims(lhsDim, rhsDim))
    return success();

  return op->emitOpError("expected ")
         << lhsName << " (" << lhsDim << ") to match " << rhsName << " ("
         << rhsDim << ")";
}

} // namespace

mlir::LogicalResult mlir::sculptor::MVMOp::verify() {
  RankedTensorType vectorType = getVector().getType();
  RankedTensorType matrixType = getMatrix().getType();
  RankedTensorType resultType = getResult().getType();

  if (failed(verifyRank2F32Tensor(*this, vectorType, "vector")))
    return failure();

  if (failed(verifyRank2F32Tensor(*this, matrixType, "matrix")))
    return failure();

  if (failed(verifyRank2F32Tensor(*this, resultType, "result")))
    return failure();

  if (failed(verifyStaticDimEquals(*this, vectorType.getDimSize(0), 1,
                                   "vector leading dimension")))
    return failure();

  if (failed(verifyStaticDimEquals(*this, resultType.getDimSize(0), 1,
                                   "result leading dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(
          *this, vectorType.getDimSize(1), matrixType.getDimSize(1),
          "vector feature dimension", "matrix input dimension")))
    return failure();

  return verifyMatchingStaticDims(*this, resultType.getDimSize(1),
                                  matrixType.getDimSize(0),
                                  "result output dimension",
                                  "matrix output dimension");
}
