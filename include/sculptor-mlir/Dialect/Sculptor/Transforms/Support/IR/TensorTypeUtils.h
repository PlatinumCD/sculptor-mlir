#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_TENSORTYPEUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_TENSORTYPEUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace tensor_type {

inline FailureOr<RankedTensorType> getStaticF32Tensor(Type type,
                                                      int64_t expectedRank) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(type);
  if (!tensorTy || !tensorTy.hasStaticShape() ||
      tensorTy.getRank() != expectedRank || !tensorTy.getElementType().isF32())
    return failure();

  return tensorTy;
}

inline FailureOr<RankedTensorType>
getPositiveStaticF32Tensor(Type type, int64_t expectedRank) {
  auto tensorTy = getStaticF32Tensor(type, expectedRank);
  if (failed(tensorTy))
    return failure();

  if (llvm::any_of((*tensorTy).getShape(),
                   [](int64_t dim) { return dim <= 0; }))
    return failure();

  return *tensorTy;
}

inline FailureOr<RankedTensorType> getStaticRank1F32Tensor(Type type) {
  return getStaticF32Tensor(type, /*expectedRank=*/1);
}

inline FailureOr<RankedTensorType> getStaticRank2F32Tensor(Type type) {
  return getStaticF32Tensor(type, /*expectedRank=*/2);
}

inline FailureOr<RankedTensorType> getStaticRank3F32Tensor(Type type) {
  return getStaticF32Tensor(type, /*expectedRank=*/3);
}

inline FailureOr<RankedTensorType> getPositiveStaticRank1F32Tensor(Type type) {
  return getPositiveStaticF32Tensor(type, /*expectedRank=*/1);
}

inline FailureOr<RankedTensorType> getPositiveStaticRank2F32Tensor(Type type) {
  return getPositiveStaticF32Tensor(type, /*expectedRank=*/2);
}

inline bool hasStaticF32Shape(Value value, ArrayRef<int64_t> shape) {
  auto tensorTy =
      getStaticF32Tensor(value.getType(), static_cast<int64_t>(shape.size()));
  return succeeded(tensorTy) && tensorTy->getShape() == shape;
}

} // namespace tensor_type
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_TENSORTYPEUTILS_H
