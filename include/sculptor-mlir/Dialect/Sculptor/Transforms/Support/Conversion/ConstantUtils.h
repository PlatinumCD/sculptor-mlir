#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_CONSTANTUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_CONSTANTUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace sculptor {
namespace converter_constant {

inline bool isResourceBackedF32Constant(arith::ConstantOp constant) {
  return constant &&
         llvm::isa<DenseF32ResourceElementsAttr>(constant.getValue());
}

inline FailureOr<llvm::SmallVector<float>>
getF32ConstantValues(arith::ConstantOp constant) {
  if (!constant)
    return failure();

  if (auto denseAttr = llvm::dyn_cast<DenseFPElementsAttr>(constant.getValue())) {
    llvm::SmallVector<float> values;
    values.reserve(denseAttr.getNumElements());
    for (const llvm::APFloat &value : denseAttr.getValues<llvm::APFloat>())
      values.push_back(value.convertToFloat());
    return values;
  }

  if (auto denseResourceAttr =
          llvm::dyn_cast<DenseF32ResourceElementsAttr>(constant.getValue())) {
    std::optional<llvm::ArrayRef<float>> values =
        denseResourceAttr.tryGetAsArrayRef();
    if (!values)
      return failure();
    return llvm::SmallVector<float>(values->begin(), values->end());
  }

  return failure();
}

inline TypedAttr buildF32ElementsAttr(RankedTensorType type,
                                      llvm::ArrayRef<float> values,
                                      llvm::StringRef resourcePrefix,
                                      bool useResource) {
  if (static_cast<int64_t>(values.size()) != type.getNumElements())
    return {};

  if (useResource) {
    static uint64_t nextResourceId = 0;
    std::string resourceName =
        resourcePrefix.str() + std::to_string(nextResourceId++);
    auto blob = HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
        values, /*dataIsMutable=*/false);
    return llvm::cast<TypedAttr>(DenseF32ResourceElementsAttr::get(
        type, resourceName, std::move(blob)));
  }

  return llvm::cast<TypedAttr>(DenseElementsAttr::get(type, values));
}

inline TypedAttr reshapeDenseOrResourceAttr(arith::ConstantOp constant,
                                            RankedTensorType targetType) {
  if (!constant)
    return {};

  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(constant.getValue()))
    return denseAttr.reshape(targetType);

  if (auto resourceAttr =
          llvm::dyn_cast<DenseResourceElementsAttr>(constant.getValue()))
    return DenseResourceElementsAttr::get(targetType,
                                          resourceAttr.getRawHandle());

  return {};
}

} // namespace converter_constant
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_CONSTANTUTILS_H
