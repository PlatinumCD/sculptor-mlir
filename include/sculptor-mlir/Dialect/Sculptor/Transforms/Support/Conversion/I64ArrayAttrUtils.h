#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_I64ARRAYATTRUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_I64ARRAYATTRUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cstddef>
#include <cstdint>

namespace mlir {
namespace sculptor {
namespace i64_array_attr {

// Reads an i64 array attribute into a plain vector for converter validation.
inline bool extract(ArrayAttr attr, size_t expectedSize,
                    SmallVectorImpl<int64_t> &values) {
  values.clear();
  if (!attr || attr.size() != expectedSize)
    return false;

  values.reserve(expectedSize);
  for (Attribute value : attr) {
    auto integer = llvm::dyn_cast<IntegerAttr>(value);
    if (!integer)
      return false;
    values.push_back(integer.getInt());
  }

  return true;
}

inline bool allEqual(ArrayRef<int64_t> values, int64_t expected) {
  return llvm::all_of(values,
                      [expected](int64_t value) { return value == expected; });
}

inline bool allPositive(ArrayRef<int64_t> values) {
  return llvm::all_of(values, [](int64_t value) { return value > 0; });
}

} // namespace i64_array_attr
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_I64ARRAYATTRUTILS_H
