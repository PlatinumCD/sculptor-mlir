#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_CONVERTSCULPTORTOGOLEMBACKEND_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_CONVERTSCULPTORTOGOLEMBACKEND_H

#include "sculptor-mlir/Dialect/Sculptor/Conversion/LowerGolemToLLVMShims.h"

namespace mlir {
namespace sculptor {

// Compatibility alias for the Sculptor Golem -> LLVM shim lowering pass. The
// sculptor.mvm -> Golem execution expansion is a separate transform stage.
using ConvertSculptorToGolemBackendPass = LowerGolemToLLVMShimsPass;

inline void registerConvertSculptorToGolemBackendPass() {
  registerLowerGolemToLLVMShimsPass();
}

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_CONVERTSCULPTORTOGOLEMBACKEND_H
