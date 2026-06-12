#include "sculptor-mlir/Dialect/Sculptor/Conversion/EmitRuntimeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/FinalizeGolemIntrinsics.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/LowerGolemToLLVMShims.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/Passes.h"

namespace mlir {
namespace sculptor {

// Registers the conversion pass bundle exposed by this library entry point.
void registerSculptorConversionPasses() {
  registerLowerGolemToLLVMShimsPass();
  registerFinalizeGolemIntrinsicsPass();
  registerEmitRuntimeGraphPass();
}

} // namespace sculptor
} // namespace mlir
