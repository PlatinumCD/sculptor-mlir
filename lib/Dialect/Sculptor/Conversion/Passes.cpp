#include "sculptor-mlir/Dialect/Sculptor/Conversion/Passes.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/EmitGolemTileABI.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/FinalizeGolemIntrinsics.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/LowerGolemToLLVMShims.h"

namespace mlir {
namespace sculptor {

// Registers the conversion pass bundle exposed by this library entry point.
void registerSculptorConversionPasses() {
  registerLowerGolemToLLVMShimsPass();
  registerFinalizeGolemIntrinsicsPass();
  registerEmitGolemTileABIPass();
}

} // namespace sculptor
} // namespace mlir
