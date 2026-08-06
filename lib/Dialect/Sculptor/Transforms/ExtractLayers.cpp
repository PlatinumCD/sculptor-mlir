#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sculptor {

// The pivot pipeline preserves canonical layer operations inline. This pass
// remains as a stable checkpoint while ConvertLayers owns decomposition.
void ExtractLayersPass::runOnOperation() {}

// Registers the layer extraction pass with MLIR's global pass registry.
void registerExtractLayersPass() { PassRegistration<ExtractLayersPass>(); }

} // namespace sculptor
} // namespace mlir
