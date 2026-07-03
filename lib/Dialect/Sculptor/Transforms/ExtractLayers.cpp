#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"

// ExtractLayers outlines inline sculptor.nn ops from forward into layer functions
// that later passes can lower independently.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sculptor {

// Outlines layer-shaped regions in forward through the registered extractor
// set.
void ExtractLayersPass::runOnOperation() {

  // Build the extractor registry in the order patterns should inspect forward.
  mlir::sculptor::LayerExtractors extractors;
  mlir::sculptor::registerConv1DExtractor(extractors, &getContext());
  mlir::sculptor::registerConv2DExtractor(extractors, &getContext());
  mlir::sculptor::registerConv2DGroupedExtractor(extractors, &getContext());
  mlir::sculptor::registerConv3DExtractor(extractors, &getContext());
  mlir::sculptor::registerRNNExtractor(extractors, &getContext());
  mlir::sculptor::registerLSTMExtractor(extractors, &getContext());
  mlir::sculptor::registerGRUExtractor(extractors, &getContext());
  mlir::sculptor::registerTransformerExtractor(extractors, &getContext());
  mlir::sculptor::registerRNNCellExtractor(extractors, &getContext());
  mlir::sculptor::registerLSTMCellExtractor(extractors, &getContext());
  mlir::sculptor::registerGRUCellExtractor(extractors, &getContext());
  mlir::sculptor::registerLinearExtractor(extractors, &getContext());

  // Only forward is treated as the source function for layer outlining.
  for (mlir::func::FuncOp func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func.getName() != "forward")
      continue;

    // Let each extractor rewrite all matches it owns before the next family
    // runs.
    for (const auto &extractor : extractors) {
      extractor->extract(func);
    }
  }
}

// Registers the layer extraction pass with MLIR's global pass registry.
void registerExtractLayersPass() { PassRegistration<ExtractLayersPass>(); }

} // namespace sculptor
} // namespace mlir
