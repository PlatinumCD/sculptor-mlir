#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

// CanonicalizeLayers matches supported tensor/linalg layer IR in forward and
// rewrites it to inline sculptor.nn ops.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sculptor {

// Rewrites layer-shaped regions in forward through the registered canonicalizer
// set.
void CanonicalizeLayersPass::runOnOperation() {
  // Build the canonicalizer registry in the order patterns should inspect
  // forward.
  mlir::sculptor::LayerCanonicalizers canonicalizers;
  mlir::sculptor::registerConv1DCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerConv2DCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerConv2DGroupedCanonicalizer(canonicalizers,
                                                   &getContext());
  mlir::sculptor::registerConv3DCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerRNNCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerLSTMCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerGRUCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerLSTMCellCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerGRUCellCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerRNNCellCanonicalizer(canonicalizers, &getContext());
  mlir::sculptor::registerTransformerCanonicalizer(canonicalizers,
                                                   &getContext());
  mlir::sculptor::registerLinearCanonicalizer(canonicalizers, &getContext());

  // Only forward is treated as the source function for layer canonicalization.
  for (mlir::func::FuncOp func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func.getName() != "forward")
      continue;

    // Let each canonicalizer rewrite all matches it owns before the next
    // family runs.
    for (const auto &canonicalizer : canonicalizers) {
      canonicalizer->canonicalize(func);
    }
  }
}

// Registers the layer canonicalization pass with MLIR's global pass registry.
void registerCanonicalizeLayersPass() {
  PassRegistration<CanonicalizeLayersPass>();
}

} // namespace sculptor
} // namespace mlir
