#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"

// ConvertLayers performs semantic decomposition only. It preserves
// sculptor.mvm as the backend-neutral analog execution primitive, keeps
// computation inline, and does not choose tasks, physical resources, or
// placement.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sculptor {

// Decomposes supported inline semantic operations without outlining them or
// erasing the sculptor.mvm boundary.
void ConvertLayersPass::runOnOperation() {
  for (mlir::func::FuncOp func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (mlir::failed(decomposeInlineLinearLayers(func)) ||
        mlir::failed(decomposeInlineConv1DLayers(func)) ||
        mlir::failed(decomposeInlineConv2DLayers(func)) ||
        mlir::failed(decomposeInlineGroupedConv2DLayers(func)) ||
        mlir::failed(decomposeInlineConv3DLayers(func)) ||
        mlir::failed(decomposeInlineRNNCellLayers(func)) ||
        mlir::failed(decomposeInlineRNNLayers(func)) ||
        mlir::failed(decomposeInlineGRUCellLayers(func)) ||
        mlir::failed(decomposeInlineGRULayers(func)) ||
        mlir::failed(decomposeInlineLSTMCellLayers(func)) ||
        mlir::failed(decomposeInlineLSTMLayers(func)) ||
        mlir::failed(decomposeInlineTransformerStacks(func)) ||
        mlir::failed(decomposeInlineTransformerBlocks(func))) {
      signalPassFailure();
      return;
    }
  }
}

// Registers the sculptor.nn-to-MVM lowering pass with MLIR's global pass
// registry.
void registerConvertLayersPass() { PassRegistration<ConvertLayersPass>(); }

} // namespace sculptor
} // namespace mlir
