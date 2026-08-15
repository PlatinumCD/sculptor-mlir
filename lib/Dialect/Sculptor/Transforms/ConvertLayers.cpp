#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/SemanticLayerIdentity.h"

// ConvertLayers performs semantic decomposition only. It preserves
// sculptor.mvm as the backend-neutral analog execution primitive, keeps
// computation inline, and does not choose tasks, physical resources, or
// placement.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sculptor {

namespace {

constexpr llvm::StringLiteral kCanonicalLayerPrefix = "sculptor.nn.";

// Gives every canonical layer a module-unique identity based only on stable IR
// walk order. Re-running this pass over the same canonical input therefore
// produces the same IDs without relying on source names or pointer identity.
void assignSemanticLayerIdentities(ModuleOp module) {
  int64_t nextLayerId = 0;
  module.walk([&](Operation *operation) {
    llvm::StringRef operationName = operation->getName().getStringRef();
    if (!operationName.starts_with(kCanonicalLayerPrefix))
      return;

    operation->setAttr(kSemanticLayerIdAttrName,
                       IntegerAttr::get(IntegerType::get(module.getContext(),
                                                         /*width=*/64),
                                        nextLayerId++));
    operation->setAttr(
        kSemanticLayerKindAttrName,
        StringAttr::get(module.getContext(),
                        operationName.drop_front(kCanonicalLayerPrefix.size())));
  });
}

} // namespace

// Decomposes supported inline semantic operations without outlining them or
// erasing the sculptor.mvm boundary.
void ConvertLayersPass::runOnOperation() {
  assignSemanticLayerIdentities(getOperation());

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
