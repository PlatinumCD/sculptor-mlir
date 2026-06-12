#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"

// ConvertLayers lowers extracted sculptor.nn layer functions called by forward
// into sculptor.mvm plus standard tensor/linalg/math/scf glue.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

namespace nn_layer_match = mlir::sculptor::nn_layer_match;

namespace mlir {
namespace sculptor {

// Dispatches each extracted sculptor.nn layer called by forward to the converter
// that lowers its layer_type to sculptor.mvm.
void ConvertLayersPass::runOnOperation() {
  // Build the sculptor.nn-to-MVM registry once so calls can dispatch by
  // layer_type.
  mlir::sculptor::LayerToMVMConverters converters;
  mlir::sculptor::LayerToMVMConverterMap converterMap;
  mlir::sculptor::registerLinearConverter(converters, converterMap,
                                        &getContext());
  mlir::sculptor::registerConv1DConverter(converters, converterMap,
                                        &getContext());
  mlir::sculptor::registerConv2DConverter(converters, converterMap,
                                        &getContext());
  mlir::sculptor::registerConv2DGroupedConverter(converters, converterMap,
                                               &getContext());
  mlir::sculptor::registerConv3DConverter(converters, converterMap,
                                        &getContext());
  mlir::sculptor::registerRNNCellConverter(converters, converterMap,
                                         &getContext());
  mlir::sculptor::registerLSTMCellConverter(converters, converterMap,
                                          &getContext());
  mlir::sculptor::registerLSTMConverter(converters, converterMap,
                                      &getContext());
  mlir::sculptor::registerGRUCellConverter(converters, converterMap,
                                         &getContext());
  mlir::sculptor::registerGRUConverter(converters, converterMap,
                                     &getContext());
  mlir::sculptor::registerRNNConverter(converters, converterMap,
                                     &getContext());

  // Only forward owns the executable layer call sequence for this pass.
  for (mlir::func::FuncOp func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func.getName() != "forward")
      continue;

    // Snapshot calls before dispatch. Most converters only mutate the callee,
    // but some layer-level rewrites also replace the forward call itself.
    mlir::SmallVector<mlir::func::CallOp> forwardCalls;
    func.walk([&](mlir::func::CallOp call) { forwardCalls.push_back(call); });

    for (mlir::func::CallOp call : forwardCalls) {
      if (!call || !call->getBlock())
        continue;

      // Resolve the callee to an extracted layer function with usable metadata.
      auto calleeAttr = call->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
      if (!calleeAttr)
        continue;

      auto layerFunc = getOperation().lookupSymbol<mlir::func::FuncOp>(
          calleeAttr.getValue());
      if (!layerFunc)
        continue;

      auto layerType = nn_layer_match::getLayerType(layerFunc);
      if (!layerType)
        continue;

      if (!nn_layer_match::isSculptorLayer(layerFunc))
        continue;

      // Leave unknown layer types untouched so other sculptor.nn-to-MVM
      // converters can be added without changing the traversal contract.
      auto converterIt = converterMap.find(layerType.getValue());
      if (converterIt == converterMap.end())
        continue;

      const mlir::sculptor::LayerToMVMConverter *converter = converterIt->second;
      converter->lowerToMVM(layerFunc);
    }
  }
}

// Registers the sculptor.nn-to-MVM lowering pass with MLIR's global pass registry.
void registerConvertLayersPass() { PassRegistration<ConvertLayersPass>(); }

} // namespace sculptor
} // namespace mlir
