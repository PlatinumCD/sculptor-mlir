#include "sculptor-mlir/Dialect/Sculptor/Conversion/EmitRuntimeGraph.h"

#include "sculptor-mlir/Dialect/Sculptor/Conversion/runtime/RuntimeGraphEmission.h"

#include "llvm/ADT/DenseMap.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sculptor {

namespace {

class EmitRuntimeGraphPassImpl final : public EmitRuntimeGraphPass {};

} // namespace

void EmitRuntimeGraphPass::runOnOperation() {
  ModuleOp module = getOperation();

  auto taskGraphFunc = findTaskGraphFunc(module);
  if (failed(taskGraphFunc)) {
    signalPassFailure();
    return;
  }
  if (!*taskGraphFunc)
    return;

  GraphModel model;
  DenseMap<Value, unsigned> resourceIndexByValue;
  if (failed(collectGraphModel(module, *taskGraphFunc, model,
                               resourceIndexByValue))) {
    signalPassFailure();
    return;
  }

  RuntimeDecls decls = getRuntimeDecls(module);
  auto shims = emitEntryShims(module, model, resourceIndexByValue, decls);
  if (failed(shims)) {
    signalPassFailure();
    return;
  }

  LLVM::LLVMFuncOp graphBuilder =
      emitGraphBuilder(module, model, *shims, decls);
  if (failed(emitPublicWrappers(module, graphBuilder, decls))) {
    signalPassFailure();
    return;
  }

  // The symbolic task graph has been compiled into runtime metadata and
  // cannot participate in the final LLVM translation.
  taskGraphFunc.value()->erase();
}

void registerEmitRuntimeGraphPass() {
  PassRegistration<EmitRuntimeGraphPass>();
}

} // namespace sculptor
} // namespace mlir
