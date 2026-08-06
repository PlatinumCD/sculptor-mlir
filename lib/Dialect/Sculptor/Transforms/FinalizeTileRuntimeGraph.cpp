#include "sculptor-mlir/Dialect/Sculptor/Transforms/FinalizeTileRuntimeGraph.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeLayout.h"

#include "mlir/Pass/PassRegistry.h"

namespace {

static bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

} // namespace

namespace mlir {
namespace sculptor {

void FinalizeTileRuntimeGraphPass::runOnOperation() {
  ModuleOp module = getOperation();
  func::FuncOp taskGraph;

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    if (taskGraph) {
      func.emitError("expected exactly one tile runtime graph");
      signalPassFailure();
      return;
    }
    taskGraph = func;
  }

  if (!taskGraph) {
    module.emitError("expected a tile runtime graph returning "
                     "!sculptor.task_graph");
    signalPassFailure();
    return;
  }

  if (failed(rebuildTileRuntimeLayout(taskGraph))) {
    taskGraph.emitError("failed to finalize tile runtime layout");
    signalPassFailure();
  }
}

void registerFinalizeTileRuntimeGraphPass() {
  PassRegistration<FinalizeTileRuntimeGraphPass>();
}

} // namespace sculptor
} // namespace mlir
