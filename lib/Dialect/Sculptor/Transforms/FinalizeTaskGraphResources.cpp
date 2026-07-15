#include "sculptor-mlir/Dialect/Sculptor/Transforms/FinalizeTaskGraphResources.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphExecutionPlan.h"

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

void FinalizeTaskGraphResourcesPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool foundTaskGraph = false;

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    foundTaskGraph = true;

    if (failed(rebuildTaskGraphExecutionPlan(func))) {
      func.emitError("failed to finalize task graph resources");
      signalPassFailure();
      return;
    }
  }

  if (!foundTaskGraph) {
    module.emitError("expected at least one task graph function returning "
                     "!sculptor.task_graph");
    signalPassFailure();
  }
}

void registerFinalizeTaskGraphResourcesPass() {
  PassRegistration<FinalizeTaskGraphResourcesPass>();
}

} // namespace sculptor
} // namespace mlir
