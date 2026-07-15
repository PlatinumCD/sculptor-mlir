#include "sculptor-mlir/Dialect/Sculptor/Transforms/BuildTaskGraphIslands.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"

#include "mlir/Pass/PassRegistry.h"

namespace {

bool returnsTaskGraph(mlir::func::FuncOp func) {
  mlir::FunctionType functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

} // namespace

namespace mlir {
namespace sculptor {

void BuildTaskGraphIslandsPass::runOnOperation() {
  bool foundTaskGraph = false;
  for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;

    auto parsedDag = task_graph::parseTaskGraphDAG(func);
    if (failed(parsedDag)) {
      signalPassFailure();
      return;
    }

    auto executionGraph =
        task_graph::buildTaskExecutionGraph(func, *parsedDag);
    if (failed(executionGraph)) {
      signalPassFailure();
      return;
    }

    auto islandGraph = task_graph::buildLogicalPlacementIslandGraph(
        *parsedDag, *executionGraph);
    if (failed(islandGraph)) {
      func.emitError("failed to build logical placement islands");
      signalPassFailure();
      return;
    }

    if (failed(task_graph::attachLogicalPlacementIslandIds(
            func, *parsedDag, *islandGraph))) {
      signalPassFailure();
      return;
    }

    foundTaskGraph = true;
  }

  if (!foundTaskGraph) {
    getOperation().emitError(
        "expected a task graph function when building placement islands");
    signalPassFailure();
  }
}

void registerBuildTaskGraphIslandsPass() {
  PassRegistration<BuildTaskGraphIslandsPass>();
}

} // namespace sculptor
} // namespace mlir
