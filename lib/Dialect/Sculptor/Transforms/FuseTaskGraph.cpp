#include "sculptor-mlir/Dialect/Sculptor/Transforms/FuseTaskGraph.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphCleanup.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphRoutineFuser.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_graph = mlir::sculptor::task_graph;

static bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

static void
refreshStructuralMetadata(mlir::func::FuncOp func,
                          const task_graph::TaskGraphDAG &dag) {
  mlir::Builder builder(func.getContext());
  func->setAttr(
      schedule_attrs::kTaskCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag.nodes.size())));
  func->setAttr(
      schedule_attrs::kDependencyCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag.dependencyCount)));
}

} // namespace

namespace mlir {
namespace sculptor {

void FuseTaskGraphPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool foundTaskGraph = false;

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    foundTaskGraph = true;

    auto dag = task_graph::parseTaskGraphDAG(func);
    if (failed(dag) ||
        failed(task_graph::fuseTaskGraphRoutines(module, func, *dag)) ||
        failed(
            task_graph::eraseUnusedTaskGraphIntermediateResources(func))) {
      func.emitError("failed to fuse scheduled task graph");
      signalPassFailure();
      return;
    }

    auto fusedDag = task_graph::parseTaskGraphDAG(func);
    if (failed(fusedDag)) {
      signalPassFailure();
      return;
    }
    refreshStructuralMetadata(func, *fusedDag);
  }

  if (!foundTaskGraph) {
    module.emitError("expected at least one task graph function returning "
                     "!sculptor.task_graph");
    signalPassFailure();
    return;
  }

  task_graph::eraseUnusedTaskCallees(module);
}

void registerFuseTaskGraphPass() { PassRegistration<FuseTaskGraphPass>(); }

} // namespace sculptor
} // namespace mlir
