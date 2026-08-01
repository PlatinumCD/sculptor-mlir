#include "sculptor-mlir/Dialect/Sculptor/Transforms/BuildTaskGraphIslands.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "mlir/Pass/PassRegistry.h"

namespace {

bool returnsTaskGraph(mlir::func::FuncOp func) {
  mlir::FunctionType functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

mlir::FailureOr<mlir::sculptor::task_graph::DigitalIslandAssignmentPolicy>
parseDigitalAssignmentPolicy(mlir::StringRef value) {
  using Policy = mlir::sculptor::task_graph::DigitalIslandAssignmentPolicy;
  if (value == "legacy")
    return Policy::Legacy;
  if (value == "multi-terminal-balanced")
    return Policy::MultiTerminalBalanced;
  return mlir::failure();
}

} // namespace

namespace mlir {
namespace sculptor {

void BuildTaskGraphIslandsPass::runOnOperation() {
  auto assignmentPolicy = parseDigitalAssignmentPolicy(digitalAssignment);
  if (failed(assignmentPolicy)) {
    getOperation().emitError("unknown digital island assignment policy '")
        << digitalAssignment << "'; expected legacy or multi-terminal-balanced";
    signalPassFailure();
    return;
  }

  bool foundTaskGraph = false;
  for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;

    auto parsedDag = task_graph::parseTaskGraphDAG(func);
    if (failed(parsedDag)) {
      signalPassFailure();
      return;
    }

    auto executionGraph = task_graph::buildTaskExecutionGraph(func, *parsedDag);
    if (failed(executionGraph)) {
      signalPassFailure();
      return;
    }

    auto islandGraph = task_graph::buildLogicalPlacementIslandGraph(
        *parsedDag, *executionGraph, *assignmentPolicy);
    if (failed(islandGraph)) {
      func.emitError("failed to build logical placement islands");
      signalPassFailure();
      return;
    }

    if (failed(task_graph::attachLogicalPlacementIslandIds(func, *parsedDag,
                                                           *islandGraph))) {
      signalPassFailure();
      return;
    }
    func->setAttr(schedule_attrs::kIslandAssignmentPolicyAttrName,
                  StringAttr::get(&getContext(), digitalAssignment));
    task_timing::invalidateTaskGraphTiming(func, /*advanceGeneration=*/false);

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
