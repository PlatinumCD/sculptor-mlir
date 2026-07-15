#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementConstraints.h"

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<PlacementConstraints> buildPlacementConstraints(
    Operation *diagnosticOp,
    const task_graph::TaskExecutionGraph &executionGraph,
    const task_graph::LogicalPlacementIslandGraph &islandGraph) {
  PlacementConstraints constraints;
  if (executionGraph.topologicalOrder.empty())
    return constraints;

  unsigned startTask = executionGraph.topologicalOrder.front();
  unsigned terminalTask = executionGraph.topologicalOrder.back();
  auto startIsland = islandGraph.islandByTaskIndex.find(startTask);
  auto terminalIsland = islandGraph.islandByTaskIndex.find(terminalTask);
  bool hasStartIsland = startIsland != islandGraph.islandByTaskIndex.end();
  bool hasTerminalIsland =
      terminalIsland != islandGraph.islandByTaskIndex.end();
  if (hasStartIsland != hasTerminalIsland) {
    diagnosticOp->emitError(
        "expected execution endpoints to have consistent island coverage");
    return failure();
  }

  SharedMeshBoundaryConstraint boundary{startTask, terminalTask, std::nullopt};
  if (hasStartIsland) {
    boundary.islands =
        IslandBoundaryEndpoints{startIsland->second, terminalIsland->second};
  }
  constraints.sharedEndpointBoundary = boundary;
  return constraints;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
