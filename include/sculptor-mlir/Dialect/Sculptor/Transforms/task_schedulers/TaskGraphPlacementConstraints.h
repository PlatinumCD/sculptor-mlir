#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTCONSTRAINTS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTCONSTRAINTS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct IslandBoundaryEndpoints {
  unsigned startIsland = 0;
  unsigned terminalIsland = 0;
};

struct SharedMeshBoundaryConstraint {
  unsigned startTask = 0;
  unsigned terminalTask = 0;
  std::optional<IslandBoundaryEndpoints> islands;
};

struct PlacementConstraints {
  std::optional<SharedMeshBoundaryConstraint> sharedEndpointBoundary;
};

FailureOr<PlacementConstraints> buildPlacementConstraints(
    Operation *diagnosticOp,
    const task_graph::TaskExecutionGraph &executionGraph,
    const task_graph::LogicalPlacementIslandGraph &islandGraph);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTCONSTRAINTS_H
