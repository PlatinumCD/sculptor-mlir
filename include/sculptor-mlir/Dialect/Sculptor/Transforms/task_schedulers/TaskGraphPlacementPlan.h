#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTPLAN_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTPLAN_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementConstraints.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct TaskGraphPlacementProblem {
  TaskGraphPlacementProblem(
      func::FuncOp taskGraphFunc, const HardwareBudget &budget,
      const TaskGraphDAG &dag,
      const task_graph::TaskExecutionGraph &executionGraph,
      const LogicalPlacementIslandGraph &islandGraph,
      const PlacementConstraints &constraints)
      : taskGraphFunc(taskGraphFunc),
        diagnosticOp(taskGraphFunc.getOperation()), budget(budget), dag(dag),
        executionGraph(executionGraph), islandGraph(islandGraph),
        constraints(constraints) {}

  func::FuncOp taskGraphFunc;
  Operation *diagnosticOp = nullptr;
  const HardwareBudget &budget;
  const TaskGraphDAG &dag;
  const task_graph::TaskExecutionGraph &executionGraph;
  const LogicalPlacementIslandGraph &islandGraph;
  const PlacementConstraints &constraints;
};

// Indexed by LogicalPlacementIslandGraph::islands. Repeated physical arrays
// are legal so order-based schedulers retain their existing round-robin
// behavior when a graph contains more islands than available arrays.
struct IslandPlacementPlan {
  llvm::SmallVector<int64_t, 16> physicalArrayByIsland;
};

LogicalResult validatePlacementPlan(const TaskGraphPlacementProblem &problem,
                                    const IslandPlacementPlan &plan);

FailureOr<IslandPlacementPlan> buildPlacementPlanFromPhysicalArrayOrder(
    const TaskGraphPlacementProblem &problem,
    llvm::ArrayRef<int64_t> physicalArrayOrder);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTPLAN_H
