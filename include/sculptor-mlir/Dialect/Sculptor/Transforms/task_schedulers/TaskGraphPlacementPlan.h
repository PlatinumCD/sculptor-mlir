#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTPLAN_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTPLAN_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementConstraints.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

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

struct LogicalIslandPlacement {
  int64_t coreId = -1;
  std::optional<int64_t> physicalArrayId;
};

struct SearchProxyObjective {
  double completionTime = 0.0;
  double communication = 0.0;
  double resourceLoad = 0.0;
};

struct FullDeploymentTimingObjective {
  double makespanNs = 0.0;
  double exposedContentionNs = 0.0;
  double exposedTransportNs = 0.0;
  int64_t totalWordHops = 0;
};

// Indexed by LogicalPlacementIslandGraph::islands. Analog islands own an array;
// digital-only reduction islands own a core but no analog array.
struct IslandPlacementPlan {
  llvm::SmallVector<LogicalIslandPlacement, 16> placements;
  std::optional<SearchProxyObjective> searchProxyObjective;
  std::optional<FullDeploymentTimingObjective> fullTimingObjective;
  std::optional<unsigned> timingRerankCandidateCount;
  std::optional<unsigned> timingRerankSelectedProxyRank;
};

struct IslandPlacementResources {
  llvm::SmallVector<int64_t, 16> analogPhysicalArrayOrder;
};

LogicalResult validatePlacementPlan(const TaskGraphPlacementProblem &problem,
                                    const IslandPlacementPlan &plan);

FailureOr<IslandPlacementPlan> buildPlacementPlanFromPhysicalArrayOrder(
    const TaskGraphPlacementProblem &problem,
    llvm::ArrayRef<int64_t> physicalArrayOrder);

FailureOr<IslandPlacementResources>
buildIslandPlacementResources(const TaskGraphPlacementProblem &problem,
                              llvm::ArrayRef<int64_t> physicalArrayOrder);

FailureOr<IslandPlacementPlan> buildPlacementPlanFromAnalogPlacements(
    const TaskGraphPlacementProblem &problem,
    const IslandPlacementResources &resources,
    const llvm::DenseMap<unsigned, int64_t> &physicalArrayByAnalogIsland);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTPLAN_H
