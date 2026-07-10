#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTOBJECTIVE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTOBJECTIVE_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct IslandPlacementScore {
  int64_t transferCost = 0;
  int64_t boundaryPenalty = 0;
  int64_t total = 0;
};

int64_t computeIslandTransferCost(
    const HardwareBudget &budget,
    llvm::ArrayRef<LogicalIslandCommunicationEdge> communicationEdges,
    const llvm::DenseMap<unsigned, int64_t> &coreByIsland);

IslandPlacementScore evaluateIslandCorePlacement(
    const HardwareBudget &budget,
    llvm::ArrayRef<LogicalIslandCommunicationEdge> communicationEdges,
    const llvm::DenseMap<unsigned, int64_t> &coreByIsland,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland);

class IslandPlacementObjective {
public:
  IslandPlacementObjective(
      const TaskGraphPlacementProblem &problem,
      std::optional<unsigned> firstTaskIsland = std::nullopt,
      std::optional<unsigned> lastTaskIsland = std::nullopt)
      : problem(problem), firstTaskIsland(firstTaskIsland),
        lastTaskIsland(lastTaskIsland) {}

  FailureOr<IslandPlacementScore>
  evaluate(const IslandPlacementPlan &plan) const;

private:
  const TaskGraphPlacementProblem &problem;
  std::optional<unsigned> firstTaskIsland;
  std::optional<unsigned> lastTaskIsland;
};

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENTOBJECTIVE_H
