#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementObjective.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"

#include <limits>

namespace {

static int64_t saturatingAdd(int64_t lhs, int64_t rhs) {
  if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs)
    return std::numeric_limits<int64_t>::max();
  return lhs + rhs;
}

static int64_t saturatingMultiply(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0)
    return 0;
  if (lhs > std::numeric_limits<int64_t>::max() / rhs)
    return std::numeric_limits<int64_t>::max();
  return lhs * rhs;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

int64_t computeIslandTransferCost(
    const HardwareBudget &budget,
    llvm::ArrayRef<IslandAffinityEdge> affinityEdges,
    const llvm::DenseMap<unsigned, int64_t> &coreByIsland) {
  int64_t transferCost = 0;
  for (const IslandAffinityEdge &edge : affinityEdges) {
    auto producerIt = coreByIsland.find(edge.firstIsland);
    auto consumerIt = coreByIsland.find(edge.secondIsland);
    if (producerIt == coreByIsland.end() || consumerIt == coreByIsland.end() ||
        producerIt->second == consumerIt->second)
      continue;

    int64_t distance =
        getMeshDistance(producerIt->second, consumerIt->second, budget);
    transferCost = saturatingAdd(transferCost,
                                 saturatingMultiply(edge.byteSize, distance));
  }
  return transferCost;
}

IslandPlacementScore evaluateIslandCorePlacement(
    const HardwareBudget &budget,
    llvm::ArrayRef<IslandAffinityEdge> affinityEdges,
    const llvm::DenseMap<unsigned, int64_t> &coreByIsland,
    const PlacementConstraints &constraints) {
  IslandPlacementScore score;
  score.transferCost =
      computeIslandTransferCost(budget, affinityEdges, coreByIsland);

  if (constraints.sharedEndpointBoundary) {
    const SharedMeshBoundaryConstraint &boundary =
        *constraints.sharedEndpointBoundary;
    if (!boundary.islands)
      return score;
    auto firstIt = coreByIsland.find(boundary.islands->startIsland);
    auto lastIt = coreByIsland.find(boundary.islands->terminalIsland);
    if (firstIt != coreByIsland.end() && lastIt != coreByIsland.end()) {
      unsigned firstBoundary = getMeshBoundaryMask(firstIt->second, budget);
      unsigned lastBoundary = getMeshBoundaryMask(lastIt->second, budget);
      if ((firstBoundary & lastBoundary) == 0)
        score.boundaryPenalty = getBoundaryPenalty(score.transferCost);
    }
  }

  score.total = saturatingAdd(score.transferCost, score.boundaryPenalty);
  return score;
}

FailureOr<IslandPlacementScore>
IslandPlacementObjective::evaluate(const IslandPlacementPlan &plan) const {
  if (failed(validatePlacementPlan(problem, plan)))
    return failure();

  llvm::DenseMap<unsigned, int64_t> coreByIsland;
  for (auto indexedIsland : llvm::enumerate(problem.islandGraph.islands)) {
    int64_t physicalArrayId = plan.physicalArrayByIsland[indexedIsland.index()];
    coreByIsland[indexedIsland.value().islandIndex] =
        physicalArrayId / problem.budget.arraysPerCore;
  }

  return evaluateIslandCorePlacement(problem.budget,
                                     problem.islandGraph.affinityGraph.edges,
                                     coreByIsland, problem.constraints);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
