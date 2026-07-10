#include "GreedySearchInternals.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementObjective.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;
using IslandPlacement = task_schedulers::greedy_detail::IslandPlacement;

static int64_t
getPhysicalArrayCore(int64_t physicalArrayId,
                     const task_schedulers::HardwareBudget &budget) {
  return physicalArrayId / budget.arraysPerCore;
}

static std::optional<unsigned>
findIslandPlacementIndex(llvm::ArrayRef<IslandPlacement> islandPlacements,
                         unsigned island) {
  for (auto indexedPlacement : llvm::enumerate(islandPlacements)) {
    if (indexedPlacement.value().island == island)
      return static_cast<unsigned>(indexedPlacement.index());
  }
  return std::nullopt;
}

static llvm::DenseMap<unsigned, int64_t>
buildCoreByIsland(llvm::ArrayRef<IslandPlacement> islandPlacements,
                  const task_schedulers::HardwareBudget &budget) {
  llvm::DenseMap<unsigned, int64_t> coreByIsland;
  for (const IslandPlacement &placement : islandPlacements) {
    coreByIsland[placement.island] =
        getPhysicalArrayCore(placement.physicalArrayId, budget);
  }
  return coreByIsland;
}

static task_schedulers::IslandPlacementScore scoreIslandPlacement(
    llvm::ArrayRef<IslandPlacement> islandPlacements,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<task_schedulers::LogicalIslandCommunicationEdge>
        islandCommunicationEdges,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland) {
  return task_schedulers::evaluateIslandCorePlacement(
      budget, islandCommunicationEdges,
      buildCoreByIsland(islandPlacements, budget), firstTaskIsland,
      lastTaskIsland);
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace greedy_detail {

void repairBoundaryRegretPlacement(
    llvm::SmallVectorImpl<IslandPlacement> &islandPlacements,
    const HardwareBudget &budget, llvm::ArrayRef<int64_t> physicalArrayOrder,
    llvm::ArrayRef<LogicalIslandCommunicationEdge> islandCommunicationEdges,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland) {
  if (!firstTaskIsland || !lastTaskIsland || islandPlacements.empty())
    return;

  std::optional<unsigned> terminalPlacementIndex =
      findIslandPlacementIndex(islandPlacements, *lastTaskIsland);
  if (!terminalPlacementIndex)
    return;

  llvm::DenseSet<int64_t> usedPhysicalArrays;
  for (const IslandPlacement &placement : islandPlacements)
    usedPhysicalArrays.insert(placement.physicalArrayId);

  llvm::DenseMap<unsigned, int64_t> coreByIsland =
      buildCoreByIsland(islandPlacements, budget);
  auto firstCoreIt = coreByIsland.find(*firstTaskIsland);
  auto lastCoreIt = coreByIsland.find(*lastTaskIsland);
  if (firstCoreIt == coreByIsland.end() || lastCoreIt == coreByIsland.end())
    return;

  unsigned firstBoundaryMask = getMeshBoundaryMask(firstCoreIt->second, budget);
  unsigned lastBoundaryMask = getMeshBoundaryMask(lastCoreIt->second, budget);
  if (firstBoundaryMask == 0 || (firstBoundaryMask & lastBoundaryMask) != 0)
    return;

  IslandPlacementScore bestScore =
      scoreIslandPlacement(islandPlacements, budget, islandCommunicationEdges,
                           firstTaskIsland, lastTaskIsland);
  int64_t bestPhysicalArrayId =
      islandPlacements[*terminalPlacementIndex].physicalArrayId;

  for (int64_t candidatePhysicalArrayId : physicalArrayOrder) {
    if (usedPhysicalArrays.contains(candidatePhysicalArrayId))
      continue;

    int64_t candidateCore =
        getPhysicalArrayCore(candidatePhysicalArrayId, budget);
    unsigned candidateBoundaryMask = getMeshBoundaryMask(candidateCore, budget);
    if ((candidateBoundaryMask & firstBoundaryMask) == 0)
      continue;

    llvm::SmallVector<IslandPlacement, 8> candidatePlacements(
        islandPlacements.begin(), islandPlacements.end());
    candidatePlacements[*terminalPlacementIndex].physicalArrayId =
        candidatePhysicalArrayId;
    IslandPlacementScore candidateScore = scoreIslandPlacement(
        candidatePlacements, budget, islandCommunicationEdges, firstTaskIsland,
        lastTaskIsland);
    if (candidateScore.total > bestScore.total)
      continue;
    if (candidateScore.total == bestScore.total &&
        candidateScore.transferCost >= bestScore.transferCost)
      continue;

    bestScore = candidateScore;
    bestPhysicalArrayId = candidatePhysicalArrayId;
  }

  islandPlacements[*terminalPlacementIndex].physicalArrayId =
      bestPhysicalArrayId;
}

} // namespace greedy_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
