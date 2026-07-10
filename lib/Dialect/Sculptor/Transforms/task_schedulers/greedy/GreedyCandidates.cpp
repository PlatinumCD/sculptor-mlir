#include "GreedySearchInternals.h"

#include "GreedyHeuristic.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <utility>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

using TaskGraphNode = task_schedulers::TaskGraphNode;
using CorePhysicalArraySlots =
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16>;

using IslandCommunicationEdge =
    mlir::sculptor::task_schedulers::LogicalIslandCommunicationEdge;
struct GreedyCandidateScore {
  int64_t coreId = -1;
  int64_t score = 0;
};

struct GreedyTransferOpportunity {
  std::optional<int64_t> bestTransferCost;
};

using GreedyPlacementState = task_schedulers::greedy_detail::PlacementState;

constexpr unsigned kGreedyMaxRecursiveCandidates = 8;
constexpr unsigned kGreedyMaxAnchorIslands = 8;

static bool
hasAvailableCoreSlot(int64_t coreId,
                     const CorePhysicalArraySlots &physicalArraysByCore,
                     llvm::ArrayRef<unsigned> usedSlotsByCore) {
  if (coreId < 0 || coreId >= static_cast<int64_t>(physicalArraysByCore.size()))
    return false;

  return usedSlotsByCore[coreId] < physicalArraysByCore[coreId].size();
}

static mlir::FailureOr<CorePhysicalArraySlots>
buildCorePhysicalArraySlotsImpl(mlir::Operation *diagnosticOp,
                                const task_schedulers::HardwareBudget &budget,
                                llvm::ArrayRef<int64_t> physicalArrayOrder) {
  CorePhysicalArraySlots physicalArraysByCore(
      static_cast<size_t>(budget.numCores));
  for (int64_t physicalArrayId : physicalArrayOrder) {
    auto placement = task_schedulers::resolvePhysicalArrayPlacement(
        diagnosticOp, budget, physicalArrayId);
    if (mlir::failed(placement))
      return mlir::failure();

    physicalArraysByCore[placement->coreId].push_back(physicalArrayId);
  }

  return physicalArraysByCore;
}

static int64_t evaluateGreedyHeuristic(
    const task_schedulers::GreedyHeuristic &heuristic,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland,
    unsigned activeIsland, unsigned activePlacementIndex,
    unsigned totalPlacementCount, std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland,
    std::optional<int64_t> bestTransferCost) {
  return heuristic.evaluate(task_schedulers::GreedyHeuristicContext{
      budget, islandCommunicationEdges, coreByPlacedIsland, activeIsland,
      activePlacementIndex, totalPlacementCount, firstTaskIsland,
      lastTaskIsland, bestTransferCost});
}

static void
appendAvailableCoreCandidate(llvm::SmallVectorImpl<int64_t> &candidates,
                             int64_t coreId,
                             const CorePhysicalArraySlots &physicalArraysByCore,
                             llvm::ArrayRef<unsigned> usedSlotsByCore) {
  if (!hasAvailableCoreSlot(coreId, physicalArraysByCore, usedSlotsByCore))
    return;

  for (int64_t candidate : candidates) {
    if (candidate == coreId)
      return;
  }

  candidates.push_back(coreId);
}

static void appendGreedyLocalCandidates(
    llvm::SmallVectorImpl<int64_t> &candidates, int64_t currentCore,
    const task_schedulers::HardwareBudget &budget,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<unsigned> usedSlotsByCore, bool includeDiagonals) {
  appendAvailableCoreCandidate(candidates, currentCore, physicalArraysByCore,
                               usedSlotsByCore);

  int64_t row = currentCore / budget.meshCols;
  int64_t col = currentCore % budget.meshCols;
  auto appendNeighbor = [&](int64_t neighborRow, int64_t neighborCol) {
    if (neighborRow < 0 || neighborRow >= budget.meshRows || neighborCol < 0 ||
        neighborCol >= budget.meshCols)
      return;

    int64_t neighborCore = neighborRow * budget.meshCols + neighborCol;
    if (neighborCore >= budget.numCores)
      return;

    appendAvailableCoreCandidate(candidates, neighborCore, physicalArraysByCore,
                                 usedSlotsByCore);
  };

  appendNeighbor(row - 1, col);
  appendNeighbor(row, col - 1);
  appendNeighbor(row + 1, col);
  appendNeighbor(row, col + 1);
  if (!includeDiagonals)
    return;

  appendNeighbor(row - 1, col - 1);
  appendNeighbor(row - 1, col + 1);
  appendNeighbor(row + 1, col - 1);
  appendNeighbor(row + 1, col + 1);
}

static llvm::SmallVector<std::pair<unsigned, int64_t>, 8>
collectPlacedCommunicationAnchors(
    unsigned island,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  llvm::DenseMap<unsigned, int64_t> weightByIsland;
  for (const IslandCommunicationEdge &edge : islandCommunicationEdges) {
    unsigned otherIsland = std::numeric_limits<unsigned>::max();
    if (edge.producerIsland == island) {
      otherIsland = edge.consumerIsland;
    } else if (edge.consumerIsland == island) {
      otherIsland = edge.producerIsland;
    } else {
      continue;
    }

    if (!coreByPlacedIsland.contains(otherIsland))
      continue;
    weightByIsland[otherIsland] += edge.byteSize;
  }

  llvm::SmallVector<std::pair<unsigned, int64_t>, 8> anchors;
  anchors.reserve(weightByIsland.size());
  for (const auto &entry : weightByIsland)
    anchors.push_back({entry.first, entry.second});

  llvm::sort(anchors, [](const auto &lhs, const auto &rhs) {
    if (lhs.second != rhs.second)
      return lhs.second > rhs.second;
    return lhs.first < rhs.first;
  });
  return anchors;
}

static int64_t minDistanceToPlacedRegionImpl(
    int64_t candidateCore, const task_schedulers::HardwareBudget &budget,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland);

static void appendNearestAvailableCoreCandidates(
    llvm::SmallVectorImpl<int64_t> &candidates, int64_t startCore,
    const task_schedulers::HardwareBudget &budget,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<unsigned> usedSlotsByCore);

static GreedyTransferOpportunity computeGreedyTransferOpportunity(
    llvm::ArrayRef<int64_t> candidateCores, unsigned island,
    unsigned placementIndex, unsigned totalPlacementCount,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    llvm::SmallVectorImpl<unsigned> &usedSlotsByCore,
    llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  GreedyTransferOpportunity opportunity;
  if (candidateCores.empty())
    return opportunity;

  task_schedulers::TransferCostGreedyHeuristic transferCost;
  for (int64_t candidateCore : candidateCores) {
    ++usedSlotsByCore[candidateCore];
    coreByPlacedIsland[island] = candidateCore;
    int64_t score = evaluateGreedyHeuristic(
        transferCost, budget, islandCommunicationEdges, coreByPlacedIsland,
        island, placementIndex, totalPlacementCount, firstTaskIsland,
        lastTaskIsland, std::nullopt);
    coreByPlacedIsland.erase(island);
    --usedSlotsByCore[candidateCore];

    if (!opportunity.bestTransferCost ||
        score < *opportunity.bestTransferCost) {
      opportunity.bestTransferCost = score;
      continue;
    }
  }

  return opportunity;
}

static void retainBestGreedyImmediateCandidates(
    llvm::SmallVectorImpl<int64_t> &candidateCores, unsigned island,
    unsigned placementIndex, unsigned totalPlacementCount,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland, unsigned maxCandidates,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    llvm::SmallVectorImpl<unsigned> &usedSlotsByCore,
    llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  if (candidateCores.size() <= maxCandidates)
    return;

  task_schedulers::TransferCostGreedyHeuristic transferCost;
  llvm::SmallVector<GreedyCandidateScore, 16> scoredCandidates;
  scoredCandidates.reserve(candidateCores.size());
  for (int64_t candidateCore : candidateCores) {
    ++usedSlotsByCore[candidateCore];
    coreByPlacedIsland[island] = candidateCore;
    int64_t score = evaluateGreedyHeuristic(
        transferCost, budget, islandCommunicationEdges, coreByPlacedIsland,
        island, placementIndex, totalPlacementCount, firstTaskIsland,
        lastTaskIsland, std::nullopt);
    coreByPlacedIsland.erase(island);
    --usedSlotsByCore[candidateCore];
    scoredCandidates.push_back(GreedyCandidateScore{candidateCore, score});
  }

  llvm::sort(scoredCandidates, [&](const GreedyCandidateScore &lhs,
                                   const GreedyCandidateScore &rhs) {
    if (lhs.score != rhs.score)
      return lhs.score < rhs.score;

    int64_t lhsRegionDistance =
        minDistanceToPlacedRegionImpl(lhs.coreId, budget, coreByPlacedIsland);
    int64_t rhsRegionDistance =
        minDistanceToPlacedRegionImpl(rhs.coreId, budget, coreByPlacedIsland);
    if (lhsRegionDistance != rhsRegionDistance)
      return lhsRegionDistance < rhsRegionDistance;

    int64_t lhsRow = lhs.coreId / budget.meshCols;
    int64_t rhsRow = rhs.coreId / budget.meshCols;
    int64_t lhsCol = lhs.coreId % budget.meshCols;
    int64_t rhsCol = rhs.coreId % budget.meshCols;
    if (lhsRow + lhsCol != rhsRow + rhsCol)
      return lhsRow + lhsCol < rhsRow + rhsCol;

    return lhs.coreId < rhs.coreId;
  });

  candidateCores.clear();
  unsigned keptCandidates =
      std::min(maxCandidates, static_cast<unsigned>(scoredCandidates.size()));
  for (unsigned index = 0; index < keptCandidates; ++index)
    candidateCores.push_back(scoredCandidates[index].coreId);
}

static void appendGreedyProducerConsumerCandidates(
    llvm::SmallVectorImpl<int64_t> &candidates, unsigned island,
    int64_t currentCore, const task_schedulers::HardwareBudget &budget,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<unsigned> usedSlotsByCore,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  appendAvailableCoreCandidate(candidates, currentCore, physicalArraysByCore,
                               usedSlotsByCore);

  auto anchors = collectPlacedCommunicationAnchors(
      island, islandCommunicationEdges, coreByPlacedIsland);
  unsigned visitedAnchors = 0;
  for (const auto &anchor : anchors) {
    if (visitedAnchors++ >= kGreedyMaxAnchorIslands)
      break;

    auto coreIt = coreByPlacedIsland.find(anchor.first);
    if (coreIt == coreByPlacedIsland.end())
      continue;

    appendGreedyLocalCandidates(candidates, coreIt->second, budget,
                                physicalArraysByCore, usedSlotsByCore,
                                /*includeDiagonals=*/true);
  }
}

static unsigned
getGreedyLocalCandidateRank(int64_t candidateCore, int64_t currentCore,
                            const task_schedulers::HardwareBudget &budget) {
  if (candidateCore == currentCore)
    return 0;

  int64_t currentRow = currentCore / budget.meshCols;
  int64_t currentCol = currentCore % budget.meshCols;
  int64_t candidateRow = candidateCore / budget.meshCols;
  int64_t candidateCol = candidateCore % budget.meshCols;

  if (candidateRow == currentRow - 1 && candidateCol == currentCol)
    return 1;
  if (candidateRow == currentRow && candidateCol == currentCol - 1)
    return 2;
  if (candidateRow == currentRow + 1 && candidateCol == currentCol)
    return 3;
  if (candidateRow == currentRow && candidateCol == currentCol + 1)
    return 4;
  if (candidateRow == currentRow - 1 && candidateCol == currentCol - 1)
    return 5;
  if (candidateRow == currentRow - 1 && candidateCol == currentCol + 1)
    return 6;
  if (candidateRow == currentRow + 1 && candidateCol == currentCol - 1)
    return 7;
  if (candidateRow == currentRow + 1 && candidateCol == currentCol + 1)
    return 8;

  return 9;
}

static int64_t minDistanceToPlacedRegionImpl(
    int64_t candidateCore, const task_schedulers::HardwareBudget &budget,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  int64_t bestDistance = std::numeric_limits<int64_t>::max();
  for (const auto &placedIsland : coreByPlacedIsland) {
    int64_t distance = task_schedulers::getMeshDistance(
        candidateCore, placedIsland.second, budget);
    bestDistance = std::min(bestDistance, distance);
  }

  return bestDistance == std::numeric_limits<int64_t>::max() ? 0 : bestDistance;
}

static bool isBetterChoiceImpl(
    bool hasBest, int64_t candidateScore, int64_t candidateCore,
    int64_t bestScore, int64_t bestCore, int64_t currentCore,
    const task_schedulers::HardwareBudget &budget,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  if (!hasBest)
    return true;
  if (candidateScore != bestScore)
    return candidateScore < bestScore;

  unsigned candidateRank =
      getGreedyLocalCandidateRank(candidateCore, currentCore, budget);
  unsigned bestRank =
      getGreedyLocalCandidateRank(bestCore, currentCore, budget);
  if (candidateRank != bestRank)
    return candidateRank < bestRank;

  int64_t candidateRegionDistance =
      minDistanceToPlacedRegionImpl(candidateCore, budget, coreByPlacedIsland);
  int64_t bestRegionDistance =
      minDistanceToPlacedRegionImpl(bestCore, budget, coreByPlacedIsland);
  if (candidateRegionDistance != bestRegionDistance)
    return candidateRegionDistance < bestRegionDistance;

  int64_t candidateRow = candidateCore / budget.meshCols;
  int64_t bestRow = bestCore / budget.meshCols;
  int64_t candidateCol = candidateCore % budget.meshCols;
  int64_t bestCol = bestCore % budget.meshCols;
  if (candidateRow + candidateCol != bestRow + bestCol)
    return candidateRow + candidateCol < bestRow + bestCol;

  return candidateCore < bestCore;
}

static void appendNearestAvailableCoreCandidates(
    llvm::SmallVectorImpl<int64_t> &candidates, int64_t startCore,
    const task_schedulers::HardwareBudget &budget,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<unsigned> usedSlotsByCore) {
  if (budget.numCores <= 0)
    return;

  llvm::SmallVector<int64_t, 16> distance(static_cast<size_t>(budget.numCores),
                                          -1);
  std::queue<int64_t> worklist;
  distance[startCore] = 0;
  worklist.push(startCore);
  int64_t selectedDistance = -1;

  while (!worklist.empty()) {
    int64_t coreId = worklist.front();
    worklist.pop();

    int64_t currentDistance = distance[coreId];
    if (selectedDistance >= 0 && currentDistance > selectedDistance)
      break;

    if (hasAvailableCoreSlot(coreId, physicalArraysByCore, usedSlotsByCore)) {
      candidates.push_back(coreId);
      selectedDistance = currentDistance;
      continue;
    }

    int64_t row = coreId / budget.meshCols;
    int64_t col = coreId % budget.meshCols;
    auto pushNeighbor = [&](int64_t neighborRow, int64_t neighborCol) {
      if (neighborRow < 0 || neighborRow >= budget.meshRows ||
          neighborCol < 0 || neighborCol >= budget.meshCols)
        return;

      int64_t neighborCore = neighborRow * budget.meshCols + neighborCol;
      if (neighborCore >= budget.numCores || distance[neighborCore] >= 0)
        return;

      distance[neighborCore] = currentDistance + 1;
      worklist.push(neighborCore);
    };

    pushNeighbor(row - 1, col);
    pushNeighbor(row + 1, col);
    pushNeighbor(row, col - 1);
    pushNeighbor(row, col + 1);
  }
}

static void appendGreedyCandidateCores(
    llvm::SmallVectorImpl<int64_t> &candidateCores, unsigned setupIndex,
    unsigned island, int64_t currentCore,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const task_schedulers::HardwareBudget &budget,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    llvm::ArrayRef<unsigned> usedSlotsByCore,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  (void)matrixSetupTasks;
  if (setupIndex == 0 && coreByPlacedIsland.empty()) {
    if (hasAvailableCoreSlot(/*coreId=*/0, physicalArraysByCore,
                             usedSlotsByCore))
      candidateCores.push_back(0);
    return;
  }

  if (budget.greedy.candidateScope ==
      task_schedulers::GreedyCandidateScope::ProducerConsumer) {
    appendGreedyProducerConsumerCandidates(
        candidateCores, island, currentCore, budget, physicalArraysByCore,
        usedSlotsByCore, islandCommunicationEdges, coreByPlacedIsland);
  } else {
    bool includeDiagonals = budget.greedy.candidateScope ==
                            task_schedulers::GreedyCandidateScope::Diagonal;
    appendGreedyLocalCandidates(candidateCores, currentCore, budget,
                                physicalArraysByCore, usedSlotsByCore,
                                includeDiagonals);
  }

  if (candidateCores.empty())
    appendNearestAvailableCoreCandidates(candidateCores, currentCore, budget,
                                         physicalArraysByCore, usedSlotsByCore);
}

static bool
applyCandidateImpl(GreedyPlacementState &state, const TaskGraphNode &setupNode,
                   unsigned island, int64_t candidateCore,
                   const CorePhysicalArraySlots &physicalArraysByCore) {
  unsigned localSlot = state.usedSlotsByCore[candidateCore]++;
  if (localSlot >= physicalArraysByCore[candidateCore].size()) {
    --state.usedSlotsByCore[candidateCore];
    return false;
  }

  int64_t physicalArrayId = physicalArraysByCore[candidateCore][localSlot];
  state.islandPlacements.push_back(
      task_schedulers::greedy_detail::IslandPlacement{setupNode.index,
                                                      physicalArrayId});
  state.coreByPlacedIsland[island] = candidateCore;
  state.currentCore = candidateCore;
  return true;
}

static llvm::SmallVector<GreedyPlacementState, 16> expandStateImpl(
    const GreedyPlacementState &state, unsigned setupIndex,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const task_schedulers::HardwareBudget &budget,
    const task_schedulers::GreedyHeuristic &heuristic,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland, bool pruneCandidates) {
  llvm::SmallVector<GreedyPlacementState, 16> expandedStates;
  if (setupIndex >= matrixSetupTasks.size())
    return expandedStates;

  const TaskGraphNode *setupNode = matrixSetupTasks[setupIndex];
  unsigned island = setupNode->index;
  unsigned totalPlacementCount = static_cast<unsigned>(matrixSetupTasks.size());

  llvm::SmallVector<int64_t, 8> candidateCores;
  appendGreedyCandidateCores(candidateCores, setupIndex, island,
                             state.currentCore, matrixSetupTasks, budget,
                             physicalArraysByCore, islandCommunicationEdges,
                             state.usedSlotsByCore, state.coreByPlacedIsland);
  if (candidateCores.empty())
    return expandedStates;

  if (pruneCandidates) {
    GreedyPlacementState pruningScratchState = state;
    retainBestGreedyImmediateCandidates(
        candidateCores, island, setupIndex, totalPlacementCount,
        firstTaskIsland, lastTaskIsland, kGreedyMaxRecursiveCandidates, budget,
        islandCommunicationEdges, pruningScratchState.usedSlotsByCore,
        pruningScratchState.coreByPlacedIsland);
  }

  GreedyPlacementState transferScratchState = state;
  GreedyTransferOpportunity transferOpportunity =
      computeGreedyTransferOpportunity(
          candidateCores, island, setupIndex, totalPlacementCount,
          firstTaskIsland, lastTaskIsland, budget, islandCommunicationEdges,
          transferScratchState.usedSlotsByCore,
          transferScratchState.coreByPlacedIsland);

  expandedStates.reserve(candidateCores.size());
  for (int64_t candidateCore : candidateCores) {
    GreedyPlacementState expandedState = state;
    if (!applyCandidateImpl(expandedState, *setupNode, island, candidateCore,
                            physicalArraysByCore))
      continue;

    expandedState.score = evaluateGreedyHeuristic(
        heuristic, budget, islandCommunicationEdges,
        expandedState.coreByPlacedIsland, island, setupIndex,
        totalPlacementCount, firstTaskIsland, lastTaskIsland,
        transferOpportunity.bestTransferCost);
    expandedStates.push_back(std::move(expandedState));
  }
  return expandedStates;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace greedy_detail {

FailureOr<CorePhysicalArraySlots>
buildCorePhysicalArraySlots(Operation *diagnosticOp,
                            const HardwareBudget &budget,
                            llvm::ArrayRef<int64_t> physicalArrayOrder) {
  return buildCorePhysicalArraySlotsImpl(diagnosticOp, budget,
                                         physicalArrayOrder);
}

int64_t minDistanceToPlacedRegion(
    int64_t candidateCore, const HardwareBudget &budget,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  return minDistanceToPlacedRegionImpl(candidateCore, budget,
                                       coreByPlacedIsland);
}

bool isBetterChoice(
    bool hasBest, int64_t candidateScore, int64_t candidateCore,
    int64_t bestScore, int64_t bestCore, int64_t currentCore,
    const HardwareBudget &budget,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  return isBetterChoiceImpl(hasBest, candidateScore, candidateCore, bestScore,
                            bestCore, currentCore, budget, coreByPlacedIsland);
}

bool applyCandidate(PlacementState &state, const TaskGraphNode &setupNode,
                    unsigned island, int64_t candidateCore,
                    const CorePhysicalArraySlots &physicalArraysByCore) {
  return applyCandidateImpl(state, setupNode, island, candidateCore,
                            physicalArraysByCore);
}

llvm::SmallVector<PlacementState, 16> expandState(
    const PlacementState &state, unsigned setupIndex,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const HardwareBudget &budget, const GreedyHeuristic &heuristic,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<LogicalIslandCommunicationEdge> islandCommunicationEdges,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland, bool pruneCandidates) {
  return expandStateImpl(state, setupIndex, matrixSetupTasks, budget, heuristic,
                         physicalArraysByCore, islandCommunicationEdges,
                         firstTaskIsland, lastTaskIsland, pruneCandidates);
}

} // namespace greedy_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
