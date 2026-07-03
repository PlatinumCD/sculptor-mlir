#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedyPlacement.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

using TaskGraphNode = task_schedulers::TaskGraphNode;
using CorePhysicalArraySlots =
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16>;

using IslandCommunicationEdge =
    mlir::sculptor::task_schedulers::LogicalIslandCommunicationEdge;
struct GreedyLookaheadChoice {
  int64_t coreId = -1;
  int64_t score = 0;
};

struct GreedyCandidateScore {
  int64_t coreId = -1;
  int64_t score = 0;
};

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
buildCorePhysicalArraySlots(mlir::Operation *diagnosticOp,
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

static int64_t recomputePlacedIslandTransferCost(
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  int64_t score = 0;
  for (const IslandCommunicationEdge &edge : islandCommunicationEdges) {
    auto producerCoreIt = coreByPlacedIsland.find(edge.producerIsland);
    auto consumerCoreIt = coreByPlacedIsland.find(edge.consumerIsland);
    if (producerCoreIt == coreByPlacedIsland.end() ||
        consumerCoreIt == coreByPlacedIsland.end())
      continue;

    score += edge.byteSize *
             task_schedulers::getMeshDistance(producerCoreIt->second,
                                              consumerCoreIt->second, budget);
  }

  return score;
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

static int64_t minDistanceToPlacedRegion(
    int64_t candidateCore, const task_schedulers::HardwareBudget &budget,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland);

static void appendNearestAvailableCoreCandidates(
    llvm::SmallVectorImpl<int64_t> &candidates, int64_t startCore,
    const task_schedulers::HardwareBudget &budget,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<unsigned> usedSlotsByCore);

static void retainBestGreedyImmediateCandidates(
    llvm::SmallVectorImpl<int64_t> &candidateCores, unsigned island,
    unsigned maxCandidates, const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    llvm::SmallVectorImpl<unsigned> &usedSlotsByCore,
    llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  if (candidateCores.size() <= maxCandidates)
    return;

  llvm::SmallVector<GreedyCandidateScore, 16> scoredCandidates;
  scoredCandidates.reserve(candidateCores.size());
  for (int64_t candidateCore : candidateCores) {
    ++usedSlotsByCore[candidateCore];
    coreByPlacedIsland[island] = candidateCore;
    int64_t score = recomputePlacedIslandTransferCost(
        budget, islandCommunicationEdges, coreByPlacedIsland);
    coreByPlacedIsland.erase(island);
    --usedSlotsByCore[candidateCore];
    scoredCandidates.push_back(GreedyCandidateScore{candidateCore, score});
  }

  llvm::sort(scoredCandidates, [&](const GreedyCandidateScore &lhs,
                                   const GreedyCandidateScore &rhs) {
    if (lhs.score != rhs.score)
      return lhs.score < rhs.score;

    int64_t lhsRegionDistance =
        minDistanceToPlacedRegion(lhs.coreId, budget, coreByPlacedIsland);
    int64_t rhsRegionDistance =
        minDistanceToPlacedRegion(rhs.coreId, budget, coreByPlacedIsland);
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

static int64_t minDistanceToPlacedRegion(
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

static bool isBetterGreedyChoice(
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
      minDistanceToPlacedRegion(candidateCore, budget, coreByPlacedIsland);
  int64_t bestRegionDistance =
      minDistanceToPlacedRegion(bestCore, budget, coreByPlacedIsland);
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

static void pruneGreedyRecursiveCandidates(
    llvm::SmallVectorImpl<int64_t> &candidateCores, unsigned island,
    int64_t remainingLookahead, const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    llvm::SmallVectorImpl<unsigned> &usedSlotsByCore,
    llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  if (remainingLookahead <= 1 ||
      candidateCores.size() <= kGreedyMaxRecursiveCandidates)
    return;

  retainBestGreedyImmediateCandidates(
      candidateCores, island, kGreedyMaxRecursiveCandidates, budget,
      islandCommunicationEdges, usedSlotsByCore, coreByPlacedIsland);
}

static mlir::FailureOr<GreedyLookaheadChoice> chooseGreedyLookaheadPlacement(
    unsigned setupIndex, int64_t remainingLookahead, int64_t currentCore,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const task_schedulers::HardwareBudget &budget,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    llvm::SmallVectorImpl<unsigned> &usedSlotsByCore,
    llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland) {
  if (setupIndex >= matrixSetupTasks.size() || remainingLookahead <= 0) {
    return GreedyLookaheadChoice{
        currentCore, recomputePlacedIslandTransferCost(
                         budget, islandCommunicationEdges, coreByPlacedIsland)};
  }

  const TaskGraphNode *setupNode = matrixSetupTasks[setupIndex];
  unsigned island = setupNode->index;

  llvm::SmallVector<int64_t, 8> candidateCores;
  if (setupIndex == 0 && coreByPlacedIsland.empty()) {
    if (hasAvailableCoreSlot(/*coreId=*/0, physicalArraysByCore,
                             usedSlotsByCore))
      candidateCores.push_back(0);
  } else {
    if (budget.greedyCandidateScope == "producer-consumer") {
      appendGreedyProducerConsumerCandidates(
          candidateCores, island, currentCore, budget, physicalArraysByCore,
          usedSlotsByCore, islandCommunicationEdges, coreByPlacedIsland);
    } else {
      bool includeDiagonals = budget.greedyCandidateScope == "diagonal";
      appendGreedyLocalCandidates(candidateCores, currentCore, budget,
                                  physicalArraysByCore, usedSlotsByCore,
                                  includeDiagonals);
    }
    if (candidateCores.empty()) {
      appendNearestAvailableCoreCandidates(candidateCores, currentCore, budget,
                                           physicalArraysByCore,
                                           usedSlotsByCore);
    }
  }

  if (candidateCores.empty())
    return mlir::failure();

  pruneGreedyRecursiveCandidates(candidateCores, island, remainingLookahead,
                                 budget, islandCommunicationEdges,
                                 usedSlotsByCore, coreByPlacedIsland);

  bool hasBest = false;
  GreedyLookaheadChoice best;
  for (int64_t candidateCore : candidateCores) {
    ++usedSlotsByCore[candidateCore];
    coreByPlacedIsland[island] = candidateCore;

    int64_t totalScore = 0;
    if (remainingLookahead > 1 && setupIndex + 1 < matrixSetupTasks.size()) {
      auto futureChoice = chooseGreedyLookaheadPlacement(
          setupIndex + 1, remainingLookahead - 1, candidateCore,
          matrixSetupTasks, budget, physicalArraysByCore,
          islandCommunicationEdges, usedSlotsByCore, coreByPlacedIsland);
      if (mlir::failed(futureChoice)) {
        coreByPlacedIsland.erase(island);
        --usedSlotsByCore[candidateCore];
        continue;
      }
      totalScore = futureChoice->score;
    } else {
      totalScore = recomputePlacedIslandTransferCost(
          budget, islandCommunicationEdges, coreByPlacedIsland);
    }

    coreByPlacedIsland.erase(island);
    --usedSlotsByCore[candidateCore];

    if (isBetterGreedyChoice(hasBest, totalScore, candidateCore, best.score,
                             best.coreId, currentCore, budget,
                             coreByPlacedIsland)) {
      hasBest = true;
      best.coreId = candidateCore;
      best.score = totalScore;
    }
  }

  if (!hasBest)
    return mlir::failure();

  return best;
}

static mlir::FailureOr<
    llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>>
buildGreedyIslandGroupPlacements(
    mlir::func::FuncOp taskGraphFunc,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<int64_t> physicalArrayOrder,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges) {
  if (matrixSetupTasks.empty())
    return llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>();

  if (budget.topology != "mesh" || budget.meshRows <= 0 ||
      budget.meshCols <= 0 || budget.numCores <= 0 ||
      budget.arraysPerCore <= 0) {
    taskGraphFunc.emitError("expected greedy island placement to use a "
                            "non-empty mesh hardware budget");
    return mlir::failure();
  }

  auto physicalArraysByCore = buildCorePhysicalArraySlots(
      taskGraphFunc.getOperation(), budget, physicalArrayOrder);
  if (mlir::failed(physicalArraysByCore))
    return mlir::failure();

  llvm::SmallVector<unsigned, 16> emptyUsedSlotsByCore(
      static_cast<size_t>(budget.numCores), 0);
  if (!hasAvailableCoreSlot(/*coreId=*/0, *physicalArraysByCore,
                            emptyUsedSlotsByCore)) {
    taskGraphFunc.emitError("expected greedy island placement to have an "
                            "available analog array on core 0");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 16> usedSlotsByCore(
      static_cast<size_t>(budget.numCores), 0);
  llvm::DenseMap<unsigned, int64_t> coreByPlacedIsland;
  llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>
      groupPlacements;
  groupPlacements.reserve(matrixSetupTasks.size());

  int64_t currentCore = 0;
  for (auto indexedSetup : llvm::enumerate(matrixSetupTasks)) {
    const TaskGraphNode *setupNode = indexedSetup.value();
    unsigned island = setupNode->index;

    auto selected = chooseGreedyLookaheadPlacement(
        static_cast<unsigned>(indexedSetup.index()), budget.greedyLookahead,
        currentCore, matrixSetupTasks, budget, *physicalArraysByCore,
        islandCommunicationEdges, usedSlotsByCore, coreByPlacedIsland);
    if (mlir::failed(selected)) {
      taskGraphFunc.emitError("failed to find an available core for greedy "
                              "island placement");
      return mlir::failure();
    }

    int64_t selectedCore = selected->coreId;
    unsigned localSlot = usedSlotsByCore[selectedCore]++;
    int64_t physicalArrayId = (*physicalArraysByCore)[selectedCore][localSlot];
    groupPlacements.push_back(task_schedulers::MatrixSetupGroupPlacement{
        setupNode->index, physicalArrayId});
    coreByPlacedIsland[island] = selectedCore;
    currentCore = selectedCore;
  }

  return groupPlacements;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

LogicalResult
runGreedyIslandPlacement(ModuleOp module, func::FuncOp taskGraphFunc,
                         const HardwareBudget &budget, const TaskGraphDAG &dag,
                         llvm::ArrayRef<int64_t> physicalArrayOrder) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks =
      collectMatrixSetupTasks(dag);
  if (!matrixSetupTasks.empty() && physicalArrayOrder.empty()) {
    taskGraphFunc.emitError("expected matrix setup island placement to have at "
                            "least one physical analog array");
    return failure();
  }

  auto islandGraph = buildLogicalPlacementIslandGraph(dag);
  if (failed(islandGraph))
    return failure();

  auto groupPlacements = buildGreedyIslandGroupPlacements(
      taskGraphFunc, matrixSetupTasks, budget, physicalArrayOrder,
      islandGraph->communicationEdges);
  if (failed(groupPlacements))
    return failure();

  return placeLogicalPlacementIslands(module, taskGraphFunc, budget, dag,
                                      *islandGraph, *groupPlacements);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
