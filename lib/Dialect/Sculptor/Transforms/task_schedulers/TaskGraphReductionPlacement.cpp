#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphReductionPlacement.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace {

namespace task_graph = mlir::sculptor::task_graph;
namespace task_schedulers = mlir::sculptor::task_schedulers;

struct CoreAnchor {
  int64_t coreId = 0;
  uint64_t weight = 0;
};

struct ReductionTree {
  int64_t treeId = 0;
  const task_graph::LogicalPlacementIsland *root = nullptr;
  llvm::SmallVector<const task_graph::LogicalPlacementIsland *, 4> lanes;
};

static uint64_t saturatingAdd(uint64_t lhs, uint64_t rhs) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return std::numeric_limits<uint64_t>::max();
  return lhs + rhs;
}

static uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0 || rhs == 0)
    return 0;
  if (lhs > std::numeric_limits<uint64_t>::max() / rhs)
    return std::numeric_limits<uint64_t>::max();
  return lhs * rhs;
}

static void appendAnchor(llvm::SmallVectorImpl<CoreAnchor> &anchors,
                         int64_t coreId, int64_t bytes) {
  if (bytes <= 0)
    return;
  uint64_t weight = static_cast<uint64_t>(bytes);
  for (CoreAnchor &anchor : anchors) {
    if (anchor.coreId != coreId)
      continue;
    anchor.weight = saturatingAdd(anchor.weight, weight);
    return;
  }
  anchors.push_back(CoreAnchor{coreId, weight});
}

static void appendPlacedAffinityAnchors(
    unsigned islandId, const task_graph::IslandAffinityGraph &affinityGraph,
    const llvm::DenseMap<unsigned, int64_t> &coreByIsland,
    llvm::SmallVectorImpl<CoreAnchor> &anchors) {
  for (const task_graph::IslandAffinityEdge &edge : affinityGraph.edges) {
    unsigned adjacentIsland = 0;
    if (edge.firstIsland == islandId)
      adjacentIsland = edge.secondIsland;
    else if (edge.secondIsland == islandId)
      adjacentIsland = edge.firstIsland;
    else
      continue;

    auto adjacentCore = coreByIsland.find(adjacentIsland);
    if (adjacentCore == coreByIsland.end())
      continue;
    appendAnchor(anchors, adjacentCore->second, edge.byteSize);
  }
}

static uint64_t
getWeightedCoordinate(llvm::ArrayRef<CoreAnchor> anchors,
                      llvm::function_ref<int64_t(int64_t)> getCoordinate) {
  llvm::SmallVector<std::pair<int64_t, uint64_t>, 8> weightedCoordinates;
  uint64_t totalWeight = 0;
  for (const CoreAnchor &anchor : anchors) {
    weightedCoordinates.push_back(
        {getCoordinate(anchor.coreId), anchor.weight});
    totalWeight = saturatingAdd(totalWeight, anchor.weight);
  }
  llvm::sort(weightedCoordinates, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  uint64_t prefixWeight = 0;
  for (const auto &[coordinate, weight] : weightedCoordinates) {
    prefixWeight = saturatingAdd(prefixWeight, weight);
    if (prefixWeight >= totalWeight - prefixWeight)
      return static_cast<uint64_t>(coordinate);
  }
  return weightedCoordinates.empty()
             ? 0
             : static_cast<uint64_t>(weightedCoordinates.back().first);
}

static int64_t getPreferredCore(llvm::ArrayRef<CoreAnchor> anchors,
                                const task_schedulers::HardwareBudget &budget) {
  if (anchors.empty()) {
    int64_t row = (budget.meshRows - 1) / 2;
    int64_t col = (budget.meshCols - 1) / 2;
    return row * budget.meshCols + col;
  }

  int64_t row =
      static_cast<int64_t>(getWeightedCoordinate(anchors, [&](int64_t coreId) {
        return task_schedulers::getMeshRow(coreId, budget);
      }));
  int64_t col =
      static_cast<int64_t>(getWeightedCoordinate(anchors, [&](int64_t coreId) {
        return task_schedulers::getMeshCol(coreId, budget);
      }));
  return row * budget.meshCols + col;
}

struct CoreScore {
  uint64_t communication = std::numeric_limits<uint64_t>::max();
  uint64_t existingWork = std::numeric_limits<uint64_t>::max();
  int64_t preferredDistance = std::numeric_limits<int64_t>::max();
  int64_t coreId = std::numeric_limits<int64_t>::max();
};

static bool isBetterCoreScore(const CoreScore &candidate,
                              const CoreScore &current) {
  if (candidate.communication != current.communication)
    return candidate.communication < current.communication;
  if (candidate.existingWork != current.existingWork)
    return candidate.existingWork < current.existingWork;
  if (candidate.preferredDistance != current.preferredDistance)
    return candidate.preferredDistance < current.preferredDistance;
  return candidate.coreId < current.coreId;
}

static mlir::FailureOr<int64_t>
chooseReductionCore(const task_schedulers::TaskGraphPlacementProblem &problem,
                    llvm::ArrayRef<CoreAnchor> anchors,
                    const llvm::DenseSet<int64_t> &excludedCores,
                    const llvm::DenseMap<int64_t, uint64_t> &workByCore) {
  int64_t preferredCore = getPreferredCore(anchors, problem.budget);
  std::optional<CoreScore> best;
  for (int64_t coreId = 0; coreId < problem.budget.numCores; ++coreId) {
    if (excludedCores.contains(coreId))
      continue;

    uint64_t communication = 0;
    for (const CoreAnchor &anchor : anchors) {
      int64_t distance = task_schedulers::getMeshDistance(anchor.coreId, coreId,
                                                          problem.budget);
      communication = saturatingAdd(
          communication,
          saturatingMultiply(anchor.weight, static_cast<uint64_t>(distance)));
    }
    CoreScore score{
        communication, workByCore.lookup(coreId),
        task_schedulers::getMeshDistance(preferredCore, coreId, problem.budget),
        coreId};
    if (!best || isBetterCoreScore(score, *best))
      best = score;
  }

  if (!best) {
    problem.diagnosticOp->emitError(
        "could not find a distinct core for a reduction lane");
    return mlir::failure();
  }
  return best->coreId;
}

static mlir::FailureOr<llvm::SmallVector<ReductionTree, 4>>
collectReductionTrees(
    const task_schedulers::TaskGraphPlacementProblem &problem) {
  llvm::SmallVector<ReductionTree, 4> trees;
  llvm::DenseMap<int64_t, unsigned> treeOrdinalById;
  for (const task_graph::LogicalPlacementIsland &island :
       problem.islandGraph.islands) {
    if (!task_graph::isReductionIsland(island))
      continue;
    if (!island.reductionTreeId || !island.reductionLevel ||
        !island.reductionWidth) {
      problem.diagnosticOp->emitError(
          "expected reduction islands to carry tree, level, and width "
          "metadata");
      return mlir::failure();
    }

    auto inserted =
        treeOrdinalById.try_emplace(*island.reductionTreeId, trees.size());
    if (inserted.second)
      trees.push_back(ReductionTree{*island.reductionTreeId});
    ReductionTree &tree = trees[inserted.first->second];
    if (task_graph::isReductionLaneIsland(island)) {
      if (!island.reductionLane) {
        problem.diagnosticOp->emitError(
            "expected a first-stage reduction island to carry a lane");
        return mlir::failure();
      }
      tree.lanes.push_back(&island);
      continue;
    }
    if (!task_graph::isReductionRootIsland(island) || tree.root) {
      problem.diagnosticOp->emitError(
          "expected one root island per balanced reduction tree");
      return mlir::failure();
    }
    tree.root = &island;
  }

  llvm::sort(trees, [](const ReductionTree &lhs, const ReductionTree &rhs) {
    return lhs.treeId < rhs.treeId;
  });
  for (ReductionTree &tree : trees) {
    if (!tree.root) {
      problem.diagnosticOp->emitError(
          "expected one root island per balanced reduction tree");
      return mlir::failure();
    }
    llvm::sort(tree.lanes, [](const auto *lhs, const auto *rhs) {
      if (lhs->reductionLane != rhs->reductionLane)
        return lhs->reductionLane < rhs->reductionLane;
      return lhs->islandIndex < rhs->islandIndex;
    });
  }
  return trees;
}

static uint64_t
getIslandWork(const task_graph::LogicalPlacementIsland &island) {
  return std::max<uint64_t>(1, island.taskIndices.size());
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<llvm::DenseMap<unsigned, int64_t>>
buildSpatialReductionCorePlacements(
    const TaskGraphPlacementProblem &problem,
    const llvm::DenseMap<unsigned, int64_t> &coreByAnalogIsland) {
  auto trees = collectReductionTrees(problem);
  if (failed(trees))
    return failure();

  llvm::DenseMap<unsigned, int64_t> coreByIsland = coreByAnalogIsland;
  llvm::DenseMap<int64_t, uint64_t> workByCore;
  for (const task_graph::LogicalPlacementIsland &island :
       problem.islandGraph.islands) {
    auto core = coreByAnalogIsland.find(island.islandIndex);
    if (core == coreByAnalogIsland.end())
      continue;
    workByCore[core->second] =
        saturatingAdd(workByCore.lookup(core->second), getIslandWork(island));
  }

  llvm::DenseMap<unsigned, int64_t> reductionCoreByIsland;
  for (const ReductionTree &tree : *trees) {
    llvm::SmallVector<CoreAnchor, 8> rootTargetAnchors;
    appendPlacedAffinityAnchors(tree.root->islandIndex,
                                problem.islandGraph.affinityGraph, coreByIsland,
                                rootTargetAnchors);
    for (const task_graph::LogicalPlacementIsland *lane : tree.lanes) {
      appendPlacedAffinityAnchors(lane->islandIndex,
                                  problem.islandGraph.affinityGraph,
                                  coreByIsland, rootTargetAnchors);
    }
    int64_t provisionalRootCore =
        getPreferredCore(rootTargetAnchors, problem.budget);

    llvm::DenseSet<int64_t> laneCores;
    for (const task_graph::LogicalPlacementIsland *lane : tree.lanes) {
      llvm::SmallVector<CoreAnchor, 8> laneAnchors;
      appendPlacedAffinityAnchors(lane->islandIndex,
                                  problem.islandGraph.affinityGraph,
                                  coreByIsland, laneAnchors);
      for (const task_graph::IslandAffinityEdge &edge :
           problem.islandGraph.affinityGraph.edges) {
        bool connectsRoot = (edge.firstIsland == lane->islandIndex &&
                             edge.secondIsland == tree.root->islandIndex) ||
                            (edge.secondIsland == lane->islandIndex &&
                             edge.firstIsland == tree.root->islandIndex);
        if (connectsRoot) {
          appendAnchor(laneAnchors, provisionalRootCore, edge.byteSize);
        }
      }

      auto core =
          chooseReductionCore(problem, laneAnchors, laneCores, workByCore);
      if (failed(core))
        return failure();
      laneCores.insert(*core);
      coreByIsland[lane->islandIndex] = *core;
      reductionCoreByIsland[lane->islandIndex] = *core;
      workByCore[*core] =
          saturatingAdd(workByCore.lookup(*core), getIslandWork(*lane));
    }

    llvm::SmallVector<CoreAnchor, 8> rootAnchors;
    appendPlacedAffinityAnchors(tree.root->islandIndex,
                                problem.islandGraph.affinityGraph, coreByIsland,
                                rootAnchors);
    llvm::DenseSet<int64_t> noExcludedCores;
    auto rootCore =
        chooseReductionCore(problem, rootAnchors, noExcludedCores, workByCore);
    if (failed(rootCore))
      return failure();
    coreByIsland[tree.root->islandIndex] = *rootCore;
    reductionCoreByIsland[tree.root->islandIndex] = *rootCore;
    workByCore[*rootCore] =
        saturatingAdd(workByCore.lookup(*rootCore), getIslandWork(*tree.root));
  }

  return reductionCoreByIsland;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
