#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphDigitalShardPlacement.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

namespace {

namespace task_graph = mlir::sculptor::task_graph;
namespace task_schedulers = mlir::sculptor::task_schedulers;

struct CoreAnchor {
  int64_t coreId = 0;
  uint64_t bytes = 0;
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
  for (CoreAnchor &anchor : anchors) {
    if (anchor.coreId != coreId)
      continue;
    anchor.bytes = saturatingAdd(anchor.bytes, static_cast<uint64_t>(bytes));
    return;
  }
  anchors.push_back(CoreAnchor{coreId, static_cast<uint64_t>(bytes)});
}

static llvm::SmallVector<CoreAnchor, 8>
collectAnchors(unsigned islandId,
               const task_graph::IslandAffinityGraph &affinityGraph,
               const llvm::DenseMap<unsigned, int64_t> &coreByIsland) {
  llvm::SmallVector<CoreAnchor, 8> anchors;
  for (const task_graph::IslandAffinityEdge &edge : affinityGraph.edges) {
    unsigned adjacent = 0;
    if (edge.firstIsland == islandId)
      adjacent = edge.secondIsland;
    else if (edge.secondIsland == islandId)
      adjacent = edge.firstIsland;
    else
      continue;
    auto core = coreByIsland.find(adjacent);
    if (core != coreByIsland.end())
      appendAnchor(anchors, core->second, edge.byteSize);
  }
  return anchors;
}

static int64_t getPreferredCore(llvm::ArrayRef<CoreAnchor> anchors,
                                const task_schedulers::HardwareBudget &budget) {
  if (anchors.empty())
    return ((budget.meshRows - 1) / 2) * budget.meshCols +
           (budget.meshCols - 1) / 2;

  uint64_t totalBytes = 0;
  uint64_t weightedRows = 0;
  uint64_t weightedColumns = 0;
  for (const CoreAnchor &anchor : anchors) {
    totalBytes = saturatingAdd(totalBytes, anchor.bytes);
    weightedRows = saturatingAdd(
        weightedRows,
        saturatingMultiply(anchor.bytes,
                           static_cast<uint64_t>(task_schedulers::getMeshRow(
                               anchor.coreId, budget))));
    weightedColumns = saturatingAdd(
        weightedColumns,
        saturatingMultiply(anchor.bytes,
                           static_cast<uint64_t>(task_schedulers::getMeshCol(
                               anchor.coreId, budget))));
  }
  if (totalBytes == 0)
    return 0;
  int64_t row = static_cast<int64_t>(weightedRows / totalBytes);
  int64_t column = static_cast<int64_t>(weightedColumns / totalBytes);
  row = std::min(row, budget.meshRows - 1);
  column = std::min(column, budget.meshCols - 1);
  return row * budget.meshCols + column;
}

struct CoreScore {
  uint64_t communicationAndPolicy = std::numeric_limits<uint64_t>::max();
  uint64_t existingWork = std::numeric_limits<uint64_t>::max();
  int64_t preferredDistance = std::numeric_limits<int64_t>::max();
  int64_t coreId = std::numeric_limits<int64_t>::max();
};

static bool isBetter(const CoreScore &candidate, const CoreScore &current) {
  return std::tie(candidate.communicationAndPolicy, candidate.existingWork,
                  candidate.preferredDistance, candidate.coreId) <
         std::tie(current.communicationAndPolicy, current.existingWork,
                  current.preferredDistance, current.coreId);
}

static mlir::FailureOr<int64_t>
chooseCore(const task_schedulers::TaskGraphPlacementProblem &problem,
           llvm::ArrayRef<CoreAnchor> anchors,
           mlir::sculptor::DistributionPlacementPolicy policy,
           const llvm::DenseSet<int64_t> &groupCores,
           const llvm::DenseMap<int64_t, uint64_t> &workByCore) {
  int64_t preferredCore = getPreferredCore(anchors, problem.budget);
  uint64_t totalAnchorBytes = 0;
  for (const CoreAnchor &anchor : anchors)
    totalAnchorBytes = saturatingAdd(totalAnchorBytes, anchor.bytes);

  std::optional<CoreScore> best;
  for (int64_t coreId = 0; coreId < problem.budget.numCores; ++coreId) {
    bool reusesGroupCore = groupCores.contains(coreId);
    if (policy ==
            mlir::sculptor::DistributionPlacementPolicy::RequireDistinct &&
        reusesGroupCore)
      continue;

    uint64_t communication = 0;
    for (const CoreAnchor &anchor : anchors) {
      communication = saturatingAdd(
          communication,
          saturatingMultiply(
              anchor.bytes,
              static_cast<uint64_t>(task_schedulers::getMeshDistance(
                  anchor.coreId, coreId, problem.budget))));
    }
    if (policy == mlir::sculptor::DistributionPlacementPolicy::PreferDistinct &&
        reusesGroupCore) {
      communication =
          saturatingAdd(communication, std::max<uint64_t>(1, totalAnchorBytes));
    }

    CoreScore score{
        communication, workByCore.lookup(coreId),
        task_schedulers::getMeshDistance(preferredCore, coreId, problem.budget),
        coreId};
    if (!best || isBetter(score, *best))
      best = score;
  }
  if (!best) {
    problem.diagnosticOp->emitError(
        "could not satisfy digital shard core placement policy");
    return mlir::failure();
  }
  return best->coreId;
}

static uint64_t
getIslandWork(const task_graph::LogicalPlacementIsland &island) {
  return std::max<uint64_t>(1, island.taskIndices.size());
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<llvm::DenseMap<unsigned, int64_t>> buildDigitalShardCorePlacements(
    const TaskGraphPlacementProblem &problem,
    const llvm::DenseMap<unsigned, int64_t> &initialCoreByIsland) {
  llvm::DenseMap<unsigned, const task_graph::LogicalPlacementIsland *>
      islandById;
  for (const task_graph::LogicalPlacementIsland &island :
       problem.islandGraph.islands)
    islandById.try_emplace(island.islandIndex, &island);

  llvm::DenseMap<unsigned, int64_t> coreByIsland = initialCoreByIsland;
  llvm::DenseMap<int64_t, uint64_t> workByCore;
  for (const auto &entry : initialCoreByIsland) {
    auto island = islandById.find(entry.first);
    if (island == islandById.end())
      continue;
    workByCore[entry.second] = saturatingAdd(workByCore.lookup(entry.second),
                                             getIslandWork(*island->second));
  }

  llvm::DenseMap<unsigned, int64_t> shardCoreByIsland;
  for (const DistributedShardPlacementConstraint &group :
       problem.constraints.distributedShardGroups) {
    if (group.policy == DistributionPlacementPolicy::RequireDistinct &&
        group.shardCount > problem.budget.numCores) {
      problem.diagnosticOp->emitError()
          << "digital distribution group " << group.groupId << " requires "
          << group.shardCount << " distinct cores, but the hardware has "
          << problem.budget.numCores;
      return failure();
    }

    llvm::DenseSet<int64_t> groupCores;
    for (unsigned islandId : group.islandsByShard) {
      auto island = islandById.find(islandId);
      if (island == islandById.end()) {
        problem.diagnosticOp->emitError(
            "digital shard constraint references an unknown island");
        return failure();
      }
      llvm::SmallVector<CoreAnchor, 8> anchors = collectAnchors(
          islandId, problem.islandGraph.affinityGraph, coreByIsland);
      auto core =
          chooseCore(problem, anchors, group.policy, groupCores, workByCore);
      if (failed(core))
        return failure();
      groupCores.insert(*core);
      coreByIsland[islandId] = *core;
      shardCoreByIsland[islandId] = *core;
      workByCore[*core] = saturatingAdd(workByCore.lookup(*core),
                                        getIslandWork(*island->second));
    }
  }
  return shardCoreByIsland;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
