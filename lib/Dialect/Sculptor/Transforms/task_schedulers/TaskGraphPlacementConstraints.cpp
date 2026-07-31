#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementConstraints.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<PlacementConstraints> buildPlacementConstraints(
    Operation *diagnosticOp,
    const task_graph::TaskExecutionGraph &executionGraph,
    const task_graph::LogicalPlacementIslandGraph &islandGraph) {
  PlacementConstraints constraints;
  if (executionGraph.topologicalOrder.empty())
    return constraints;

  unsigned startTask = executionGraph.topologicalOrder.front();
  unsigned terminalTask = executionGraph.topologicalOrder.back();
  auto startIsland = islandGraph.islandByTaskIndex.find(startTask);
  auto terminalIsland = islandGraph.islandByTaskIndex.find(terminalTask);
  bool hasStartIsland = startIsland != islandGraph.islandByTaskIndex.end();
  bool hasTerminalIsland =
      terminalIsland != islandGraph.islandByTaskIndex.end();
  if (hasStartIsland != hasTerminalIsland) {
    diagnosticOp->emitError(
        "expected execution endpoints to have consistent island coverage");
    return failure();
  }

  SharedMeshBoundaryConstraint boundary{startTask, terminalTask, std::nullopt};
  if (hasStartIsland) {
    boundary.islands =
        IslandBoundaryEndpoints{startIsland->second, terminalIsland->second};
  }
  constraints.sharedEndpointBoundary = boundary;

  llvm::DenseMap<int64_t, unsigned> groupOrdinalById;
  llvm::SmallVector<llvm::SmallVector<std::pair<int64_t, unsigned>, 8>, 4>
      shardsByGroup;
  for (const task_graph::LogicalPlacementIsland &island : islandGraph.islands) {
    if (!task_graph::isDigitalShardIsland(island))
      continue;
    if (!island.distributionGroupId || !island.distributionShardId ||
        !island.distributionShardCount || !island.distributionPlacement) {
      diagnosticOp->emitError(
          "expected digital shard island to carry complete distribution "
          "metadata");
      return failure();
    }

    auto inserted = groupOrdinalById.try_emplace(
        *island.distributionGroupId,
        static_cast<unsigned>(constraints.distributedShardGroups.size()));
    if (inserted.second) {
      constraints.distributedShardGroups.push_back(
          DistributedShardPlacementConstraint{*island.distributionGroupId,
                                              *island.distributionShardCount,
                                              *island.distributionPlacement,
                                              {}});
      shardsByGroup.emplace_back();
    }
    unsigned ordinal = inserted.first->second;
    DistributedShardPlacementConstraint &group =
        constraints.distributedShardGroups[ordinal];
    if (group.shardCount != *island.distributionShardCount ||
        group.policy != *island.distributionPlacement) {
      diagnosticOp->emitError(
          "expected one shard count and placement policy per digital "
          "distribution group");
      return failure();
    }
    shardsByGroup[ordinal].push_back(
        {*island.distributionShardId, island.islandIndex});
  }

  for (auto indexedGroup :
       llvm::enumerate(constraints.distributedShardGroups)) {
    auto &shards = shardsByGroup[indexedGroup.index()];
    llvm::sort(shards);
    if (static_cast<int64_t>(shards.size()) !=
        indexedGroup.value().shardCount) {
      diagnosticOp->emitError()
          << "expected digital distribution group "
          << indexedGroup.value().groupId << " to contain "
          << indexedGroup.value().shardCount << " shard islands";
      return failure();
    }
    for (auto indexedShard : llvm::enumerate(shards)) {
      if (indexedShard.value().first !=
          static_cast<int64_t>(indexedShard.index())) {
        diagnosticOp->emitError()
            << "expected contiguous shard IDs in digital distribution group "
            << indexedGroup.value().groupId;
        return failure();
      }
      indexedGroup.value().islandsByShard.push_back(
          indexedShard.value().second);
    }
  }
  llvm::sort(constraints.distributedShardGroups,
             [](const DistributedShardPlacementConstraint &lhs,
                const DistributedShardPlacementConstraint &rhs) {
               return lhs.groupId < rhs.groupId;
             });
  return constraints;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
