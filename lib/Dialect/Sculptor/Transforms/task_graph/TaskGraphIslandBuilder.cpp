#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseSet.h"

#include "TaskGraphIslandInternals.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <queue>
#include <utility>

namespace {

namespace task_graph = mlir::sculptor::task_graph;

using TaskGraphNode = task_graph::TaskGraphNode;
using IslandAffinityEdge = task_graph::IslandAffinityEdge;
using MatrixSetupMVMMap =
    llvm::DenseMap<unsigned, llvm::SmallVector<const TaskGraphNode *, 4>>;

static MatrixSetupMVMMap buildMVMTasksByMatrixSetupTask(
    const task_graph::TaskGraphDAG &dag,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks) {
  MatrixSetupMVMMap mvmTasksByMatrixSetupTask;
  for (const TaskGraphNode *setupNode : matrixSetupTasks) {
    llvm::SmallVector<const TaskGraphNode *, 4> &mvmTasks =
        mvmTasksByMatrixSetupTask[setupNode->index];

    for (unsigned successorIndex : setupNode->successors) {
      const TaskGraphNode &successorNode = dag.nodes[successorIndex];
      if (task_graph::isAnalogComputeTask(successorNode.op))
        mvmTasks.push_back(&successorNode);
    }
  }
  return mvmTasksByMatrixSetupTask;
}

static void recordInitialMatrixSetupIslands(
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const MatrixSetupMVMMap &mvmTasksByMatrixSetupTask,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  for (const TaskGraphNode *setupNode : matrixSetupTasks) {
    islandByTaskIndex.try_emplace(setupNode->index, setupNode->index);

    auto mvmTasksIt = mvmTasksByMatrixSetupTask.find(setupNode->index);
    if (mvmTasksIt == mvmTasksByMatrixSetupTask.end())
      continue;

    for (const TaskGraphNode *mvmNode : mvmTasksIt->second)
      islandByTaskIndex.try_emplace(mvmNode->index, setupNode->index);
  }
}

struct ReductionIslandMetadata {
  int64_t treeId = 0;
  int64_t level = 0;
  std::optional<int64_t> lane;
  int64_t width = 0;
};

static mlir::FailureOr<ReductionIslandMetadata>
getReductionIslandMetadata(const TaskGraphNode &node) {
  auto treeId = node.op->getAttrOfType<mlir::IntegerAttr>(
      mlir::sculptor::task_graph_attrs::kTaskReductionTreeIdAttrName);
  auto level = node.op->getAttrOfType<mlir::IntegerAttr>(
      mlir::sculptor::task_graph_attrs::kTaskReductionLevelAttrName);
  auto lane = node.op->getAttrOfType<mlir::IntegerAttr>(
      mlir::sculptor::task_graph_attrs::kTaskReductionLaneAttrName);
  auto width = node.op->getAttrOfType<mlir::IntegerAttr>(
      mlir::sculptor::task_graph_attrs::kTaskReductionWidthAttrName);
  if (!treeId || !level || !width || treeId.getInt() < 0 ||
      level.getInt() < 0 || level.getInt() > 1 || width.getInt() < 2) {
    node.op->emitError(
        "expected valid reduction tree, level, and width metadata");
    return mlir::failure();
  }

  if (level.getInt() == 0 &&
      (!lane || lane.getInt() < 0 || lane.getInt() >= width.getInt())) {
    node.op->emitError(
        "expected a first-stage reduction to carry a valid lane");
    return mlir::failure();
  }
  if (level.getInt() == 1 && lane && lane.getInt() != 0) {
    node.op->emitError(
        "expected a legacy root reduction lane, when present, to be zero");
    return mlir::failure();
  }

  return ReductionIslandMetadata{treeId.getInt(), level.getInt(),
                                 level.getInt() == 0
                                     ? std::optional<int64_t>(lane.getInt())
                                     : std::nullopt,
                                 width.getInt()};
}

static mlir::LogicalResult recordInitialReductionIslands(
    const task_graph::TaskGraphDAG &dag,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::DenseMap<std::pair<int64_t, int64_t>, unsigned> islandByTreeLane;
  for (const TaskGraphNode &node : dag.nodes) {
    if (!task_graph::isReductionTask(node.op))
      continue;

    auto metadata = getReductionIslandMetadata(node);
    if (mlir::failed(metadata))
      return mlir::failure();
    int64_t branch = metadata->lane.value_or(metadata->width);
    auto key = std::make_pair(metadata->treeId, branch);
    auto inserted = islandByTreeLane.try_emplace(key, node.index);
    islandByTaskIndex[node.index] = inserted.first->second;
  }
  return mlir::success();
}

static mlir::FailureOr<mlir::sculptor::TaskDistributionAttr>
getShardDistributionMetadata(const TaskGraphNode &node) {
  auto distribution =
      node.op->getAttrOfType<mlir::sculptor::TaskDistributionAttr>(
          mlir::sculptor::task_graph_attrs::kTaskDistributionAttrName);
  if (!distribution ||
      distribution.getRole() != mlir::sculptor::TaskDistributionRole::Shard)
    return mlir::failure();

  int64_t groupId = distribution.getGroupId().getInt();
  int64_t shardId = distribution.getShardId().getInt();
  int64_t shardCount = distribution.getShardCount().getInt();
  int64_t rowShard = distribution.getRowShard().getInt();
  int64_t rowShards = distribution.getRowShards().getInt();
  int64_t columnShard = distribution.getColumnShard().getInt();
  int64_t columnShards = distribution.getColumnShards().getInt();
  if (groupId < 0 || shardCount < 2 || shardId < 0 || shardId >= shardCount ||
      rowShards <= 0 || columnShards <= 0 || rowShard < 0 ||
      rowShard >= rowShards || columnShard < 0 || columnShard >= columnShards ||
      rowShards > std::numeric_limits<int64_t>::max() / columnShards ||
      rowShards * columnShards != shardCount) {
    node.op->emitError("expected valid digital matmul shard metadata");
    return mlir::failure();
  }
  return distribution;
}

static mlir::LogicalResult recordInitialDigitalShardIslands(
    const task_graph::TaskGraphDAG &dag,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::DenseSet<std::pair<int64_t, int64_t>> seenShards;
  for (const TaskGraphNode &node : dag.nodes) {
    auto distribution =
        node.op->getAttrOfType<mlir::sculptor::TaskDistributionAttr>(
            mlir::sculptor::task_graph_attrs::kTaskDistributionAttrName);
    if (!distribution ||
        distribution.getRole() != mlir::sculptor::TaskDistributionRole::Shard)
      continue;

    auto metadata = getShardDistributionMetadata(node);
    if (mlir::failed(metadata))
      return mlir::failure();
    auto key = std::make_pair(metadata->getGroupId().getInt(),
                              metadata->getShardId().getInt());
    if (!seenShards.insert(key).second) {
      node.op->emitError(
          "expected one digital matmul task for each distribution shard");
      return mlir::failure();
    }
    islandByTaskIndex.try_emplace(node.index, node.index);
  }
  return mlir::success();
}

static void appendUniqueTaskIndex(llvm::SmallVectorImpl<unsigned> &tasks,
                                  unsigned taskIndex) {
  for (unsigned existingTask : tasks) {
    if (existingTask == taskIndex)
      return;
  }

  tasks.push_back(taskIndex);
}

static void assignUncoveredExecutionEndpointIsland(
    const task_graph::TaskExecutionGraph &executionGraph,
    const llvm::DenseSet<unsigned> &anchoredIslands, unsigned endpoint,
    bool traverseForward,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  if (islandByTaskIndex.contains(endpoint) ||
      endpoint >= executionGraph.topologicalOrder.size())
    return;

  llvm::SmallVector<bool, 16> visited(executionGraph.topologicalOrder.size(),
                                      false);
  std::queue<unsigned> worklist;
  visited[endpoint] = true;
  worklist.push(endpoint);
  std::optional<unsigned> selectedIsland;
  while (!worklist.empty() && !selectedIsland) {
    size_t levelSize = worklist.size();
    llvm::SmallVector<unsigned, 4> candidates;
    for (size_t item = 0; item < levelSize; ++item) {
      unsigned current = worklist.front();
      worklist.pop();
      const auto &edgeIndices = traverseForward
                                    ? executionGraph.outgoingEdges[current]
                                    : executionGraph.incomingEdges[current];
      for (unsigned edgeIndex : edgeIndices) {
        const task_graph::TaskExecutionEdge &edge =
            executionGraph.edges[edgeIndex];
        unsigned adjacent =
            traverseForward ? edge.consumerTask : edge.producerTask;
        auto island = islandByTaskIndex.find(adjacent);
        if (island != islandByTaskIndex.end() &&
            anchoredIslands.contains(island->second)) {
          candidates.push_back(island->second);
          continue;
        }
        if (!visited[adjacent]) {
          visited[adjacent] = true;
          worklist.push(adjacent);
        }
      }
    }
    if (!candidates.empty()) {
      selectedIsland =
          traverseForward
              ? *std::min_element(candidates.begin(), candidates.end())
              : *std::max_element(candidates.begin(), candidates.end());
    }
  }
  if (selectedIsland)
    islandByTaskIndex[endpoint] = *selectedIsland;
}

static void assignExecutionEndpointIslands(
    const task_graph::TaskExecutionGraph &executionGraph,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  if (executionGraph.topologicalOrder.empty())
    return;

  llvm::DenseSet<unsigned> anchoredIslands;
  for (const auto &entry : islandByTaskIndex)
    anchoredIslands.insert(entry.second);
  assignUncoveredExecutionEndpointIsland(
      executionGraph, anchoredIslands, executionGraph.topologicalOrder.front(),
      /*traverseForward=*/true, islandByTaskIndex);
  assignUncoveredExecutionEndpointIsland(
      executionGraph, anchoredIslands, executionGraph.topologicalOrder.back(),
      /*traverseForward=*/false, islandByTaskIndex);
}

static mlir::LogicalResult assignUncoveredDigitalTasksToNearestIsland(
    const task_graph::TaskGraphDAG &dag,
    const task_graph::TaskExecutionGraph &executionGraph,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::DenseSet<unsigned> anchoredIslands;
  for (const auto &entry : islandByTaskIndex)
    anchoredIslands.insert(entry.second);
  if (anchoredIslands.empty())
    return mlir::success();

  for (const TaskGraphNode &node : dag.nodes) {
    if (islandByTaskIndex.contains(node.index))
      continue;
    if (!task_graph::isDigitalTask(node.op)) {
      node.op->emitError(
          "expected every non-digital task to have an anchored logical island");
      return mlir::failure();
    }

    llvm::SmallVector<bool, 16> visited(dag.nodes.size(), false);
    std::queue<unsigned> worklist;
    visited[node.index] = true;
    worklist.push(node.index);
    std::optional<unsigned> selectedIsland;
    while (!worklist.empty() && !selectedIsland) {
      size_t levelSize = worklist.size();
      llvm::SmallVector<unsigned, 4> candidates;
      for (size_t item = 0; item < levelSize; ++item) {
        unsigned current = worklist.front();
        worklist.pop();
        auto visitAdjacent = [&](unsigned adjacent) {
          auto island = islandByTaskIndex.find(adjacent);
          if (island != islandByTaskIndex.end() &&
              anchoredIslands.contains(island->second)) {
            candidates.push_back(island->second);
            return;
          }
          if (!visited[adjacent]) {
            visited[adjacent] = true;
            worklist.push(adjacent);
          }
        };
        for (unsigned edgeIndex : executionGraph.incomingEdges[current])
          visitAdjacent(executionGraph.edges[edgeIndex].producerTask);
        for (unsigned edgeIndex : executionGraph.outgoingEdges[current])
          visitAdjacent(executionGraph.edges[edgeIndex].consumerTask);
      }
      if (!candidates.empty())
        selectedIsland =
            *std::min_element(candidates.begin(), candidates.end());
    }

    if (!selectedIsland) {
      node.op->emitError(
          "could not assign digital task to a reachable logical island");
      return mlir::failure();
    }
    islandByTaskIndex[node.index] = *selectedIsland;
  }
  return mlir::success();
}

static mlir::FailureOr<task_graph::LogicalPlacementIslandGraph>
assembleLogicalPlacementIslandGraph(
    const task_graph::TaskGraphDAG &dag,
    const task_graph::TaskExecutionGraph &executionGraph,
    llvm::ArrayRef<IslandAffinityEdge> affinityEdges,
    const llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  task_graph::LogicalPlacementIslandGraph graph;
  graph.islandByTaskIndex = islandByTaskIndex;
  graph.affinityGraph.edges.append(affinityEdges.begin(), affinityEdges.end());
  graph.executionGraph =
      task_graph::buildIslandExecutionGraph(executionGraph, islandByTaskIndex);

  llvm::DenseMap<unsigned, llvm::SmallVector<const TaskGraphNode *, 8>>
      nodesByIsland;
  for (const TaskGraphNode &node : dag.nodes) {
    auto islandIt = islandByTaskIndex.find(node.index);
    if (islandIt != islandByTaskIndex.end())
      nodesByIsland[islandIt->second].push_back(&node);
  }

  llvm::SmallVector<unsigned, 16> islandIndices;
  islandIndices.reserve(nodesByIsland.size());
  for (const auto &entry : nodesByIsland)
    islandIndices.push_back(entry.first);
  llvm::sort(islandIndices);

  llvm::DenseMap<unsigned, unsigned> islandOrdinalByIndex;
  graph.islands.reserve(islandIndices.size());
  for (unsigned islandIndex : islandIndices) {
    task_graph::LogicalPlacementIsland island;
    island.islandIndex = islandIndex;
    bool foundReductionTask = false;
    bool foundDigitalShardTask = false;
    for (const TaskGraphNode *node : nodesByIsland.lookup(islandIndex)) {
      if (task_graph::isMatrixSetupTask(node->op)) {
        if (island.matrixSetupTaskIndex) {
          node->op->emitError(
              "expected one matrix setup task per analog placement island");
          return mlir::failure();
        }
        island.matrixSetupTaskIndex = node->index;
      }

      if (!task_graph::isReductionTask(node->op)) {
        auto distribution =
            node->op->getAttrOfType<mlir::sculptor::TaskDistributionAttr>(
                mlir::sculptor::task_graph_attrs::kTaskDistributionAttrName);
        if (!distribution || distribution.getRole() !=
                                 mlir::sculptor::TaskDistributionRole::Shard)
          continue;
        auto metadata = getShardDistributionMetadata(*node);
        if (mlir::failed(metadata))
          return mlir::failure();
        if (foundDigitalShardTask) {
          node->op->emitError(
              "expected one digital shard anchor per placement island");
          return mlir::failure();
        }
        island.distributionGroupId = metadata->getGroupId().getInt();
        island.distributionShardId = metadata->getShardId().getInt();
        island.distributionShardCount = metadata->getShardCount().getInt();
        island.distributionPlacement = metadata->getPlacement();
        foundDigitalShardTask = true;
        continue;
      } else {
        auto metadata = getReductionIslandMetadata(*node);
        if (mlir::failed(metadata))
          return mlir::failure();
        if (!foundReductionTask) {
          island.reductionTreeId = metadata->treeId;
          island.reductionLevel = metadata->level;
          island.reductionLane = metadata->lane;
          island.reductionWidth = metadata->width;
          foundReductionTask = true;
        } else if (island.reductionTreeId != metadata->treeId ||
                   island.reductionLevel != metadata->level ||
                   island.reductionLane != metadata->lane ||
                   island.reductionWidth != metadata->width) {
          node->op->emitError(
              "expected a reduction island to contain one tree branch");
          return mlir::failure();
        }
      }
    }

    unsigned anchorKindCount =
        static_cast<unsigned>(island.matrixSetupTaskIndex.has_value()) +
        static_cast<unsigned>(foundReductionTask) +
        static_cast<unsigned>(foundDigitalShardTask);
    if (anchorKindCount > 1) {
      nodesByIsland.lookup(islandIndex)
          .front()
          ->op->emitError(
              "expected analog, reduction, and digital shard tasks to use "
              "separate islands");
      return mlir::failure();
    }
    if (foundReductionTask) {
      island.kind = task_graph::LogicalPlacementIslandKind::Reduction;
      for (const TaskGraphNode *node : nodesByIsland.lookup(islandIndex)) {
        if (!task_graph::isReductionTask(node->op)) {
          node->op->emitError(
              "expected reduction islands to contain only reduction tasks");
          return mlir::failure();
        }
      }
    } else if (foundDigitalShardTask) {
      island.kind = task_graph::LogicalPlacementIslandKind::DigitalShard;
      for (const TaskGraphNode *node : nodesByIsland.lookup(islandIndex)) {
        if (!task_graph::isDigitalTask(node->op)) {
          node->op->emitError(
              "expected digital shard islands to contain only digital tasks");
          return mlir::failure();
        }
      }
    } else if (!island.matrixSetupTaskIndex) {
      dag.nodes[nodesByIsland.lookup(islandIndex).front()->index].op->emitError(
          "expected logical island to have an analog, reduction, or digital "
          "shard anchor");
      return mlir::failure();
    }

    unsigned ordinal = static_cast<unsigned>(graph.islands.size());
    graph.islands.push_back(std::move(island));
    islandOrdinalByIndex.try_emplace(islandIndex, ordinal);
  }

  for (const TaskGraphNode &node : dag.nodes) {
    auto islandIt = islandByTaskIndex.find(node.index);
    if (islandIt == islandByTaskIndex.end())
      continue;

    auto ordinalIt = islandOrdinalByIndex.find(islandIt->second);
    if (ordinalIt == islandOrdinalByIndex.end())
      continue;

    task_graph::LogicalPlacementIsland &island =
        graph.islands[ordinalIt->second];
    appendUniqueTaskIndex(island.taskIndices, node.index);

    if (island.matrixSetupTaskIndex &&
        node.index == *island.matrixSetupTaskIndex)
      continue;

    if (task_graph::isAnalogComputeTask(node.op)) {
      appendUniqueTaskIndex(island.mvmTaskIndices, node.index);
      continue;
    }

    if (task_graph::isDigitalTask(node.op))
      appendUniqueTaskIndex(island.digitalTaskIndices, node.index);
  }

  return graph;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_graph {

FailureOr<LogicalPlacementIslandGraph>
buildLogicalPlacementIslandGraph(const TaskGraphDAG &dag,
                                 const TaskExecutionGraph &executionGraph,
                                 DigitalIslandAssignmentPolicy policy) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks =
      collectMatrixSetupTasks(dag);
  MatrixSetupMVMMap mvmTasksByMatrixSetupTask =
      buildMVMTasksByMatrixSetupTask(dag, matrixSetupTasks);

  llvm::DenseMap<unsigned, unsigned> islandByTaskIndex;
  recordInitialMatrixSetupIslands(matrixSetupTasks, mvmTasksByMatrixSetupTask,
                                  islandByTaskIndex);
  if (failed(recordInitialReductionIslands(dag, islandByTaskIndex)))
    return failure();
  if (failed(recordInitialDigitalShardIslands(dag, islandByTaskIndex)))
    return failure();
  if (policy == DigitalIslandAssignmentPolicy::Legacy &&
      failed(assignPrePlacementMinCutDigitalIslands(dag, islandByTaskIndex)))
    return failure();

  auto resourceEdges = collectResourceEdges(dag);
  if (failed(resourceEdges))
    return failure();

  if (policy == DigitalIslandAssignmentPolicy::MultiTerminalBalanced &&
      failed(assignMultiTerminalBalancedDigitalIslands(dag, *resourceEdges,
                                                       islandByTaskIndex))) {
    return failure();
  }

  if (failed(assignRemainingDigitalIslandsByLocalAffinity(dag, *resourceEdges,
                                                          islandByTaskIndex)))
    return failure();

  assignExecutionEndpointIslands(executionGraph, islandByTaskIndex);
  if (failed(assignUncoveredDigitalTasksToNearestIsland(dag, executionGraph,
                                                        islandByTaskIndex)))
    return failure();

  llvm::SmallVector<IslandAffinityEdge, 16> affinityEdges =
      buildIslandAffinityEdges(dag, *resourceEdges, islandByTaskIndex);

  return assembleLogicalPlacementIslandGraph(dag, executionGraph, affinityEdges,
                                             islandByTaskIndex);
}

LogicalResult attachLogicalPlacementIslandIds(
    func::FuncOp taskGraphFunc, const TaskGraphDAG &dag,
    const LogicalPlacementIslandGraph &islandGraph) {
  Builder builder(taskGraphFunc.getContext());
  for (const TaskGraphNode &node : dag.nodes) {
    auto islandIt = islandGraph.islandByTaskIndex.find(node.index);
    if (islandIt == islandGraph.islandByTaskIndex.end())
      continue;

    node.op->setAttr(schedule_attrs::kIslandIdAttrName,
                     builder.getI64IntegerAttr(islandIt->second));
  }

  return success();
}

FailureOr<LogicalPlacementIslandGraph>
loadLogicalPlacementIslandGraph(const TaskGraphDAG &dag,
                                const TaskExecutionGraph &executionGraph) {
  llvm::DenseMap<unsigned, unsigned> islandByTaskIndex;
  for (const TaskGraphNode &node : dag.nodes) {
    auto islandId =
        node.op->getAttrOfType<IntegerAttr>(schedule_attrs::kIslandIdAttrName);
    if (!islandId) {
      auto distribution = node.op->getAttrOfType<TaskDistributionAttr>(
          task_graph_attrs::kTaskDistributionAttrName);
      if (isMatrixSetupTask(node.op) ||
          (distribution &&
           distribution.getRole() == TaskDistributionRole::Shard)) {
        node.op->emitError("expected logical island ID; run "
                           "--sculptor-build-task-graph-islands before "
                           "--sculptor-schedule-task-graph");
        return failure();
      }
      continue;
    }

    int64_t islandValue = islandId.getInt();
    if (islandValue < 0 || static_cast<uint64_t>(islandValue) >
                               std::numeric_limits<unsigned>::max()) {
      node.op->emitError("logical island ID is outside the supported range");
      return failure();
    }

    unsigned islandIndex = static_cast<unsigned>(islandValue);
    islandByTaskIndex.try_emplace(node.index, islandIndex);
  }

  auto resourceEdges = collectResourceEdges(dag);
  if (failed(resourceEdges))
    return failure();

  llvm::SmallVector<IslandAffinityEdge, 16> affinityEdges =
      buildIslandAffinityEdges(dag, *resourceEdges, islandByTaskIndex);
  return assembleLogicalPlacementIslandGraph(dag, executionGraph, affinityEdges,
                                             islandByTaskIndex);
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
