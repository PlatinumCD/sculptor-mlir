#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"

#include "mlir/IR/Builders.h"

#include "TaskGraphIslandInternals.h"

#include <limits>
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

static void appendUniqueTaskIndex(llvm::SmallVectorImpl<unsigned> &tasks,
                                  unsigned taskIndex) {
  for (unsigned existingTask : tasks) {
    if (existingTask == taskIndex)
      return;
  }

  tasks.push_back(taskIndex);
}

static task_graph::LogicalPlacementIslandGraph
assembleLogicalPlacementIslandGraph(
    const task_graph::TaskGraphDAG &dag,
    const task_graph::TaskExecutionGraph &executionGraph,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    llvm::ArrayRef<IslandAffinityEdge> affinityEdges,
    const llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  task_graph::LogicalPlacementIslandGraph graph;
  graph.islandByTaskIndex = islandByTaskIndex;
  graph.affinityGraph.edges.append(affinityEdges.begin(), affinityEdges.end());
  graph.executionGraph =
      task_graph::buildIslandExecutionGraph(executionGraph, islandByTaskIndex);

  llvm::DenseMap<unsigned, unsigned> islandOrdinalByIndex;
  graph.islands.reserve(matrixSetupTasks.size());
  for (const TaskGraphNode *setupNode : matrixSetupTasks) {
    task_graph::LogicalPlacementIsland island;
    island.islandIndex = setupNode->index;
    island.matrixSetupTaskIndex = setupNode->index;
    unsigned ordinal = static_cast<unsigned>(graph.islands.size());
    graph.islands.push_back(std::move(island));
    islandOrdinalByIndex.try_emplace(setupNode->index, ordinal);
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

    if (node.index == island.matrixSetupTaskIndex)
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
                                 const TaskExecutionGraph &executionGraph) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks =
      collectMatrixSetupTasks(dag);
  MatrixSetupMVMMap mvmTasksByMatrixSetupTask =
      buildMVMTasksByMatrixSetupTask(dag, matrixSetupTasks);

  llvm::DenseMap<unsigned, unsigned> islandByTaskIndex;
  recordInitialMatrixSetupIslands(matrixSetupTasks, mvmTasksByMatrixSetupTask,
                                  islandByTaskIndex);
  if (failed(assignPrePlacementMinCutDigitalIslands(dag, islandByTaskIndex)))
    return failure();

  auto resourceEdges = collectResourceEdges(dag);
  if (failed(resourceEdges))
    return failure();

  if (failed(assignRemainingDigitalIslandsByLocalAffinity(dag, *resourceEdges,
                                                          islandByTaskIndex)))
    return failure();

  llvm::SmallVector<IslandAffinityEdge, 16> affinityEdges =
      buildIslandAffinityEdges(dag, *resourceEdges, islandByTaskIndex);

  return assembleLogicalPlacementIslandGraph(dag, executionGraph,
                                             matrixSetupTasks, affinityEdges,
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
  llvm::DenseMap<unsigned, const TaskGraphNode *> setupByIsland;
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks;

  for (const TaskGraphNode &node : dag.nodes) {
    auto islandId =
        node.op->getAttrOfType<IntegerAttr>(schedule_attrs::kIslandIdAttrName);
    if (!islandId) {
      if (isMatrixSetupTask(node.op)) {
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
    if (!isMatrixSetupTask(node.op))
      continue;

    if (islandIndex != node.index) {
      node.op->emitError(
          "expected logical island ID to match its matrix setup task index");
      return failure();
    }
    if (!setupByIsland.try_emplace(islandIndex, &node).second) {
      node.op->emitError(
          "expected one matrix setup task per logical placement island");
      return failure();
    }
    matrixSetupTasks.push_back(&node);
  }

  for (const auto &entry : islandByTaskIndex) {
    if (setupByIsland.contains(entry.second))
      continue;
    dag.nodes[entry.first].op->emitError(
        "expected logical island to be anchored by a matrix setup task");
    return failure();
  }

  auto resourceEdges = collectResourceEdges(dag);
  if (failed(resourceEdges))
    return failure();

  llvm::SmallVector<IslandAffinityEdge, 16> affinityEdges =
      buildIslandAffinityEdges(dag, *resourceEdges, islandByTaskIndex);
  return assembleLogicalPlacementIslandGraph(dag, executionGraph,
                                             matrixSetupTasks, affinityEdges,
                                             islandByTaskIndex);
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
