#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTaskKinds.h"

#include "TaskGraphIslandInternals.h"

#include <utility>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

using TaskGraphNode = task_schedulers::TaskGraphNode;
using IslandCommunicationEdge = task_schedulers::LogicalIslandCommunicationEdge;
using MatrixSetupMVMMap =
    llvm::DenseMap<unsigned, llvm::SmallVector<const TaskGraphNode *, 4>>;

static MatrixSetupMVMMap buildMVMTasksByMatrixSetupTask(
    const task_schedulers::TaskGraphDAG &dag,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks) {
  MatrixSetupMVMMap mvmTasksByMatrixSetupTask;
  for (const TaskGraphNode *setupNode : matrixSetupTasks) {
    llvm::SmallVector<const TaskGraphNode *, 4> &mvmTasks =
        mvmTasksByMatrixSetupTask[setupNode->index];

    for (unsigned successorIndex : setupNode->successors) {
      const TaskGraphNode &successorNode = dag.nodes[successorIndex];
      if (task_schedulers::isAnalogComputeTask(successorNode.op))
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

static task_schedulers::LogicalPlacementIslandGraph
assembleLogicalPlacementIslandGraph(
    const task_schedulers::TaskGraphDAG &dag,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    const llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  task_schedulers::LogicalPlacementIslandGraph graph;
  graph.islandByTaskIndex = islandByTaskIndex;
  graph.communicationEdges.append(islandCommunicationEdges.begin(),
                                  islandCommunicationEdges.end());

  llvm::DenseMap<unsigned, unsigned> islandOrdinalByIndex;
  graph.islands.reserve(matrixSetupTasks.size());
  for (const TaskGraphNode *setupNode : matrixSetupTasks) {
    task_schedulers::LogicalPlacementIsland island;
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

    task_schedulers::LogicalPlacementIsland &island =
        graph.islands[ordinalIt->second];
    appendUniqueTaskIndex(island.taskIndices, node.index);

    if (node.index == island.matrixSetupTaskIndex)
      continue;

    if (task_schedulers::isAnalogComputeTask(node.op)) {
      appendUniqueTaskIndex(island.mvmTaskIndices, node.index);
      continue;
    }

    if (task_schedulers::isDigitalTask(node.op))
      appendUniqueTaskIndex(island.digitalTaskIndices, node.index);
  }

  return graph;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<LogicalPlacementIslandGraph>
buildLogicalPlacementIslandGraph(const TaskGraphDAG &dag) {
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

  llvm::SmallVector<IslandCommunicationEdge, 16> islandCommunicationEdges =
      buildIslandCommunicationEdges(dag, *resourceEdges, islandByTaskIndex);

  return assembleLogicalPlacementIslandGraph(
      dag, matrixSetupTasks, islandCommunicationEdges, islandByTaskIndex);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
