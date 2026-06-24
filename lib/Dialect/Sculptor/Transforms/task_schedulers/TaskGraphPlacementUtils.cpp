#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "llvm/ADT/StringSet.h"

#include <limits>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace task_schedulers = mlir::sculptor::task_schedulers;

using TaskGraphNode = mlir::sculptor::task_schedulers::TaskGraphNode;
using MatrixSetupMVMMap =
    llvm::DenseMap<unsigned, llvm::SmallVector<const TaskGraphNode *, 4>>;

static void recordSourceLayerCore(mlir::sculptor::TaskCreateOp taskOp,
                                  int64_t coreId,
                                  llvm::StringMap<int64_t> &coreIdBySourceLayer,
                                  llvm::StringSet<> &ambiguousSourceLayers) {
  llvm::StringRef sourceLayer = taskOp.getSourceLayer();
  auto inserted = coreIdBySourceLayer.try_emplace(sourceLayer, coreId);
  if (!inserted.second && inserted.first->second != coreId)
    ambiguousSourceLayers.insert(sourceLayer);
}

static mlir::IntegerAttr getCoreIdAttr(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp->getAttrOfType<mlir::IntegerAttr>(
      runtime_attrs::kTaskCoreIdAttrName);
}

static mlir::IntegerAttr getTaskIndexAttr(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp->getAttrOfType<mlir::IntegerAttr>(
      runtime_attrs::kTaskIndexAttrName);
}

static const TaskGraphNode *
findMostRecentMappedProducer(const TaskGraphNode &node,
                             const task_schedulers::TaskGraphDAG &dag) {
  const TaskGraphNode *selectedProducer = nullptr;
  int64_t selectedTaskIndex = std::numeric_limits<int64_t>::min();

  mlir::sculptor::TaskCreateOp taskOp = node.op;
  for (mlir::Value dependency : taskOp.getDependencies()) {
    auto producerIndexIt = dag.nodeIndexByTaskResult.find(dependency);
    if (producerIndexIt == dag.nodeIndexByTaskResult.end())
      continue;

    const TaskGraphNode &producer = dag.nodes[producerIndexIt->second];
    mlir::sculptor::TaskCreateOp producerTask = producer.op;
    if (!getCoreIdAttr(producerTask))
      continue;

    mlir::IntegerAttr taskIndex = getTaskIndexAttr(producerTask);
    if (!taskIndex)
      continue;

    if (!selectedProducer || taskIndex.getInt() > selectedTaskIndex) {
      selectedProducer = &producer;
      selectedTaskIndex = taskIndex.getInt();
    }
  }

  return selectedProducer;
}

static const TaskGraphNode *
findEarliestMappedConsumer(const TaskGraphNode &node,
                           const task_schedulers::TaskGraphDAG &dag) {
  const TaskGraphNode *selectedConsumer = nullptr;
  int64_t selectedTaskIndex = std::numeric_limits<int64_t>::max();

  for (unsigned successorIndex : node.successors) {
    const TaskGraphNode &consumer = dag.nodes[successorIndex];
    mlir::sculptor::TaskCreateOp consumerTask = consumer.op;
    if (!getCoreIdAttr(consumerTask))
      continue;

    mlir::IntegerAttr taskIndex = getTaskIndexAttr(consumerTask);
    if (!taskIndex)
      continue;

    if (!selectedConsumer || taskIndex.getInt() < selectedTaskIndex) {
      selectedConsumer = &consumer;
      selectedTaskIndex = taskIndex.getInt();
    }
  }

  return selectedConsumer;
}

static mlir::LogicalResult attachSourceLayerCorePlacements(
    mlir::ModuleOp module, const task_schedulers::HardwareBudget &budget,
    const task_schedulers::TaskGraphDAG &dag,
    const llvm::StringMap<int64_t> &coreIdBySourceLayer,
    const llvm::StringSet<> &ambiguousSourceLayers) {
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (getCoreIdAttr(taskOp))
      continue;

    llvm::StringRef sourceLayer = taskOp.getSourceLayer();
    if (ambiguousSourceLayers.contains(sourceLayer))
      continue;

    auto coreIt = coreIdBySourceLayer.find(sourceLayer);
    if (coreIt == coreIdBySourceLayer.end())
      continue;

    if (mlir::failed(task_schedulers::attachTaskCorePlacement(
            module, taskOp, budget, coreIt->second)))
      return mlir::failure();
  }

  return mlir::success();
}

static mlir::LogicalResult attachMostRecentProducerCorePlacements(
    mlir::ModuleOp module, const task_schedulers::HardwareBudget &budget,
    const task_schedulers::TaskGraphDAG &dag) {
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (getCoreIdAttr(taskOp))
      continue;

    const TaskGraphNode *producer = findMostRecentMappedProducer(node, dag);
    if (!producer)
      continue;

    int64_t coreId = getCoreIdAttr(producer->op).getInt();
    if (mlir::failed(task_schedulers::attachTaskCorePlacement(module, taskOp,
                                                              budget, coreId)))
      return mlir::failure();
  }

  return mlir::success();
}

static mlir::LogicalResult attachEarliestConsumerCorePlacements(
    mlir::ModuleOp module, const task_schedulers::HardwareBudget &budget,
    const task_schedulers::TaskGraphDAG &dag) {
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (getCoreIdAttr(taskOp))
      continue;

    const TaskGraphNode *consumer = findEarliestMappedConsumer(node, dag);
    if (!consumer)
      continue;

    int64_t coreId = getCoreIdAttr(consumer->op).getInt();
    if (mlir::failed(task_schedulers::attachTaskCorePlacement(module, taskOp,
                                                              budget, coreId)))
      return mlir::failure();
  }

  return mlir::success();
}

static llvm::SmallVector<const TaskGraphNode *, 8>
collectMatrixSetupTasks(const task_schedulers::TaskGraphDAG &dag) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks;
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (taskOp.getTaskKind() == task_graph_names::kMatrixSetupTaskKind)
      matrixSetupTasks.push_back(&node);
  }
  return matrixSetupTasks;
}

static MatrixSetupMVMMap buildMVMTasksByMatrixSetupTask(
    const task_schedulers::TaskGraphDAG &dag,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks) {
  MatrixSetupMVMMap mvmTasksByMatrixSetupTask;
  for (const TaskGraphNode *setupNode : matrixSetupTasks) {
    llvm::SmallVector<const TaskGraphNode *, 4> &mvmTasks =
        mvmTasksByMatrixSetupTask[setupNode->index];

    for (unsigned successorIndex : setupNode->successors) {
      const TaskGraphNode &successorNode = dag.nodes[successorIndex];
      mlir::sculptor::TaskCreateOp successorTask = successorNode.op;
      if (successorTask.getTaskKind() == task_graph_names::kMVMTaskKind ||
          successorTask.getTaskKind() == task_graph_names::kConvTileMVMTaskKind)
        mvmTasks.push_back(&successorNode);
    }
  }
  return mvmTasksByMatrixSetupTask;
}

static mlir::LogicalResult buildMatrixSetupPlacementMap(
    mlir::func::FuncOp taskGraphFunc,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    llvm::ArrayRef<task_schedulers::MatrixSetupGroupPlacement> groupPlacements,
    llvm::DenseMap<unsigned, int64_t> &physicalArrayByMatrixSetupTask) {
  for (const task_schedulers::MatrixSetupGroupPlacement &placement :
       groupPlacements) {
    if (!physicalArrayByMatrixSetupTask
             .try_emplace(placement.matrixSetupTaskIndex,
                          placement.physicalArrayId)
             .second) {
      taskGraphFunc.emitError("expected matrix setup group placement to be "
                              "specified once per task");
      return mlir::failure();
    }
  }

  for (const TaskGraphNode *setupNode : matrixSetupTasks) {
    if (physicalArrayByMatrixSetupTask.contains(setupNode->index))
      continue;

    mlir::sculptor::TaskCreateOp setupTask = setupNode->op;
    setupTask.emitError("expected matrix setup task to have an assigned "
                        "physical analog array");
    return mlir::failure();
  }

  return mlir::success();
}

static mlir::LogicalResult materializeMatrixSetupGroupPlacements(
    mlir::ModuleOp module, const task_schedulers::HardwareBudget &budget,
    const task_schedulers::TaskGraphDAG &dag,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const MatrixSetupMVMMap &mvmTasksByMatrixSetupTask,
    const llvm::DenseMap<unsigned, int64_t> &physicalArrayByMatrixSetupTask) {
  llvm::StringMap<int64_t> coreIdBySourceLayer;
  llvm::StringSet<> ambiguousSourceLayers;

  for (const TaskGraphNode *setupNode : matrixSetupTasks) {
    auto physicalArrayIt =
        physicalArrayByMatrixSetupTask.find(setupNode->index);
    if (physicalArrayIt == physicalArrayByMatrixSetupTask.end()) {
      mlir::sculptor::TaskCreateOp setupTask = setupNode->op;
      setupTask.emitError("expected matrix setup task to have an assigned "
                          "physical analog array");
      return mlir::failure();
    }

    int64_t physicalArrayId = physicalArrayIt->second;
    mlir::sculptor::TaskCreateOp setupTask = setupNode->op;
    auto placement = task_schedulers::resolvePhysicalArrayPlacement(
        setupTask.getOperation(), budget, physicalArrayId);
    if (mlir::failed(placement))
      return mlir::failure();

    recordSourceLayerCore(setupTask, placement->coreId, coreIdBySourceLayer,
                          ambiguousSourceLayers);

    if (mlir::failed(task_schedulers::attachTaskAnalogArrayPlacement(
            module, setupTask, budget, physicalArrayId)))
      return mlir::failure();

    auto mvmTasksIt = mvmTasksByMatrixSetupTask.find(setupNode->index);
    if (mvmTasksIt == mvmTasksByMatrixSetupTask.end())
      continue;

    for (const TaskGraphNode *mvmNode : mvmTasksIt->second) {
      if (mlir::failed(task_schedulers::attachTaskAnalogArrayPlacement(
              module, mvmNode->op, budget, physicalArrayId)))
        return mlir::failure();
      recordSourceLayerCore(mvmNode->op, placement->coreId, coreIdBySourceLayer,
                            ambiguousSourceLayers);
    }
  }

  if (mlir::failed(attachSourceLayerCorePlacements(
          module, budget, dag, coreIdBySourceLayer, ambiguousSourceLayers)))
    return mlir::failure();

  if (mlir::failed(attachMostRecentProducerCorePlacements(module, budget, dag)))
    return mlir::failure();

  if (mlir::failed(attachEarliestConsumerCorePlacements(module, budget, dag)))
    return mlir::failure();

  return mlir::success();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

LogicalResult placeMatrixSetupGroupsAndSurroundingTasks(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag, llvm::ArrayRef<int64_t> physicalArrayOrder) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks =
      collectMatrixSetupTasks(dag);
  if (!matrixSetupTasks.empty() && physicalArrayOrder.empty()) {
    taskGraphFunc.emitError("expected matrix setup group placement to have at "
                            "least one physical analog array");
    return failure();
  }

  llvm::SmallVector<MatrixSetupGroupPlacement, 8> groupPlacements;
  groupPlacements.reserve(matrixSetupTasks.size());
  for (auto indexedSetup : llvm::enumerate(matrixSetupTasks)) {
    const TaskGraphNode *setupNode = indexedSetup.value();
    int64_t physicalArrayId =
        physicalArrayOrder[indexedSetup.index() % physicalArrayOrder.size()];
    groupPlacements.push_back(
        MatrixSetupGroupPlacement{setupNode->index, physicalArrayId});
  }

  return placeMatrixSetupGroupsAndSurroundingTasks(
      module, taskGraphFunc, budget, dag, groupPlacements);
}

LogicalResult placeMatrixSetupGroupsAndSurroundingTasks(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag,
    llvm::ArrayRef<MatrixSetupGroupPlacement> groupPlacements) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks =
      collectMatrixSetupTasks(dag);
  MatrixSetupMVMMap mvmTasksByMatrixSetupTask =
      buildMVMTasksByMatrixSetupTask(dag, matrixSetupTasks);

  llvm::DenseMap<unsigned, int64_t> physicalArrayByMatrixSetupTask;
  if (failed(buildMatrixSetupPlacementMap(taskGraphFunc, matrixSetupTasks,
                                          groupPlacements,
                                          physicalArrayByMatrixSetupTask)))
    return failure();

  return materializeMatrixSetupGroupPlacements(
      module, budget, dag, matrixSetupTasks, mvmTasksByMatrixSetupTask,
      physicalArrayByMatrixSetupTask);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
