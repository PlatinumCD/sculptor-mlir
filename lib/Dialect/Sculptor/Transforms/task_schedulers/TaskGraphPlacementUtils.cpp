#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTaskKinds.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"

#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <limits>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_schedulers = mlir::sculptor::task_schedulers;

using TaskGraphNode = mlir::sculptor::task_schedulers::TaskGraphNode;

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
    const task_schedulers::TaskGraphDAG &dag, bool &changed) {
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
    changed = true;
  }

  return mlir::success();
}

static mlir::LogicalResult attachEarliestConsumerCorePlacements(
    mlir::ModuleOp module, const task_schedulers::HardwareBudget &budget,
    const task_schedulers::TaskGraphDAG &dag, bool &changed) {
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
    changed = true;
  }

  return mlir::success();
}

static mlir::LogicalResult attachAdjacentCorePlacementsToFixedPoint(
    mlir::ModuleOp module, const task_schedulers::HardwareBudget &budget,
    const task_schedulers::TaskGraphDAG &dag) {
  bool changed = false;
  do {
    changed = false;
    if (mlir::failed(attachMostRecentProducerCorePlacements(module, budget, dag,
                                                            changed)))
      return mlir::failure();

    if (mlir::failed(
            attachEarliestConsumerCorePlacements(module, budget, dag, changed)))
      return mlir::failure();
  } while (changed);

  return mlir::success();
}

static mlir::LogicalResult buildMatrixSetupPlacementMap(
    mlir::func::FuncOp taskGraphFunc,
    const task_schedulers::TaskGraphDAG &dag,
    const task_schedulers::LogicalPlacementIslandGraph &islandGraph,
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

  for (const task_schedulers::LogicalPlacementIsland &island :
       islandGraph.islands) {
    if (physicalArrayByMatrixSetupTask.contains(island.matrixSetupTaskIndex))
      continue;

    if (island.matrixSetupTaskIndex < dag.nodes.size()) {
      mlir::sculptor::TaskCreateOp setupTask =
          dag.nodes[island.matrixSetupTaskIndex].op;
      setupTask.emitError("expected matrix setup task to have an assigned "
                          "physical analog array");
      return mlir::failure();
    }

    taskGraphFunc.emitError("expected matrix setup task to have an assigned "
                            "physical analog array");
    return mlir::failure();
  }

  return mlir::success();
}

static mlir::LogicalResult materializeMatrixSetupIslandPlacements(
    mlir::ModuleOp module, const task_schedulers::HardwareBudget &budget,
    const task_schedulers::TaskGraphDAG &dag,
    const task_schedulers::LogicalPlacementIslandGraph &islandGraph,
    const llvm::DenseMap<unsigned, int64_t> &physicalArrayByMatrixSetupTask,
    const llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  mlir::Builder builder(module.getContext());
  llvm::StringMap<int64_t> coreIdBySourceLayer;
  llvm::StringSet<> ambiguousSourceLayers;

  for (const task_schedulers::LogicalPlacementIsland &island :
       islandGraph.islands) {
    if (island.matrixSetupTaskIndex >= dag.nodes.size())
      continue;

    const TaskGraphNode *setupNode = &dag.nodes[island.matrixSetupTaskIndex];
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
    setupTask->setAttr(schedule_attrs::kIslandIndexAttrName,
                       builder.getI64IntegerAttr(setupNode->index));
    auto placement = task_schedulers::resolvePhysicalArrayPlacement(
        setupTask.getOperation(), budget, physicalArrayId);
    if (mlir::failed(placement))
      return mlir::failure();

    recordSourceLayerCore(setupTask, placement->coreId, coreIdBySourceLayer,
                          ambiguousSourceLayers);

    if (mlir::failed(task_schedulers::attachTaskAnalogArrayPlacement(
            module, setupTask, budget, physicalArrayId)))
      return mlir::failure();

    for (unsigned mvmTaskIndex : island.mvmTaskIndices) {
      if (mvmTaskIndex >= dag.nodes.size())
        continue;

      const TaskGraphNode &mvmNode = dag.nodes[mvmTaskIndex];
      mvmNode.op->setAttr(schedule_attrs::kIslandIndexAttrName,
                          builder.getI64IntegerAttr(setupNode->index));
      if (mlir::failed(task_schedulers::attachTaskAnalogArrayPlacement(
              module, mvmNode.op, budget, physicalArrayId)))
        return mlir::failure();
      recordSourceLayerCore(mvmNode.op, placement->coreId, coreIdBySourceLayer,
                            ambiguousSourceLayers);
    }
  }

  for (const auto &islandEntry : islandByTaskIndex) {
    unsigned taskIndex = islandEntry.first;
    unsigned setupTaskIndex = islandEntry.second;
    if (taskIndex == setupTaskIndex)
      continue;

    if (taskIndex >= dag.nodes.size())
      continue;

    mlir::sculptor::TaskCreateOp taskOp = dag.nodes[taskIndex].op;
    if (getCoreIdAttr(taskOp) || !task_schedulers::isDigitalTask(taskOp))
      continue;

    taskOp->setAttr(schedule_attrs::kIslandIndexAttrName,
                    builder.getI64IntegerAttr(setupTaskIndex));

    auto physicalArrayIt = physicalArrayByMatrixSetupTask.find(setupTaskIndex);
    if (physicalArrayIt == physicalArrayByMatrixSetupTask.end())
      continue;

    auto placement = task_schedulers::resolvePhysicalArrayPlacement(
        taskOp.getOperation(), budget, physicalArrayIt->second);
    if (mlir::failed(placement))
      return mlir::failure();

    if (mlir::failed(task_schedulers::attachTaskCorePlacement(
            module, taskOp, budget, placement->coreId)))
      return mlir::failure();
  }

  if (mlir::failed(attachSourceLayerCorePlacements(
          module, budget, dag, coreIdBySourceLayer, ambiguousSourceLayers)))
    return mlir::failure();

  return attachAdjacentCorePlacementsToFixedPoint(module, budget, dag);
}

static mlir::FailureOr<
    llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>>
buildRoundRobinIslandGroupPlacements(
    mlir::func::FuncOp taskGraphFunc,
    const task_schedulers::LogicalPlacementIslandGraph &islandGraph,
    llvm::ArrayRef<int64_t> physicalArrayOrder) {
  if (!islandGraph.islands.empty() && physicalArrayOrder.empty()) {
    taskGraphFunc.emitError("expected logical island placement to have at "
                            "least one physical analog array");
    return mlir::failure();
  }

  llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>
      groupPlacements;
  groupPlacements.reserve(islandGraph.islands.size());
  for (auto indexedIsland : llvm::enumerate(islandGraph.islands)) {
    const task_schedulers::LogicalPlacementIsland &island =
        indexedIsland.value();
    int64_t physicalArrayId =
        physicalArrayOrder[indexedIsland.index() % physicalArrayOrder.size()];
    groupPlacements.push_back(task_schedulers::MatrixSetupGroupPlacement{
        island.matrixSetupTaskIndex, physicalArrayId});
  }

  return groupPlacements;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

LogicalResult placeLogicalPlacementIslands(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag, llvm::ArrayRef<int64_t> physicalArrayOrder) {
  auto islandGraph = buildLogicalPlacementIslandGraph(dag);
  if (failed(islandGraph))
    return failure();

  auto groupPlacements =
      buildRoundRobinIslandGroupPlacements(taskGraphFunc, *islandGraph,
                                           physicalArrayOrder);
  if (failed(groupPlacements))
    return failure();

  return placeLogicalPlacementIslands(module, taskGraphFunc, budget, dag,
                                      *islandGraph, *groupPlacements);
}

LogicalResult placeLogicalPlacementIslands(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag, const LogicalPlacementIslandGraph &islandGraph,
    llvm::ArrayRef<MatrixSetupGroupPlacement> groupPlacements) {
  llvm::DenseMap<unsigned, int64_t> physicalArrayByMatrixSetupTask;
  if (failed(buildMatrixSetupPlacementMap(taskGraphFunc, dag, islandGraph,
                                          groupPlacements,
                                          physicalArrayByMatrixSetupTask)))
    return failure();

  return materializeMatrixSetupIslandPlacements(
      module, budget, dag, islandGraph, physicalArrayByMatrixSetupTask,
      islandGraph.islandByTaskIndex);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
