#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTaskKinds.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <limits>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_schedulers = mlir::sculptor::task_schedulers;

using TaskGraphNode = mlir::sculptor::task_schedulers::TaskGraphNode;

static mlir::FailureOr<mlir::func::FuncOp>
lookupTaskCallee(mlir::ModuleOp module, mlir::sculptor::TaskCreateOp taskOp,
                 llvm::StringRef placementKind) {
  auto callee = module.lookupSymbol<mlir::func::FuncOp>(
      taskOp.getCalleeAttr().getValue());
  if (!callee) {
    return taskOp.emitError("expected task callee '")
           << taskOp.getCalleeAttr().getValue()
           << "' to resolve to a function for " << placementKind
           << " placement";
  }
  return callee;
}

static void attachCorePlacementAttrs(mlir::Operation *op,
                                     mlir::Builder &builder, int64_t coreId) {
  op->setAttr(runtime_attrs::kTaskCoreIdAttrName,
              builder.getI64IntegerAttr(coreId));
}

static void attachAnalogArrayPlacementAttrs(
    mlir::Operation *op, mlir::Builder &builder,
    const task_schedulers::PhysicalArrayPlacement &placement) {
  attachCorePlacementAttrs(op, builder, placement.coreId);
  op->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
              builder.getI64IntegerAttr(placement.physicalArrayId));
  op->setAttr(runtime_attrs::kTaskLocalArrayIdAttrName,
              builder.getI64IntegerAttr(placement.localArrayId));
}

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

static void buildMatrixSetupPlacementMap(
    const task_schedulers::LogicalPlacementIslandGraph &islandGraph,
    const task_schedulers::IslandPlacementPlan &plan,
    llvm::DenseMap<unsigned, int64_t> &physicalArrayByMatrixSetupTask) {
  for (auto indexedIsland : llvm::enumerate(islandGraph.islands)) {
    physicalArrayByMatrixSetupTask.try_emplace(
        indexedIsland.value().matrixSetupTaskIndex,
        plan.physicalArrayByIsland[indexedIsland.index()]);
  }
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

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<PhysicalArrayPlacement>
resolvePhysicalArrayPlacement(Operation *diagnosticOp,
                              const HardwareBudget &budget,
                              int64_t physicalArrayId) {
  int64_t coreId = physicalArrayId / budget.arraysPerCore;
  int64_t localArrayId = physicalArrayId % budget.arraysPerCore;
  if (physicalArrayId < 0 || coreId < 0 || coreId >= budget.numCores ||
      localArrayId < 0 || localArrayId >= budget.arraysPerCore) {
    if (diagnosticOp)
      diagnosticOp->emitError("assigned analog array is outside the hardware "
                              "budget");
    return failure();
  }
  return PhysicalArrayPlacement{physicalArrayId, coreId, localArrayId};
}

LogicalResult attachTaskCorePlacement(ModuleOp module,
                                      sculptor::TaskCreateOp taskOp,
                                      const HardwareBudget &budget,
                                      int64_t coreId) {
  if (coreId < 0 || coreId >= budget.numCores) {
    taskOp.emitError("assigned core is outside the hardware budget");
    return failure();
  }

  auto callee = lookupTaskCallee(module, taskOp, "core");
  if (failed(callee))
    return failure();

  Builder builder(module.getContext());
  attachCorePlacementAttrs(taskOp.getOperation(), builder, coreId);
  attachCorePlacementAttrs(callee->getOperation(), builder, coreId);
  return success();
}

LogicalResult attachTaskAnalogArrayPlacement(ModuleOp module,
                                             sculptor::TaskCreateOp taskOp,
                                             const HardwareBudget &budget,
                                             int64_t physicalArrayId) {
  auto placement = resolvePhysicalArrayPlacement(taskOp.getOperation(), budget,
                                                 physicalArrayId);
  if (failed(placement))
    return failure();

  auto callee = lookupTaskCallee(module, taskOp, "analog array");
  if (failed(callee))
    return failure();

  Builder builder(module.getContext());
  attachAnalogArrayPlacementAttrs(taskOp.getOperation(), builder, *placement);
  attachAnalogArrayPlacementAttrs(callee->getOperation(), builder, *placement);
  return success();
}

LogicalResult commitPlacementPlan(ModuleOp module, func::FuncOp taskGraphFunc,
                                  const TaskGraphPlacementProblem &problem,
                                  const IslandPlacementPlan &plan) {
  if (failed(validatePlacementPlan(problem, plan)))
    return failure();

  llvm::DenseMap<unsigned, int64_t> physicalArrayByMatrixSetupTask;
  buildMatrixSetupPlacementMap(problem.islandGraph, plan,
                               physicalArrayByMatrixSetupTask);

  return materializeMatrixSetupIslandPlacements(
      module, problem.budget, problem.dag, problem.islandGraph,
      physicalArrayByMatrixSetupTask, problem.islandGraph.islandByTaskIndex);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
