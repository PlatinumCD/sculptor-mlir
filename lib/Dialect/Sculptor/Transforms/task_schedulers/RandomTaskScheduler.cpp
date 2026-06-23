#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <random>

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

class RandomTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "random"; }

  mlir::LogicalResult schedule(
      mlir::ModuleOp module, mlir::func::FuncOp taskGraphFunc,
      const mlir::sculptor::task_schedulers::HardwareBudget &budget,
      const mlir::sculptor::task_schedulers::TaskGraphDAG &dag) const final {
    llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks;
    for (const TaskGraphNode &node : dag.nodes) {
      mlir::sculptor::TaskCreateOp taskOp = node.op;
      if (taskOp.getTaskKind() == task_graph_names::kMatrixSetupTaskKind)
        matrixSetupTasks.push_back(&node);
    }

    MatrixSetupMVMMap mvmTasksByMatrixSetupTask;
    for (const mlir::sculptor::task_schedulers::TaskGraphNode *setupNode :
         matrixSetupTasks) {
      llvm::SmallVector<const TaskGraphNode *, 4> &mvmTasks =
          mvmTasksByMatrixSetupTask[setupNode->index];

      for (unsigned successorIndex : setupNode->successors) {
        const mlir::sculptor::task_schedulers::TaskGraphNode &successorNode =
            dag.nodes[successorIndex];
        mlir::sculptor::TaskCreateOp successorTask = successorNode.op;
        if (successorTask.getTaskKind() == task_graph_names::kMVMTaskKind)
          mvmTasks.push_back(&successorNode);
      }
    }

    if (budget.analogArrays.empty()) {
      taskGraphFunc.emitError("expected random task scheduler to have at "
                              "least one analog array");
      return mlir::failure();
    }

    llvm::SmallVector<int64_t, 8> shuffledAnalogArrays = budget.analogArrays;
    std::mt19937 randomEngine(0);
    std::shuffle(shuffledAnalogArrays.begin(), shuffledAnalogArrays.end(),
                 randomEngine);

    llvm::StringMap<int64_t> coreIdBySourceLayer;
    llvm::StringSet<> ambiguousSourceLayers;
    for (auto indexedSetup : llvm::enumerate(matrixSetupTasks)) {
      const TaskGraphNode *setupNode = indexedSetup.value();
      int64_t physicalArrayId =
          shuffledAnalogArrays[indexedSetup.index() %
                               shuffledAnalogArrays.size()];

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
        recordSourceLayerCore(mvmNode->op, placement->coreId,
                              coreIdBySourceLayer, ambiguousSourceLayers);
      }
    }

    if (mlir::failed(attachSourceLayerCorePlacements(
            module, budget, dag, coreIdBySourceLayer, ambiguousSourceLayers)))
      return mlir::failure();

    if (mlir::failed(attachMostRecentProducerCorePlacements(
            module, budget, dag)))
      return mlir::failure();

    if (mlir::failed(attachEarliestConsumerCorePlacements(module, budget, dag)))
      return mlir::failure();

    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerRandomTaskScheduler(TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(registry,
                                   std::make_unique<RandomTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
