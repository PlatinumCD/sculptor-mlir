#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleHelpers.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;
namespace task_graph_names = mlir::sculptor::task_graph_names;

struct LayerBoundary {
  int64_t id = 0;
  llvm::StringRef sourceLayer;
  mlir::Value logicalArrayResource;
  int64_t logicalArrayIndex = 0;
  llvm::SmallVector<mlir::sculptor::TaskCreateOp, 16> tasks;
};

struct LayerBoundaryFormation {
  llvm::SmallVector<LayerBoundary, 16> boundaries;
  llvm::DenseMap<mlir::Operation *, unsigned> boundaryByTask;
  llvm::DenseMap<mlir::Value, unsigned> boundaryByLogicalArray;
};

static bool isMatrixSetupTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getTaskKind() == task_graph_names::kMatrixSetupTaskKind;
}

static bool isMVMTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getTaskKind() == task_graph_names::kMVMTaskKind;
}

static bool isLogicalArrayResource(mlir::Value value) {
  auto resourceType =
      mlir::dyn_cast<mlir::sculptor::TaskResourceType>(value.getType());
  return resourceType &&
         llvm::isa<mlir::sculptor::LogicalArrayType>(resourceType.getValueType());
}

static mlir::FailureOr<mlir::Value>
getSingleLogicalArrayResource(mlir::sculptor::TaskCreateOp taskOp,
                              mlir::OperandRange resources,
                              llvm::StringRef resourceRole) {
  std::optional<mlir::Value> logicalArrayResource;

  for (mlir::Value resource : resources) {
    if (!isLogicalArrayResource(resource))
      continue;

    if (logicalArrayResource && *logicalArrayResource != resource) {
      taskOp.emitError("expected ")
          << taskOp.getTaskKind() << " task to reference one logical array in "
          << resourceRole;
      return mlir::failure();
    }

    logicalArrayResource = resource;
  }

  if (!logicalArrayResource) {
    taskOp.emitError("expected ")
        << taskOp.getTaskKind() << " task to reference a logical array in "
        << resourceRole;
    return mlir::failure();
  }

  return *logicalArrayResource;
}

static mlir::FailureOr<unsigned> createLogicalArrayBoundary(
    LayerBoundaryFormation &formation, mlir::sculptor::TaskCreateOp taskOp,
    mlir::Value logicalArrayResource, int64_t logicalArrayIndex) {
  if (formation.boundaryByLogicalArray.contains(logicalArrayResource)) {
    taskOp.emitError("logical array already has a layer-placement boundary");
    return mlir::failure();
  }

  unsigned index = formation.boundaries.size();
  LayerBoundary boundary;
  boundary.id = static_cast<int64_t>(index + 1);
  boundary.sourceLayer = taskOp.getSourceLayer();
  boundary.logicalArrayResource = logicalArrayResource;
  boundary.logicalArrayIndex = logicalArrayIndex;
  formation.boundaries.push_back(std::move(boundary));
  formation.boundaryByLogicalArray.try_emplace(logicalArrayResource, index);
  return index;
}

static mlir::LogicalResult
addTaskToBoundary(LayerBoundaryFormation &formation, unsigned boundaryIndex,
                  mlir::sculptor::TaskCreateOp taskOp) {
  if (boundaryIndex >= formation.boundaries.size()) {
    taskOp.emitError("internal layer-placement boundary index out of range");
    return mlir::failure();
  }

  auto inserted = formation.boundaryByTask.try_emplace(taskOp.getOperation(),
                                                       boundaryIndex);
  if (!inserted.second) {
    taskOp.emitError("internal layer-placement task assigned to multiple "
                     "boundaries");
    return mlir::failure();
  }

  formation.boundaries[boundaryIndex].tasks.push_back(taskOp);
  return mlir::success();
}

static void appendUniqueBoundary(llvm::SmallVectorImpl<unsigned> &boundaries,
                                 unsigned boundaryIndex) {
  for (unsigned existing : boundaries) {
    if (existing == boundaryIndex)
      return;
  }
  boundaries.push_back(boundaryIndex);
}

static llvm::SmallVector<unsigned, 4>
collectAssignedPredecessorBoundaries(const task_schedulers::TaskGraphDAG &dag,
                                     const task_schedulers::TaskGraphNode &node,
                                     const LayerBoundaryFormation &formation) {
  llvm::SmallVector<unsigned, 4> boundaries;

  for (unsigned predecessorIndex : node.predecessors) {
    const task_schedulers::TaskGraphNode &predecessor =
        dag.nodes[predecessorIndex];
    mlir::sculptor::TaskCreateOp predecessorOp = predecessor.op;
    auto boundaryIt =
        formation.boundaryByTask.find(predecessorOp.getOperation());
    if (boundaryIt == formation.boundaryByTask.end())
      continue;
    appendUniqueBoundary(boundaries, boundaryIt->second);
  }

  return boundaries;
}

static llvm::SmallVector<unsigned, 4>
collectAssignedSuccessorBoundaries(const task_schedulers::TaskGraphDAG &dag,
                                   const task_schedulers::TaskGraphNode &node,
                                   const LayerBoundaryFormation &formation) {
  llvm::SmallVector<unsigned, 4> boundaries;

  for (unsigned successorIndex : node.successors) {
    const task_schedulers::TaskGraphNode &successor = dag.nodes[successorIndex];
    mlir::sculptor::TaskCreateOp successorOp = successor.op;
    auto boundaryIt = formation.boundaryByTask.find(successorOp.getOperation());
    if (boundaryIt == formation.boundaryByTask.end())
      continue;
    appendUniqueBoundary(boundaries, boundaryIt->second);
  }

  return boundaries;
}

static std::optional<unsigned>
chooseBoundaryForRemainingTask(const task_schedulers::TaskGraphDAG &dag,
                               const task_schedulers::TaskGraphNode &node,
                               const LayerBoundaryFormation &formation) {
  // A multi-input task is the split/merge point after shard-local work, so it
  // is owned by the last contributing shard. Shared prep before the split is
  // owned by the first shard that consumes it.
  llvm::SmallVector<unsigned, 4> predecessorBoundaries =
      collectAssignedPredecessorBoundaries(dag, node, formation);
  if (!predecessorBoundaries.empty()) {
    unsigned owner = predecessorBoundaries.front();
    for (unsigned boundaryIndex : predecessorBoundaries)
      owner = std::max(owner, boundaryIndex);
    return owner;
  }

  llvm::SmallVector<unsigned, 4> successorBoundaries =
      collectAssignedSuccessorBoundaries(dag, node, formation);
  if (!successorBoundaries.empty()) {
    unsigned owner = successorBoundaries.front();
    for (unsigned boundaryIndex : successorBoundaries)
      owner = std::min(owner, boundaryIndex);
    return owner;
  }

  return std::nullopt;
}

static mlir::FailureOr<LayerBoundaryFormation>
formLayerBoundaries(mlir::func::FuncOp taskGraphFunc,
                    const task_schedulers::TaskGraphDAG &dag) {
  LayerBoundaryFormation formation;

  llvm::DenseMap<mlir::Value, int64_t> logicalArrayIndexByResource;
  for (auto indexedResource : llvm::enumerate(dag.logicalArrayResources)) {
    logicalArrayIndexByResource.try_emplace(
        indexedResource.value(), static_cast<int64_t>(indexedResource.index()));
  }

  for (const task_schedulers::TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (!isMatrixSetupTask(taskOp))
      continue;

    mlir::FailureOr<mlir::Value> logicalArrayResource =
        getSingleLogicalArrayResource(taskOp, taskOp.getOutputs(), "outputs");
    if (mlir::failed(logicalArrayResource))
      return mlir::failure();

    auto indexIt = logicalArrayIndexByResource.find(*logicalArrayResource);
    if (indexIt == logicalArrayIndexByResource.end()) {
      taskOp.emitError("expected matrix setup output to be a task graph "
                       "logical array resource");
      return mlir::failure();
    }

    mlir::FailureOr<unsigned> boundaryIndex = createLogicalArrayBoundary(
        formation, taskOp, *logicalArrayResource, indexIt->second);
    if (mlir::failed(boundaryIndex))
      return mlir::failure();

    if (mlir::failed(addTaskToBoundary(formation, *boundaryIndex, taskOp)))
      return mlir::failure();
  }

  for (const task_schedulers::TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (!isMVMTask(taskOp))
      continue;

    mlir::FailureOr<mlir::Value> logicalArrayResource =
        getSingleLogicalArrayResource(taskOp, taskOp.getInputs(), "inputs");
    if (mlir::failed(logicalArrayResource))
      return mlir::failure();

    auto boundaryIt =
        formation.boundaryByLogicalArray.find(*logicalArrayResource);
    if (boundaryIt == formation.boundaryByLogicalArray.end()) {
      taskOp.emitError("expected MVM logical array input to have a "
                       "layer-placement boundary");
      return mlir::failure();
    }

    if (mlir::failed(addTaskToBoundary(formation, boundaryIt->second, taskOp)))
      return mlir::failure();
  }

  if (formation.boundaries.empty()) {
    taskGraphFunc.emitError()
        << "layer-placement could not form any layer boundaries";
    return mlir::failure();
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (const task_schedulers::TaskGraphNode &node : dag.nodes) {
      mlir::sculptor::TaskCreateOp taskOp = node.op;
      if (formation.boundaryByTask.contains(taskOp.getOperation()))
        continue;

      std::optional<unsigned> owner =
          chooseBoundaryForRemainingTask(dag, node, formation);
      if (!owner)
        continue;

      if (mlir::failed(addTaskToBoundary(formation, *owner, taskOp)))
        return mlir::failure();
      changed = true;
    }
  }

  for (const task_schedulers::TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (formation.boundaryByTask.contains(taskOp.getOperation()))
      continue;

    taskOp.emitError("layer-placement could not attach task to a logical "
                     "array boundary");
    return mlir::failure();
  }

  return formation;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 8>> validateBoundaryPlacement(
    mlir::func::FuncOp taskGraphFunc, const LayerBoundaryFormation &formation,
    const task_schedulers::HardwareBudget &budget,
    const task_schedulers::TaskGraphScheduleOptions &options) {
  if (options.placement.empty()) {
    taskGraphFunc.emitError("layer-placement requires explicit "
                            "boundary-to-core placement");
    return mlir::failure();
  }

  if (static_cast<int64_t>(formation.boundaries.size()) != budget.numCores) {
    taskGraphFunc.emitError("layer-placement one-boundary-per-core mode "
                            "requires boundary count to match core count: ")
        << formation.boundaries.size() << " boundaries for " << budget.numCores
        << " cores";
    return mlir::failure();
  }

  if (options.placement.size() != formation.boundaries.size()) {
    taskGraphFunc.emitError("layer-placement placement vector length must "
                            "match boundary count: got ")
        << options.placement.size() << " entries for "
        << formation.boundaries.size() << " boundaries";
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 8> seenCores(static_cast<size_t>(budget.numCores),
                                          0);
  for (auto indexedCore : llvm::enumerate(options.placement)) {
    int64_t coreId = indexedCore.value();
    if (coreId < 0 || coreId >= budget.numCores) {
      taskGraphFunc.emitError("layer-placement placement entry ")
          << indexedCore.index() << " maps to core " << coreId
          << ", outside the scheduled core range [0, " << budget.numCores
          << ")";
      return mlir::failure();
    }

    if (seenCores[static_cast<size_t>(coreId)] != 0) {
      taskGraphFunc.emitError("layer-placement placement repeats core ")
          << coreId << "; one-boundary-per-core placement requires each core "
          << "exactly once";
      return mlir::failure();
    }
    seenCores[static_cast<size_t>(coreId)] = 1;
  }

  return options.placement;
}

static mlir::FailureOr<
    llvm::SmallVector<task_schedulers::LogicalArrayPlacement>>
buildBoundaryLogicalArrayPlacements(
    const LayerBoundaryFormation &formation,
    llvm::ArrayRef<int64_t> boundaryToCore,
    const task_schedulers::TaskGraphDAG &dag,
    const task_schedulers::HardwareBudget &budget) {
  llvm::SmallVector<task_schedulers::LogicalArrayPlacement> placements;
  placements.resize(dag.logicalArrayResources.size());
  llvm::SmallVector<int64_t, 8> seenLogicalArrays(
      dag.logicalArrayResources.size(), 0);

  for (auto indexedBoundary : llvm::enumerate(formation.boundaries)) {
    const LayerBoundary &boundary = indexedBoundary.value();
    if (boundary.logicalArrayIndex < 0 ||
        boundary.logicalArrayIndex >=
            static_cast<int64_t>(dag.logicalArrayResources.size())) {
      return mlir::failure();
    }

    int64_t coreId = boundaryToCore[indexedBoundary.index()];
    int64_t physicalArrayId = coreId * budget.arraysPerCore;
    placements[boundary.logicalArrayIndex] =
        task_schedulers::LogicalArrayPlacement{boundary.logicalArrayIndex,
                                               physicalArrayId};
    seenLogicalArrays[boundary.logicalArrayIndex] = 1;
  }

  for (auto seen : llvm::enumerate(seenLogicalArrays)) {
    if (seen.value() != 0)
      continue;
    mlir::Operation *resourceOp =
        dag.logicalArrayResources[seen.index()].getDefiningOp();
    if (resourceOp)
      resourceOp->emitError("logical array has no layer-placement boundary");
    return mlir::failure();
  }

  return placements;
}

static mlir::FailureOr<task_schedulers::TaskCoreAssignments>
buildLayerCoreAssignments(
    const LayerBoundaryFormation &formation,
    llvm::ArrayRef<int64_t> boundaryToCore,
    const llvm::DenseMap<mlir::Value,
                         task_schedulers::LogicalArrayRuntimePlacement>
        &placementByLogicalArray) {
  task_schedulers::TaskCoreAssignments assignments;

  for (auto indexedBoundary : llvm::enumerate(formation.boundaries)) {
    const LayerBoundary &boundary = indexedBoundary.value();
    int64_t coreId = boundaryToCore[indexedBoundary.index()];

    for (mlir::sculptor::TaskCreateOp taskOp : boundary.tasks) {
      assignments.coreByTask.try_emplace(taskOp.getOperation(), coreId);

      std::optional<task_schedulers::LogicalArrayRuntimePlacement>
          arrayPlacement;
      if (isMatrixSetupTask(taskOp)) {
        mlir::FailureOr<task_schedulers::LogicalArrayRuntimePlacement>
            placement = task_schedulers::getSingleLogicalArrayPlacement(
                taskOp, taskOp.getOutputs(), placementByLogicalArray,
                "outputs");
        if (mlir::failed(placement))
          return mlir::failure();
        arrayPlacement = *placement;
      } else if (isMVMTask(taskOp)) {
        mlir::FailureOr<task_schedulers::LogicalArrayRuntimePlacement>
            placement = task_schedulers::getSingleLogicalArrayPlacement(
                taskOp, taskOp.getInputs(), placementByLogicalArray, "inputs");
        if (mlir::failed(placement))
          return mlir::failure();
        arrayPlacement = *placement;
      }

      if (!arrayPlacement)
        continue;

      if (arrayPlacement->coreId != coreId) {
        taskOp.emitError("logical array placement core does not match owning "
                         "layer-placement boundary");
        return mlir::failure();
      }

      assignments.physicalArrayByTask.try_emplace(
          taskOp.getOperation(), arrayPlacement->physicalArrayId);
    }
  }

  return assignments;
}

class LayerPlacementTaskScheduler final
    : public task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "layer-placement"; }

  mlir::LogicalResult schedule(
      mlir::ModuleOp module, mlir::func::FuncOp taskGraphFunc,
      const task_schedulers::HardwareBudget &budget,
      const task_schedulers::TaskGraphDAG &dag,
      const task_schedulers::TaskGraphScheduleOptions &options) const final {
    if (mlir::failed(task_schedulers::verifyLogicalArraysFitAnalogBudget(
            taskGraphFunc.getOperation(),
            static_cast<int64_t>(dag.logicalArrayResources.size()),
            budget.numAnalogArrays)))
      return mlir::failure();

    mlir::FailureOr<LayerBoundaryFormation> boundaries =
        formLayerBoundaries(taskGraphFunc, dag);
    if (mlir::failed(boundaries))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<int64_t, 8>> boundaryToCore =
        validateBoundaryPlacement(taskGraphFunc, *boundaries, budget, options);
    if (mlir::failed(boundaryToCore))
      return mlir::failure();

    mlir::Builder builder(taskGraphFunc.getContext());
    mlir::FailureOr<llvm::SmallVector<task_schedulers::LogicalArrayPlacement>>
        placements = buildBoundaryLogicalArrayPlacements(
            *boundaries, *boundaryToCore, dag, budget);
    if (mlir::failed(placements))
      return mlir::failure();
    task_schedulers::attachLogicalArrayPlacementAttrs(dag.logicalArrayResources,
                                                      *placements, builder);

    mlir::FailureOr<llvm::DenseMap<
        mlir::Value, task_schedulers::LogicalArrayRuntimePlacement>>
        placementByLogicalArray =
            task_schedulers::buildLogicalArrayRuntimePlacementMap(
                dag.logicalArrayResources, *placements, budget);
    if (mlir::failed(placementByLogicalArray))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<mlir::sculptor::TaskCreateOp>>
        scheduledTasks = task_schedulers::buildMatrixSetupFirstTaskOrder(dag);
    if (mlir::failed(scheduledTasks))
      return mlir::failure();
    task_schedulers::attachTaskIndices(*scheduledTasks, builder);

    mlir::FailureOr<task_schedulers::TaskCoreAssignments> coreAssignments =
        buildLayerCoreAssignments(*boundaries, *boundaryToCore,
                                  *placementByLogicalArray);
    if (mlir::failed(coreAssignments))
      return mlir::failure();
    task_schedulers::attachTaskCoreIds(*coreAssignments, builder);
    if (mlir::failed(task_schedulers::attachTaskFunctionPlacementAttrs(
            module, *scheduledTasks, *coreAssignments, budget)))
      return mlir::failure();

    mlir::FailureOr<int64_t> totalDigitalOps =
        task_schedulers::attachTaskDigitalOps(module, *scheduledTasks, builder);
    if (mlir::failed(totalDigitalOps))
      return mlir::failure();

    task_schedulers::CoreTransferSummary transferSummary =
        task_schedulers::computeCoreTransferSummary(*scheduledTasks, budget,
                                                    *coreAssignments);
    task_schedulers::attachTaskGraphScheduleSummaryAttrs(
        taskGraphFunc, dag, *placements, transferSummary, *totalDigitalOps,
        builder);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerLayerPlacementTaskScheduler(TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(
      registry, std::make_unique<LayerPlacementTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
