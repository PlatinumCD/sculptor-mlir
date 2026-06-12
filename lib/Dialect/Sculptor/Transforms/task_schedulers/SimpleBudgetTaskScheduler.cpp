#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleHelpers.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <optional>

namespace {

namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace task_schedulers = mlir::sculptor::task_schedulers;

struct TaskCoreAssignments {
  llvm::DenseMap<mlir::Operation *, int64_t> coreByTask;
  llvm::DenseMap<mlir::Operation *, int64_t> physicalArrayByTask;
  llvm::StringMap<int64_t> coreBySourceLayer;
};

struct LogicalArrayRuntimePlacement {
  int64_t physicalArrayId = 0;
  int64_t coreId = 0;
};

struct CoreTransferSummary {
  llvm::SmallVector<int64_t> transferBytes;
  llvm::SmallVector<int64_t> transferCost;
  int64_t interCoreTransferBytes = 0;
  int64_t totalTransferCost = 0;
};

static mlir::ArrayAttr
buildSimpleBudgetI64ArrayAttr(mlir::Builder &builder,
                              llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

static llvm::SmallVector<task_schedulers::LogicalArrayPlacement>
buildSimpleLogicalArrayPlacements(
    const task_schedulers::TaskGraphDAG &dag,
    const task_schedulers::HardwareBudget &budget) {
  llvm::SmallVector<task_schedulers::LogicalArrayPlacement> placements;
  placements.reserve(dag.logicalArrayResources.size());
  for (int64_t index = 0,
               count = static_cast<int64_t>(dag.logicalArrayResources.size());
       index < count; ++index) {
    placements.push_back({index, budget.analogArrays[index]});
  }
  return placements;
}

static void attachSimpleBudgetLogicalArrayPlacementAttrs(
    const llvm::SmallVectorImpl<mlir::Value> &logicalArrayResources,
    llvm::ArrayRef<task_schedulers::LogicalArrayPlacement> placements,
    mlir::Builder &builder) {
  for (const task_schedulers::LogicalArrayPlacement &placement : placements) {
    mlir::Operation *resourceOp =
        logicalArrayResources[placement.logicalArrayIndex].getDefiningOp();
    if (!resourceOp)
      continue;

    resourceOp->setAttr(schedule_attrs::kLogicalArrayIndexAttrName,
                        builder.getI64IntegerAttr(placement.logicalArrayIndex));
  }
}

static mlir::ArrayAttr buildSimpleBudgetLogicalArrayToAnalogArrayAttr(
    llvm::ArrayRef<task_schedulers::LogicalArrayPlacement> placements,
    mlir::Builder &builder) {
  llvm::SmallVector<int64_t> analogArrayByLogicalArray;
  analogArrayByLogicalArray.reserve(placements.size());
  for (const task_schedulers::LogicalArrayPlacement &placement : placements)
    analogArrayByLogicalArray.push_back(placement.analogArrayIndex);
  return buildSimpleBudgetI64ArrayAttr(builder, analogArrayByLogicalArray);
}

static bool isMatrixSetupTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getTaskKind() == task_graph_names::kMatrixSetupTaskKind;
}

static bool isAnalogTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() == task_graph_names::kAnalogDomain;
}

static bool isDigitalTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() == task_graph_names::kDigitalDomain;
}

static bool isMVMTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getTaskKind() == task_graph_names::kMVMTaskKind;
}

static bool isForwardTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getSourceLayer() == task_graph_names::kForwardSourceLayer;
}

static bool isLogicalArrayResource(mlir::Value value) {
  auto resourceType =
      mlir::dyn_cast<mlir::sculptor::TaskResourceType>(value.getType());
  return resourceType &&
         llvm::isa<mlir::sculptor::LogicalArrayType>(resourceType.getValueType());
}

static std::optional<int64_t> getStaticElementCount(mlir::Type type) {
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shapedType || !shapedType.hasRank() || !shapedType.hasStaticShape())
    return std::nullopt;
  return shapedType.getNumElements();
}

static std::optional<int64_t>
getFirstStaticResultElementCount(mlir::Operation *op) {
  for (mlir::Type resultType : op->getResultTypes()) {
    std::optional<int64_t> count = getStaticElementCount(resultType);
    if (count)
      return count;
  }
  return std::nullopt;
}

static bool hasShapedResult(mlir::Operation *op) {
  return getFirstStaticResultElementCount(op).has_value();
}

static bool hasLinalgAncestor(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef().starts_with("linalg."))
      return true;
  }
  return false;
}

static bool isLinalgElementwiseNamedOp(mlir::Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();
  return opName == "linalg.add" || opName == "linalg.sub" ||
         opName == "linalg.mul" || opName == "linalg.div" ||
         opName == "linalg.max" || opName == "linalg.min";
}

static bool isElementwiseScalarComputeOp(mlir::Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();
  return opName == "arith.addf" || opName == "arith.subf" ||
         opName == "arith.mulf" || opName == "arith.divf" ||
         opName == "arith.negf" || opName == "arith.addi" ||
         opName == "arith.subi" || opName == "arith.muli" ||
         opName == "arith.divsi" || opName == "arith.divui" ||
         opName == "arith.maximumf" || opName == "arith.minimumf" ||
         opName == "arith.maxnumf" || opName == "arith.minnumf" ||
         opName == "math.absf" || opName == "math.ceil" ||
         opName == "math.exp" || opName == "math.expm1" ||
         opName == "math.floor" || opName == "math.log" ||
         opName == "math.log1p" || opName == "math.log2" ||
         opName == "math.log10" || opName == "math.powf" ||
         opName == "math.rsqrt" || opName == "math.sqrt" ||
         opName == "math.tanh";
}

static bool hasOnlyParallelIterators(mlir::Operation *op) {
  auto iteratorTypes = op->getAttrOfType<mlir::ArrayAttr>("iterator_types");
  if (!iteratorTypes)
    return false;

  for (mlir::Attribute attr : iteratorTypes) {
    if (auto iteratorType = mlir::dyn_cast<mlir::StringAttr>(attr)) {
      if (iteratorType.getValue() != "parallel")
        return false;
      continue;
    }

    std::string printedAttr;
    llvm::raw_string_ostream os(printedAttr);
    attr.print(os);
    if (!llvm::StringRef(os.str()).contains("parallel"))
      return false;
  }
  return true;
}

static int64_t countScalarOpsInLinalgGeneric(mlir::Operation *genericOp) {
  int64_t scalarOps = 0;
  genericOp->walk([&](mlir::Operation *nestedOp) {
    if (nestedOp == genericOp)
      return;
    if (isElementwiseScalarComputeOp(nestedOp) && !hasShapedResult(nestedOp))
      ++scalarOps;
  });
  return scalarOps;
}

static int64_t countElementwiseDigitalOps(mlir::func::FuncOp callee) {
  int64_t digitalOps = 0;

  callee.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "linalg.generic") {
      if (!hasOnlyParallelIterators(op))
        return;

      std::optional<int64_t> elementCount =
          getFirstStaticResultElementCount(op);
      if (!elementCount)
        return;

      digitalOps += *elementCount * countScalarOpsInLinalgGeneric(op);
      return;
    }

    if (hasLinalgAncestor(op))
      return;

    if (isLinalgElementwiseNamedOp(op)) {
      if (std::optional<int64_t> elementCount =
              getFirstStaticResultElementCount(op))
        digitalOps += *elementCount;
      return;
    }

    if (!isElementwiseScalarComputeOp(op))
      return;

    if (std::optional<int64_t> elementCount =
            getFirstStaticResultElementCount(op)) {
      digitalOps += *elementCount;
      return;
    }

    if (op->getNumResults() != 0)
      ++digitalOps;
  });

  return digitalOps;
}

static mlir::FailureOr<
    llvm::DenseMap<mlir::Value, LogicalArrayRuntimePlacement>>
buildSimpleBudgetLogicalArrayRuntimePlacementMap(
    const llvm::SmallVectorImpl<mlir::Value> &logicalArrayResources,
    llvm::ArrayRef<task_schedulers::LogicalArrayPlacement> placements,
    const task_schedulers::HardwareBudget &budget) {
  llvm::DenseMap<mlir::Value, LogicalArrayRuntimePlacement>
      placementByLogicalArray;

  for (const task_schedulers::LogicalArrayPlacement &placement : placements) {
    if (placement.logicalArrayIndex < 0 ||
        placement.logicalArrayIndex >=
            static_cast<int64_t>(logicalArrayResources.size())) {
      return mlir::failure();
    }

    int64_t analogArrayIndex = placement.analogArrayIndex;
    int64_t coreId = analogArrayIndex / budget.arraysPerCore;
    if (coreId < 0 || coreId >= budget.numCores) {
      mlir::Operation *resourceOp =
          logicalArrayResources[placement.logicalArrayIndex].getDefiningOp();
      if (resourceOp)
        resourceOp->emitError("physical analog array placement maps outside "
                              "the runtime core budget");
      return mlir::failure();
    }

    placementByLogicalArray.try_emplace(
        logicalArrayResources[placement.logicalArrayIndex],
        LogicalArrayRuntimePlacement{analogArrayIndex, coreId});
  }

  return placementByLogicalArray;
}

static mlir::FailureOr<LogicalArrayRuntimePlacement>
getSingleLogicalArrayPlacement(
    mlir::sculptor::TaskCreateOp taskOp, mlir::OperandRange resources,
    const llvm::DenseMap<mlir::Value, LogicalArrayRuntimePlacement>
        &placementByLogicalArray,
    llvm::StringRef resourceRole) {
  std::optional<LogicalArrayRuntimePlacement> logicalArrayPlacement;

  for (mlir::Value resource : resources) {
    if (!isLogicalArrayResource(resource))
      continue;

    auto placementIt = placementByLogicalArray.find(resource);
    if (placementIt == placementByLogicalArray.end()) {
      taskOp.emitError("expected logical array resource in ")
          << resourceRole << " to have physical analog array placement";
      return mlir::failure();
    }

    if (logicalArrayPlacement && logicalArrayPlacement->physicalArrayId !=
                                     placementIt->second.physicalArrayId) {
      taskOp.emitError("expected logical array resources in ")
          << resourceRole << " to map to one physical analog array";
      return mlir::failure();
    }

    logicalArrayPlacement = placementIt->second;
  }

  if (!logicalArrayPlacement) {
    taskOp.emitError("expected ")
        << taskOp.getTaskKind() << " task to reference a logical array in "
        << resourceRole;
    return mlir::failure();
  }

  return *logicalArrayPlacement;
}

static mlir::FailureOr<llvm::SmallVector<mlir::sculptor::TaskCreateOp>>
buildSimpleBudgetMatrixSetupFirstTaskOrder(
    const task_schedulers::TaskGraphDAG &dag) {
  llvm::SmallVector<mlir::sculptor::TaskCreateOp> matrixSetupTasks;
  llvm::SmallVector<mlir::sculptor::TaskCreateOp> remainingTasks;

  for (const task_schedulers::TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (!isMatrixSetupTask(taskOp)) {
      remainingTasks.push_back(taskOp);
      continue;
    }

    if (!taskOp.getDependencies().empty()) {
      taskOp.emitError("expected sculptor.matrix_setup task to have no task "
                       "dependencies before matrix setup front-loading");
      return mlir::failure();
    }
    matrixSetupTasks.push_back(taskOp);
  }

  llvm::SmallVector<mlir::sculptor::TaskCreateOp> scheduledTasks;
  scheduledTasks.reserve(dag.nodes.size());
  scheduledTasks.append(matrixSetupTasks.begin(), matrixSetupTasks.end());
  scheduledTasks.append(remainingTasks.begin(), remainingTasks.end());
  return scheduledTasks;
}

static void attachSimpleBudgetTaskIndices(
    llvm::ArrayRef<mlir::sculptor::TaskCreateOp> scheduledTasks,
    mlir::Builder &builder) {
  for (auto indexedTask : llvm::enumerate(scheduledTasks)) {
    indexedTask.value()->setAttr(
        runtime_attrs::kTaskIndexAttrName,
        builder.getI64IntegerAttr(static_cast<int64_t>(indexedTask.index())));
  }
}

static mlir::FailureOr<TaskCoreAssignments> buildCoreAssignments(
    llvm::ArrayRef<mlir::sculptor::TaskCreateOp> scheduledTasks,
    const task_schedulers::HardwareBudget &budget,
    const llvm::DenseMap<mlir::Value, LogicalArrayRuntimePlacement>
        &placementByLogicalArray) {
  TaskCoreAssignments assignments;
  int64_t currentCore = 0;
  bool forwardEndedCurrentSegment = false;

  for (mlir::sculptor::TaskCreateOp taskOp : scheduledTasks) {
    if (isMatrixSetupTask(taskOp))
      continue;

    if (isForwardTask(taskOp)) {
      assignments.coreByTask.try_emplace(taskOp.getOperation(), currentCore);
      forwardEndedCurrentSegment = true;
      continue;
    }

    llvm::StringRef sourceLayer = taskOp.getSourceLayer();
    auto sourceIt = assignments.coreBySourceLayer.find(sourceLayer);
    if (sourceIt == assignments.coreBySourceLayer.end()) {
      if (forwardEndedCurrentSegment) {
        if (currentCore + 1 < budget.numCores)
          ++currentCore;
        forwardEndedCurrentSegment = false;
      }
      sourceIt =
          assignments.coreBySourceLayer.try_emplace(sourceLayer, currentCore)
              .first;
    }

    int64_t coreId = sourceIt->getValue();
    if (isMVMTask(taskOp) || isAnalogTask(taskOp)) {
      mlir::FailureOr<LogicalArrayRuntimePlacement> analogPlacement =
          getSingleLogicalArrayPlacement(taskOp, taskOp.getInputs(),
                                         placementByLogicalArray, "inputs");
      if (mlir::failed(analogPlacement))
        return mlir::failure();
      coreId = analogPlacement->coreId;
      assignments.physicalArrayByTask.try_emplace(
          taskOp.getOperation(), analogPlacement->physicalArrayId);
    }

    assignments.coreByTask.try_emplace(taskOp.getOperation(), coreId);
  }

  for (mlir::sculptor::TaskCreateOp taskOp : scheduledTasks) {
    if (!isMatrixSetupTask(taskOp))
      continue;

    mlir::FailureOr<LogicalArrayRuntimePlacement> analogPlacement =
        getSingleLogicalArrayPlacement(taskOp, taskOp.getOutputs(),
                                       placementByLogicalArray, "outputs");
    if (mlir::failed(analogPlacement))
      return mlir::failure();

    assignments.coreByTask.try_emplace(taskOp.getOperation(),
                                       analogPlacement->coreId);
    assignments.physicalArrayByTask.try_emplace(
        taskOp.getOperation(), analogPlacement->physicalArrayId);
  }

  return assignments;
}

static void
attachSimpleBudgetTaskCoreIds(const TaskCoreAssignments &assignments,
                              mlir::Builder &builder) {
  for (auto &assignment : assignments.coreByTask) {
    assignment.first->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                              builder.getI64IntegerAttr(assignment.second));
  }
  for (auto &assignment : assignments.physicalArrayByTask) {
    assignment.first->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
                              builder.getI64IntegerAttr(assignment.second));
  }
}

static bool hasAnalogArrayOps(mlir::func::FuncOp func) {
  bool found = false;
  func.walk([&](mlir::Operation *op) {
    if (mlir::isa<mlir::sculptor::ArraySetOp, mlir::sculptor::ArrayLoadOp,
                  mlir::sculptor::ArrayExecuteOp, mlir::sculptor::ArrayStoreOp>(
            op)) {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

struct FunctionPlacement {
  mlir::sculptor::TaskCreateOp representativeTask;
  std::optional<int64_t> coreId;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> localArrayId;
  bool hasCoreConflict = false;
  bool hasArrayConflict = false;
};

static void recordFunctionPlacement(FunctionPlacement &placement,
                                    mlir::sculptor::TaskCreateOp taskOp,
                                    int64_t coreId) {
  if (!placement.representativeTask)
    placement.representativeTask = taskOp;
  if (placement.coreId && *placement.coreId != coreId)
    placement.hasCoreConflict = true;
  placement.coreId = placement.coreId.value_or(coreId);
}

static void recordFunctionArrayPlacement(FunctionPlacement &placement,
                                         int64_t physicalArrayId,
                                         int64_t localArrayId) {
  if (placement.physicalArrayId &&
      (*placement.physicalArrayId != physicalArrayId ||
       *placement.localArrayId != localArrayId)) {
    placement.hasArrayConflict = true;
  }
  placement.physicalArrayId =
      placement.physicalArrayId.value_or(physicalArrayId);
  placement.localArrayId = placement.localArrayId.value_or(localArrayId);
}

static mlir::LogicalResult attachSimpleBudgetTaskFunctionPlacementAttrs(
    mlir::ModuleOp module, llvm::ArrayRef<mlir::sculptor::TaskCreateOp> tasks,
    const TaskCoreAssignments &assignments,
    const task_schedulers::HardwareBudget &budget) {
  llvm::StringMap<FunctionPlacement> placementByCallee;
  for (mlir::sculptor::TaskCreateOp taskOp : tasks) {
    auto coreIt = assignments.coreByTask.find(taskOp.getOperation());
    if (coreIt == assignments.coreByTask.end())
      continue;

    FunctionPlacement &placement =
        placementByCallee[taskOp.getCalleeAttr().getValue()];
    recordFunctionPlacement(placement, taskOp, coreIt->second);

    auto arrayIt = assignments.physicalArrayByTask.find(taskOp.getOperation());
    if (arrayIt == assignments.physicalArrayByTask.end())
      continue;

    int64_t physicalArrayId = arrayIt->second;
    int64_t localArrayId = physicalArrayId % budget.arraysPerCore;
    recordFunctionArrayPlacement(placement, physicalArrayId, localArrayId);
  }

  mlir::Builder builder(module.getContext());
  for (auto &entry : placementByCallee) {
    mlir::func::FuncOp callee =
        module.lookupSymbol<mlir::func::FuncOp>(entry.getKey());
    if (!callee)
      continue;

    FunctionPlacement &placement = entry.getValue();
    if (placement.hasCoreConflict || placement.hasArrayConflict) {
      if (!callee.isDeclaration() && hasAnalogArrayOps(callee)) {
        placement.representativeTask.emitError("expected task callee '")
            << entry.getKey()
            << "' to have one scheduler placement before shim lowering";
        return mlir::failure();
      }
      continue;
    }

    if (placement.coreId)
      callee->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                      builder.getI64IntegerAttr(*placement.coreId));
    if (placement.physicalArrayId)
      callee->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
                      builder.getI64IntegerAttr(*placement.physicalArrayId));
    if (placement.localArrayId)
      callee->setAttr(runtime_attrs::kTaskLocalArrayIdAttrName,
                      builder.getI64IntegerAttr(*placement.localArrayId));
  }

  return mlir::success();
}

static mlir::FailureOr<int64_t> attachSimpleBudgetTaskDigitalOps(
    mlir::ModuleOp module,
    llvm::ArrayRef<mlir::sculptor::TaskCreateOp> scheduledTasks,
    mlir::Builder &builder) {
  int64_t totalDigitalOps = 0;

  for (mlir::sculptor::TaskCreateOp taskOp : scheduledTasks) {
    int64_t taskDigitalOps = 0;
    if (isDigitalTask(taskOp)) {
      auto callee = module.lookupSymbol<mlir::func::FuncOp>(
          taskOp.getCalleeAttr().getValue());
      if (!callee) {
        taskOp.emitError("expected digital task callee '")
            << taskOp.getCalleeAttr().getValue()
            << "' to resolve to a function";
        return mlir::failure();
      }

      taskDigitalOps = countElementwiseDigitalOps(callee);
      totalDigitalOps += taskDigitalOps;
    }

    taskOp->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                    builder.getI64IntegerAttr(taskDigitalOps));
  }

  return totalDigitalOps;
}

static int64_t getResourceByteSize(mlir::Value resource) {
  mlir::Operation *resourceOp = resource.getDefiningOp();
  if (!resourceOp)
    return 0;

  auto byteSizeAttr = resourceOp->getAttrOfType<mlir::IntegerAttr>(
      runtime_attrs::kResourceByteSizeAttrName);
  if (!byteSizeAttr)
    return 0;

  return byteSizeAttr.getInt();
}

static int64_t getMeshDistance(int64_t sourceCore, int64_t destinationCore,
                               const task_schedulers::HardwareBudget &budget) {
  int64_t sourceRow = sourceCore / budget.meshCols;
  int64_t sourceCol = sourceCore % budget.meshCols;
  int64_t destinationRow = destinationCore / budget.meshCols;
  int64_t destinationCol = destinationCore % budget.meshCols;
  return std::llabs(sourceRow - destinationRow) +
         std::llabs(sourceCol - destinationCol);
}

static CoreTransferSummary computeSimpleBudgetCoreTransferSummary(
    llvm::ArrayRef<mlir::sculptor::TaskCreateOp> scheduledTasks,
    const task_schedulers::HardwareBudget &budget,
    const TaskCoreAssignments &assignments) {
  CoreTransferSummary summary;
  int64_t matrixSize = budget.numCores * budget.numCores;
  summary.transferBytes.assign(static_cast<size_t>(matrixSize), 0);
  summary.transferCost.assign(static_cast<size_t>(matrixSize), 0);

  llvm::DenseMap<mlir::Value, mlir::sculptor::TaskCreateOp> producerByResource;
  for (mlir::sculptor::TaskCreateOp taskOp : scheduledTasks) {
    for (mlir::Value output : taskOp.getOutputs())
      producerByResource[output] = taskOp;
  }

  for (mlir::sculptor::TaskCreateOp consumer : scheduledTasks) {
    auto consumerCoreIt = assignments.coreByTask.find(consumer.getOperation());
    if (consumerCoreIt == assignments.coreByTask.end())
      continue;

    int64_t consumerCore = consumerCoreIt->second;
    for (mlir::Value input : consumer.getInputs()) {
      auto producerIt = producerByResource.find(input);
      if (producerIt == producerByResource.end())
        continue;

      auto producerCoreIt =
          assignments.coreByTask.find(producerIt->second.getOperation());
      if (producerCoreIt == assignments.coreByTask.end())
        continue;

      int64_t producerCore = producerCoreIt->second;
      if (producerCore == consumerCore)
        continue;

      int64_t byteSize = getResourceByteSize(input);
      if (byteSize <= 0)
        continue;

      size_t matrixIndex =
          static_cast<size_t>(producerCore * budget.numCores + consumerCore);
      int64_t transferCost =
          byteSize * getMeshDistance(producerCore, consumerCore, budget);
      summary.transferBytes[matrixIndex] += byteSize;
      summary.transferCost[matrixIndex] += transferCost;
      summary.interCoreTransferBytes += byteSize;
      summary.totalTransferCost += transferCost;
    }
  }

  return summary;
}

class SimpleBudgetTaskScheduler final
    : public task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "simple-budget"; }

  mlir::LogicalResult schedule(
      mlir::ModuleOp module, mlir::func::FuncOp taskGraphFunc,
      const task_schedulers::HardwareBudget &budget,
      const task_schedulers::TaskGraphDAG &dag,
      const task_schedulers::TaskGraphScheduleOptions &options) const final {
    (void)module;
    (void)options;

    if (mlir::failed(task_schedulers::verifyLogicalArraysFitAnalogBudget(
            taskGraphFunc.getOperation(),
            static_cast<int64_t>(dag.logicalArrayResources.size()),
            budget.numAnalogArrays)))
      return mlir::failure();

    mlir::Builder builder(taskGraphFunc.getContext());
    llvm::SmallVector<task_schedulers::LogicalArrayPlacement> placements =
        buildSimpleLogicalArrayPlacements(dag, budget);
    attachSimpleBudgetLogicalArrayPlacementAttrs(dag.logicalArrayResources,
                                                 placements, builder);
    mlir::FailureOr<llvm::DenseMap<mlir::Value, LogicalArrayRuntimePlacement>>
        placementByLogicalArray =
            buildSimpleBudgetLogicalArrayRuntimePlacementMap(
                dag.logicalArrayResources, placements, budget);
    if (mlir::failed(placementByLogicalArray))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<mlir::sculptor::TaskCreateOp>>
        scheduledTasks = buildSimpleBudgetMatrixSetupFirstTaskOrder(dag);
    if (mlir::failed(scheduledTasks))
      return mlir::failure();
    attachSimpleBudgetTaskIndices(*scheduledTasks, builder);

    mlir::FailureOr<TaskCoreAssignments> coreAssignments =
        buildCoreAssignments(*scheduledTasks, budget, *placementByLogicalArray);
    if (mlir::failed(coreAssignments))
      return mlir::failure();
    attachSimpleBudgetTaskCoreIds(*coreAssignments, builder);
    if (mlir::failed(attachSimpleBudgetTaskFunctionPlacementAttrs(
            module, *scheduledTasks, *coreAssignments, budget)))
      return mlir::failure();

    mlir::FailureOr<int64_t> totalDigitalOps =
        attachSimpleBudgetTaskDigitalOps(module, *scheduledTasks, builder);
    if (mlir::failed(totalDigitalOps))
      return mlir::failure();

    CoreTransferSummary transferSummary =
        computeSimpleBudgetCoreTransferSummary(*scheduledTasks, budget,
                                               *coreAssignments);

    taskGraphFunc->setAttr(
        schedule_attrs::kTaskCountAttrName,
        builder.getI64IntegerAttr(static_cast<int64_t>(dag.nodes.size())));
    taskGraphFunc->setAttr(
        schedule_attrs::kDependencyCountAttrName,
        builder.getI64IntegerAttr(static_cast<int64_t>(dag.dependencyCount)));
    taskGraphFunc->setAttr(
        schedule_attrs::kCoreTransferBytesAttrName,
        buildSimpleBudgetI64ArrayAttr(builder, transferSummary.transferBytes));
    taskGraphFunc->setAttr(
        schedule_attrs::kInterCoreTransferBytesAttrName,
        builder.getI64IntegerAttr(transferSummary.interCoreTransferBytes));
    taskGraphFunc->setAttr(
        schedule_attrs::kCoreTransferCostAttrName,
        buildSimpleBudgetI64ArrayAttr(builder, transferSummary.transferCost));
    taskGraphFunc->setAttr(
        schedule_attrs::kTotalTransferCostAttrName,
        builder.getI64IntegerAttr(transferSummary.totalTransferCost));
    taskGraphFunc->setAttr(schedule_attrs::kTotalDigitalOpsAttrName,
                           builder.getI64IntegerAttr(*totalDigitalOps));
    taskGraphFunc->setAttr(schedule_attrs::kNumLogicalArraysAttrName,
                           builder.getI64IntegerAttr(static_cast<int64_t>(
                               dag.logicalArrayResources.size())));
    taskGraphFunc->setAttr(
        schedule_attrs::kLogicalArrayToAnalogArrayAttrName,
        buildSimpleBudgetLogicalArrayToAnalogArrayAttr(placements, builder));
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerSimpleBudgetTaskScheduler(TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(
      registry, std::make_unique<SimpleBudgetTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
