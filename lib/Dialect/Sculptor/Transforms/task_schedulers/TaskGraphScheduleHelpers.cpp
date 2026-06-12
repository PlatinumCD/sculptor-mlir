#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleHelpers.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdlib>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;

std::optional<int64_t> getStaticElementCount(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.hasRank() || !shapedType.hasStaticShape())
    return std::nullopt;
  return shapedType.getNumElements();
}

std::optional<int64_t> getFirstStaticResultElementCount(Operation *op) {
  for (Type resultType : op->getResultTypes()) {
    std::optional<int64_t> count = getStaticElementCount(resultType);
    if (count)
      return count;
  }
  return std::nullopt;
}

bool hasShapedResult(Operation *op) {
  return getFirstStaticResultElementCount(op).has_value();
}

bool hasLinalgAncestor(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef().starts_with("linalg."))
      return true;
  }
  return false;
}

bool isLinalgElementwiseNamedOp(Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();
  return opName == "linalg.add" || opName == "linalg.sub" ||
         opName == "linalg.mul" || opName == "linalg.div" ||
         opName == "linalg.max" || opName == "linalg.min";
}

bool isElementwiseScalarComputeOp(Operation *op) {
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

bool hasOnlyParallelIterators(Operation *op) {
  auto iteratorTypes = op->getAttrOfType<ArrayAttr>("iterator_types");
  if (!iteratorTypes)
    return false;

  for (Attribute attr : iteratorTypes) {
    if (auto iteratorType = dyn_cast<StringAttr>(attr)) {
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

int64_t countScalarOpsInLinalgGeneric(Operation *genericOp) {
  int64_t scalarOps = 0;
  genericOp->walk([&](Operation *nestedOp) {
    if (nestedOp == genericOp)
      return;
    if (isElementwiseScalarComputeOp(nestedOp) && !hasShapedResult(nestedOp))
      ++scalarOps;
  });
  return scalarOps;
}

int64_t countElementwiseDigitalOps(func::FuncOp callee) {
  int64_t digitalOps = 0;

  callee.walk([&](Operation *op) {
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

bool hasAnalogArrayOps(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<ArraySetOp, ArrayLoadOp, ArrayExecuteOp, ArrayStoreOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

struct FunctionPlacement {
  sculptor::TaskCreateOp representativeTask;
  std::optional<int64_t> coreId;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> localArrayId;
  bool hasCoreConflict = false;
  bool hasArrayConflict = false;
};

void recordFunctionPlacement(FunctionPlacement &placement,
                             sculptor::TaskCreateOp taskOp, int64_t coreId) {
  if (!placement.representativeTask)
    placement.representativeTask = taskOp;
  if (placement.coreId && *placement.coreId != coreId)
    placement.hasCoreConflict = true;
  placement.coreId = placement.coreId.value_or(coreId);
}

void recordFunctionArrayPlacement(FunctionPlacement &placement,
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

int64_t getResourceByteSize(Value resource) {
  Operation *resourceOp = resource.getDefiningOp();
  if (!resourceOp)
    return 0;

  auto byteSizeAttr = resourceOp->getAttrOfType<IntegerAttr>(
      runtime_attrs::kResourceByteSizeAttrName);
  if (!byteSizeAttr)
    return 0;

  return byteSizeAttr.getInt();
}

int64_t getMeshDistance(int64_t sourceCore, int64_t destinationCore,
                        const HardwareBudget &budget) {
  int64_t sourceRow = sourceCore / budget.meshCols;
  int64_t sourceCol = sourceCore % budget.meshCols;
  int64_t destinationRow = destinationCore / budget.meshCols;
  int64_t destinationCol = destinationCore % budget.meshCols;
  return std::llabs(sourceRow - destinationRow) +
         std::llabs(sourceCol - destinationCol);
}

} // namespace

LogicalResult verifyLogicalArraysFitAnalogBudget(Operation *diagnosticOp,
                                                 int64_t numLogicalArrays,
                                                 int64_t numAnalogArrays) {
  if (numLogicalArrays <= numAnalogArrays)
    return success();

  assert(diagnosticOp && "expected diagnostic operation");
  diagnosticOp->emitError()
      << "logical analog array count exceeds physical analog array budget: "
      << numLogicalArrays << " logical arrays require more than "
      << numAnalogArrays << " physical analog arrays";
  return failure();
}

bool isMatrixSetupTask(sculptor::TaskCreateOp taskOp) {
  return taskOp.getTaskKind() == task_graph_names::kMatrixSetupTaskKind;
}

bool isMVMTask(sculptor::TaskCreateOp taskOp) {
  return taskOp.getTaskKind() == task_graph_names::kMVMTaskKind;
}

bool isAnalogTask(sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() == task_graph_names::kAnalogDomain;
}

bool isDigitalTask(sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() == task_graph_names::kDigitalDomain;
}

bool isLogicalArrayResource(Value value) {
  auto resourceType = dyn_cast<sculptor::TaskResourceType>(value.getType());
  return resourceType &&
         isa<sculptor::LogicalArrayType>(resourceType.getValueType());
}

ArrayAttr buildI64ArrayAttr(Builder &builder, llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

void attachLogicalArrayPlacementAttrs(
    const llvm::SmallVectorImpl<Value> &logicalArrayResources,
    llvm::ArrayRef<LogicalArrayPlacement> placements, Builder &builder) {
  for (const LogicalArrayPlacement &placement : placements) {
    Operation *resourceOp =
        logicalArrayResources[placement.logicalArrayIndex].getDefiningOp();
    if (!resourceOp)
      continue;

    resourceOp->setAttr(schedule_attrs::kLogicalArrayIndexAttrName,
                        builder.getI64IntegerAttr(placement.logicalArrayIndex));
  }
}

ArrayAttr buildLogicalArrayToAnalogArrayAttr(
    llvm::ArrayRef<LogicalArrayPlacement> placements, Builder &builder) {
  llvm::SmallVector<int64_t> analogArrayByLogicalArray;
  analogArrayByLogicalArray.reserve(placements.size());
  for (const LogicalArrayPlacement &placement : placements)
    analogArrayByLogicalArray.push_back(placement.analogArrayIndex);
  return buildI64ArrayAttr(builder, analogArrayByLogicalArray);
}

FailureOr<llvm::DenseMap<Value, LogicalArrayRuntimePlacement>>
buildLogicalArrayRuntimePlacementMap(
    const llvm::SmallVectorImpl<Value> &logicalArrayResources,
    llvm::ArrayRef<LogicalArrayPlacement> placements,
    const HardwareBudget &budget) {
  llvm::DenseMap<Value, LogicalArrayRuntimePlacement> placementByLogicalArray;

  for (const LogicalArrayPlacement &placement : placements) {
    if (placement.logicalArrayIndex < 0 ||
        placement.logicalArrayIndex >=
            static_cast<int64_t>(logicalArrayResources.size())) {
      return failure();
    }

    int64_t analogArrayIndex = placement.analogArrayIndex;
    int64_t coreId = analogArrayIndex / budget.arraysPerCore;
    if (coreId < 0 || coreId >= budget.numCores) {
      Operation *resourceOp =
          logicalArrayResources[placement.logicalArrayIndex].getDefiningOp();
      if (resourceOp)
        resourceOp->emitError("physical analog array placement maps outside "
                              "the runtime core budget");
      return failure();
    }

    placementByLogicalArray.try_emplace(
        logicalArrayResources[placement.logicalArrayIndex],
        LogicalArrayRuntimePlacement{analogArrayIndex, coreId});
  }

  return placementByLogicalArray;
}

FailureOr<LogicalArrayRuntimePlacement> getSingleLogicalArrayPlacement(
    sculptor::TaskCreateOp taskOp, OperandRange resources,
    const llvm::DenseMap<Value, LogicalArrayRuntimePlacement>
        &placementByLogicalArray,
    llvm::StringRef resourceRole) {
  std::optional<LogicalArrayRuntimePlacement> logicalArrayPlacement;

  for (Value resource : resources) {
    if (!isLogicalArrayResource(resource))
      continue;

    auto placementIt = placementByLogicalArray.find(resource);
    if (placementIt == placementByLogicalArray.end()) {
      taskOp.emitError("expected logical array resource in ")
          << resourceRole << " to have physical analog array placement";
      return failure();
    }

    if (logicalArrayPlacement && logicalArrayPlacement->physicalArrayId !=
                                     placementIt->second.physicalArrayId) {
      taskOp.emitError("expected logical array resources in ")
          << resourceRole << " to map to one physical analog array";
      return failure();
    }

    logicalArrayPlacement = placementIt->second;
  }

  if (!logicalArrayPlacement) {
    taskOp.emitError("expected ")
        << taskOp.getTaskKind() << " task to reference a logical array in "
        << resourceRole;
    return failure();
  }

  return *logicalArrayPlacement;
}

FailureOr<llvm::SmallVector<sculptor::TaskCreateOp>>
buildMatrixSetupFirstTaskOrder(const TaskGraphDAG &dag) {
  llvm::SmallVector<sculptor::TaskCreateOp> matrixSetupTasks;
  llvm::SmallVector<sculptor::TaskCreateOp> remainingTasks;

  for (const TaskGraphNode &node : dag.nodes) {
    sculptor::TaskCreateOp taskOp = node.op;
    if (!isMatrixSetupTask(taskOp)) {
      remainingTasks.push_back(taskOp);
      continue;
    }

    if (!taskOp.getDependencies().empty()) {
      taskOp.emitError("expected sculptor.matrix_setup task to have no task "
                       "dependencies before matrix setup front-loading");
      return failure();
    }
    matrixSetupTasks.push_back(taskOp);
  }

  llvm::SmallVector<sculptor::TaskCreateOp> scheduledTasks;
  scheduledTasks.reserve(dag.nodes.size());
  scheduledTasks.append(matrixSetupTasks.begin(), matrixSetupTasks.end());
  scheduledTasks.append(remainingTasks.begin(), remainingTasks.end());
  return scheduledTasks;
}

void attachTaskIndices(llvm::ArrayRef<sculptor::TaskCreateOp> scheduledTasks,
                       Builder &builder) {
  for (auto indexedTask : llvm::enumerate(scheduledTasks)) {
    indexedTask.value()->setAttr(
        runtime_attrs::kTaskIndexAttrName,
        builder.getI64IntegerAttr(static_cast<int64_t>(indexedTask.index())));
  }
}

void attachTaskCoreIds(const TaskCoreAssignments &assignments,
                       Builder &builder) {
  for (auto &assignment : assignments.coreByTask) {
    assignment.first->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                              builder.getI64IntegerAttr(assignment.second));
  }
  for (auto &assignment : assignments.physicalArrayByTask) {
    assignment.first->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
                              builder.getI64IntegerAttr(assignment.second));
  }
}

LogicalResult attachTaskFunctionPlacementAttrs(
    ModuleOp module, llvm::ArrayRef<sculptor::TaskCreateOp> tasks,
    const TaskCoreAssignments &assignments, const HardwareBudget &budget) {
  llvm::StringMap<FunctionPlacement> placementByCallee;
  for (sculptor::TaskCreateOp taskOp : tasks) {
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

  Builder builder(module.getContext());
  for (auto &entry : placementByCallee) {
    func::FuncOp callee = module.lookupSymbol<func::FuncOp>(entry.getKey());
    if (!callee)
      continue;

    FunctionPlacement &placement = entry.getValue();
    if (placement.hasCoreConflict || placement.hasArrayConflict) {
      if (!callee.isDeclaration() && hasAnalogArrayOps(callee)) {
        placement.representativeTask.emitError("expected task callee '")
            << entry.getKey()
            << "' to have one scheduler placement before shim lowering";
        return failure();
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

  return success();
}

FailureOr<int64_t>
attachTaskDigitalOps(ModuleOp module,
                     llvm::ArrayRef<sculptor::TaskCreateOp> scheduledTasks,
                     Builder &builder) {
  int64_t totalDigitalOps = 0;

  for (sculptor::TaskCreateOp taskOp : scheduledTasks) {
    int64_t taskDigitalOps = 0;
    if (isDigitalTask(taskOp)) {
      auto callee =
          module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
      if (!callee) {
        taskOp.emitError("expected digital task callee '")
            << taskOp.getCalleeAttr().getValue()
            << "' to resolve to a function";
        return failure();
      }

      taskDigitalOps = countElementwiseDigitalOps(callee);
      totalDigitalOps += taskDigitalOps;
    }

    taskOp->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                    builder.getI64IntegerAttr(taskDigitalOps));
  }

  return totalDigitalOps;
}

CoreTransferSummary
computeCoreTransferSummary(llvm::ArrayRef<sculptor::TaskCreateOp> scheduledTasks,
                           const HardwareBudget &budget,
                           const TaskCoreAssignments &assignments) {
  CoreTransferSummary summary;
  int64_t matrixSize = budget.numCores * budget.numCores;
  summary.transferBytes.assign(static_cast<size_t>(matrixSize), 0);
  summary.transferCost.assign(static_cast<size_t>(matrixSize), 0);

  llvm::DenseMap<Value, sculptor::TaskCreateOp> producerByResource;
  for (sculptor::TaskCreateOp taskOp : scheduledTasks) {
    for (Value output : taskOp.getOutputs())
      producerByResource[output] = taskOp;
  }

  for (sculptor::TaskCreateOp consumer : scheduledTasks) {
    auto consumerCoreIt = assignments.coreByTask.find(consumer.getOperation());
    if (consumerCoreIt == assignments.coreByTask.end())
      continue;

    int64_t consumerCore = consumerCoreIt->second;
    for (Value input : consumer.getInputs()) {
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
      int64_t matrixIndex = producerCore * budget.numCores + consumerCore;
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

void attachTaskGraphScheduleSummaryAttrs(
    func::FuncOp taskGraphFunc, const TaskGraphDAG &dag,
    llvm::ArrayRef<LogicalArrayPlacement> placements,
    const CoreTransferSummary &transferSummary, int64_t totalDigitalOps,
    Builder &builder) {
  taskGraphFunc->setAttr(
      schedule_attrs::kTaskCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag.nodes.size())));
  taskGraphFunc->setAttr(
      schedule_attrs::kDependencyCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag.dependencyCount)));
  taskGraphFunc->setAttr(
      schedule_attrs::kCoreTransferBytesAttrName,
      buildI64ArrayAttr(builder, transferSummary.transferBytes));
  taskGraphFunc->setAttr(
      schedule_attrs::kInterCoreTransferBytesAttrName,
      builder.getI64IntegerAttr(transferSummary.interCoreTransferBytes));
  taskGraphFunc->setAttr(
      schedule_attrs::kCoreTransferCostAttrName,
      buildI64ArrayAttr(builder, transferSummary.transferCost));
  taskGraphFunc->setAttr(
      schedule_attrs::kTotalTransferCostAttrName,
      builder.getI64IntegerAttr(transferSummary.totalTransferCost));
  taskGraphFunc->setAttr(schedule_attrs::kTotalDigitalOpsAttrName,
                         builder.getI64IntegerAttr(totalDigitalOps));
  taskGraphFunc->setAttr(schedule_attrs::kNumLogicalArraysAttrName,
                         builder.getI64IntegerAttr(static_cast<int64_t>(
                             dag.logicalArrayResources.size())));
  taskGraphFunc->setAttr(
      schedule_attrs::kLogicalArrayToAnalogArrayAttrName,
      buildLogicalArrayToAnalogArrayAttr(placements, builder));
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
