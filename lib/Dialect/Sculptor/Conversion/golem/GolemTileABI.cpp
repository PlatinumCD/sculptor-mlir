#include "GolemTileABI.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResourceUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace mlir {
namespace sculptor {
namespace golem_tile_abi {

namespace {

template <typename Integer>
FailureOr<Integer> getRequiredUnsignedAttr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr) {
    op->emitError("expected required attribute '") << name << "'";
    return failure();
  }

  int64_t value = attr.getInt();
  if (value < 0 ||
      static_cast<uint64_t>(value) >
          static_cast<uint64_t>(std::numeric_limits<Integer>::max())) {
    op->emitError("expected attribute '")
        << name
        << "' to be a non-negative value representable by the Golem "
           "tile ABI";
    return failure();
  }
  return static_cast<Integer>(value);
}

template <typename Integer>
FailureOr<std::optional<Integer>> getOptionalUnsignedAttr(Operation *op,
                                                          StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return std::optional<Integer>();

  int64_t value = attr.getInt();
  if (value < 0 ||
      static_cast<uint64_t>(value) >
          static_cast<uint64_t>(std::numeric_limits<Integer>::max())) {
    op->emitError("expected attribute '")
        << name
        << "' to be a non-negative value representable by the Golem "
           "tile ABI";
    return failure();
  }
  return std::optional<Integer>(static_cast<Integer>(value));
}

FailureOr<SmallVector<uint32_t>> getRequiredU32ArrayAttr(Operation *op,
                                                         StringRef name) {
  Attribute attr = op->getAttr(name);
  if (!attr) {
    op->emitError("expected required attribute '") << name << "'";
    return failure();
  }

  SmallVector<uint32_t> result;
  auto appendValue = [&](int64_t value) -> LogicalResult {
    if (value < 0 ||
        static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max()) {
      op->emitError("expected attribute '")
          << name << "' to contain non-negative 32-bit integers";
      return failure();
    }
    result.push_back(static_cast<uint32_t>(value));
    return success();
  };

  if (auto dense = dyn_cast<DenseI64ArrayAttr>(attr)) {
    result.reserve(dense.size());
    for (int64_t value : dense.asArrayRef())
      if (failed(appendValue(value)))
        return failure();
    return result;
  }

  auto array = dyn_cast<ArrayAttr>(attr);
  if (!array) {
    op->emitError("expected attribute '") << name << "' to be an integer array";
    return failure();
  }
  result.reserve(array.size());
  for (Attribute entry : array) {
    auto integer = dyn_cast<IntegerAttr>(entry);
    if (!integer || failed(appendValue(integer.getInt())))
      return failure();
  }
  return result;
}

bool hasRepresentableContiguousStrides(ShapedType shapedType) {
  uint64_t stride = 1;
  constexpr uint64_t maxStride =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  for (int64_t dim = shapedType.getRank() - 1; dim >= 0; --dim) {
    uint64_t size = static_cast<uint64_t>(shapedType.getDimSize(dim));
    if (size != 0 && stride > maxStride / size)
      return false;
    stride *= size;
  }
  return true;
}

FailureOr<ShapedType> getSupportedTensorType(Operation *op, Type type) {
  auto resourceType = dyn_cast<TaskResourceType>(type);
  Type valueType = resourceType ? resourceType.getValueType() : type;
  auto shapedType = dyn_cast<ShapedType>(valueType);
  if (!shapedType || !shapedType.hasStaticShape() ||
      !shapedType.getElementType().isF32()) {
    op->emitError(
        "Golem tile task ABI supports only statically shaped f32 tensors");
    return failure();
  }
  if (!hasRepresentableContiguousStrides(shapedType)) {
    op->emitError(
        "Golem tile task ABI tensor strides must fit in a signed 64-bit value");
    return failure();
  }
  return shapedType;
}

bool isRouteResourceKind(ResourceKind kind) {
  return kind == ResourceKind::RouteInput || kind == ResourceKind::RouteOutput;
}

std::optional<std::pair<ResourceKind, Value>>
getResourceKindAndValue(Operation &op) {
  if (auto resource = dyn_cast<TaskGraphInputOp>(&op))
    return std::make_pair(ResourceKind::ModelInput, resource.getResult());
  if (auto resource = dyn_cast<TaskGraphOutputOp>(&op))
    return std::make_pair(ResourceKind::ModelOutput, resource.getResult());
  if (auto resource = dyn_cast<TaskGraphIntermediateOp>(&op))
    return std::make_pair(ResourceKind::Intermediate, resource.getResult());
  if (auto resource = dyn_cast<TaskGraphPersistentOp>(&op))
    return std::make_pair(ResourceKind::Persistent, resource.getResult());
  if (auto resource = dyn_cast<TaskGraphRouteInputOp>(&op))
    return std::make_pair(ResourceKind::RouteInput, resource.getResult());
  if (auto resource = dyn_cast<TaskGraphRouteOutputOp>(&op))
    return std::make_pair(ResourceKind::RouteOutput, resource.getResult());
  return std::nullopt;
}

FailureOr<func::FuncOp> findTaskGraphFunction(ModuleOp module) {
  func::FuncOp taskGraphFunc;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    FunctionType type = func.getFunctionType();
    bool returnsGraph =
        type.getNumResults() == 1 && isa<TaskGraphType>(type.getResult(0));
    if (!returnsGraph) {
      func.emitError("expected every task implementation to be an llvm.func "
                     "before emitting the Golem tile ABI");
      return failure();
    }
    if (taskGraphFunc) {
      func.emitError(
          "expected exactly one declarative task graph in an isolated core");
      return failure();
    }
    taskGraphFunc = func;
  }

  if (!taskGraphFunc) {
    module.emitError(
        "expected @generate_task_graph in an isolated finalized core module");
    return failure();
  }
  if (taskGraphFunc.getName() != "generate_task_graph") {
    taskGraphFunc.emitError("expected the declarative task graph symbol to be "
                            "@generate_task_graph");
    return failure();
  }
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError(
        "expected @generate_task_graph to contain exactly one block");
    return failure();
  }
  return taskGraphFunc;
}

LogicalResult collectResources(TileModel &model) {
  // Global IDs identify logical payloads, while route IDs identify individual
  // transfers of those payloads. Keep the identity domains separate so fan-out
  // routes can share a global resource without sharing a local buffer.
  DenseSet<uint32_t> nonRouteGlobalIds;
  DenseSet<uint32_t> routeIds;
  DenseSet<uint32_t> slots;

  for (Operation &op : model.taskGraphFunc.getBody().front()) {
    auto kindAndValue = getResourceKindAndValue(op);
    if (!kindAndValue)
      continue;
    if (kindAndValue->first == ResourceKind::Persistent) {
      op.emitError("Golem tile ABI does not support persistent task-graph "
                   "resource storage");
      return failure();
    }

    auto shapedType =
        getSupportedTensorType(&op, kindAndValue->second.getType());
    auto globalId = getRequiredUnsignedAttr<uint32_t>(
        &op, deployment_attrs::kGlobalResourceIdAttrName);
    auto slot = getRequiredUnsignedAttr<uint32_t>(
        &op, runtime_attrs::kResourceSlotAttrName);
    auto byteSize = getRequiredUnsignedAttr<uint64_t>(
        &op, runtime_attrs::kResourceByteSizeAttrName);
    if (failed(shapedType) || failed(globalId) || failed(slot) ||
        failed(byteSize))
      return failure();

    std::optional<uint32_t> routeId;
    if (isRouteResourceKind(kindAndValue->first)) {
      auto collectedRouteId = getRequiredUnsignedAttr<uint32_t>(
          &op, deployment_attrs::kRouteIdAttrName);
      if (failed(collectedRouteId))
        return failure();
      routeId = *collectedRouteId;
      if (!routeIds.insert(*routeId).second) {
        op.emitError("duplicate sculptor.deployment.route_id ") << *routeId;
        return failure();
      }
    } else {
      if (!nonRouteGlobalIds.insert(*globalId).second) {
        op.emitError("duplicate non-route "
                     "sculptor.deployment.global_resource_id ")
            << *globalId;
        return failure();
      }
    }
    if (!slots.insert(*slot).second) {
      op.emitError("duplicate core-local sculptor.runtime.slot ") << *slot;
      return failure();
    }

    FailureOr<int64_t> computedByteSize =
        getTaskResourceByteSize(kindAndValue->second);
    if (failed(computedByteSize) || *computedByteSize < 0 ||
        static_cast<uint64_t>(*computedByteSize) != *byteSize) {
      op.emitError("resource byte size does not match its static tensor type");
      return failure();
    }

    std::optional<uint64_t> workspaceOffset;
    if (kindAndValue->first == ResourceKind::Intermediate ||
        kindAndValue->first == ResourceKind::RouteInput ||
        kindAndValue->first == ResourceKind::RouteOutput) {
      auto offset = getRequiredUnsignedAttr<uint64_t>(
          &op, runtime_attrs::kResourceTempOffsetAttrName);
      if (failed(offset))
        return failure();
      workspaceOffset = *offset;
    }

    unsigned resourceIndex = model.resources.size();
    model.resourceIndexByValue.try_emplace(kindAndValue->second, resourceIndex);
    model.resources.push_back(ResourceModel{
        &op, kindAndValue->second, *shapedType, kindAndValue->first, *globalId,
        routeId, *slot, 0, *byteSize, workspaceOffset});
    if (kindAndValue->first == ResourceKind::RouteInput) {
      model.routeInputResourceIndexByRouteId.try_emplace(*routeId,
                                                         resourceIndex);
    } else if (kindAndValue->first == ResourceKind::RouteOutput) {
      model.routeOutputResourceIndexByRouteId.try_emplace(*routeId,
                                                          resourceIndex);
    } else {
      model.nonRouteResourceIndexByGlobalId.try_emplace(*globalId,
                                                        resourceIndex);
    }
  }

  if (model.resources.size() > std::numeric_limits<uint32_t>::max()) {
    model.taskGraphFunc.emitError(
        "local resource count exceeds the Golem tile ABI");
    return failure();
  }
  uint32_t resourceCount = static_cast<uint32_t>(model.resources.size());

  model.resourceIndicesBySlot.assign(resourceCount,
                                     std::numeric_limits<unsigned>::max());
  for (auto indexedResource : llvm::enumerate(model.resources)) {
    ResourceModel &resource = indexedResource.value();
    if (resource.slot >= resourceCount) {
      resource.op->emitError("core-local resource slot ")
          << resource.slot << " is outside the finalized resource table";
      return failure();
    }
    model.resourceIndicesBySlot[resource.slot] = indexedResource.index();
  }
  for (auto indexedResource : llvm::enumerate(model.resourceIndicesBySlot)) {
    if (indexedResource.value() == std::numeric_limits<unsigned>::max()) {
      model.taskGraphFunc.emitError("finalized resource table is missing "
                                    "core-local slot ")
          << indexedResource.index();
      return failure();
    }

    ResourceModel &resource = model.resources[indexedResource.value()];
    int64_t rank = resource.shapedType.getRank();
    uint64_t dimensionOffset = model.resourceDimensions.size();
    if (rank < 0 ||
        static_cast<uint64_t>(rank) > std::numeric_limits<uint32_t>::max() ||
        dimensionOffset > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(rank) >
            std::numeric_limits<uint32_t>::max() - dimensionOffset) {
      resource.op->emitError(
          "resource rank or dimension offset exceeds the Golem tile ABI");
      return failure();
    }
    resource.dimensionOffset = static_cast<uint32_t>(dimensionOffset);
    model.resourceDimensions.append(resource.shapedType.getShape().begin(),
                                    resource.shapedType.getShape().end());
  }

  return success();
}

FailureOr<SmallVector<unsigned>> getResultIndices(TaskCreateOp task,
                                                  unsigned outputCount) {
  SmallVector<unsigned> result;
  auto attr =
      task->getAttrOfType<ArrayAttr>(runtime_attrs::kTaskResultIndicesAttrName);
  if (!attr) {
    result.reserve(outputCount);
    for (unsigned index = 0; index < outputCount; ++index)
      result.push_back(index);
    return result;
  }

  if (attr.size() != outputCount) {
    task.emitError("expected '") << runtime_attrs::kTaskResultIndicesAttrName
                                 << "' to match the number of task outputs";
    return failure();
  }

  for (Attribute entry : attr) {
    auto integer = dyn_cast<IntegerAttr>(entry);
    if (!integer || integer.getInt() < 0 ||
        static_cast<uint64_t>(integer.getInt()) >
            std::numeric_limits<unsigned>::max()) {
      task.emitError("expected '")
          << runtime_attrs::kTaskResultIndicesAttrName
          << "' to contain non-negative result indexes";
      return failure();
    }
    result.push_back(static_cast<unsigned>(integer.getInt()));
  }
  return result;
}

static void appendFlattenedMemRefParameters(SmallVectorImpl<Type> &parameters,
                                            ShapedType shapedType, Type ptrType,
                                            Type i64Type) {
  parameters.push_back(ptrType);
  parameters.push_back(ptrType);
  parameters.push_back(i64Type);
  parameters.append(shapedType.getRank(), i64Type);
  parameters.append(shapedType.getRank(), i64Type);
}

LogicalResult validateCalleeSignature(TaskModel &task) {
  auto functionType = task.callee.getFunctionType();
  SmallVector<Type> expectedInputParameters;
  Type ptrType = LLVM::LLVMPointerType::get(task.op.getContext());
  Type i64Type = IntegerType::get(task.op.getContext(), 64);
  for (ShapedType inputType : task.inputTypes)
    appendFlattenedMemRefParameters(expectedInputParameters, inputType, ptrType,
                                    i64Type);

  bool hasDenseResultIndices = true;
  if (!task.outputTypes.empty()) {
    unsigned maximumResultIndex = *llvm::max_element(task.resultIndices);
    if (maximumResultIndex == std::numeric_limits<unsigned>::max()) {
      task.op.emitError("task result index exceeds the Golem tile ABI");
      return failure();
    }
    unsigned resultCount = maximumResultIndex + 1;
    task.canonicalOutputIndices.assign(resultCount,
                                       std::numeric_limits<unsigned>::max());
    for (auto indexedOutput : llvm::enumerate(task.outputTypes)) {
      unsigned outputIndex = indexedOutput.index();
      unsigned resultIndex = task.resultIndices[outputIndex];
      unsigned &canonicalOutputIndex = task.canonicalOutputIndices[resultIndex];
      if (canonicalOutputIndex == std::numeric_limits<unsigned>::max()) {
        canonicalOutputIndex = outputIndex;
        continue;
      }
      if (task.outputTypes[canonicalOutputIndex] != indexedOutput.value()) {
        task.op.emitError("task outputs selecting result index ")
            << resultIndex << " must have the same tensor type";
        return failure();
      }
    }
    hasDenseResultIndices =
        llvm::none_of(task.canonicalOutputIndices, [](unsigned outputIndex) {
          return outputIndex == std::numeric_limits<unsigned>::max();
        });
  }

  SmallVector<Type> expectedOutputParameters(expectedInputParameters);
  if (hasDenseResultIndices) {
    for (unsigned outputIndex : task.canonicalOutputIndices)
      appendFlattenedMemRefParameters(expectedOutputParameters,
                                      task.outputTypes[outputIndex], ptrType,
                                      i64Type);
  }

  bool hasInputParameters =
      functionType.getParams() == ArrayRef<Type>(expectedInputParameters);
  bool hasOutputParameters =
      !task.outputTypes.empty() && hasDenseResultIndices &&
      functionType.getParams() == ArrayRef<Type>(expectedOutputParameters) &&
      isa<LLVM::LLVMVoidType>(functionType.getReturnType());
  if (!hasInputParameters && !hasOutputParameters) {
    task.op.emitError("LLVM task callee '")
        << task.callee.getName()
        << "' does not match the flattened memref input or output-parameter "
           "ABI";
    return failure();
  }

  task.usesOutputParameters = hasOutputParameters;

  Type returnType = functionType.getReturnType();
  if (task.outputTypes.empty()) {
    if (!isa<LLVM::LLVMVoidType>(returnType)) {
      task.op.emitError("expected a void LLVM callee for a task without "
                        "tensor outputs");
      return failure();
    }
    return success();
  }

  if (task.usesOutputParameters)
    return success();

  if (isa<LLVM::LLVMVoidType>(returnType)) {
    task.op.emitError("expected an LLVM descriptor result for a task with "
                      "tensor outputs");
    return failure();
  }

  auto aggregateType = dyn_cast<LLVM::LLVMStructType>(returnType);
  for (auto indexedOutput : llvm::enumerate(task.outputTypes)) {
    unsigned resultIndex = task.resultIndices[indexedOutput.index()];
    Type descriptorType =
        getMemRefDescriptorType(task.op.getContext(), indexedOutput.value());
    if (returnType == descriptorType) {
      if (resultIndex != 0) {
        task.op.emitError(
            "a scalar LLVM descriptor result must use result index zero");
        return failure();
      }
      continue;
    }

    if (!aggregateType || aggregateType.isOpaque() ||
        resultIndex >= aggregateType.getBody().size() ||
        aggregateType.getBody()[resultIndex] != descriptorType) {
      task.op.emitError("result index ")
          << resultIndex
          << " does not select the expected LLVM memref descriptor";
      return failure();
    }
  }
  return success();
}

LogicalResult collectTasks(ModuleOp module, TileModel &model) {
  DenseSet<uint32_t> globalIds;
  DenseSet<uint32_t> localIndices;
  DenseMap<Value, unsigned> taskIndexByValue;

  for (Operation &op : model.taskGraphFunc.getBody().front()) {
    auto task = dyn_cast<TaskCreateOp>(&op);
    if (!task)
      continue;

    auto globalId = getRequiredUnsignedAttr<uint32_t>(
        task, deployment_attrs::kGlobalTaskIdAttrName);
    auto localIndex = getRequiredUnsignedAttr<uint32_t>(
        task, runtime_attrs::kTaskIndexAttrName);
    auto coreId = getRequiredUnsignedAttr<uint32_t>(
        task, runtime_attrs::kTaskCoreIdAttrName);
    auto localArrayId = getOptionalUnsignedAttr<uint32_t>(
        task, runtime_attrs::kTaskLocalArrayIdAttrName);
    auto physicalArrayId = getOptionalUnsignedAttr<uint32_t>(
        task, runtime_attrs::kTaskPhysicalArrayIdAttrName);
    auto inputSlots =
        getRequiredU32ArrayAttr(task, runtime_attrs::kTaskInputSlotsAttrName);
    auto outputSlots =
        getRequiredU32ArrayAttr(task, runtime_attrs::kTaskOutputSlotsAttrName);
    if (failed(globalId) || failed(localIndex) || failed(coreId) ||
        failed(localArrayId) || failed(physicalArrayId) || failed(inputSlots) ||
        failed(outputSlots))
      return failure();

    if (*coreId != model.coreId) {
      task.emitError("task core ID does not match isolated module core ID ")
          << model.coreId;
      return failure();
    }
    if (!globalIds.insert(*globalId).second) {
      task.emitError("duplicate sculptor.deployment.global_task_id ")
          << *globalId;
      return failure();
    }
    if (!localIndices.insert(*localIndex).second) {
      task.emitError("duplicate core-local sculptor.runtime.task_index ")
          << *localIndex;
      return failure();
    }

    LLVM::LLVMFuncOp callee =
        module.lookupSymbol<LLVM::LLVMFuncOp>(task.getCallee());
    if (!callee) {
      task.emitError("task callee '")
          << task.getCallee()
          << "' is missing or is not an llvm.func; run standard function "
             "lowering before sculptor-emit-golem-tile-abi";
      return failure();
    }

    if (inputSlots->size() != task.getInputs().size() ||
        outputSlots->size() != task.getOutputs().size()) {
      task.emitError("finalized task slot arrays must match task input and "
                     "output counts");
      return failure();
    }

    TaskModel taskModel;
    taskModel.op = task;
    taskModel.callee = callee;
    taskModel.globalId = *globalId;
    taskModel.localIndex = *localIndex;
    taskModel.coreId = *coreId;
    taskModel.localArrayId = *localArrayId;
    taskModel.physicalArrayId = *physicalArrayId;
    taskModel.isBoot = task_graph::isMatrixSetupTask(task);
    taskModel.inputSlots.assign(inputSlots->begin(), inputSlots->end());
    taskModel.outputSlots.assign(outputSlots->begin(), outputSlots->end());

    for (auto indexedInput : llvm::enumerate(task.getInputs())) {
      auto resourceIt = model.resourceIndexByValue.find(indexedInput.value());
      if (resourceIt == model.resourceIndexByValue.end()) {
        task.emitError("task input does not reference a finalized tensor "
                       "resource in this core");
        return failure();
      }
      const ResourceModel &resource = model.resources[resourceIt->second];
      uint32_t inputSlot = (*inputSlots)[indexedInput.index()];
      if (inputSlot >= model.resourceIndicesBySlot.size()) {
        task.emitError("task input references nonexistent resource slot ")
            << inputSlot;
        return failure();
      }
      if (inputSlot != resource.slot) {
        task.emitError("task input slot metadata does not match its resource");
        return failure();
      }
      taskModel.inputTypes.push_back(resource.shapedType);
    }

    for (auto indexedOutput : llvm::enumerate(task.getOutputs())) {
      auto resourceIt = model.resourceIndexByValue.find(indexedOutput.value());
      if (resourceIt == model.resourceIndexByValue.end()) {
        task.emitError("task output does not reference a finalized tensor "
                       "resource in this core");
        return failure();
      }
      const ResourceModel &resource = model.resources[resourceIt->second];
      uint32_t outputSlot = (*outputSlots)[indexedOutput.index()];
      if (outputSlot >= model.resourceIndicesBySlot.size()) {
        task.emitError("task output references nonexistent resource slot ")
            << outputSlot;
        return failure();
      }
      if (outputSlot != resource.slot) {
        task.emitError("task output slot metadata does not match its resource");
        return failure();
      }
      taskModel.outputTypes.push_back(resource.shapedType);
    }

    auto resultIndices = getResultIndices(task, taskModel.outputTypes.size());
    if (failed(resultIndices))
      return failure();
    taskModel.resultIndices = std::move(*resultIndices);

    if (taskModel.isBoot) {
      if (!taskModel.inputTypes.empty() || !taskModel.outputTypes.empty()) {
        task.emitError("matrix setup task must have zero tensor inputs and "
                       "zero tensor outputs");
        return failure();
      }
      if (!taskModel.localArrayId || !taskModel.physicalArrayId) {
        task.emitError("matrix setup task requires local and physical array "
                       "placement attributes");
        return failure();
      }
    }

    if (failed(validateCalleeSignature(taskModel)))
      return failure();

    taskIndexByValue.try_emplace(task.getResult(), model.tasks.size());
    model.tasks.push_back(std::move(taskModel));
  }

  for (auto indexedTask : llvm::enumerate(model.tasks)) {
    TaskModel &task = indexedTask.value();
    if (task.isBoot)
      model.bootTaskIndices.push_back(indexedTask.index());
    else
      model.dispatchTaskIndices.push_back(indexedTask.index());

    for (Value dependency : task.op.getDependencies()) {
      auto dependencyIt = taskIndexByValue.find(dependency);
      if (dependencyIt == taskIndexByValue.end()) {
        task.op.emitError(
            "task dependency does not reference a local task in this core");
        return failure();
      }
      const TaskModel &producer = model.tasks[dependencyIt->second];
      if (task.isBoot && !producer.isBoot) {
        task.op.emitError("boot task with global task ID ")
            << task.globalId << " depends on dispatch task "
            << producer.globalId;
        return failure();
      }
      if (!task.isBoot && !producer.isBoot)
        task.dispatchDependencyIds.push_back(producer.globalId);
    }
  }

  llvm::sort(model.bootTaskIndices, [&](unsigned lhs, unsigned rhs) {
    return model.tasks[lhs].localIndex < model.tasks[rhs].localIndex;
  });
  llvm::sort(model.dispatchTaskIndices, [&](unsigned lhs, unsigned rhs) {
    return model.tasks[lhs].globalId < model.tasks[rhs].globalId;
  });
  return success();
}

LogicalResult collectWorkspaceMetadata(TileModel &model) {
  auto workspaceSize = getRequiredUnsignedAttr<uint64_t>(
      model.taskGraphFunc, runtime_attrs::kTaskGraphWorkspaceSizeAttrName);
  if (failed(workspaceSize))
    return failure();
  model.workspaceSize = *workspaceSize;

  for (const ResourceModel &resource : model.resources) {
    bool usesWorkspace = resource.kind == ResourceKind::Intermediate ||
                         resource.kind == ResourceKind::RouteInput ||
                         resource.kind == ResourceKind::RouteOutput;
    if (!usesWorkspace)
      continue;

    if (!resource.workspaceOffset) {
      resource.op->emitError(
          "workspace resource requires sculptor.runtime.temp_offset");
      return failure();
    }
    if (*resource.workspaceOffset > model.workspaceSize ||
        resource.byteSize > model.workspaceSize - *resource.workspaceOffset) {
      resource.op->emitError("workspace resource range exceeds "
                             "sculptor.runtime.workspace_size");
      return failure();
    }
  }
  return success();
}

LogicalResult appendTaskBindingRange(TileModel &model, TaskCreateOp task,
                                     ArrayRef<uint32_t> values,
                                     uint32_t &offset, uint32_t &count) {
  uint64_t currentSize = model.taskBindingData.size();
  uint64_t valueCount = values.size();
  constexpr uint64_t maxCount = std::numeric_limits<uint32_t>::max();
  if (currentSize > maxCount || valueCount > maxCount ||
      valueCount > maxCount - currentSize) {
    task.emitError("task binding offsets or counts exceed the Golem tile ABI");
    return failure();
  }

  offset = static_cast<uint32_t>(currentSize);
  count = static_cast<uint32_t>(valueCount);
  model.taskBindingData.append(values.begin(), values.end());
  return success();
}

LogicalResult buildTaskBindings(TileModel &model) {
  if (model.dispatchTaskIndices.size() > std::numeric_limits<uint32_t>::max()) {
    model.taskGraphFunc.emitError(
        "dispatch task count exceeds the Golem tile ABI");
    return failure();
  }

  DenseSet<uint32_t> dispatchTaskIds;
  for (unsigned taskIndex : model.dispatchTaskIndices)
    dispatchTaskIds.insert(model.tasks[taskIndex].globalId);

  model.taskBindings.reserve(model.dispatchTaskIndices.size());
  for (unsigned taskIndex : model.dispatchTaskIndices) {
    TaskModel &task = model.tasks[taskIndex];
    DenseSet<uint32_t> dependencyIds;
    for (uint32_t dependencyId : task.dispatchDependencyIds) {
      if (!dispatchTaskIds.contains(dependencyId)) {
        task.op.emitError("task binding dependency ")
            << dependencyId << " does not reference a local dispatch task";
        return failure();
      }
      if (!dependencyIds.insert(dependencyId).second) {
        task.op.emitError("task binding contains duplicate dependency ")
            << dependencyId;
        return failure();
      }
    }

    TaskBindingModel binding;
    binding.taskId = task.globalId;
    if (failed(appendTaskBindingRange(model, task.op, task.inputSlots,
                                      binding.inputOffset,
                                      binding.inputCount)) ||
        failed(appendTaskBindingRange(model, task.op, task.outputSlots,
                                      binding.outputOffset,
                                      binding.outputCount)) ||
        failed(appendTaskBindingRange(
            model, task.op, task.dispatchDependencyIds,
            binding.dependencyOffset, binding.dependencyCount)))
      return failure();
    model.taskBindings.push_back(binding);
  }
  return success();
}

LogicalResult validateDeploymentPlan(TileModel &model) {
  for (const ResourceModel &resource : model.resources) {
    uint64_t offset = resource.dimensionOffset;
    uint64_t rank = static_cast<uint64_t>(resource.shapedType.getRank());
    if (offset > model.resourceDimensions.size() ||
        rank > model.resourceDimensions.size() - offset) {
      resource.op->emitError(
          "resource dimension range is outside the flattened dimension table");
      return failure();
    }
    ArrayRef<int64_t> dimensions =
        ArrayRef<int64_t>(model.resourceDimensions).slice(offset, rank);
    if (dimensions != resource.shapedType.getShape()) {
      resource.op->emitError(
          "resource dimension table does not match its static tensor type");
      return failure();
    }
  }

  if (model.taskBindings.size() != model.dispatchTaskIndices.size()) {
    model.taskGraphFunc.emitError(
        "task binding count does not match the dispatch task table");
    return failure();
  }
  DenseSet<uint32_t> resourceSlots;
  for (const ResourceModel &resource : model.resources)
    resourceSlots.insert(resource.slot);

  for (auto indexedBinding : llvm::enumerate(model.taskBindings)) {
    const TaskBindingModel &binding = indexedBinding.value();
    TaskModel &task =
        model.tasks[model.dispatchTaskIndices[indexedBinding.index()]];
    if (binding.taskId != task.globalId ||
        binding.inputCount != task.inputTypes.size() ||
        binding.outputCount != task.outputTypes.size()) {
      task.op.emitError(
          "task binding counts disagree with the existing Task record");
      return failure();
    }

    auto validateRange =
        [&](uint32_t offset, uint32_t count,
            StringRef description) -> FailureOr<ArrayRef<uint32_t>> {
      if (offset > model.taskBindingData.size() ||
          count > model.taskBindingData.size() - offset) {
        task.op.emitError("task binding ")
            << description
            << " range is outside the flattened binding data table";
        return failure();
      }
      return ArrayRef<uint32_t>(model.taskBindingData).slice(offset, count);
    };

    auto inputs =
        validateRange(binding.inputOffset, binding.inputCount, "input");
    auto outputs =
        validateRange(binding.outputOffset, binding.outputCount, "output");
    auto dependencies = validateRange(binding.dependencyOffset,
                                      binding.dependencyCount, "dependency");
    if (failed(inputs) || failed(outputs) || failed(dependencies))
      return failure();
    for (uint32_t slot : llvm::concat<const uint32_t>(*inputs, *outputs)) {
      if (!resourceSlots.contains(slot)) {
        task.op.emitError("task binding references nonexistent resource slot ")
            << slot;
        return failure();
      }
    }
  }
  return success();
}

FailureOr<ArrayAttr> getRequiredArrayAttr(ModuleOp module, StringRef name) {
  auto attr = module->getAttrOfType<ArrayAttr>(name);
  if (!attr) {
    module.emitError("expected isolated core module attribute '")
        << name << "'";
    return failure();
  }
  return attr;
}

FailureOr<ResourceModel *>
findNonRouteResourceByGlobalId(TileModel &model, uint32_t globalId,
                               ResourceKind expectedKind,
                               Operation *diagnosticOp) {
  auto resourceIt = model.nonRouteResourceIndexByGlobalId.find(globalId);
  if (resourceIt == model.nonRouteResourceIndexByGlobalId.end()) {
    diagnosticOp->emitError("cannot match deployment metadata to local "
                            "non-route resource with global ID ")
        << globalId;
    return failure();
  }

  ResourceModel &resource = model.resources[resourceIt->second];
  if (resource.kind != expectedKind) {
    diagnosticOp->emitError("global resource ")
        << globalId << " has the wrong boundary kind";
    return failure();
  }
  return &resource;
}

FailureOr<ResourceModel *> findRouteResourceById(TileModel &model,
                                                 uint32_t routeId,
                                                 ResourceKind expectedKind,
                                                 Operation *diagnosticOp) {
  const DenseMap<uint32_t, unsigned> *resourcesByRouteId = nullptr;
  if (expectedKind == ResourceKind::RouteInput) {
    resourcesByRouteId = &model.routeInputResourceIndexByRouteId;
  } else if (expectedKind == ResourceKind::RouteOutput) {
    resourcesByRouteId = &model.routeOutputResourceIndexByRouteId;
  } else {
    diagnosticOp->emitError(
        "internal error: route lookup requested for a non-route resource");
    return failure();
  }

  auto resourceIt = resourcesByRouteId->find(routeId);
  if (resourceIt == resourcesByRouteId->end()) {
    diagnosticOp->emitError("cannot match deployment route ")
        << routeId << " to a local route boundary resource";
    return failure();
  }
  return &model.resources[resourceIt->second];
}

LogicalResult collectRoutes(ModuleOp module, TileModel &model,
                            StringRef attrName, ResourceKind resourceKind,
                            bool incoming,
                            SmallVectorImpl<RouteModel> &routes) {
  auto manifest = getRequiredArrayAttr(module, attrName);
  if (failed(manifest))
    return failure();

  DenseSet<uint32_t> routeIds;
  for (Attribute entry : *manifest) {
    auto route = dyn_cast<DeploymentRouteAttr>(entry);
    if (!route) {
      module.emitError("expected '")
          << attrName << "' to contain #sculptor.deployment_route attributes";
      return failure();
    }

    int64_t routeIdValue = route.getId().getInt();
    int64_t sourceCoreValue = route.getSourceCore().getInt();
    int64_t sourceTaskValue = route.getSourceTask().getInt();
    int64_t sourceOutputValue = route.getSourceOutput().getInt();
    int64_t destinationCoreValue = route.getDestinationCore().getInt();
    int64_t destinationTaskValue = route.getDestinationTask().getInt();
    int64_t destinationInputValue = route.getDestinationInput().getInt();
    int64_t routeCore = incoming ? route.getDestinationCore().getInt()
                                 : route.getSourceCore().getInt();
    int64_t resourceIdValue = route.getResourceId().getInt();
    int64_t byteSizeValue = route.getByteSize().getInt();
    auto isU32 = [](int64_t value) {
      return value >= 0 && static_cast<uint64_t>(value) <=
                               std::numeric_limits<uint32_t>::max();
    };
    if (!isU32(routeIdValue) || !isU32(sourceCoreValue) ||
        !isU32(sourceTaskValue) || !isU32(sourceOutputValue) ||
        !isU32(destinationCoreValue) || !isU32(destinationTaskValue) ||
        !isU32(destinationInputValue) || routeCore != model.coreId ||
        !isU32(resourceIdValue) || byteSizeValue < 0) {
      module.emitError("deployment route contains values incompatible with "
                       "this isolated Golem core");
      return failure();
    }

    uint32_t routeId = static_cast<uint32_t>(routeIdValue);
    if (!routeIds.insert(routeId).second) {
      module.emitError("duplicate route ID ")
          << routeId << " in '" << attrName << "'";
      return failure();
    }

    auto resource = findRouteResourceById(model, routeId, resourceKind, module);
    if (failed(resource))
      return failure();
    if ((*resource)->globalId != static_cast<uint32_t>(resourceIdValue) ||
        (*resource)->byteSize != static_cast<uint64_t>(byteSizeValue)) {
      module.emitError("route metadata does not match local route resource "
                       "slot metadata for route ")
          << routeId;
      return failure();
    }
    routes.push_back(RouteModel{route, (*resource)->slot});
  }

  unsigned localRouteCount =
      llvm::count_if(model.resources, [&](const ResourceModel &resource) {
        return resource.kind == resourceKind;
      });
  if (localRouteCount != routes.size()) {
    module.emitError("route manifest does not account for every local route "
                     "resource in the isolated core");
    return failure();
  }

  llvm::sort(routes, [](const RouteModel &lhs, const RouteModel &rhs) {
    return lhs.route.getId().getInt() < rhs.route.getId().getInt();
  });
  return success();
}

LogicalResult collectModelIO(ModuleOp module, TileModel &model,
                             StringRef attrName, StringRef indexName,
                             ResourceKind resourceKind,
                             SmallVectorImpl<ModelIOModel> &entries) {
  auto manifest = getRequiredArrayAttr(module, attrName);
  if (failed(manifest))
    return failure();

  DenseSet<uint32_t> modelIndices;
  for (Attribute entry : *manifest) {
    auto ownership = dyn_cast<DictionaryAttr>(entry);
    auto index =
        ownership ? ownership.getAs<IntegerAttr>(indexName) : IntegerAttr();
    auto owner =
        ownership ? ownership.getAs<IntegerAttr>("owner_core") : IntegerAttr();
    auto resourceId = ownership
                          ? ownership.getAs<IntegerAttr>("global_resource_id")
                          : IntegerAttr();
    if (!index || !owner || !resourceId || index.getInt() < 0 ||
        owner.getInt() != model.coreId || resourceId.getInt() < 0 ||
        static_cast<uint64_t>(index.getInt()) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(resourceId.getInt()) >
            std::numeric_limits<uint32_t>::max()) {
      module.emitError("invalid model ownership record in '")
          << attrName << "'";
      return failure();
    }

    uint32_t modelIndex = static_cast<uint32_t>(index.getInt());
    if (!modelIndices.insert(modelIndex).second) {
      module.emitError("duplicate model ownership index ")
          << modelIndex << " in '" << attrName << "'";
      return failure();
    }

    auto resource = findNonRouteResourceByGlobalId(
        model, static_cast<uint32_t>(resourceId.getInt()), resourceKind,
        module);
    if (failed(resource))
      return failure();
    entries.push_back(ModelIOModel{modelIndex, model.coreId,
                                   (*resource)->globalId, (*resource)->slot,
                                   (*resource)->byteSize});
  }

  unsigned localResourceCount =
      llvm::count_if(model.resources, [&](const ResourceModel &resource) {
        return resource.kind == resourceKind;
      });
  if (localResourceCount != entries.size()) {
    module.emitError("model ownership manifest does not account for every "
                     "local model boundary resource");
    return failure();
  }

  llvm::sort(entries, [](const ModelIOModel &lhs, const ModelIOModel &rhs) {
    return lhs.modelIndex < rhs.modelIndex;
  });
  return success();
}

} // namespace

LLVM::LLVMStructType getTensorType(MLIRContext *context) {
  return LLVM::LLVMStructType::getLiteral(
      context, {IntegerType::get(context, 32), IntegerType::get(context, 64),
                LLVM::LLVMPointerType::get(context)});
}

LLVM::LLVMStructType getTaskType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  return LLVM::LLVMStructType::getLiteral(
      context,
      {i32Type, LLVM::LLVMPointerType::get(context), i32Type, i32Type});
}

LLVM::LLVMStructType getResourceType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type,
                i32Type, i64Type, i64Type});
}

LLVM::LLVMStructType getTaskBindingType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  return LLVM::LLVMStructType::getLiteral(
      context,
      {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type});
}

LLVM::LLVMStructType getRouteType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type,
                i32Type, i32Type, i64Type});
}

LLVM::LLVMStructType getModelIOType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i64Type});
}

LLVM::LLVMStructType getMemRefDescriptorType(MLIRContext *context,
                                             ShapedType shapedType) {
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i64Type = IntegerType::get(context, 64);
  auto indexArrayType = LLVM::LLVMArrayType::get(i64Type, shapedType.getRank());
  return LLVM::LLVMStructType::getLiteral(
      context, {ptrType, ptrType, i64Type, indexArrayType, indexArrayType});
}

FailureOr<TileModel> collectTileModel(ModuleOp module) {
  if (!module.getOps<ModuleOp>().empty()) {
    module.emitError("sculptor-emit-golem-tile-abi requires one extracted "
                     "core and cannot package nested core modules");
    return failure();
  }

  auto coreId = getRequiredUnsignedAttr<uint32_t>(
      module, runtime_attrs::kTaskCoreIdAttrName);
  auto taskGraphFunc = findTaskGraphFunction(module);
  if (failed(coreId) || failed(taskGraphFunc))
    return failure();

  TileModel model;
  model.coreId = *coreId;
  model.taskGraphFunc = *taskGraphFunc;

  if (failed(collectResources(model)) || failed(collectTasks(module, model)) ||
      failed(collectRoutes(
          module, model, deployment_attrs::kIncomingRoutesAttrName,
          ResourceKind::RouteInput, true, model.incomingRoutes)) ||
      failed(collectRoutes(
          module, model, deployment_attrs::kOutgoingRoutesAttrName,
          ResourceKind::RouteOutput, false, model.outgoingRoutes)) ||
      failed(collectModelIO(
          module, model, deployment_attrs::kModelInputsAttrName, "input_index",
          ResourceKind::ModelInput, model.modelInputs)) ||
      failed(collectModelIO(
          module, model, deployment_attrs::kModelOutputsAttrName,
          "output_index", ResourceKind::ModelOutput, model.modelOutputs)) ||
      failed(collectWorkspaceMetadata(model)) ||
      failed(buildTaskBindings(model)) || failed(validateDeploymentPlan(model)))
    return failure();

  return model;
}

} // namespace golem_tile_abi
} // namespace sculptor
} // namespace mlir
