#include "GolemTileABI.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileScratchpadAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeResourceUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeTaskKinds.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

namespace mlir {
namespace sculptor {
namespace golem_tile_abi {

namespace scratchpad_attrs = mlir::sculptor::scratchpad_attrs;
namespace tile_memory = mlir::sculptor::tile_memory;

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
  if (!shapedType || !shapedType.hasStaticShape()) {
    op->emitError(
        "Golem tile task ABI supports only statically shaped tensors");
    return failure();
  }
  Type elementType = shapedType.getElementType();
  auto integerType = dyn_cast<IntegerType>(elementType);
  bool supportedInteger =
      integerType && llvm::is_contained({8u, 16u, 32u, 64u},
                                       integerType.getWidth());
  if (!elementType.isF32() && !supportedInteger) {
    op->emitError("Golem tile task ABI supports f32 and byte-addressable "
                  "i8/i16/i32/i64 tensor elements");
    return failure();
  }
  if (!hasRepresentableContiguousStrides(shapedType)) {
    op->emitError(
        "Golem tile task ABI tensor strides must fit in a signed 64-bit value");
    return failure();
  }
  return shapedType;
}

FailureOr<uint32_t> getElementTypeCode(Operation *op, Type elementType) {
  if (elementType.isF32())
    return kFloat32ElementType;
  auto integerType = dyn_cast<IntegerType>(elementType);
  if (!integerType)
    return op->emitError("unsupported Golem tile ABI element type ")
               << elementType,
           failure();
  bool isUnsigned = integerType.isUnsigned();
  switch (integerType.getWidth()) {
  case 8:
    return isUnsigned ? kUInt8ElementType : kInt8ElementType;
  case 16:
    return isUnsigned ? kUInt16ElementType : kInt16ElementType;
  case 32:
    return isUnsigned ? kUInt32ElementType : kInt32ElementType;
  case 64:
    return isUnsigned ? kUInt64ElementType : kInt64ElementType;
  default:
    return op->emitError("unsupported Golem tile ABI integer width ")
               << integerType.getWidth(),
           failure();
  }
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
  // transfers of those payloads. Incoming transfers have distinct receive
  // buffers. Outgoing fan-out transfers of one payload share one send buffer.
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
        &op, tile_runtime_attrs::kResourceSlotAttrName);
    auto byteSize = getRequiredUnsignedAttr<uint64_t>(
        &op, tile_runtime_attrs::kResourceByteSizeAttrName);
    auto elementType = succeeded(shapedType)
                           ? getElementTypeCode(&op,
                                                (*shapedType).getElementType())
                           : FailureOr<uint32_t>(failure());
    if (failed(shapedType) || failed(elementType) || failed(globalId) || failed(slot) ||
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
    auto storageClass =
        op.getAttrOfType<StringAttr>(scratchpad_attrs::kStorageClassAttrName);
    bool scratchpad =
        storageClass &&
        storageClass.getValue() == scratchpad_attrs::kScratchpadStorageClass;
    if (scratchpad) {
      auto offset = getRequiredUnsignedAttr<uint64_t>(
          &op, scratchpad_attrs::kScratchpadOffsetAttrName);
      if (failed(offset))
        return failure();
      workspaceOffset = *offset;
    } else if (kindAndValue->first == ResourceKind::Intermediate ||
               kindAndValue->first == ResourceKind::RouteInput ||
               kindAndValue->first == ResourceKind::RouteOutput) {
      auto offset = getRequiredUnsignedAttr<uint64_t>(
          &op, tile_runtime_attrs::kResourceTempOffsetAttrName);
      if (failed(offset))
        return failure();
      workspaceOffset = *offset;
    }

    unsigned resourceIndex = model.resources.size();
    model.resourceIndexByValue.try_emplace(kindAndValue->second, resourceIndex);
    model.resources.push_back(ResourceModel{
        &op, kindAndValue->second, *shapedType, *elementType,
        kindAndValue->first, *globalId, routeId, *slot, 0, *byteSize,
        workspaceOffset, scratchpad});
    if (kindAndValue->first == ResourceKind::RouteInput) {
      model.routeInputResourceIndexByRouteId.try_emplace(*routeId,
                                                         resourceIndex);
    } else if (kindAndValue->first == ResourceKind::RouteOutput) {
      if (!model.routeOutputResourceIndexByGlobalId
               .try_emplace(*globalId, resourceIndex)
               .second) {
        op.emitError("duplicate outgoing route resource with global ID ")
            << *globalId;
        return failure();
      }
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
  auto attr = task->getAttrOfType<ArrayAttr>(
      tile_runtime_attrs::kTaskResultIndicesAttrName);
  if (!attr) {
    result.reserve(outputCount);
    for (unsigned index = 0; index < outputCount; ++index)
      result.push_back(index);
    return result;
  }

  if (attr.size() != outputCount) {
    task.emitError("expected '")
        << tile_runtime_attrs::kTaskResultIndicesAttrName
        << "' to match the number of task outputs";
    return failure();
  }

  for (Attribute entry : attr) {
    auto integer = dyn_cast<IntegerAttr>(entry);
    if (!integer || integer.getInt() < 0 ||
        static_cast<uint64_t>(integer.getInt()) >
            std::numeric_limits<unsigned>::max()) {
      task.emitError("expected '")
          << tile_runtime_attrs::kTaskResultIndicesAttrName
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
        task, tile_runtime_attrs::kTaskIndexAttrName);
    auto coreId = getRequiredUnsignedAttr<uint32_t>(
        task, tile_runtime_attrs::kTaskCoreIdAttrName);
    auto localArrayId = getOptionalUnsignedAttr<uint32_t>(
        task, tile_runtime_attrs::kTaskLocalArrayIdAttrName);
    auto physicalArrayId = getOptionalUnsignedAttr<uint32_t>(
        task, tile_runtime_attrs::kTaskPhysicalArrayIdAttrName);
    auto inputSlots = getRequiredU32ArrayAttr(
        task, tile_runtime_attrs::kTaskInputSlotsAttrName);
    auto outputSlots = getRequiredU32ArrayAttr(
        task, tile_runtime_attrs::kTaskOutputSlotsAttrName);
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
    taskModel.isBoot = tile_runtime::isMatrixSetupTask(task);
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
      model.taskGraphFunc, tile_runtime_attrs::kTaskGraphWorkspaceSizeAttrName);
  if (failed(workspaceSize))
    return failure();
  model.workspaceSize = *workspaceSize;

  for (const ResourceModel &resource : model.resources) {
    if (resource.scratchpad)
      continue;
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

LogicalResult collectScratchpadMetadata(TileModel &model) {
  auto requiredBytes = model.taskGraphFunc->getAttrOfType<IntegerAttr>(
      scratchpad_attrs::kScratchpadRequiredBytesAttrName);
  auto descriptors = model.taskGraphFunc->getAttrOfType<ArrayAttr>(
      scratchpad_attrs::kScratchpadDMADescriptorsAttrName);
  auto abiVersion = model.taskGraphFunc->getAttrOfType<IntegerAttr>(
      scratchpad_attrs::kScratchpadABIVersionAttrName);
  auto featureBits = model.taskGraphFunc->getAttrOfType<IntegerAttr>(
      scratchpad_attrs::kScratchpadFeatureBitsAttrName);
  if (!requiredBytes && !descriptors)
    return success();
  if (!requiredBytes || requiredBytes.getInt() < 0 || !descriptors ||
      !abiVersion || abiVersion.getInt() != 2 || !featureBits ||
      featureBits.getInt() != kScratchpadDMAFeature) {
    model.taskGraphFunc.emitError(
        "incomplete scratchpad plan metadata on task graph");
    return failure();
  }
  model.scratchpadRequiredBytes = static_cast<uint64_t>(requiredBytes.getInt());
  model.abiFeatures |= kScratchpadDMAFeature;
  for (const ResourceModel &resource : model.resources) {
    if (!resource.scratchpad)
      continue;
    if (!resource.workspaceOffset ||
        *resource.workspaceOffset > model.scratchpadRequiredBytes ||
        resource.byteSize >
            model.scratchpadRequiredBytes - *resource.workspaceOffset) {
      resource.op->emitError(
          "scratchpad resource range exceeds required scratchpad bytes");
      return failure();
    }
  }

  auto getU32 = [&](DictionaryAttr descriptor,
                    StringRef name) -> FailureOr<uint32_t> {
    auto value = descriptor.getAs<IntegerAttr>(name);
    if (!value || value.getInt() < 0 ||
        static_cast<uint64_t>(value.getInt()) > UINT32_MAX) {
      model.taskGraphFunc.emitError("invalid scratchpad DMA field '")
          << name << "'";
      return failure();
    }
    return static_cast<uint32_t>(value.getInt());
  };
  auto getU64 = [&](DictionaryAttr descriptor,
                    StringRef name) -> FailureOr<uint64_t> {
    auto value = descriptor.getAs<IntegerAttr>(name);
    if (!value || value.getInt() < 0) {
      model.taskGraphFunc.emitError("invalid scratchpad DMA field '")
          << name << "'";
      return failure();
    }
    return static_cast<uint64_t>(value.getInt());
  };

  DenseSet<uint32_t> descriptorIds;
  DenseSet<uint32_t> completionTokens;
  for (Attribute entry : descriptors) {
    auto descriptor = dyn_cast<DictionaryAttr>(entry);
    if (!descriptor) {
      model.taskGraphFunc.emitError(
          "scratchpad DMA descriptors must be dictionaries");
      return failure();
    }
    auto descriptorId = getU32(descriptor, scratchpad_attrs::kDMAIdFieldName);
    auto direction =
        getU32(descriptor, scratchpad_attrs::kDMADirectionFieldName);
    auto globalResourceId =
        getU32(descriptor, scratchpad_attrs::kDMAGlobalResourceIdFieldName);
    auto routeId = getU32(descriptor, scratchpad_attrs::kDMARouteIdFieldName);
    auto scratchpadOffset =
        getU64(descriptor, scratchpad_attrs::kDMAScratchpadOffsetFieldName);
    auto byteSize = getU64(descriptor, scratchpad_attrs::kDMAByteSizeFieldName);
    auto completionToken =
        getU32(descriptor, scratchpad_attrs::kDMACompletionTokenFieldName);
    auto triggerKind =
        getU32(descriptor, scratchpad_attrs::kDMATriggerKindFieldName);
    auto triggerId =
        getU32(descriptor, scratchpad_attrs::kDMATriggerIdFieldName);
    auto flags = getU32(descriptor, scratchpad_attrs::kDMAFlagsFieldName);
    auto sourceStorage =
        getU32(descriptor, scratchpad_attrs::kDMASourceStorageFieldName);
    auto destinationStorage =
        getU32(descriptor, scratchpad_attrs::kDMADestinationStorageFieldName);
    auto reserved = getU64(descriptor, scratchpad_attrs::kDMAReservedFieldName);
    if (failed(descriptorId) || failed(direction) || failed(globalResourceId) ||
        failed(routeId) || failed(scratchpadOffset) || failed(byteSize) ||
        failed(completionToken) || failed(triggerKind) || failed(triggerId) ||
        failed(flags) || failed(sourceStorage) || failed(destinationStorage) ||
        failed(reserved))
      return failure();
    if (!descriptorIds.insert(*descriptorId).second ||
        !completionTokens.insert(*completionToken).second) {
      model.taskGraphFunc.emitError(
          "scratchpad DMA descriptor and completion token IDs must be unique");
      return failure();
    }
    if (*direction > scratchpad_attrs::kDirectionScratchpadToNic ||
        *sourceStorage > scratchpad_attrs::kStorageNic ||
        *destinationStorage > scratchpad_attrs::kStorageNic ||
        *triggerKind > scratchpad_attrs::kTriggerTaskComplete ||
        *flags != scratchpad_attrs::kDMAAsynchronous || *reserved != 0) {
      model.taskGraphFunc.emitError(
          "scratchpad DMA descriptor violates Platform v0.2");
      return failure();
    }

    ResourceModel *resource = nullptr;
    for (ResourceModel &candidate : model.resources) {
      if (candidate.globalId != *globalResourceId)
        continue;
      if (*routeId != UINT32_MAX &&
          candidate.routeId != std::optional<uint32_t>(*routeId))
        continue;
      resource = &candidate;
      break;
    }
    if (!resource || !resource->scratchpad || !resource->workspaceOffset ||
        *resource->workspaceOffset != *scratchpadOffset ||
        resource->byteSize != *byteSize) {
      model.taskGraphFunc.emitError(
          "scratchpad DMA descriptor does not match a planned resource");
      return failure();
    }
    if (*scratchpadOffset > model.scratchpadRequiredBytes ||
        *byteSize > model.scratchpadRequiredBytes - *scratchpadOffset) {
      resource->op->emitError(
          "scratchpad resource range exceeds required scratchpad bytes");
      return failure();
    }
    model.dmaDescriptors.push_back(DMADescriptorModel{
        *descriptorId, *direction, resource->slot, *routeId, *scratchpadOffset,
        *byteSize, *completionToken, *triggerKind, *triggerId, *flags,
        *sourceStorage, *destinationStorage, *reserved});
  }
  llvm::sort(model.dmaDescriptors,
             [](const DMADescriptorModel &lhs, const DMADescriptorModel &rhs) {
               return lhs.descriptorId < rhs.descriptorId;
             });
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

LogicalResult buildStaticRuntime(TileModel &model) {
  const size_t taskCount = model.taskBindings.size();
  const size_t resourceCount = model.resources.size();
  if (taskCount > UINT32_MAX || resourceCount > UINT32_MAX)
    return model.taskGraphFunc.emitError(
        "static runtime table count exceeds the Golem tile ABI");

  DenseMap<uint32_t, uint32_t> taskIndexById;
  model.taskIdIndex.reserve(taskCount);
  for (auto indexed : llvm::enumerate(model.taskBindings)) {
    const uint32_t taskIndex = static_cast<uint32_t>(indexed.index());
    const uint32_t taskId = indexed.value().taskId;
    if (!taskIndexById.try_emplace(taskId, taskIndex).second)
      return model.taskGraphFunc.emitError(
          "static runtime found a duplicate task ID");
    model.taskIdIndex.push_back({taskId, taskIndex});
  }
  llvm::sort(model.taskIdIndex,
             [](const IdIndexModel &left, const IdIndexModel &right) {
               return left.id < right.id;
             });

  for (const RouteModel &route : model.outgoingRoutes) {
    const uint32_t sourceTask =
        static_cast<uint32_t>(route.route.getSourceTask().getInt());
    if (!taskIndexById.count(sourceTask))
      return model.taskGraphFunc.emitError(
          "outgoing route source is not a local dispatch task");
  }
  // Group routes by local source task. One task can then transmit from one
  // contiguous range without searching the complete outgoing route table.
  llvm::sort(model.outgoingRoutes,
             [&](const RouteModel &left, const RouteModel &right) {
               const uint32_t leftId = static_cast<uint32_t>(
                   left.route.getSourceTask().getInt());
               const uint32_t rightId = static_cast<uint32_t>(
                   right.route.getSourceTask().getInt());
               const uint32_t leftTask = taskIndexById.lookup(leftId);
               const uint32_t rightTask = taskIndexById.lookup(rightId);
               return std::make_tuple(leftTask,
                                      left.route.getId().getInt()) <
                      std::make_tuple(rightTask,
                                      right.route.getId().getInt());
             });

  model.staticTasks.resize(taskCount);
  model.staticResources.resize(resourceCount);
  SmallVector<SmallVector<uint32_t>> readyConsumers(resourceCount);
  SmallVector<SmallVector<uint32_t>> boundConsumers(resourceCount);
  SmallVector<SmallVector<uint32_t>> dependents(taskCount);

  for (auto indexed : llvm::enumerate(model.taskBindings)) {
    const uint32_t taskIndex = static_cast<uint32_t>(indexed.index());
    const TaskBindingModel &binding = indexed.value();
    const uint64_t readiness = static_cast<uint64_t>(binding.inputCount) +
                               binding.outputCount + binding.dependencyCount;
    if (readiness > UINT32_MAX)
      return model.taskGraphFunc.emitError(
          "static task readiness count exceeds the Golem tile ABI");
    model.staticTasks[taskIndex].initialReadiness =
        static_cast<uint32_t>(readiness);

    ArrayRef<uint32_t> inputs(model.taskBindingData.data() +
                                  binding.inputOffset,
                              binding.inputCount);
    ArrayRef<uint32_t> outputs(model.taskBindingData.data() +
                                   binding.outputOffset,
                               binding.outputCount);
    ArrayRef<uint32_t> dependencies(model.taskBindingData.data() +
                                        binding.dependencyOffset,
                                    binding.dependencyCount);
    for (uint32_t slot : inputs)
      readyConsumers[slot].push_back(taskIndex);
    for (uint32_t slot : outputs)
      boundConsumers[slot].push_back(taskIndex);
    for (uint32_t dependencyId : dependencies) {
      auto dependency = taskIndexById.find(dependencyId);
      if (dependency == taskIndexById.end())
        return model.taskGraphFunc.emitError(
            "static runtime dependency is not a local task");
      dependents[dependency->second].push_back(taskIndex);
    }
  }

  uint32_t routeIndex = 0;
  for (uint32_t taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
    StaticTaskModel &task = model.staticTasks[taskIndex];
    task.outgoingRouteOffset = routeIndex;
    while (routeIndex < model.outgoingRoutes.size()) {
      const uint32_t sourceId = static_cast<uint32_t>(
          model.outgoingRoutes[routeIndex].route.getSourceTask().getInt());
      if (taskIndexById.lookup(sourceId) != taskIndex)
        break;
      ++routeIndex;
    }
    task.outgoingRouteCount = routeIndex - task.outgoingRouteOffset;

    llvm::sort(dependents[taskIndex]);
    task.dependentOffset =
        static_cast<uint32_t>(model.staticRuntimeData.size());
    task.dependentCount = static_cast<uint32_t>(dependents[taskIndex].size());
    model.staticRuntimeData.append(dependents[taskIndex]);
  }
  if (routeIndex != model.outgoingRoutes.size())
    return model.taskGraphFunc.emitError(
        "outgoing route table is not grouped by local task");

  for (uint32_t slot = 0; slot < resourceCount; ++slot) {
    StaticResourceModel &resource = model.staticResources[slot];
    llvm::sort(readyConsumers[slot]);
    llvm::sort(boundConsumers[slot]);
    resource.readyConsumerOffset =
        static_cast<uint32_t>(model.staticRuntimeData.size());
    resource.readyConsumerCount =
        static_cast<uint32_t>(readyConsumers[slot].size());
    model.staticRuntimeData.append(readyConsumers[slot]);
    resource.boundConsumerOffset =
        static_cast<uint32_t>(model.staticRuntimeData.size());
    resource.boundConsumerCount =
        static_cast<uint32_t>(boundConsumers[slot].size());
    model.staticRuntimeData.append(boundConsumers[slot]);
  }
  if (model.staticRuntimeData.size() > UINT32_MAX)
    return model.taskGraphFunc.emitError(
        "static runtime data exceeds the Golem tile ABI");

  auto buildRouteIndex = [](ArrayRef<RouteModel> routes,
                            SmallVectorImpl<IdIndexModel> &entries) {
    entries.reserve(routes.size());
    for (auto indexed : llvm::enumerate(routes))
      entries.push_back(
          {static_cast<uint32_t>(indexed.value().route.getId().getInt()),
           static_cast<uint32_t>(indexed.index())});
    llvm::sort(entries,
               [](const IdIndexModel &left, const IdIndexModel &right) {
                 return left.id < right.id;
               });
  };
  buildRouteIndex(model.incomingRoutes, model.incomingRouteIdIndex);
  buildRouteIndex(model.outgoingRoutes, model.outgoingRouteIdIndex);
  model.abiFeatures |= kStaticRuntimeFeature;
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

  DenseSet<uint32_t> ownerIds;
  for (const MemoryOwnerModel &owner : model.memoryOwners) {
    if (!ownerIds.insert(owner.id).second ||
        (owner.localSlot != kInvalidABIId &&
         !resourceSlots.contains(owner.localSlot)))
      return model.taskGraphFunc.emitError("invalid memory owner ABI record");
  }
  DenseSet<uint32_t> viewIds;
  for (const MemoryViewModel &view : model.memoryViews) {
    uint64_t geometryCount = static_cast<uint64_t>(view.rank) * 3;
    if (!viewIds.insert(view.id).second || !ownerIds.contains(view.ownerId) ||
        view.geometryOffset > model.memoryViewGeometry.size() ||
        geometryCount > model.memoryViewGeometry.size() - view.geometryOffset ||
        (view.ownerSlot != kInvalidABIId &&
         !resourceSlots.contains(view.ownerSlot)))
      return model.taskGraphFunc.emitError("invalid memory view ABI record");
  }
  DenseSet<uint32_t> routeViewIds;
  for (const RouteViewModel &view : model.routeViews) {
    if (!routeViewIds.insert(view.routeId).second ||
        !resourceSlots.contains(view.localResourceSlot) ||
        !resourceSlots.contains(view.ownerSlot) ||
        !viewIds.contains(view.viewId) || view.byteSize == 0)
      return model.taskGraphFunc.emitError("invalid route-view ABI record");
  }
  DenseSet<uint32_t> assemblyIds;
  for (const AssemblyModel &assembly : model.assemblies) {
    if (!assemblyIds.insert(assembly.id).second ||
        !resourceSlots.contains(assembly.ownerSlot) ||
        assembly.contributionOffset > model.assemblyContributions.size() ||
        assembly.contributionCount >
            model.assemblyContributions.size() - assembly.contributionOffset)
      return model.taskGraphFunc.emitError("invalid assembly ABI record");
  }
  for (const AssemblyContributionModel &contribution :
       model.assemblyContributions) {
    if (!assemblyIds.contains(contribution.assemblyId) ||
        !viewIds.contains(contribution.destinationViewId))
      return model.taskGraphFunc.emitError(
          "invalid assembly contribution ABI record");
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

FailureOr<ResourceModel *> findRouteResource(TileModel &model, uint32_t routeId,
                                             uint32_t globalResourceId,
                                             ResourceKind expectedKind,
                                             Operation *diagnosticOp) {
  if (expectedKind == ResourceKind::RouteInput) {
    auto resourceIt = model.routeInputResourceIndexByRouteId.find(routeId);
    if (resourceIt == model.routeInputResourceIndexByRouteId.end()) {
      diagnosticOp->emitError("cannot match incoming deployment route ")
          << routeId << " to a local route boundary resource";
      return failure();
    }
    return &model.resources[resourceIt->second];
  }
  if (expectedKind == ResourceKind::RouteOutput) {
    auto resourceIt =
        model.routeOutputResourceIndexByGlobalId.find(globalResourceId);
    if (resourceIt != model.routeOutputResourceIndexByGlobalId.end())
      return &model.resources[resourceIt->second];

    // A model output can also be consumed by a remote tile. In that case the
    // materialized graph intentionally routes from the external model-output
    // slot, avoiding a duplicate workspace allocation and result copy.
    auto boundaryIt = model.nonRouteResourceIndexByGlobalId.find(globalResourceId);
    if (boundaryIt != model.nonRouteResourceIndexByGlobalId.end()) {
      ResourceModel &resource = model.resources[boundaryIt->second];
      if (resource.kind == ResourceKind::ModelOutput)
        return &resource;
    }
    diagnosticOp->emitError("cannot match outgoing deployment route ")
        << routeId << " to a local route or model-output resource with global ID "
        << globalResourceId;
    return failure();
  }
  diagnosticOp->emitError(
      "internal error: route lookup requested for a non-route resource");
  return failure();
}

LogicalResult collectRoutes(ModuleOp module, TileModel &model,
                            StringRef attrName, ResourceKind resourceKind,
                            bool incoming,
                            SmallVectorImpl<RouteModel> &routes) {
  auto manifest = getRequiredArrayAttr(module, attrName);
  if (failed(manifest))
    return failure();

  DenseSet<uint32_t> routeIds;
  DenseSet<unsigned> matchedResourceIndices;
  std::map<uint32_t, std::tuple<uint32_t, uint32_t, uint64_t>>
      outgoingSourceByResource;
  std::map<uint32_t, uint32_t> minimumOutgoingRouteByResource;

  auto findTask = [&](uint32_t globalId) -> TaskModel * {
    for (TaskModel &task : model.tasks)
      if (task.globalId == globalId)
        return &task;
    return nullptr;
  };

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

    uint32_t resourceId = static_cast<uint32_t>(resourceIdValue);
    auto resource =
        findRouteResource(model, routeId, resourceId, resourceKind, module);
    if (failed(resource))
      return failure();
    if ((*resource)->globalId != resourceId ||
        (*resource)->byteSize != static_cast<uint64_t>(byteSizeValue)) {
      module.emitError("route metadata does not match local route resource "
                       "slot metadata for route ")
          << routeId;
      return failure();
    }

    unsigned resourceIndex =
        model.resourceIndexByValue.lookup((*resource)->value);
    matchedResourceIndices.insert(resourceIndex);

    if (incoming) {
      if ((*resource)->routeId != std::optional<uint32_t>(routeId)) {
        module.emitError("incoming route resource does not preserve route ID ")
            << routeId;
        return failure();
      }
      TaskModel *destination =
          findTask(static_cast<uint32_t>(destinationTaskValue));
      uint32_t destinationInput = static_cast<uint32_t>(destinationInputValue);
      if (!destination || destinationInput >= destination->inputSlots.size() ||
          destination->inputSlots[destinationInput] != (*resource)->slot) {
        module.emitError("incoming route ")
            << routeId << " does not match its destination task input binding";
        return failure();
      }
    } else {
      uint32_t sourceTask = static_cast<uint32_t>(sourceTaskValue);
      uint32_t sourceOutput = static_cast<uint32_t>(sourceOutputValue);
      auto sourceMetadata = std::make_tuple(
          sourceTask, sourceOutput, static_cast<uint64_t>(byteSizeValue));
      auto [sourceIt, inserted] =
          outgoingSourceByResource.emplace(resourceId, sourceMetadata);
      if (!inserted && sourceIt->second != sourceMetadata) {
        module.emitError("outgoing routes for global resource ")
            << resourceId
            << " do not share the same producer result and byte size";
        return failure();
      }
      auto minimum = minimumOutgoingRouteByResource.find(resourceId);
      if (minimum == minimumOutgoingRouteByResource.end())
        minimumOutgoingRouteByResource.emplace(resourceId, routeId);
      else
        minimum->second = std::min(minimum->second, routeId);

      TaskModel *source = findTask(sourceTask);
      bool matchesOutput = false;
      if (source) {
        for (auto indexedSlot : llvm::enumerate(source->outputSlots)) {
          if (indexedSlot.value() == (*resource)->slot &&
              source->resultIndices[indexedSlot.index()] == sourceOutput) {
            matchesOutput = true;
            break;
          }
        }
      }
      if (!matchesOutput) {
        module.emitError("outgoing route ")
            << routeId << " does not match its source task result binding";
        return failure();
      }
    }
    routes.push_back(RouteModel{route, (*resource)->slot});
  }

  unsigned localRouteCount =
      llvm::count_if(model.resources, [&](const ResourceModel &resource) {
        return resource.kind == resourceKind;
      });
  unsigned matchedLocalRouteCount =
      llvm::count_if(matchedResourceIndices, [&](unsigned index) {
        return model.resources[index].kind == resourceKind;
      });
  bool completeResourceCoverage =
      incoming ? (localRouteCount == routes.size() &&
                  matchedResourceIndices.size() == routes.size())
               : (localRouteCount == matchedLocalRouteCount);
  if (!completeResourceCoverage) {
    module.emitError("route manifest does not account for every local route "
                     "resource in the isolated core");
    return failure();
  }

  if (!incoming) {
    for (const auto &[resourceId, minimumRouteId] :
         minimumOutgoingRouteByResource) {
      auto routeResource =
          model.routeOutputResourceIndexByGlobalId.find(resourceId);
      if (routeResource == model.routeOutputResourceIndexByGlobalId.end())
        continue;
      ResourceModel &resource = model.resources[routeResource->second];
      if (resource.routeId != std::optional<uint32_t>(minimumRouteId)) {
        resource.op->emitError(
            "coalesced route output must use the minimum route ID as its "
            "canonical route ID");
        return failure();
      }
    }
  }

  llvm::sort(routes, [](const RouteModel &lhs, const RouteModel &rhs) {
    return lhs.route.getId().getInt() < rhs.route.getId().getInt();
  });
  return success();
}

template <typename AttrTy>
FailureOr<SmallVector<AttrTy>> getTypedMemoryArray(ModuleOp module,
                                                   StringRef name) {
  auto values = module->getAttrOfType<ArrayAttr>(name);
  if (!values) {
    module.emitError("expected tile memory-plan attribute '") << name << "'";
    return failure();
  }
  SmallVector<AttrTy> result;
  result.reserve(values.size());
  for (Attribute value : values) {
    auto typed = dyn_cast<AttrTy>(value);
    if (!typed) {
      module.emitError("tile memory-plan attribute '")
          << name << "' contains an invalid record";
      return failure();
    }
    result.push_back(typed);
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
getMemoryI64Array(ModuleOp module, Attribute value, StringRef description) {
  auto array = dyn_cast_or_null<ArrayAttr>(value);
  if (!array) {
    module.emitError("expected integer array for ") << description;
    return failure();
  }
  SmallVector<int64_t> result;
  result.reserve(array.size());
  for (Attribute element : array) {
    auto integer = dyn_cast<IntegerAttr>(element);
    if (!integer) {
      module.emitError("expected integer element in ") << description;
      return failure();
    }
    result.push_back(integer.getInt());
  }
  return result;
}

FailureOr<uint32_t> getMemoryU32(ModuleOp module, IntegerAttr value,
                                 StringRef description,
                                 bool allowInvalid = false) {
  int64_t integer = value.getInt();
  if (allowInvalid && integer == -1)
    return kInvalidABIId;
  if (integer < 0 || static_cast<uint64_t>(integer) > UINT32_MAX) {
    module.emitError("memory-plan ")
        << description << " is outside the Golem tile ABI range";
    return failure();
  }
  return static_cast<uint32_t>(integer);
}

ResourceModel *findMemoryOwnerResource(TileModel &model,
                                       TileMemoryOwnerAttr owner) {
  if (owner.getResourceId().getInt() < 0)
    return nullptr;
  uint32_t resourceId = static_cast<uint32_t>(owner.getResourceId().getInt());
  ResourceModel *match = nullptr;
  bool routeOwner = owner.getKind() == MemoryOwnerKind::RouteInput ||
                    owner.getKind() == MemoryOwnerKind::RouteOutput;
  for (ResourceModel &resource : model.resources) {
    if (resource.globalId != resourceId)
      continue;
    bool kindMatches = true;
    if (owner.getKind() == MemoryOwnerKind::RouteInput)
      kindMatches = resource.kind == ResourceKind::RouteInput;
    else if (owner.getKind() == MemoryOwnerKind::RouteOutput)
      kindMatches = resource.kind == ResourceKind::RouteOutput;
    else
      kindMatches = resource.kind != ResourceKind::RouteInput &&
                    resource.kind != ResourceKind::RouteOutput;
    if (!kindMatches)
      continue;
    if (resource.byteSize !=
        static_cast<uint64_t>(owner.getByteSize().getInt()))
      continue;
    if (match && !routeOwner)
      return nullptr;
    if (!match || resource.slot < match->slot)
      match = &resource;
  }
  return match;
}

LogicalResult collectTileMemoryABI(ModuleOp module, TileModel &model) {
  if (failed(tile_memory::verifyTileMemoryPlan(module)))
    return failure();
  auto owners = getTypedMemoryArray<TileMemoryOwnerAttr>(
      module, tile_memory::kOwnersAttrName);
  auto views = getTypedMemoryArray<TileMemoryViewAttr>(
      module, tile_memory::kViewsAttrName);
  auto movements = getTypedMemoryArray<TileMemoryMovementAttr>(
      module, tile_memory::kMovementsAttrName);
  auto segments = getTypedMemoryArray<TileMemorySegmentAttr>(
      module, tile_memory::kSegmentsAttrName);
  auto assemblies = getTypedMemoryArray<TileMemoryAssemblyAttr>(
      module, tile_memory::kAssembliesAttrName);
  auto completions = getTypedMemoryArray<TileMemoryCompletionEventAttr>(
      module, tile_memory::kCompletionEventsAttrName);
  if (failed(owners) || failed(views) || failed(movements) ||
      failed(segments) || failed(assemblies) || failed(completions))
    return failure();

  DenseMap<uint32_t, unsigned> ownerIndexById;
  DenseMap<uint32_t, unsigned> viewIndexById;
  DenseMap<uint32_t, TileMemoryCompletionEventAttr> completionById;
  for (TileMemoryOwnerAttr owner : *owners) {
    auto id = getMemoryU32(module, owner.getId(), "owner ID");
    auto resourceId =
        getMemoryU32(module, owner.getResourceId(), "owner resource ID", true);
    if (failed(id) || failed(resourceId) || owner.getByteSize().getInt() < 0)
      return failure();
    if (ownerIndexById.count(*id))
      return module.emitError("duplicate tile memory owner ID ") << *id;
    ResourceModel *resource = findMemoryOwnerResource(model, owner);
    bool metadataOnly = owner.getKind() == MemoryOwnerKind::Persistent;
    if (*resourceId != kInvalidABIId && !resource && !metadataOnly) {
      module.emitError("cannot match tile memory owner ")
          << *id << " to one local runtime resource";
      return failure();
    }
    MemoryOwnerModel record;
    record.id = *id;
    record.globalResourceId = *resourceId;
    record.localSlot = resource ? resource->slot : kInvalidABIId;
    record.kind = static_cast<uint32_t>(owner.getKind());
    record.byteSize = static_cast<uint64_t>(owner.getByteSize().getInt());
    ownerIndexById[*id] = model.memoryOwners.size();
    model.memoryOwners.push_back(record);
  }

  for (TileMemoryViewAttr view : *views) {
    auto id = getMemoryU32(module, view.getId(), "view ID");
    auto ownerId = getMemoryU32(module, view.getOwnerId(), "view owner ID");
    auto offsets = getMemoryI64Array(module, view.getOffsets(), "view offsets");
    auto sizes = getMemoryI64Array(module, view.getSizes(), "view sizes");
    auto strides = getMemoryI64Array(module, view.getStrides(), "view strides");
    if (failed(id) || failed(ownerId) || failed(offsets) || failed(sizes) ||
        failed(strides) || view.getByteOffset().getInt() < 0 ||
        view.getByteSize().getInt() < 0)
      return failure();
    auto ownerIt = ownerIndexById.find(*ownerId);
    const uint64_t geometryCount = static_cast<uint64_t>(offsets->size()) * 3;
    if (ownerIt == ownerIndexById.end() || viewIndexById.count(*id) ||
        offsets->size() != sizes->size() || sizes->size() != strides->size() ||
        offsets->size() > UINT32_MAX || geometryCount > UINT32_MAX ||
        model.memoryViewGeometry.size() > UINT32_MAX - geometryCount) {
      module.emitError("invalid tile memory view ") << *id;
      return failure();
    }
    MemoryViewModel record;
    record.id = *id;
    record.ownerId = *ownerId;
    record.ownerSlot = model.memoryOwners[ownerIt->second].localSlot;
    record.rank = static_cast<uint32_t>(offsets->size());
    record.geometryOffset = model.memoryViewGeometry.size();
    record.contiguity = static_cast<uint32_t>(view.getContiguity());
    record.byteOffset = static_cast<uint64_t>(view.getByteOffset().getInt());
    record.byteSize = static_cast<uint64_t>(view.getByteSize().getInt());
    model.memoryViewGeometry.append(offsets->begin(), offsets->end());
    model.memoryViewGeometry.append(sizes->begin(), sizes->end());
    model.memoryViewGeometry.append(strides->begin(), strides->end());
    viewIndexById[*id] = model.memoryViews.size();
    model.memoryViews.push_back(record);
  }

  for (TileMemoryCompletionEventAttr completion : *completions) {
    auto id = getMemoryU32(module, completion.getId(), "completion event ID");
    if (failed(id) || completionById.count(*id))
      return module.emitError("invalid or duplicate completion event ID");
    completionById[*id] = completion;
  }

  DenseMap<uint32_t, unsigned> assemblyIndexById;
  DenseMap<uint32_t, uint32_t> assemblyByRoute;
  for (TileMemoryAssemblyAttr assembly : *assemblies) {
    auto id = getMemoryU32(module, assembly.getId(), "assembly ID");
    auto ownerId =
        getMemoryU32(module, assembly.getOwnerId(), "assembly owner ID");
    auto sources = getMemoryI64Array(module, assembly.getContributingViewIds(),
                                     "assembly source views");
    auto destinations = getMemoryI64Array(
        module, assembly.getDestinationViewIds(), "assembly destination views");
    auto events = getMemoryI64Array(module, assembly.getCompletionEventIds(),
                                    "assembly completion events");
    auto readiness = getMemoryU32(module, assembly.getReadinessEventId(),
                                  "assembly readiness event ID");
    if (failed(id) || failed(ownerId) || failed(sources) ||
        failed(destinations) || failed(events) || failed(readiness) ||
        sources->size() != destinations->size() ||
        sources->size() != events->size() || sources->size() > UINT32_MAX ||
        model.assemblyContributions.size() > UINT32_MAX - sources->size() ||
        assemblyIndexById.count(*id))
      return module.emitError("invalid tile memory assembly");
    auto ownerIt = ownerIndexById.find(*ownerId);
    if (ownerIt == ownerIndexById.end() ||
        model.memoryOwners[ownerIt->second].localSlot == kInvalidABIId)
      return module.emitError("assembly owner has no local runtime slot");

    AssemblyModel record;
    record.id = *id;
    record.ownerSlot = model.memoryOwners[ownerIt->second].localSlot;
    record.contributionOffset = model.assemblyContributions.size();
    record.contributionCount = sources->size();
    record.readinessEventId = *readiness;
    for (size_t index = 0; index < sources->size(); ++index) {
      if ((*sources)[index] < 0 || (*destinations)[index] < 0 ||
          (*events)[index] < 0 ||
          static_cast<uint64_t>((*sources)[index]) > UINT32_MAX ||
          static_cast<uint64_t>((*destinations)[index]) > UINT32_MAX ||
          static_cast<uint64_t>((*events)[index]) > UINT32_MAX)
        return module.emitError("assembly contribution exceeds ABI range");
      uint32_t sourceView = static_cast<uint32_t>((*sources)[index]);
      uint32_t destinationView = static_cast<uint32_t>((*destinations)[index]);
      uint32_t eventId = static_cast<uint32_t>((*events)[index]);
      if (!viewIndexById.count(destinationView) ||
          !completionById.count(eventId))
        return module.emitError(
            "assembly contribution references unknown metadata");
      int64_t routeValue = completionById[eventId].getRouteId().getInt();
      uint32_t routeId =
          routeValue < 0 ? kInvalidABIId : static_cast<uint32_t>(routeValue);
      uint32_t flags = routeId == kInvalidABIId
                           ? 0U
                           : static_cast<uint32_t>(RouteViewIncoming);
      model.assemblyContributions.push_back(AssemblyContributionModel{
          *id, sourceView, destinationView, eventId, routeId, flags});
      if (routeId != kInvalidABIId) {
        if (assemblyByRoute.count(routeId))
          return module.emitError(
              "one route contributes to multiple assemblies");
        assemblyByRoute[routeId] = *id;
      }
    }
    assemblyIndexById[*id] = model.assemblies.size();
    model.assemblies.push_back(record);
  }

  auto findRouteMovement =
      [&](uint32_t routeId,
          bool incoming) -> std::optional<TileMemoryMovementAttr> {
    std::optional<TileMemoryMovementAttr> match;
    std::optional<TileMemoryMovementAttr> assemblyMatch;
    for (TileMemoryMovementAttr movement : *movements) {
      if (movement.getRouteId().getInt() != routeId)
        continue;
      if (movement.getAssemblyId().getInt() >= 0) {
        if (incoming && movement.getMode() != MemoryMovementMode::Packed &&
            assemblyByRoute.lookup(routeId) ==
                static_cast<uint32_t>(movement.getAssemblyId().getInt())) {
          if (assemblyMatch)
            return std::nullopt;
          assemblyMatch = movement;
        }
        continue;
      }
      if (match)
        return std::nullopt;
      match = movement;
    }
    return assemblyMatch ? assemblyMatch : match;
  };
  auto appendRouteViews = [&](ArrayRef<RouteModel> routes,
                              bool incoming) -> LogicalResult {
    for (const RouteModel &routeModel : routes) {
      uint32_t routeId = routeModel.route.getId().getInt();
      std::optional<TileMemoryMovementAttr> movement =
          findRouteMovement(routeId, incoming);
      if (!movement) {
        module.emitError("route ")
            << routeId << " has no unique memory movement record";
        return failure();
      }
      int64_t selectedViewValue =
          incoming ? movement->getDestinationViewId().getInt()
                   : movement->getSourceViewId().getInt();
      int64_t completionValue =
          incoming ? movement->getDestinationCompletionEventId().getInt()
                   : movement->getSourceCompletionEventId().getInt();
      if (completionValue < 0 ||
          static_cast<uint64_t>(completionValue) > UINT32_MAX)
        return module.emitError("route completion event exceeds ABI range");
      uint32_t completionEvent = static_cast<uint32_t>(completionValue);
      uint32_t assemblyId =
          movement->getAssemblyId().getInt() < 0
              ? kInvalidABIId
              : static_cast<uint32_t>(movement->getAssemblyId().getInt());
      uint32_t flags = incoming ? RouteViewIncoming : RouteViewOutgoing;
      if (incoming && assemblyId != kInvalidABIId)
        flags |= RouteViewAssemblyDestination;
      if (selectedViewValue < 0 ||
          static_cast<uint64_t>(selectedViewValue) > UINT32_MAX)
        return module.emitError("route view ID exceeds ABI range");
      uint32_t selectedView = static_cast<uint32_t>(selectedViewValue);
      auto viewIt = viewIndexById.find(selectedView);
      if (viewIt == viewIndexById.end())
        return module.emitError("route local view is absent from tile plan");
      const MemoryViewModel &view = model.memoryViews[viewIt->second];
      if (view.ownerSlot == kInvalidABIId ||
          view.byteSize !=
              static_cast<uint64_t>(routeModel.route.getByteSize().getInt()))
        return module.emitError(
            "route view has no owner slot or mismatched size");
      if (view.contiguity !=
              static_cast<uint32_t>(MemoryContiguity::Contiguous) &&
          movement->getMode() != MemoryMovementMode::Segmented &&
          movement->getMode() != MemoryMovementMode::Packed)
        return module.emitError(
            "non-contiguous route requires segmented or packed movement");
      model.routeViews.push_back(RouteViewModel{
          routeId, routeModel.localSlot, view.ownerSlot, selectedView,
          static_cast<uint32_t>(movement->getId().getInt()),
          static_cast<uint32_t>(movement->getMode()), completionEvent,
          assemblyId, flags, view.byteOffset, view.byteSize});
    }
    return success();
  };
  if (failed(appendRouteViews(model.incomingRoutes, true)) ||
      failed(appendRouteViews(model.outgoingRoutes, false)))
    return failure();
  llvm::sort(model.routeViews,
             [](const RouteViewModel &left, const RouteViewModel &right) {
               return std::tie(left.routeId, left.flags) <
                      std::tie(right.routeId, right.flags);
             });

  DenseMap<uint32_t, SmallVector<TileMemorySegmentAttr>> segmentsByMovement;
  for (TileMemorySegmentAttr segment : *segments) {
    auto movementId =
        getMemoryU32(module, segment.getMovementId(), "segment movement ID");
    if (failed(movementId))
      return failure();
    segmentsByMovement[*movementId].push_back(segment);
  }
  for (const RouteViewModel &routeView : model.routeViews) {
    if (routeView.movementMode !=
        static_cast<uint32_t>(MemoryMovementMode::Segmented))
      continue;
    auto group = segmentsByMovement.find(routeView.movementId);
    if (group == segmentsByMovement.end() || group->second.size() < 2 ||
        group->second.size() > UINT32_MAX ||
        model.memorySegments.size() > UINT32_MAX - group->second.size())
      return module.emitError(
          "segmented route has no ABI-representable segment table");
    SegmentedMovementModel record;
    record.movementId = routeView.movementId;
    record.routeId = routeView.routeId;
    record.segmentOffset = model.memorySegments.size();
    record.segmentCount = group->second.size();
    record.completionEventId = routeView.completionEventId;
    record.assemblyId = routeView.assemblyId;
    record.flags = routeView.flags;
    record.byteSize = routeView.byteSize;
    uint64_t totalBytes = 0;
    for (TileMemorySegmentAttr segment : group->second) {
      if (segment.getSourceByteOffset().getInt() < 0 ||
          segment.getDestinationByteOffset().getInt() < 0 ||
          segment.getByteSize().getInt() <= 0)
        return module.emitError("memory segment exceeds the tile ABI range");
      uint64_t bytes = static_cast<uint64_t>(segment.getByteSize().getInt());
      if (totalBytes > UINT64_MAX - bytes)
        return module.emitError("memory segment byte total overflowed");
      totalBytes += bytes;
      model.memorySegments.push_back(MemorySegmentModel{
          static_cast<uint64_t>(segment.getSourceByteOffset().getInt()),
          static_cast<uint64_t>(segment.getDestinationByteOffset().getInt()),
          bytes});
    }
    if (totalBytes != record.byteSize)
      return module.emitError(
          "segmented route byte total disagrees with route payload");
    model.segmentedMovements.push_back(record);
  }
  model.abiFeatures |= kMemoryViewFeature;
  if (!model.assemblies.empty())
    model.abiFeatures |= kAssemblyJoinFeature;
  if (!model.segmentedMovements.empty())
    model.abiFeatures |= kSegmentedMovementFeature;
  llvm::sort(model.memoryOwners,
             [](const MemoryOwnerModel &left, const MemoryOwnerModel &right) {
               return left.id < right.id;
             });
  llvm::sort(model.memoryViews,
             [](const MemoryViewModel &left, const MemoryViewModel &right) {
               return left.id < right.id;
             });
  llvm::sort(model.assemblies,
             [](const AssemblyModel &left, const AssemblyModel &right) {
               return left.id < right.id;
             });
  llvm::sort(model.segmentedMovements,
             [](const SegmentedMovementModel &left,
                const SegmentedMovementModel &right) {
               return std::tie(left.routeId, left.flags) <
                      std::tie(right.routeId, right.flags);
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

LLVM::LLVMStructType getIdIndexType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  return LLVM::LLVMStructType::getLiteral(context, {i32Type, i32Type});
}

LLVM::LLVMStructType getStaticTaskType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i32Type});
}

LLVM::LLVMStructType getStaticResourceType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type});
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

LLVM::LLVMStructType getDMADescriptorType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i64Type, i64Type, i32Type,
                i32Type, i32Type, i32Type, i32Type, i32Type, i64Type});
}

LLVM::LLVMStructType getMemoryOwnerType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i64Type});
}

LLVM::LLVMStructType getMemoryViewType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type,
                i32Type, i64Type, i64Type});
}

LLVM::LLVMStructType getRouteViewType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type,
                i32Type, i64Type, i64Type});
}

LLVM::LLVMStructType getAssemblyType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type});
}

LLVM::LLVMStructType getAssemblyContributionType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  return LLVM::LLVMStructType::getLiteral(
      context, {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type});
}

LLVM::LLVMStructType getSegmentedMovementType(MLIRContext *context) {
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(context, {i32Type, i32Type, i32Type,
                                                    i32Type, i32Type, i32Type,
                                                    i32Type, i32Type, i64Type});
}

LLVM::LLVMStructType getMemorySegmentType(MLIRContext *context) {
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(context, {i64Type, i64Type, i64Type});
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
      module, tile_runtime_attrs::kTaskCoreIdAttrName);
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
      (module->hasAttr(tile_memory::kPlanVersionAttrName) &&
       failed(collectTileMemoryABI(module, model))) ||
      failed(collectWorkspaceMetadata(model)) ||
      failed(collectScratchpadMetadata(model)) ||
      failed(buildTaskBindings(model)) || failed(buildStaticRuntime(model)) ||
      failed(validateDeploymentPlan(model)))
    return failure();

  return model;
}

} // namespace golem_tile_abi
} // namespace sculptor
} // namespace mlir
