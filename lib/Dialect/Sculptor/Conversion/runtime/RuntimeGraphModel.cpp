#include "sculptor-mlir/Dialect/Sculptor/Conversion/runtime/RuntimeGraphEmission.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace mlir {
namespace sculptor {

namespace {

FailureOr<uint64_t> getRequiredI64Attr(Operation *op, llvm::StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr) {
    op->emitError("expected required runtime attr '") << name << "'";
    return failure();
  }
  if (attr.getInt() < 0) {
    op->emitError("expected non-negative runtime attr '") << name << "'";
    return failure();
  }
  return static_cast<uint64_t>(attr.getInt());
}

uint64_t lookupTaskIndexOrMax(TaskCreateOp taskOp) {
  auto taskIndex =
      getRequiredI64Attr(taskOp, runtime_attrs::kTaskIndexAttrName);
  return succeeded(taskIndex) ? *taskIndex
                              : std::numeric_limits<uint64_t>::max();
}

FailureOr<int32_t> getTaskCoreId(TaskCreateOp taskOp) {
  auto coreIdAttr =
      taskOp->getAttrOfType<IntegerAttr>(runtime_attrs::kTaskCoreIdAttrName);
  if (!coreIdAttr)
    return static_cast<int32_t>(-1);

  int64_t coreId = coreIdAttr.getInt();
  if (coreId < 0 || coreId > std::numeric_limits<int32_t>::max()) {
    taskOp.emitError("expected runtime attr '")
        << runtime_attrs::kTaskCoreIdAttrName
        << "' to be a non-negative 32-bit integer";
    return failure();
  }

  return static_cast<int32_t>(coreId);
}

bool isHandleResourceType(Type valueType) {
  return isa<RuntimeHandleType, LogicalArrayType>(valueType);
}

bool isMetadataOnlyResourceType(Type valueType) {
  return isa<LogicalArrayType>(valueType);
}

LogicalResult initializeGraphMetadata(func::FuncOp taskGraphFunc,
                                      GraphModel &model) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError(
        "expected generated task graph to have a single block");
    return failure();
  }

  auto workspaceSize = getRequiredI64Attr(
      taskGraphFunc, runtime_attrs::kTaskGraphWorkspaceSizeAttrName);
  if (failed(workspaceSize))
    return failure();

  model.workspaceSize = *workspaceSize;
  return success();
}

std::optional<std::pair<int32_t, Value>>
getResourceStorageAndValue(Operation &op) {
  if (auto inputOp = dyn_cast<TaskGraphInputOp>(&op))
    return std::make_pair(STORAGE_INPUT, inputOp.getResult());
  if (auto outputOp = dyn_cast<TaskGraphOutputOp>(&op))
    return std::make_pair(STORAGE_OUTPUT, outputOp.getResult());
  if (auto intermediateOp = dyn_cast<TaskGraphIntermediateOp>(&op))
    return std::make_pair(STORAGE_TEMP, intermediateOp.getResult());
  if (auto persistentOp = dyn_cast<TaskGraphPersistentOp>(&op))
    return std::make_pair(STORAGE_PERSISTENT, persistentOp.getResult());
  return std::nullopt;
}

FailureOr<ResourceModel> buildResourceModel(Operation &op, int32_t storage,
                                            Value resourceValue) {
  auto slot = getRequiredI64Attr(&op, runtime_attrs::kResourceSlotAttrName);
  auto byteSize =
      getRequiredI64Attr(&op, runtime_attrs::kResourceByteSizeAttrName);
  if (failed(slot) || failed(byteSize))
    return failure();

  uint64_t workspaceOffsetValue = 0;
  if (storage == STORAGE_TEMP) {
    auto tempOffset =
        getRequiredI64Attr(&op, runtime_attrs::kResourceTempOffsetAttrName);
    if (failed(tempOffset))
      return failure();
    workspaceOffsetValue = *tempOffset;
  }

  auto resourceType = dyn_cast<TaskResourceType>(resourceValue.getType());
  if (!resourceType) {
    op.emitError("expected task graph resource handle type");
    return failure();
  }

  ResourceModel resource;
  resource.value = resourceValue;
  resource.valueType = resourceType.getValueType();
  resource.kind =
      isHandleResourceType(resource.valueType) ? RES_HANDLE : RES_BUFFER;
  resource.storage = storage;
  resource.slot = static_cast<uint32_t>(*slot);
  resource.byteSize = *byteSize;
  resource.workspaceOffset = workspaceOffsetValue;
  return resource;
}

LogicalResult
collectResourceModels(func::FuncOp taskGraphFunc, GraphModel &model,
                      DenseMap<Value, unsigned> &resourceIndexByValue) {
  for (Operation &op : taskGraphFunc.getBody().front()) {
    std::optional<std::pair<int32_t, Value>> resourceInfo =
        getResourceStorageAndValue(op);
    if (!resourceInfo)
      continue;

    auto resource =
        buildResourceModel(op, resourceInfo->first, resourceInfo->second);
    if (failed(resource))
      return failure();

    resourceIndexByValue.try_emplace(resourceInfo->second,
                                     model.resources.size());
    model.resources.push_back(*resource);
  }

  return success();
}

SmallVector<TaskCreateOp> getTasksInRuntimeOrder(func::FuncOp taskGraphFunc) {
  SmallVector<TaskCreateOp> orderedTasks;
  for (Operation &op : taskGraphFunc.getBody().front())
    if (auto taskOp = dyn_cast<TaskCreateOp>(&op))
      orderedTasks.push_back(taskOp);

  std::sort(orderedTasks.begin(), orderedTasks.end(),
            [&](TaskCreateOp lhs, TaskCreateOp rhs) {
              return lookupTaskIndexOrMax(lhs) < lookupTaskIndexOrMax(rhs);
            });

  return orderedTasks;
}

FailureOr<uint32_t>
getOrCreateCallableIndex(ModuleOp module, TaskCreateOp taskOp,
                         llvm::StringMap<uint32_t> &callableIndexBySymbol,
                         GraphModel &model) {
  auto calleeAttr = taskOp.getCalleeAttr();
  if (!calleeAttr) {
    taskOp.emitError("expected direct task callee symbol");
    return failure();
  }

  auto existingCallable = callableIndexBySymbol.find(calleeAttr.getValue());
  if (existingCallable != callableIndexBySymbol.end())
    return existingCallable->second;

  auto calleeFunc =
      module.lookupSymbol<LLVM::LLVMFuncOp>(calleeAttr.getValue());
  if (!calleeFunc) {
    taskOp.emitError("expected LLVM callable symbol for task callee '")
        << calleeAttr.getValue()
        << "'; run sculptor-emit-runtime-graph after convert-func-to-llvm";
    return failure();
  }

  uint32_t callableIndex = model.callables.size();
  callableIndexBySymbol.try_emplace(calleeAttr.getValue(), callableIndex);
  model.callables.push_back(
      CallableModel{calleeAttr.getValue().str(), calleeFunc, taskOp});
  return callableIndex;
}

LogicalResult
appendTaskBinding(TaskCreateOp taskOp, Value resourceValue, uint16_t flags,
                  const GraphModel &model,
                  const DenseMap<Value, unsigned> &resourceIndexByValue,
                  TaskModel &task, SmallVectorImpl<BindingModel> &bindings) {
  auto resourceIt = resourceIndexByValue.find(resourceValue);
  if (resourceIt == resourceIndexByValue.end()) {
    taskOp.emitError("expected every task resource to carry runtime slot "
                     "metadata");
    return failure();
  }

  const ResourceModel &resource = model.resources[resourceIt->second];
  if (isMetadataOnlyResourceType(resource.valueType))
    return success();

  BindingModel binding;
  binding.kind = resource.kind == RES_HANDLE ? ARG_HANDLE : ARG_BUFFER;
  binding.flags = flags;
  binding.source = SRC_SLOT;
  binding.sourceIndex = resource.slot;
  binding.byteSize = static_cast<uint32_t>(resource.byteSize);
  bindings.push_back(binding);
  ++task.argCount;
  return success();
}

LogicalResult
appendTaskBindings(TaskCreateOp taskOp, GraphModel &model,
                   const DenseMap<Value, unsigned> &resourceIndexByValue,
                   TaskModel &task) {
  for (Value input : taskOp.getInputs())
    if (failed(appendTaskBinding(taskOp, input, ARG_IN, model,
                                 resourceIndexByValue, task, model.bindings)))
      return failure();
  for (Value output : taskOp.getOutputs())
    if (failed(appendTaskBinding(taskOp, output, ARG_OUT, model,
                                 resourceIndexByValue, task, model.bindings)))
      return failure();
  return success();
}

LogicalResult
appendTaskDependencies(TaskCreateOp taskOp,
                       const DenseMap<Value, uint32_t> &taskIndexByValue,
                       TaskModel &task, SmallVectorImpl<uint32_t> &deps) {
  for (Value dependency : taskOp.getDependencies()) {
    auto dependencyIt = taskIndexByValue.find(dependency);
    if (dependencyIt == taskIndexByValue.end()) {
      taskOp.emitError("expected task dependencies to be emitted in task "
                       "index order");
      return failure();
    }

    deps.push_back(dependencyIt->second);
    ++task.depCount;
  }
  return success();
}

LogicalResult
collectTaskModels(ModuleOp module, func::FuncOp taskGraphFunc,
                  GraphModel &model,
                  const DenseMap<Value, unsigned> &resourceIndexByValue) {
  DenseMap<Value, uint32_t> taskIndexByValue;
  llvm::StringMap<uint32_t> callableIndexBySymbol;

  for (TaskCreateOp taskOp : getTasksInRuntimeOrder(taskGraphFunc)) {
    auto coreId = getTaskCoreId(taskOp);
    if (failed(coreId))
      return failure();
    auto callableIndex =
        getOrCreateCallableIndex(module, taskOp, callableIndexBySymbol, model);
    if (failed(callableIndex))
      return failure();

    TaskModel task;
    task.op = taskOp;
    task.callableIndex = *callableIndex;
    task.coreId = *coreId;
    task.argBegin = model.bindings.size();
    task.depBegin = model.deps.size();

    if (failed(appendTaskBindings(taskOp, model, resourceIndexByValue, task)))
      return failure();
    if (failed(
            appendTaskDependencies(taskOp, taskIndexByValue, task, model.deps)))
      return failure();

    taskIndexByValue.try_emplace(taskOp.getResult(), model.tasks.size());
    model.tasks.push_back(task);
  }

  return success();
}

} // namespace

FailureOr<func::FuncOp> findTaskGraphFunc(ModuleOp module) {
  func::FuncOp taskGraphFunc;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    auto functionType = func.getFunctionType();
    if (functionType.getNumInputs() != 0 || functionType.getNumResults() != 1)
      continue;
    if (!isa<TaskGraphType>(functionType.getResult(0)))
      continue;

    if (taskGraphFunc) {
      module.emitError("expected at most one generated task graph function");
      return failure();
    }
    taskGraphFunc = func;
  }

  return taskGraphFunc;
}

LogicalResult
collectGraphModel(ModuleOp module, func::FuncOp taskGraphFunc,
                  GraphModel &model,
                  DenseMap<Value, unsigned> &resourceIndexByValue) {
  model = GraphModel{};
  resourceIndexByValue.clear();

  if (failed(initializeGraphMetadata(taskGraphFunc, model)))
    return failure();
  if (failed(collectResourceModels(taskGraphFunc, model, resourceIndexByValue)))
    return failure();
  return collectTaskModels(module, taskGraphFunc, model, resourceIndexByValue);
}

} // namespace sculptor
} // namespace mlir
