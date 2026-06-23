#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyStep.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphExecutionPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;

constexpr size_t kWorkspaceAlignment = alignof(std::max_align_t);

struct TaskPlan {
  llvm::SmallVector<uint32_t, 4> inputSlots;
  llvm::SmallVector<uint32_t, 4> outputSlots;
};

struct ExecutablePlan {
  llvm::SmallVector<TaskPlan, 8> tasks;
  llvm::SmallVector<uint32_t, 4> inputSlots;
  llvm::SmallVector<uint32_t, 4> outputSlots;
  llvm::SmallVector<size_t, 4> tempOffsets;
  uint32_t resourceCount = 0;
  uint32_t tempBaseSlot = 0;
  uint32_t tempCount = 0;
  size_t workspaceSize = 0;
};

struct ResourceInfo {
  uint32_t slot = 0;
  size_t byteSize = 0;
  std::optional<uint32_t> tempIndex;
};

struct TempInterval {
  uint32_t slot = 0;
  uint32_t tempIndex = 0;
  size_t byteSize = 0;
  unsigned firstUse = std::numeric_limits<unsigned>::max();
  unsigned lastUse = 0;
  size_t offset = 0;
};

struct ActiveAllocation {
  unsigned lastUse = 0;
  size_t offset = 0;
  size_t size = 0;
};

struct FreeAllocation {
  size_t offset = 0;
  size_t size = 0;
};

mlir::FailureOr<size_t> getStaticByteSize(mlir::Type valueType) {
  if (llvm::isa<mlir::sculptor::RuntimeHandleType,
                mlir::sculptor::LogicalArrayType>(valueType))
    return static_cast<size_t>(0);

  auto getByteSize =
      [](mlir::ShapedType shapedType) -> mlir::FailureOr<size_t> {
    if (!shapedType.hasStaticShape() || !shapedType.getElementType().isF32())
      return mlir::failure();

    int64_t elementCount = shapedType.getNumElements();
    if (elementCount < 0)
      return mlir::failure();

    return static_cast<size_t>(elementCount) * sizeof(float);
  };

  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(valueType)) {
    unsigned bitWidth = floatType.getWidth();
    if (bitWidth == 0)
      return mlir::failure();

    return llvm::divideCeil(static_cast<size_t>(bitWidth),
                            static_cast<size_t>(CHAR_BIT));
  }

  if (auto rankedTensorType = llvm::dyn_cast<mlir::RankedTensorType>(valueType))
    return getByteSize(rankedTensorType);

  if (auto memRefType = llvm::dyn_cast<mlir::MemRefType>(valueType))
    return getByteSize(memRefType);

  return mlir::failure();
}

mlir::FailureOr<size_t> getResourceByteSize(mlir::Value resourceValue) {
  auto resourceType =
      llvm::dyn_cast<mlir::sculptor::TaskResourceType>(resourceValue.getType());
  if (!resourceType)
    return mlir::failure();

  return getStaticByteSize(resourceType.getValueType());
}

template <typename OpT>
mlir::LogicalResult
recordResource(OpT resourceOp, ExecutablePlan &plan,
               llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
               llvm::SmallVectorImpl<mlir::Value> &temporaryResources) {
  mlir::FailureOr<size_t> byteSize =
      getResourceByteSize(resourceOp.getResult());
  if (failed(byteSize)) {
    resourceOp.emitError("expected runtime resources to carry runtime handles, "
                         "float scalars, or static f32 tensor/memref payloads");
    return mlir::failure();
  }

  ResourceInfo info;
  info.slot = resourceInfoByValue.size();
  info.byteSize = *byteSize;
  if constexpr (std::is_same_v<OpT, mlir::sculptor::TaskGraphTemporaryOp>) {
    info.tempIndex = temporaryResources.size();
    temporaryResources.push_back(resourceOp.getResult());
  }

  resourceInfoByValue.try_emplace(resourceOp.getResult(), info);

  if constexpr (std::is_same_v<OpT, mlir::sculptor::TaskGraphInputOp>) {
    plan.inputSlots.push_back(info.slot);
  } else if constexpr (std::is_same_v<OpT, mlir::sculptor::TaskGraphOutputOp>) {
    plan.outputSlots.push_back(info.slot);
  }

  return mlir::success();
}

mlir::LogicalResult
collectResources(mlir::func::FuncOp taskGraphFunc, ExecutablePlan &plan,
                 llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
                 llvm::SmallVectorImpl<mlir::Value> &temporaryResources) {
  mlir::Block &block = taskGraphFunc.getBody().front();
  for (mlir::Operation &op : block) {
    if (auto inputOp = llvm::dyn_cast<mlir::sculptor::TaskGraphInputOp>(&op)) {
      if (failed(recordResource(inputOp, plan, resourceInfoByValue,
                                temporaryResources)))
        return mlir::failure();
      continue;
    }

    if (auto outputOp = llvm::dyn_cast<mlir::sculptor::TaskGraphOutputOp>(&op)) {
      if (failed(recordResource(outputOp, plan, resourceInfoByValue,
                                temporaryResources)))
        return mlir::failure();
      continue;
    }

    if (auto temporaryOp =
            llvm::dyn_cast<mlir::sculptor::TaskGraphTemporaryOp>(&op)) {
      if (failed(recordResource(temporaryOp, plan, resourceInfoByValue,
                                temporaryResources)))
        return mlir::failure();
      continue;
    }

    if (auto persistentOp =
            llvm::dyn_cast<mlir::sculptor::TaskGraphPersistentOp>(&op)) {
      if (failed(recordResource(persistentOp, plan, resourceInfoByValue,
                                temporaryResources)))
        return mlir::failure();
    }
  }

  plan.resourceCount = resourceInfoByValue.size();
  plan.tempCount = temporaryResources.size();
  plan.tempBaseSlot = plan.resourceCount;
  if (!temporaryResources.empty())
    plan.tempBaseSlot =
        resourceInfoByValue.lookup(temporaryResources.front()).slot;

  return mlir::success();
}

mlir::LogicalResult
collectTasks(mlir::func::FuncOp taskGraphFunc, ExecutablePlan &plan,
             llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue) {
  mlir::Block &block = taskGraphFunc.getBody().front();
  llvm::DenseMap<mlir::Value, unsigned> taskIndexByValue;

  for (mlir::Operation &op : block) {
    auto taskOp = llvm::dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    TaskPlan taskPlan;

    for (mlir::Value dependency : taskOp.getDependencies()) {
      auto dependencyIt = taskIndexByValue.find(dependency);
      if (dependencyIt == taskIndexByValue.end() ||
          dependencyIt->second >= plan.tasks.size()) {
        taskOp.emitError("expected dependencies to reference earlier tasks");
        return mlir::failure();
      }
    }

    for (mlir::Value input : taskOp.getInputs()) {
      auto resourceIt = resourceInfoByValue.find(input);
      if (resourceIt == resourceInfoByValue.end()) {
        taskOp.emitError("expected every task input to have a runtime slot");
        return mlir::failure();
      }

      taskPlan.inputSlots.push_back(resourceIt->second.slot);
    }

    for (mlir::Value output : taskOp.getOutputs()) {
      auto resourceIt = resourceInfoByValue.find(output);
      if (resourceIt == resourceInfoByValue.end()) {
        taskOp.emitError("expected every task output to have a runtime slot");
        return mlir::failure();
      }

      taskPlan.outputSlots.push_back(resourceIt->second.slot);
    }

    taskIndexByValue.try_emplace(taskOp.getResult(), plan.tasks.size());
    plan.tasks.push_back(std::move(taskPlan));
  }

  return mlir::success();
}

void updateIntervalUse(TempInterval &interval, unsigned taskIndex) {
  interval.firstUse = std::min(interval.firstUse, taskIndex);
  interval.lastUse = std::max(interval.lastUse, taskIndex);
}

void buildTemporaryIntervals(
    llvm::ArrayRef<mlir::Value> temporaryResources,
    const llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
    llvm::SmallVectorImpl<TempInterval> &intervals,
    llvm::DenseMap<uint32_t, uint32_t> &tempIndexBySlot) {
  intervals.reserve(temporaryResources.size());
  for (mlir::Value temporaryResource : temporaryResources) {
    const ResourceInfo &resourceInfo =
        resourceInfoByValue.lookup(temporaryResource);
    TempInterval interval;
    interval.slot = resourceInfo.slot;
    interval.tempIndex = *resourceInfo.tempIndex;
    interval.byteSize = resourceInfo.byteSize;
    tempIndexBySlot.try_emplace(interval.slot, interval.tempIndex);
    intervals.push_back(interval);
  }
}

void recordTemporarySlotUses(
    llvm::ArrayRef<uint32_t> slots, unsigned taskIndex,
    const llvm::DenseMap<uint32_t, uint32_t> &tempIndexBySlot,
    llvm::SmallVectorImpl<TempInterval> &intervals) {
  for (uint32_t slot : slots) {
    auto tempIt = tempIndexBySlot.find(slot);
    if (tempIt == tempIndexBySlot.end())
      continue;

    updateIntervalUse(intervals[tempIt->second], taskIndex);
  }
}

void recordTemporaryUses(
    llvm::ArrayRef<TaskPlan> tasks,
    const llvm::DenseMap<uint32_t, uint32_t> &tempIndexBySlot,
    llvm::SmallVectorImpl<TempInterval> &intervals) {
  for (const auto &task : llvm::enumerate(tasks)) {
    recordTemporarySlotUses(task.value().inputSlots, task.index(),
                            tempIndexBySlot, intervals);
    recordTemporarySlotUses(task.value().outputSlots, task.index(),
                            tempIndexBySlot, intervals);
  }
}

mlir::LogicalResult
verifyTemporaryUses(mlir::func::FuncOp taskGraphFunc,
                    llvm::ArrayRef<TempInterval> intervals) {
  for (const TempInterval &interval : intervals) {
    if (interval.firstUse == std::numeric_limits<unsigned>::max()) {
      taskGraphFunc.emitError("expected every temporary slot to be used by at "
                              "least one task");
      return mlir::failure();
    }
  }

  return mlir::success();
}

void sortTemporaryIntervalsByUse(
    llvm::SmallVectorImpl<TempInterval> &intervals) {
  std::sort(intervals.begin(), intervals.end(),
            [](const TempInterval &lhs, const TempInterval &rhs) {
              if (lhs.firstUse != rhs.firstUse)
                return lhs.firstUse < rhs.firstUse;
              return lhs.tempIndex < rhs.tempIndex;
            });
}

void releaseExpiredAllocations(
    unsigned firstUse,
    llvm::SmallVectorImpl<ActiveAllocation> &activeAllocations,
    llvm::SmallVectorImpl<FreeAllocation> &freeAllocations) {
  llvm::erase_if(activeAllocations, [&](const ActiveAllocation &allocation) {
    if (allocation.lastUse >= firstUse)
      return false;

    freeAllocations.push_back(
        FreeAllocation{allocation.offset, allocation.size});
    return true;
  });
}

size_t
chooseTemporaryOffset(size_t byteSize, size_t workspaceSize,
                      llvm::SmallVectorImpl<FreeAllocation> &freeAllocations) {
  auto reusableIt = std::find_if(freeAllocations.begin(), freeAllocations.end(),
                                 [&](const FreeAllocation &allocation) {
                                   return allocation.size >= byteSize;
                                 });

  if (reusableIt != freeAllocations.end()) {
    size_t chosenOffset = reusableIt->offset;
    freeAllocations.erase(reusableIt);
    return chosenOffset;
  }

  return llvm::alignTo(workspaceSize, kWorkspaceAlignment);
}

mlir::LogicalResult packTemporaryWorkspace(
    mlir::func::FuncOp taskGraphFunc, ExecutablePlan &plan,
    llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
    llvm::ArrayRef<mlir::Value> temporaryResources) {
  if (temporaryResources.empty()) {
    plan.workspaceSize = 0;
    plan.tempOffsets.clear();
    return mlir::success();
  }

  llvm::DenseMap<uint32_t, uint32_t> tempIndexBySlot;
  llvm::SmallVector<TempInterval> intervals;
  buildTemporaryIntervals(temporaryResources, resourceInfoByValue, intervals,
                          tempIndexBySlot);
  recordTemporaryUses(plan.tasks, tempIndexBySlot, intervals);
  if (failed(verifyTemporaryUses(taskGraphFunc, intervals)))
    return mlir::failure();

  sortTemporaryIntervalsByUse(intervals);

  llvm::SmallVector<ActiveAllocation> activeAllocations;
  llvm::SmallVector<FreeAllocation> freeAllocations;
  plan.tempOffsets.assign(temporaryResources.size(), 0);
  plan.workspaceSize = 0;

  for (TempInterval &interval : intervals) {
    releaseExpiredAllocations(interval.firstUse, activeAllocations,
                              freeAllocations);

    size_t chosenOffset = chooseTemporaryOffset(
        interval.byteSize, plan.workspaceSize, freeAllocations);
    interval.offset = chosenOffset;
    plan.tempOffsets[interval.tempIndex] = chosenOffset;
    plan.workspaceSize =
        std::max(plan.workspaceSize, chosenOffset + interval.byteSize);
    activeAllocations.push_back(
        ActiveAllocation{interval.lastUse, chosenOffset, interval.byteSize});
  }

  return mlir::success();
}

mlir::ArrayAttr buildI64ArrayAttr(mlir::Builder &builder,
                                  llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

template <typename T>
mlir::ArrayAttr buildIntegerArrayAttr(mlir::Builder &builder,
                                      llvm::ArrayRef<T> values) {
  llvm::SmallVector<int64_t> widenedValues;
  widenedValues.reserve(values.size());
  for (T value : values)
    widenedValues.push_back(static_cast<int64_t>(value));
  return buildI64ArrayAttr(builder, widenedValues);
}

mlir::FailureOr<ExecutablePlan>
buildExecutablePlan(mlir::func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected runtime-lowered task graph to have a "
                            "single block");
    return mlir::failure();
  }

  ExecutablePlan plan;
  llvm::DenseMap<mlir::Value, ResourceInfo> resourceInfoByValue;
  llvm::SmallVector<mlir::Value> temporaryResources;

  if (failed(collectResources(taskGraphFunc, plan, resourceInfoByValue,
                              temporaryResources)) ||
      failed(collectTasks(taskGraphFunc, plan, resourceInfoByValue)) ||
      failed(packTemporaryWorkspace(taskGraphFunc, plan, resourceInfoByValue,
                                    temporaryResources))) {
    return mlir::failure();
  }

  return plan;
}

mlir::LogicalResult
annotateTaskGraphWithExecutablePlan(mlir::func::FuncOp taskGraphFunc,
                                    const ExecutablePlan &plan) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected runtime-lowered task graph to have a "
                            "single block");
    return mlir::failure();
  }

  llvm::DenseMap<mlir::Value, ResourceInfo> resourceInfoByValue;
  llvm::SmallVector<mlir::Value> temporaryResources;
  ExecutablePlan recomputedPlan;
  if (failed(collectResources(taskGraphFunc, recomputedPlan,
                              resourceInfoByValue, temporaryResources))) {
    return mlir::failure();
  }

  mlir::Builder builder(taskGraphFunc.getContext());
  taskGraphFunc->setAttr(runtime_attrs::kTaskGraphResourceCountAttrName,
                         builder.getI64IntegerAttr(plan.resourceCount));
  taskGraphFunc->setAttr(
      runtime_attrs::kTaskGraphInputSlotsAttrName,
      buildIntegerArrayAttr(builder,
                            llvm::ArrayRef<uint32_t>(plan.inputSlots)));
  taskGraphFunc->setAttr(
      runtime_attrs::kTaskGraphOutputSlotsAttrName,
      buildIntegerArrayAttr(builder,
                            llvm::ArrayRef<uint32_t>(plan.outputSlots)));
  taskGraphFunc->setAttr(
      runtime_attrs::kTaskGraphTempOffsetsAttrName,
      buildIntegerArrayAttr(builder, llvm::ArrayRef<size_t>(plan.tempOffsets)));
  taskGraphFunc->setAttr(runtime_attrs::kTaskGraphTempBaseSlotAttrName,
                         builder.getI64IntegerAttr(plan.tempBaseSlot));
  taskGraphFunc->setAttr(runtime_attrs::kTaskGraphTempCountAttrName,
                         builder.getI64IntegerAttr(plan.tempCount));
  taskGraphFunc->setAttr(runtime_attrs::kTaskGraphWorkspaceSizeAttrName,
                         builder.getI64IntegerAttr(plan.workspaceSize));

  for (mlir::Value temporaryResource : temporaryResources) {
    const ResourceInfo &resourceInfo =
        resourceInfoByValue.lookup(temporaryResource);
    mlir::Operation *resourceOp = temporaryResource.getDefiningOp();
    resourceOp->setAttr(runtime_attrs::kResourceTempIndexAttrName,
                        builder.getI64IntegerAttr(*resourceInfo.tempIndex));
    resourceOp->setAttr(
        runtime_attrs::kResourceTempOffsetAttrName,
        builder.getI64IntegerAttr(plan.tempOffsets[*resourceInfo.tempIndex]));
  }

  for (auto &resourceIt : resourceInfoByValue) {
    mlir::Operation *resourceOp = resourceIt.first.getDefiningOp();
    resourceOp->setAttr(runtime_attrs::kResourceSlotAttrName,
                        builder.getI64IntegerAttr(resourceIt.second.slot));
    resourceOp->setAttr(runtime_attrs::kResourceByteSizeAttrName,
                        builder.getI64IntegerAttr(resourceIt.second.byteSize));
  }

  unsigned taskIndex = 0;
  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    auto taskOp = llvm::dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    const TaskPlan &taskPlan = plan.tasks[taskIndex];
    taskOp->setAttr(runtime_attrs::kTaskIndexAttrName,
                    builder.getI64IntegerAttr(taskIndex));
    taskOp->setAttr(runtime_attrs::kTaskInputSlotsAttrName,
                    buildIntegerArrayAttr(builder, llvm::ArrayRef<uint32_t>(
                                                       taskPlan.inputSlots)));
    taskOp->setAttr(runtime_attrs::kTaskOutputSlotsAttrName,
                    buildIntegerArrayAttr(builder, llvm::ArrayRef<uint32_t>(
                                                       taskPlan.outputSlots)));
    ++taskIndex;
  }

  return mlir::success();
}

class TaskGraphExecutionPlanAssembler final
    : public mlir::sculptor::TaskGraphAssemblyStep {
public:
  mlir::StringRef getName() const final { return "TaskGraphExecutionPlan"; }

  mlir::LogicalResult assemble(mlir::ModuleOp module,
                               mlir::func::FuncOp forward) const final {
    auto taskGraphFunc =
        mlir::sculptor::assembler_utils::lookupGeneratedTaskGraphFunc(module,
                                                                    forward);
    if (!taskGraphFunc) {
      forward.emitError("expected task graph scaffold to create a generator "
                        "function");
      return mlir::failure();
    }

    if (failed(mlir::sculptor::rebuildTaskGraphExecutionPlan(taskGraphFunc))) {
      return mlir::failure();
    }

    mlir::sculptor::assembler_utils::clearAssemblyAttrs(forward, taskGraphFunc);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {

LogicalResult rebuildTaskGraphExecutionPlan(func::FuncOp taskGraphFunc) {
  auto executablePlan = buildExecutablePlan(taskGraphFunc);
  if (failed(executablePlan) ||
      failed(annotateTaskGraphWithExecutablePlan(taskGraphFunc,
                                                *executablePlan))) {
    return failure();
  }

  return success();
}

void registerTaskGraphExecutionPlanAssembler(TaskGraphAssemblySteps &steps) {
  steps.push_back(std::make_unique<TaskGraphExecutionPlanAssembler>());
}

} // namespace sculptor
} // namespace mlir
