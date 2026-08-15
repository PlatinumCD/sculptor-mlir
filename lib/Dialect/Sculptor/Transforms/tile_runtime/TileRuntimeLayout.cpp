#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileScratchpadAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeLayout.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeOrder.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeResourceUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {

namespace deployment_attrs = mlir::sculptor::deployment_attrs;
namespace tile_runtime_attrs = mlir::sculptor::tile_runtime_attrs;
namespace scratchpad_attrs = mlir::sculptor::scratchpad_attrs;

constexpr size_t kWorkspaceAlignment = alignof(std::max_align_t);

struct TaskPlan {
  llvm::SmallVector<uint32_t, 4> inputSlots;
  llvm::SmallVector<uint32_t, 4> outputSlots;
};

struct RouteSlot {
  int64_t routeId = 0;
  uint32_t slot = 0;
};

struct ExecutablePlan {
  llvm::SmallVector<TaskPlan, 8> tasks;
  llvm::SmallVector<uint32_t, 4> inputSlots;
  llvm::SmallVector<uint32_t, 4> outputSlots;
  llvm::SmallVector<RouteSlot, 4> routeInputSlots;
  llvm::SmallVector<RouteSlot, 4> routeOutputSlots;
  llvm::SmallVector<size_t, 4> tempOffsets;
  llvm::SmallVector<size_t, 4> routeOffsets;
  uint32_t resourceCount = 0;
  uint32_t tempBaseSlot = 0;
  uint32_t tempCount = 0;
  size_t workspaceSize = 0;
};

struct ResourceInfo {
  uint32_t slot = 0;
  size_t byteSize = 0;
  std::optional<uint32_t> tempIndex;
  std::optional<uint32_t> routeIndex;
  std::optional<int64_t> ownerId;
  std::optional<int64_t> lifetimeId;
  bool scratchpad = false;
};

struct RouteResource {
  mlir::Value value;
  int64_t routeId = 0;
  uint32_t routeIndex = 0;
};

template <typename AttrTy>
mlir::FailureOr<llvm::SmallVector<AttrTy>>
getMemoryPlanArray(mlir::ModuleOp module, llvm::StringRef name) {
  auto values = module->getAttrOfType<mlir::ArrayAttr>(name);
  if (!values)
    return module.emitError("expected tile memory-plan array '") << name << "'";
  llvm::SmallVector<AttrTy> result;
  result.reserve(values.size());
  for (mlir::Attribute value : values) {
    auto typed = mlir::dyn_cast<AttrTy>(value);
    if (!typed)
      return module.emitError("tile memory-plan array '")
             << name << "' contains an invalid record";
    result.push_back(typed);
  }
  return result;
}

template <typename OpT>
mlir::LogicalResult
recordResource(OpT resourceOp, ExecutablePlan &plan,
               llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
               llvm::SmallVectorImpl<mlir::Value> &intermediateResources,
               llvm::SmallVectorImpl<RouteResource> &routeResources) {
  mlir::FailureOr<int64_t> byteSize =
      mlir::sculptor::getTaskResourceByteSize(resourceOp.getResult());
  if (failed(byteSize)) {
    resourceOp.emitError("expected runtime resources to carry runtime handles "
                         "or static byte-addressable numeric payloads");
    return mlir::failure();
  }

  ResourceInfo info;
  info.slot = resourceInfoByValue.size();
  info.byteSize = static_cast<size_t>(*byteSize);
  auto storageClass = resourceOp->template getAttrOfType<mlir::StringAttr>(
      scratchpad_attrs::kStorageClassAttrName);
  info.scratchpad =
      storageClass &&
      storageClass.getValue() == scratchpad_attrs::kScratchpadStorageClass;
  if constexpr (std::is_same_v<OpT, mlir::sculptor::TaskGraphIntermediateOp>) {
    if (!info.scratchpad) {
      info.tempIndex = intermediateResources.size();
      intermediateResources.push_back(resourceOp.getResult());
    }
  } else if constexpr (std::is_same_v<OpT,
                                      mlir::sculptor::TaskGraphRouteInputOp> ||
                       std::is_same_v<OpT,
                                      mlir::sculptor::TaskGraphRouteOutputOp>) {
    auto routeId = resourceOp->template getAttrOfType<mlir::IntegerAttr>(
        deployment_attrs::kRouteIdAttrName);
    if (!routeId || routeId.getInt() < 0) {
      resourceOp.emitError("expected a non-negative deployment route ID");
      return mlir::failure();
    }
    info.routeIndex = routeResources.size();
    routeResources.push_back(RouteResource{resourceOp.getResult(),
                                           routeId.getInt(), *info.routeIndex});
  }

  resourceInfoByValue.try_emplace(resourceOp.getResult(), info);

  if constexpr (std::is_same_v<OpT, mlir::sculptor::TaskGraphInputOp>) {
    plan.inputSlots.push_back(info.slot);
  } else if constexpr (std::is_same_v<OpT, mlir::sculptor::TaskGraphOutputOp>) {
    plan.outputSlots.push_back(info.slot);
  } else if constexpr (std::is_same_v<OpT,
                                      mlir::sculptor::TaskGraphRouteInputOp>) {
    plan.routeInputSlots.push_back(
        RouteSlot{resourceOp
                      ->template getAttrOfType<mlir::IntegerAttr>(
                          deployment_attrs::kRouteIdAttrName)
                      .getInt(),
                  info.slot});
  } else if constexpr (std::is_same_v<OpT,
                                      mlir::sculptor::TaskGraphRouteOutputOp>) {
    plan.routeOutputSlots.push_back(
        RouteSlot{resourceOp
                      ->template getAttrOfType<mlir::IntegerAttr>(
                          deployment_attrs::kRouteIdAttrName)
                      .getInt(),
                  info.slot});
  }

  return mlir::success();
}

mlir::LogicalResult
collectResources(mlir::func::FuncOp taskGraphFunc, ExecutablePlan &plan,
                 llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
                 llvm::SmallVectorImpl<mlir::Value> &intermediateResources,
                 llvm::SmallVectorImpl<RouteResource> &routeResources) {
  mlir::Block &block = taskGraphFunc.getBody().front();
  for (mlir::Operation &op : block) {
    if (auto inputOp = llvm::dyn_cast<mlir::sculptor::TaskGraphInputOp>(&op)) {
      if (failed(recordResource(inputOp, plan, resourceInfoByValue,
                                intermediateResources, routeResources)))
        return mlir::failure();
      continue;
    }

    if (auto outputOp =
            llvm::dyn_cast<mlir::sculptor::TaskGraphOutputOp>(&op)) {
      if (failed(recordResource(outputOp, plan, resourceInfoByValue,
                                intermediateResources, routeResources)))
        return mlir::failure();
      continue;
    }

    if (auto intermediateOp =
            llvm::dyn_cast<mlir::sculptor::TaskGraphIntermediateOp>(&op)) {
      if (failed(recordResource(intermediateOp, plan, resourceInfoByValue,
                                intermediateResources, routeResources)))
        return mlir::failure();
      continue;
    }

    if (auto persistentOp =
            llvm::dyn_cast<mlir::sculptor::TaskGraphPersistentOp>(&op)) {
      if (failed(recordResource(persistentOp, plan, resourceInfoByValue,
                                intermediateResources, routeResources)))
        return mlir::failure();
      continue;
    }

    if (auto routeInputOp =
            llvm::dyn_cast<mlir::sculptor::TaskGraphRouteInputOp>(&op)) {
      if (failed(recordResource(routeInputOp, plan, resourceInfoByValue,
                                intermediateResources, routeResources)))
        return mlir::failure();
      continue;
    }

    if (auto routeOutputOp =
            llvm::dyn_cast<mlir::sculptor::TaskGraphRouteOutputOp>(&op)) {
      if (failed(recordResource(routeOutputOp, plan, resourceInfoByValue,
                                intermediateResources, routeResources)))
        return mlir::failure();
    }
  }

  auto sortByRouteId = [](llvm::SmallVectorImpl<RouteSlot> &slots) {
    llvm::sort(slots, [](const RouteSlot &lhs, const RouteSlot &rhs) {
      return lhs.routeId < rhs.routeId;
    });
  };
  sortByRouteId(plan.routeInputSlots);
  sortByRouteId(plan.routeOutputSlots);

  plan.resourceCount = resourceInfoByValue.size();
  plan.tempCount = intermediateResources.size();
  plan.tempBaseSlot = plan.resourceCount;
  if (!intermediateResources.empty())
    plan.tempBaseSlot =
        resourceInfoByValue.lookup(intermediateResources.front()).slot;

  return mlir::success();
}

mlir::LogicalResult bindResourcesToMemoryPlan(
    mlir::func::FuncOp taskGraphFunc,
    llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue) {
  mlir::ModuleOp module = taskGraphFunc->getParentOfType<mlir::ModuleOp>();
  auto owners = getMemoryPlanArray<mlir::sculptor::TileMemoryOwnerAttr>(
      module, mlir::sculptor::tile_memory::kOwnersAttrName);
  auto lifetimes = getMemoryPlanArray<mlir::sculptor::TileMemoryLifetimeAttr>(
      module, mlir::sculptor::tile_memory::kLifetimesAttrName);
  if (failed(owners) || failed(lifetimes))
    return mlir::failure();

  llvm::DenseMap<int64_t, mlir::sculptor::TileMemoryOwnerAttr> ownerByResource;
  llvm::DenseMap<int64_t, mlir::sculptor::TileMemoryLifetimeAttr>
      lifetimeByOwner;
  for (mlir::sculptor::TileMemoryOwnerAttr owner : *owners) {
    int64_t resourceId = owner.getResourceId().getInt();
    if (resourceId < 0)
      continue;
    if (!ownerByResource.try_emplace(resourceId, owner).second)
      return module.emitError(
          "multiple local memory owners use one global resource ID");
  }
  for (mlir::sculptor::TileMemoryLifetimeAttr lifetime : *lifetimes) {
    if (lifetime.getSubjectKind() !=
        mlir::sculptor::MemoryLifetimeSubjectKind::Owner)
      continue;
    if (!lifetimeByOwner.try_emplace(lifetime.getOwnerId().getInt(), lifetime)
             .second)
      return module.emitError("memory owner has multiple lifetime records");
  }

  for (auto &[value, resource] : resourceInfoByValue) {
    auto resourceId = value.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>(
        deployment_attrs::kGlobalResourceIdAttrName);
    if (!resourceId)
      return value.getDefiningOp()->emitError(
          "runtime resource has no global resource ID");
    auto owner = ownerByResource.find(resourceId.getInt());
    if (owner == ownerByResource.end())
      return value.getDefiningOp()->emitError(
          "runtime resource has no tile memory owner");
    auto lifetime = lifetimeByOwner.find(owner->second.getId().getInt());
    if (lifetime == lifetimeByOwner.end())
      return value.getDefiningOp()->emitError(
          "runtime resource owner has no lifetime");
    if (owner->second.getByteSize().getInt() !=
        static_cast<int64_t>(resource.byteSize))
      return value.getDefiningOp()->emitError(
          "runtime resource size disagrees with its memory owner");
    resource.ownerId = owner->second.getId().getInt();
    resource.lifetimeId = lifetime->second.getId().getInt();
  }
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

void packConservativeWorkspace(
    ExecutablePlan &plan,
    const llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
    llvm::ArrayRef<mlir::Value> intermediateResources,
    llvm::ArrayRef<RouteResource> routeResources) {
  plan.workspaceSize = 0;
  plan.tempOffsets.assign(intermediateResources.size(), 0);
  for (mlir::Value value : intermediateResources) {
    const ResourceInfo &resource = resourceInfoByValue.lookup(value);
    size_t offset = llvm::alignTo(plan.workspaceSize, kWorkspaceAlignment);
    plan.tempOffsets[*resource.tempIndex] = offset;
    plan.workspaceSize = offset + std::max<size_t>(resource.byteSize, 1);
  }
  plan.routeOffsets.assign(routeResources.size(), 0);
  for (const RouteResource &route : routeResources) {
    const ResourceInfo &resource = resourceInfoByValue.lookup(route.value);
    if (resource.scratchpad)
      continue;
    size_t offset = llvm::alignTo(plan.workspaceSize, kWorkspaceAlignment);
    plan.routeOffsets[route.routeIndex] = offset;
    plan.workspaceSize = offset + std::max<size_t>(resource.byteSize, 1);
  }
}

mlir::LogicalResult packProvenWorkspace(
    mlir::func::FuncOp taskGraphFunc, ExecutablePlan &plan,
    llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
    llvm::ArrayRef<mlir::Value> intermediateResources,
    llvm::ArrayRef<RouteResource> routeResources) {
  mlir::ModuleOp module = taskGraphFunc->getParentOfType<mlir::ModuleOp>();
  auto lifetimes = getMemoryPlanArray<mlir::sculptor::TileMemoryLifetimeAttr>(
      module, mlir::sculptor::tile_memory::kLifetimesAttrName);
  if (failed(lifetimes))
    return mlir::failure();

  std::map<int64_t, mlir::sculptor::TileMemoryLifetimeAttr> lifetimeById;
  for (mlir::sculptor::TileMemoryLifetimeAttr lifetime : *lifetimes)
    lifetimeById[lifetime.getId().getInt()] = lifetime;

  llvm::SmallVector<mlir::sculptor::tile_memory::WorkspaceAllocationRequest>
      requests;
  for (auto &[value, resource] : resourceInfoByValue) {
    if (resource.scratchpad || !resource.lifetimeId)
      continue;
    bool workspaceResource =
        resource.tempIndex.has_value() || resource.routeIndex.has_value();
    if (!workspaceResource)
      continue;
    if (!resource.ownerId)
      return value.getDefiningOp()->emitError(
          "workspace resource has no memory owner");
    auto lifetime = lifetimeById.find(*resource.lifetimeId);
    if (lifetime == lifetimeById.end())
      return value.getDefiningOp()->emitError(
          "runtime resource references an unknown lifetime");
    if (lifetime->second.getStorage() !=
        mlir::sculptor::MemoryLifetimeStorage::Workspace)
      continue;
    requests.push_back({*resource.lifetimeId,
                        static_cast<int64_t>(resource.byteSize),
                        lifetime->second.getAlignment().getInt()});
  }
  auto exactLayout =
      mlir::sculptor::tile_memory::buildExactWorkspaceLayout(module, requests);
  if (failed(exactLayout))
    return mlir::failure();
  std::map<int64_t, size_t> offsetByLifetime;
  for (const auto &allocation : exactLayout->allocations)
    offsetByLifetime[allocation.lifetimeId] =
        static_cast<size_t>(allocation.offset);
  plan.workspaceSize = static_cast<size_t>(exactLayout->workspaceBytes);

  plan.tempOffsets.assign(intermediateResources.size(), 0);
  for (mlir::Value resourceValue : intermediateResources) {
    const ResourceInfo &resource = resourceInfoByValue.lookup(resourceValue);
    if (!resource.lifetimeId || !resource.tempIndex)
      return resourceValue.getDefiningOp()->emitError(
          "intermediate resource is missing lifetime metadata");
    auto offset = offsetByLifetime.find(*resource.lifetimeId);
    plan.tempOffsets[*resource.tempIndex] =
        offset == offsetByLifetime.end() ? 0 : offset->second;
  }
  plan.routeOffsets.assign(routeResources.size(), 0);
  for (const RouteResource &route : routeResources) {
    const ResourceInfo &resource = resourceInfoByValue.lookup(route.value);
    if (resource.scratchpad)
      continue;
    if (!resource.lifetimeId)
      return route.value.getDefiningOp()->emitError(
          "route resource is missing lifetime metadata");
    plan.routeOffsets[route.routeIndex] =
        offsetByLifetime.at(*resource.lifetimeId);
  }

  mlir::Builder builder(module.getContext());
  llvm::SmallVector<mlir::Attribute> updatedLifetimes;
  updatedLifetimes.reserve(lifetimes->size());
  for (mlir::sculptor::TileMemoryLifetimeAttr lifetime : *lifetimes) {
    auto offset = offsetByLifetime.find(lifetime.getId().getInt());
    updatedLifetimes.push_back(mlir::sculptor::TileMemoryLifetimeAttr::get(
        module.getContext(), lifetime.getId(), lifetime.getSubjectKind(),
        lifetime.getStorage(), lifetime.getOwnerId(), lifetime.getRoutine(),
        lifetime.getAllocationOrdinal(), lifetime.getTile(),
        lifetime.getByteSize(), lifetime.getAlignment(),
        builder.getI64IntegerAttr(offset == offsetByLifetime.end()
                                      ? lifetime.getOffset().getInt()
                                      : static_cast<int64_t>(offset->second)),
        lifetime.getViewIds(), lifetime.getAccessEventIds()));
  }
  module->setAttr(mlir::sculptor::tile_memory::kLifetimesAttrName,
                  builder.getArrayAttr(updatedLifetimes));
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
  llvm::SmallVector<mlir::Value> intermediateResources;
  llvm::SmallVector<RouteResource> routeResources;

  if (failed(collectResources(taskGraphFunc, plan, resourceInfoByValue,
                              intermediateResources, routeResources)) ||
      failed(collectTasks(taskGraphFunc, plan, resourceInfoByValue))) {
    return mlir::failure();
  }
  mlir::ModuleOp module = taskGraphFunc->getParentOfType<mlir::ModuleOp>();
  if (module->hasAttr(mlir::sculptor::tile_memory::kPlanVersionAttrName)) {
    if (failed(bindResourcesToMemoryPlan(taskGraphFunc, resourceInfoByValue)) ||
        failed(packProvenWorkspace(taskGraphFunc, plan, resourceInfoByValue,
                                   intermediateResources, routeResources)))
      return mlir::failure();
  } else {
    packConservativeWorkspace(plan, resourceInfoByValue, intermediateResources,
                              routeResources);
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
  llvm::SmallVector<mlir::Value> intermediateResources;
  llvm::SmallVector<RouteResource> routeResources;
  ExecutablePlan recomputedPlan;
  if (failed(collectResources(taskGraphFunc, recomputedPlan,
                              resourceInfoByValue, intermediateResources,
                              routeResources))) {
    return mlir::failure();
  }

  mlir::Builder builder(taskGraphFunc.getContext());
  taskGraphFunc->setAttr(tile_runtime_attrs::kTaskGraphResourceCountAttrName,
                         builder.getI64IntegerAttr(plan.resourceCount));
  taskGraphFunc->setAttr(
      tile_runtime_attrs::kTaskGraphInputSlotsAttrName,
      buildIntegerArrayAttr(builder,
                            llvm::ArrayRef<uint32_t>(plan.inputSlots)));
  taskGraphFunc->setAttr(
      tile_runtime_attrs::kTaskGraphOutputSlotsAttrName,
      buildIntegerArrayAttr(builder,
                            llvm::ArrayRef<uint32_t>(plan.outputSlots)));
  auto buildRouteSlotArrayAttr =
      [&](llvm::ArrayRef<RouteSlot> routeSlots) -> mlir::ArrayAttr {
    llvm::SmallVector<int64_t, 4> slots;
    slots.reserve(routeSlots.size());
    for (const RouteSlot &routeSlot : routeSlots)
      slots.push_back(routeSlot.slot);
    return buildI64ArrayAttr(builder, slots);
  };
  taskGraphFunc->setAttr(tile_runtime_attrs::kTaskGraphRouteInputSlotsAttrName,
                         buildRouteSlotArrayAttr(plan.routeInputSlots));
  taskGraphFunc->setAttr(tile_runtime_attrs::kTaskGraphRouteOutputSlotsAttrName,
                         buildRouteSlotArrayAttr(plan.routeOutputSlots));
  taskGraphFunc->setAttr(
      tile_runtime_attrs::kTaskGraphTempOffsetsAttrName,
      buildIntegerArrayAttr(builder, llvm::ArrayRef<size_t>(plan.tempOffsets)));
  taskGraphFunc->setAttr(tile_runtime_attrs::kTaskGraphTempBaseSlotAttrName,
                         builder.getI64IntegerAttr(plan.tempBaseSlot));
  taskGraphFunc->setAttr(tile_runtime_attrs::kTaskGraphTempCountAttrName,
                         builder.getI64IntegerAttr(plan.tempCount));
  taskGraphFunc->setAttr(tile_runtime_attrs::kTaskGraphWorkspaceSizeAttrName,
                         builder.getI64IntegerAttr(plan.workspaceSize));

  for (mlir::Value intermediateResource : intermediateResources) {
    const ResourceInfo &resourceInfo =
        resourceInfoByValue.lookup(intermediateResource);
    mlir::Operation *resourceOp = intermediateResource.getDefiningOp();
    resourceOp->setAttr(tile_runtime_attrs::kResourceTempIndexAttrName,
                        builder.getI64IntegerAttr(*resourceInfo.tempIndex));
    resourceOp->setAttr(
        tile_runtime_attrs::kResourceTempOffsetAttrName,
        builder.getI64IntegerAttr(plan.tempOffsets[*resourceInfo.tempIndex]));
  }

  for (const RouteResource &route : routeResources) {
    const ResourceInfo &resourceInfo = resourceInfoByValue.lookup(route.value);
    if (resourceInfo.scratchpad)
      continue;
    mlir::Operation *resourceOp = route.value.getDefiningOp();
    resourceOp->setAttr(
        tile_runtime_attrs::kResourceTempOffsetAttrName,
        builder.getI64IntegerAttr(plan.routeOffsets[route.routeIndex]));
  }

  for (auto &resourceIt : resourceInfoByValue) {
    mlir::Operation *resourceOp = resourceIt.first.getDefiningOp();
    resourceOp->setAttr(tile_runtime_attrs::kResourceSlotAttrName,
                        builder.getI64IntegerAttr(resourceIt.second.slot));
    resourceOp->setAttr(tile_runtime_attrs::kResourceByteSizeAttrName,
                        builder.getI64IntegerAttr(resourceIt.second.byteSize));
  }

  llvm::SmallVector<int64_t> coreByTask;
  coreByTask.reserve(plan.tasks.size());
  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    auto taskOp = llvm::dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;
    auto coreId = taskOp->getAttrOfType<mlir::IntegerAttr>(
        tile_runtime_attrs::kTaskCoreIdAttrName);
    coreByTask.push_back(coreId ? coreId.getInt() : 0);
  }
  llvm::SmallVector<unsigned> localRuntimeOrder =
      mlir::sculptor::tile_runtime::buildLocalRuntimeOrder(coreByTask);

  unsigned taskOrdinal = 0;
  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    auto taskOp = llvm::dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    const TaskPlan &taskPlan = plan.tasks[taskOrdinal];
    taskOp->setAttr(tile_runtime_attrs::kTaskIndexAttrName,
                    builder.getI64IntegerAttr(localRuntimeOrder[taskOrdinal]));
    taskOp->setAttr(tile_runtime_attrs::kTaskInputSlotsAttrName,
                    buildIntegerArrayAttr(builder, llvm::ArrayRef<uint32_t>(
                                                       taskPlan.inputSlots)));
    taskOp->setAttr(tile_runtime_attrs::kTaskOutputSlotsAttrName,
                    buildIntegerArrayAttr(builder, llvm::ArrayRef<uint32_t>(
                                                       taskPlan.outputSlots)));
    ++taskOrdinal;
  }

  return mlir::success();
}

} // namespace

namespace mlir {
namespace sculptor {

LogicalResult rebuildTileRuntimeLayout(func::FuncOp taskGraphFunc) {
  auto executablePlan = buildExecutablePlan(taskGraphFunc);
  if (failed(executablePlan) || failed(annotateTaskGraphWithExecutablePlan(
                                    taskGraphFunc, *executablePlan))) {
    return failure();
  }

  return success();
}

} // namespace sculptor
} // namespace mlir
