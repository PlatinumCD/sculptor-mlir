#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlanTileScratchpad.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileScratchpadAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeResourceUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeTaskKinds.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassRegistry.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace {

namespace deployment_attrs = mlir::sculptor::deployment_attrs;
namespace tile_runtime_attrs = mlir::sculptor::tile_runtime_attrs;
namespace scratchpad_attrs = mlir::sculptor::scratchpad_attrs;

constexpr int64_t kPlatformMaximumScratchpadBytes = 16 * 1024 * 1024;

using mlir::sculptor::TaskCreateOp;

struct ResourceLifetime {
  mlir::Value value;
  unsigned firstUse = 0;
  unsigned lastUse = 0;
  uint64_t byteSize = 0;
  uint64_t offset = 0;
  bool incoming = false;
  bool outgoing = false;
  std::optional<uint32_t> producerGlobalTaskId;
};

struct ActiveAllocation {
  unsigned lastUse = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct CandidatePlan {
  llvm::SmallVector<TaskCreateOp> tasks;
  llvm::SmallVector<ResourceLifetime> resources;
  uint64_t requiredBytes = 0;
  uint64_t internalBytes = 0;
};

bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto type = func.getFunctionType();
  return type.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(type.getResult(0));
}

mlir::FailureOr<mlir::func::FuncOp>
findTaskGraphFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp graph;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    if (graph) {
      func.emitError("expected one extracted-core task graph");
      return mlir::failure();
    }
    graph = func;
  }
  if (!graph) {
    module.emitError("expected one extracted-core task graph");
    return mlir::failure();
  }
  if (!graph.getBody().hasOneBlock()) {
    graph.emitError("expected task graph to contain one block");
    return mlir::failure();
  }
  return graph;
}

bool isSupportedResource(mlir::Value value) {
  auto resourceType =
      mlir::dyn_cast<mlir::sculptor::TaskResourceType>(value.getType());
  auto shaped =
      resourceType
          ? mlir::dyn_cast<mlir::ShapedType>(resourceType.getValueType())
          : mlir::ShapedType();
  return shaped && shaped.hasStaticShape() && shaped.getElementType().isF32();
}

std::optional<unsigned> findUniqueNextTask(
    TaskCreateOp task,
    const llvm::DenseMap<mlir::Value, llvm::SmallVector<unsigned>> &consumers,
    llvm::ArrayRef<TaskCreateOp> tasks) {
  std::optional<unsigned> next;
  for (mlir::Value output : task.getOutputs()) {
    auto it = consumers.find(output);
    if (it == consumers.end())
      continue;
    for (unsigned consumer : it->second) {
      if (next && *next != consumer)
        return std::nullopt;
      next = consumer;
    }
  }
  if (!next || mlir::sculptor::tile_runtime::isMatrixSetupTask(tasks[*next]))
    return std::nullopt;
  return next;
}

mlir::FailureOr<CandidatePlan> buildPlanForTasks(
    llvm::ArrayRef<TaskCreateOp> regionTasks, uint64_t alignment,
    bool doubleBufferBoundaries,
    const llvm::DenseMap<mlir::Value, TaskCreateOp> &producer,
    const llvm::DenseMap<mlir::Value, llvm::SmallVector<TaskCreateOp>>
        &consumers) {
  CandidatePlan plan;
  plan.tasks.assign(regionTasks.begin(), regionTasks.end());
  llvm::DenseSet<mlir::Operation *> regionSet;
  for (TaskCreateOp task : regionTasks)
    regionSet.insert(task.getOperation());

  llvm::DenseMap<mlir::Value, unsigned> resourceIndex;
  auto recordUse = [&](mlir::Value value,
                       unsigned position) -> mlir::LogicalResult {
    if (!isSupportedResource(value))
      return mlir::success();
    auto globalId = value.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>(
        deployment_attrs::kGlobalResourceIdAttrName);
    if (!globalId || globalId.getInt() < 0 ||
        static_cast<uint64_t>(globalId.getInt()) > UINT32_MAX) {
      value.getDefiningOp()->emitError(
          "scratchpad resource requires a 32-bit global resource ID");
      return mlir::failure();
    }
    auto it = resourceIndex.find(value);
    if (it == resourceIndex.end()) {
      auto byteSize = mlir::sculptor::getTaskResourceByteSize(value);
      if (mlir::failed(byteSize) || *byteSize <= 0)
        return mlir::failure();
      unsigned index = plan.resources.size();
      resourceIndex.try_emplace(value, index);
      plan.resources.push_back(ResourceLifetime{
          value, position, position, static_cast<uint64_t>(*byteSize)});
      return mlir::success();
    }
    ResourceLifetime &resource = plan.resources[it->second];
    resource.firstUse = std::min(resource.firstUse, position);
    resource.lastUse = std::max(resource.lastUse, position);
    return mlir::success();
  };

  for (auto indexedTask : llvm::enumerate(regionTasks)) {
    TaskCreateOp task = indexedTask.value();
    for (mlir::Value input : task.getInputs())
      if (mlir::failed(recordUse(input, indexedTask.index())))
        return mlir::failure();
    for (mlir::Value output : task.getOutputs())
      if (mlir::failed(recordUse(output, indexedTask.index())))
        return mlir::failure();
  }

  for (ResourceLifetime &resource : plan.resources) {
    auto producerIt = producer.find(resource.value);
    TaskCreateOp producerTask =
        producerIt == producer.end() ? TaskCreateOp() : producerIt->second;
    resource.incoming = producerIt == producer.end() ||
                        !regionSet.contains(producerTask.getOperation());
    if (producerTask) {
      auto taskId = producerTask->getAttrOfType<mlir::IntegerAttr>(
          deployment_attrs::kGlobalTaskIdAttrName);
      if (taskId && taskId.getInt() >= 0 &&
          static_cast<uint64_t>(taskId.getInt()) <=
              std::numeric_limits<uint32_t>::max())
        resource.producerGlobalTaskId = static_cast<uint32_t>(taskId.getInt());
    }
    auto consumerIt = consumers.find(resource.value);
    resource.outgoing = consumerIt == consumers.end();
    if (consumerIt != consumers.end()) {
      for (TaskCreateOp consumer : consumerIt->second)
        resource.outgoing |= !regionSet.contains(consumer.getOperation());
    }
    if (!resource.incoming && !resource.outgoing)
      plan.internalBytes += resource.byteSize;
    if (resource.incoming)
      resource.firstUse = 0;
    if (resource.outgoing)
      resource.lastUse = regionTasks.size() - 1;
  }

  llvm::sort(plan.resources, [](const ResourceLifetime &lhs,
                                const ResourceLifetime &rhs) {
    if (lhs.firstUse != rhs.firstUse)
      return lhs.firstUse < rhs.firstUse;
    auto lhsId = lhs.value.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>(
        deployment_attrs::kGlobalResourceIdAttrName);
    auto rhsId = rhs.value.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>(
        deployment_attrs::kGlobalResourceIdAttrName);
    return lhsId.getInt() < rhsId.getInt();
  });

  llvm::SmallVector<ActiveAllocation> active;
  llvm::SmallVector<std::pair<uint64_t, uint64_t>> freeRanges;
  for (ResourceLifetime &resource : plan.resources) {
    llvm::erase_if(active, [&](const ActiveAllocation &allocation) {
      if (allocation.lastUse >= resource.firstUse)
        return false;
      freeRanges.emplace_back(allocation.offset, allocation.size);
      return true;
    });
    uint64_t allocationSize = resource.byteSize;
    if (doubleBufferBoundaries && (resource.incoming || resource.outgoing)) {
      if (allocationSize > std::numeric_limits<uint64_t>::max() / 2)
        return mlir::failure();
      allocationSize *= 2;
    }
    allocationSize = llvm::alignTo(allocationSize, alignment);
    auto freeIt = llvm::find_if(
        freeRanges, [&](auto range) { return range.second >= allocationSize; });
    if (freeIt != freeRanges.end()) {
      resource.offset = freeIt->first;
      freeRanges.erase(freeIt);
    } else {
      resource.offset = llvm::alignTo(plan.requiredBytes, alignment);
      if (resource.offset >
          std::numeric_limits<uint64_t>::max() - allocationSize)
        return mlir::failure();
      plan.requiredBytes = resource.offset + allocationSize;
    }
    active.push_back(
        ActiveAllocation{resource.lastUse, resource.offset, allocationSize});
  }
  return plan;
}

mlir::DictionaryAttr buildDMADescriptor(mlir::Builder &builder,
                                        const ResourceLifetime &resource,
                                        uint32_t id) {
  mlir::Operation *resourceOp = resource.value.getDefiningOp();
  auto globalId = resourceOp->getAttrOfType<mlir::IntegerAttr>(
      deployment_attrs::kGlobalResourceIdAttrName);
  uint32_t direction = 0;
  uint32_t sourceStorage = 0;
  uint32_t destinationStorage = 0;
  uint32_t triggerKind = 0;
  uint32_t triggerId = scratchpad_attrs::kInvalidU32;
  uint32_t routeId = scratchpad_attrs::kInvalidU32;
  if (mlir::isa<mlir::sculptor::TaskGraphInputOp>(resourceOp)) {
    direction = scratchpad_attrs::kDirectionBackingToScratchpad;
    sourceStorage = scratchpad_attrs::kStorageBacking;
    destinationStorage = scratchpad_attrs::kStorageScratchpad;
    triggerKind = scratchpad_attrs::kTriggerResourceReady;
    triggerId = static_cast<uint32_t>(globalId.getInt());
  } else if (mlir::isa<mlir::sculptor::TaskGraphRouteInputOp>(resourceOp)) {
    direction = scratchpad_attrs::kDirectionNicToScratchpad;
    sourceStorage = scratchpad_attrs::kStorageNic;
    destinationStorage = scratchpad_attrs::kStorageScratchpad;
    triggerKind = scratchpad_attrs::kTriggerRouteArrival;
    routeId = static_cast<uint32_t>(resourceOp
                                        ->getAttrOfType<mlir::IntegerAttr>(
                                            deployment_attrs::kRouteIdAttrName)
                                        .getInt());
    triggerId = routeId;
  } else if (mlir::isa<mlir::sculptor::TaskGraphOutputOp>(resourceOp)) {
    direction = scratchpad_attrs::kDirectionScratchpadToBacking;
    sourceStorage = scratchpad_attrs::kStorageScratchpad;
    destinationStorage = scratchpad_attrs::kStorageBacking;
    triggerKind = scratchpad_attrs::kTriggerTaskComplete;
    triggerId =
        resource.producerGlobalTaskId.value_or(scratchpad_attrs::kInvalidU32);
  } else {
    direction = scratchpad_attrs::kDirectionScratchpadToNic;
    sourceStorage = scratchpad_attrs::kStorageScratchpad;
    destinationStorage = scratchpad_attrs::kStorageNic;
    triggerKind = scratchpad_attrs::kTriggerTaskComplete;
    triggerId =
        resource.producerGlobalTaskId.value_or(scratchpad_attrs::kInvalidU32);
    routeId = static_cast<uint32_t>(resourceOp
                                        ->getAttrOfType<mlir::IntegerAttr>(
                                            deployment_attrs::kRouteIdAttrName)
                                        .getInt());
  }
  return builder.getDictionaryAttr({
      builder.getNamedAttr(scratchpad_attrs::kDMAIdFieldName,
                           builder.getI32IntegerAttr(id)),
      builder.getNamedAttr(scratchpad_attrs::kDMADirectionFieldName,
                           builder.getI32IntegerAttr(direction)),
      builder.getNamedAttr(scratchpad_attrs::kDMAGlobalResourceIdFieldName,
                           globalId),
      builder.getNamedAttr(scratchpad_attrs::kDMARouteIdFieldName,
                           builder.getI64IntegerAttr(routeId)),
      builder.getNamedAttr(scratchpad_attrs::kDMAScratchpadOffsetFieldName,
                           builder.getI64IntegerAttr(resource.offset)),
      builder.getNamedAttr(scratchpad_attrs::kDMAByteSizeFieldName,
                           builder.getI64IntegerAttr(resource.byteSize)),
      builder.getNamedAttr(scratchpad_attrs::kDMACompletionTokenFieldName,
                           builder.getI32IntegerAttr(id)),
      builder.getNamedAttr(scratchpad_attrs::kDMATriggerKindFieldName,
                           builder.getI32IntegerAttr(triggerKind)),
      builder.getNamedAttr(scratchpad_attrs::kDMATriggerIdFieldName,
                           builder.getI64IntegerAttr(triggerId)),
      builder.getNamedAttr(
          scratchpad_attrs::kDMAFlagsFieldName,
          builder.getI32IntegerAttr(scratchpad_attrs::kDMAAsynchronous)),
      builder.getNamedAttr(scratchpad_attrs::kDMASourceStorageFieldName,
                           builder.getI32IntegerAttr(sourceStorage)),
      builder.getNamedAttr(scratchpad_attrs::kDMADestinationStorageFieldName,
                           builder.getI32IntegerAttr(destinationStorage)),
      builder.getNamedAttr(scratchpad_attrs::kDMAReservedFieldName,
                           builder.getI64IntegerAttr(0)),
  });
}

} // namespace

namespace mlir::sculptor {

void PlanTileScratchpadPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (localMemory == "workspace")
    return;
  if (localMemory != "scratchpad") {
    module.emitError("local-memory must be 'workspace' or 'scratchpad'");
    return signalPassFailure();
  }
  if (bytes <= 0) {
    module.emitError("scratchpad mode requires a positive byte capacity");
    return signalPassFailure();
  }
  if (bytes > kPlatformMaximumScratchpadBytes) {
    module.emitError(
        "scratchpad capacity exceeds the Platform v0.2 16 MiB window");
    return signalPassFailure();
  }
  if (alignment <= 0 || !llvm::isPowerOf2_64(alignment)) {
    module.emitError("scratchpad alignment must be a positive power of two");
    return signalPassFailure();
  }
  if (alignment > bytes) {
    module.emitError("scratchpad alignment cannot exceed capacity");
    return signalPassFailure();
  }
  if (!module->hasAttr(tile_runtime_attrs::kTaskCoreIdAttrName)) {
    module.emitError("scratchpad planning requires an extracted core module");
    return signalPassFailure();
  }

  auto graphOr = findTaskGraphFunction(module);
  if (failed(graphOr))
    return signalPassFailure();
  func::FuncOp graph = *graphOr;

  SmallVector<TaskCreateOp> tasks;
  DenseMap<Value, TaskCreateOp> producer;
  DenseMap<Value, SmallVector<TaskCreateOp>> consumers;
  DenseMap<Value, SmallVector<unsigned>> consumerIndices;
  for (Operation &op : graph.getBody().front()) {
    if (op.hasAttr(tile_runtime_attrs::kResourceSlotAttrName) ||
        op.hasAttr(tile_runtime_attrs::kTaskIndexAttrName)) {
      op.emitError("scratchpad planning must run before task graph resource "
                   "finalization");
      return signalPassFailure();
    }
    auto task = dyn_cast<TaskCreateOp>(&op);
    if (!task)
      continue;
    unsigned index = tasks.size();
    tasks.push_back(task);
    for (Value output : task.getOutputs())
      producer.try_emplace(output, task);
    for (Value input : task.getInputs()) {
      consumers[input].push_back(task);
      consumerIndices[input].push_back(index);
    }
  }

  std::optional<CandidatePlan> best;
  for (unsigned start = 0; start < tasks.size(); ++start) {
    if (tile_runtime::isMatrixSetupTask(tasks[start]))
      continue;
    SmallVector<TaskCreateOp> chain;
    DenseSet<unsigned> seen;
    unsigned current = start;
    while (seen.insert(current).second) {
      chain.push_back(tasks[current]);
      auto candidate = buildPlanForTasks(
          chain, alignment, doubleBufferBoundaries, producer, consumers);
      if (succeeded(candidate) &&
          candidate->requiredBytes <= static_cast<uint64_t>(bytes) &&
          (!best || candidate->internalBytes > best->internalBytes ||
           (candidate->internalBytes == best->internalBytes &&
            candidate->tasks.size() > best->tasks.size())))
        best = std::move(*candidate);
      auto next = findUniqueNextTask(tasks[current], consumerIndices, tasks);
      if (!next)
        break;
      current = *next;
    }
  }

  if (!best) {
    graph.emitError("no supported producer-consumer region fits scratchpad");
    return signalPassFailure();
  }

  Builder builder(module.getContext());
  SmallVector<Attribute> descriptors;
  uint32_t descriptorId = 0;
  for (const ResourceLifetime &resource : best->resources) {
    Operation *resourceOp = resource.value.getDefiningOp();
    if (!resourceOp->hasAttr(deployment_attrs::kGlobalResourceIdAttrName)) {
      resourceOp->emitError(
          "scratchpad resource requires a global resource ID");
      return signalPassFailure();
    }
    resourceOp->setAttr(
        scratchpad_attrs::kStorageClassAttrName,
        builder.getStringAttr(scratchpad_attrs::kScratchpadStorageClass));
    resourceOp->setAttr(scratchpad_attrs::kScratchpadOffsetAttrName,
                        builder.getI64IntegerAttr(resource.offset));
    if (isa<TaskGraphOutputOp, TaskGraphRouteOutputOp>(resourceOp) &&
        !resource.producerGlobalTaskId) {
      resourceOp->emitError(
          "scratchpad output DMA requires a producer global task ID");
      return signalPassFailure();
    }
    if (isa<TaskGraphInputOp, TaskGraphRouteInputOp, TaskGraphOutputOp,
            TaskGraphRouteOutputOp>(resourceOp))
      descriptors.push_back(
          buildDMADescriptor(builder, resource, descriptorId++));
  }

  graph->setAttr(scratchpad_attrs::kScratchpadRequiredBytesAttrName,
                 builder.getI64IntegerAttr(best->requiredBytes));
  graph->setAttr(scratchpad_attrs::kScratchpadAlignmentAttrName,
                 builder.getI64IntegerAttr(alignment));
  graph->setAttr(scratchpad_attrs::kScratchpadDMADescriptorsAttrName,
                 builder.getArrayAttr(descriptors));
  graph->setAttr(scratchpad_attrs::kScratchpadABIVersionAttrName,
                 builder.getI32IntegerAttr(2));
  graph->setAttr(scratchpad_attrs::kScratchpadFeatureBitsAttrName,
                 builder.getI32IntegerAttr(1));
}

void registerPlanTileScratchpadPass() {
  PassRegistration<PlanTileScratchpadPass>();
}

} // namespace mlir::sculptor
