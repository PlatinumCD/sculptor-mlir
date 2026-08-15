#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlanTileScratchpad.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileScratchpadAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeResourceUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeTaskKinds.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassRegistry.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>

namespace {

namespace deployment_attrs = mlir::sculptor::deployment_attrs;
namespace scratchpad_attrs = mlir::sculptor::scratchpad_attrs;
namespace tile_memory = mlir::sculptor::tile_memory;
namespace tile_runtime_attrs = mlir::sculptor::tile_runtime_attrs;

using namespace mlir;
using namespace mlir::sculptor;

constexpr int64_t kPlatformMaximumScratchpadBytes = 16 * 1024 * 1024;

struct Candidate {
  uint32_t id = 0;
  Value value;
  TileMemoryOwnerAttr owner;
  TileMemoryLifetimeAttr lifetime;
  uint64_t byteSize = 0;
  uint64_t alignment = 0;
  uint64_t capacityDemand = 0;
  uint64_t readCount = 0;
  uint64_t writeCount = 0;
  uint64_t movementCount = 0;
  uint64_t reuseCount = 0;
  uint64_t estimatedSavings = 0;
  std::optional<uint32_t> producerGlobalTaskId;
  std::optional<uint32_t> routeId;
};

struct CandidateGroup {
  uint32_t key = 0;
  SmallVector<unsigned> members;
  uint64_t byteSize = 0;
  uint64_t alignment = 1;
  uint64_t capacityDemand = 0;
  uint64_t estimatedSavings = 0;
};

struct SelectedGroup {
  CandidateGroup group;
  uint64_t offset = 0;
};

bool returnsTaskGraph(func::FuncOp func) {
  auto type = func.getFunctionType();
  return type.getNumResults() == 1 && isa<TaskGraphType>(type.getResult(0));
}

FailureOr<func::FuncOp> findTaskGraphFunction(ModuleOp module) {
  func::FuncOp graph;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    if (graph)
      return func.emitError("expected one extracted-core task graph");
    graph = func;
  }
  if (!graph)
    return module.emitError("expected one extracted-core task graph");
  if (!graph.getBody().hasOneBlock())
    return graph.emitError("expected task graph to contain one block");
  return graph;
}

template <typename AttrTy>
FailureOr<SmallVector<AttrTy>> getPlanArray(ModuleOp module, StringRef name) {
  auto values = module->getAttrOfType<ArrayAttr>(name);
  if (!values)
    return module.emitError("expected tile memory-plan array '") << name << "'";
  SmallVector<AttrTy> result;
  result.reserve(values.size());
  for (Attribute value : values) {
    auto typed = dyn_cast<AttrTy>(value);
    if (!typed)
      return module.emitError("tile memory-plan array '")
             << name << "' contains an invalid record";
    result.push_back(typed);
  }
  return result;
}

bool isSupportedResource(Value value) {
  auto resourceType = dyn_cast<TaskResourceType>(value.getType());
  auto shaped = resourceType ? dyn_cast<ShapedType>(resourceType.getValueType())
                             : ShapedType();
  return shaped && shaped.hasStaticShape() && shaped.getElementType().isF32();
}

std::optional<Value> getResourceValue(Operation &op) {
  if (auto resource = dyn_cast<TaskGraphInputOp>(&op))
    return resource.getResult();
  if (auto resource = dyn_cast<TaskGraphOutputOp>(&op))
    return resource.getResult();
  if (auto resource = dyn_cast<TaskGraphIntermediateOp>(&op))
    return resource.getResult();
  if (auto resource = dyn_cast<TaskGraphPersistentOp>(&op))
    return resource.getResult();
  if (auto resource = dyn_cast<TaskGraphRouteInputOp>(&op))
    return resource.getResult();
  if (auto resource = dyn_cast<TaskGraphRouteOutputOp>(&op))
    return resource.getResult();
  return std::nullopt;
}

bool resourceMatchesOwner(Operation *resource, MemoryOwnerKind kind) {
  if (kind == MemoryOwnerKind::RouteInput)
    return isa<TaskGraphRouteInputOp>(resource);
  if (kind == MemoryOwnerKind::RouteOutput)
    return isa<TaskGraphRouteOutputOp>(resource);
  if (kind == MemoryOwnerKind::Intermediate ||
      kind == MemoryOwnerKind::Assembly)
    return isa<TaskGraphIntermediateOp>(resource);
  return false;
}

FailureOr<std::optional<Value>> findOwnerResource(func::FuncOp graph,
                                                  TileMemoryOwnerAttr owner) {
  Value match;
  for (Operation &op : graph.getBody().front()) {
    std::optional<Value> value = getResourceValue(op);
    if (!value || !resourceMatchesOwner(&op, owner.getKind()))
      continue;
    auto resourceId = op.getAttrOfType<IntegerAttr>(
        deployment_attrs::kGlobalResourceIdAttrName);
    if (!resourceId || resourceId.getInt() != owner.getResourceId().getInt())
      continue;
    // The deployment memory plan intentionally coalesces resources by
    // (global resource ID, tile).  Materialization can nevertheless retain
    // one route-input resource per incoming route when the same value feeds
    // multiple routines on a tile.  Such resources have distinct runtime
    // slots and DMA triggers, so a single owner-level scratchpad allocation
    // cannot represent them safely.  Leave that owner in workspace until the
    // runtime graph or scratchpad ABI models the per-route resources.
    if (match)
      return std::optional<Value>();
    match = *value;
  }
  if (!match)
    return graph.emitError("scratchpad owner has no local runtime resource");
  return std::optional<Value>(match);
}

void annotateOwnerResourcesAsScratchpad(func::FuncOp graph,
                                        TileMemoryOwnerAttr owner,
                                        uint64_t offset, Builder &builder) {
  for (Operation &op : graph.getBody().front()) {
    std::optional<Value> value = getResourceValue(op);
    if (!value)
      continue;
    auto resourceId = op.getAttrOfType<IntegerAttr>(
        deployment_attrs::kGlobalResourceIdAttrName);
    if (!resourceId || resourceId.getInt() != owner.getResourceId().getInt())
      continue;

    // Materialization may represent one memory owner with both a non-route
    // resource and a route resource.  They are distinct runtime slots, but
    // both slots name the same storage.  Keep their storage annotations in
    // lockstep with the owner-level memory plan.
    op.setAttr(scratchpad_attrs::kStorageClassAttrName,
               builder.getStringAttr(
                   scratchpad_attrs::kScratchpadStorageClass));
    op.setAttr(scratchpad_attrs::kScratchpadOffsetAttrName,
               builder.getI64IntegerAttr(offset));
  }
}

FailureOr<uint64_t> checkedAdd(Operation *anchor, uint64_t left, uint64_t right,
                               StringRef description) {
  if (right > std::numeric_limits<uint64_t>::max() - left)
    return anchor->emitError("scratchpad arithmetic overflow in ")
           << description;
  return left + right;
}

FailureOr<uint64_t> checkedMul(Operation *anchor, uint64_t left, uint64_t right,
                               StringRef description) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
    return anchor->emitError("scratchpad arithmetic overflow in ")
           << description;
  return left * right;
}

FailureOr<uint64_t> alignValue(Operation *anchor, uint64_t value,
                               uint64_t alignment) {
  if (alignment == 0 || !llvm::isPowerOf2_64(alignment))
    return anchor->emitError("invalid scratchpad alignment");
  FailureOr<uint64_t> rounded =
      checkedAdd(anchor, value, alignment - 1, "alignment");
  if (failed(rounded))
    return failure();
  return *rounded & ~(alignment - 1);
}

bool rangesOverlap(uint64_t leftOffset, uint64_t leftSize, uint64_t rightOffset,
                   uint64_t rightSize) {
  return leftOffset < rightOffset + rightSize &&
         rightOffset < leftOffset + leftSize;
}

FailureOr<SmallVector<Candidate>> buildCandidates(ModuleOp module,
                                                  func::FuncOp graph,
                                                  uint64_t alignment,
                                                  bool doubleBufferBoundaries) {
  auto owners =
      getPlanArray<TileMemoryOwnerAttr>(module, tile_memory::kOwnersAttrName);
  auto views =
      getPlanArray<TileMemoryViewAttr>(module, tile_memory::kViewsAttrName);
  auto bindings = getPlanArray<TileMemoryBindingAttr>(
      module, tile_memory::kBindingsAttrName);
  auto movements = getPlanArray<TileMemoryMovementAttr>(
      module, tile_memory::kMovementsAttrName);
  auto lifetimes = getPlanArray<TileMemoryLifetimeAttr>(
      module, tile_memory::kLifetimesAttrName);
  if (failed(owners) || failed(views) || failed(bindings) ||
      failed(movements) || failed(lifetimes))
    return failure();

  std::map<int64_t, TileMemoryLifetimeAttr> lifetimeByOwner;
  for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
    if (lifetime.getSubjectKind() == MemoryLifetimeSubjectKind::Owner)
      lifetimeByOwner[lifetime.getOwnerId().getInt()] = lifetime;
  }
  std::map<int64_t, int64_t> ownerByView;
  for (TileMemoryViewAttr view : *views)
    ownerByView[view.getId().getInt()] = view.getOwnerId().getInt();

  std::map<int64_t, std::pair<uint64_t, uint64_t>> accessCounts;
  for (TileMemoryBindingAttr binding : *bindings) {
    auto &counts = accessCounts[binding.getOwnerId().getInt()];
    switch (binding.getEffect()) {
    case MemoryAccessEffect::Read:
    case MemoryAccessEffect::AsyncTransferSource:
      ++counts.first;
      break;
    case MemoryAccessEffect::Write:
    case MemoryAccessEffect::AsyncTransferDestination:
      ++counts.second;
      break;
    case MemoryAccessEffect::ReadWrite:
      ++counts.first;
      ++counts.second;
      break;
    }
  }
  std::map<int64_t, uint64_t> movementCounts;
  for (TileMemoryMovementAttr movement : *movements) {
    if (auto source = ownerByView.find(movement.getSourceViewId().getInt());
        source != ownerByView.end())
      ++movementCounts[source->second];
    if (auto destination =
            ownerByView.find(movement.getDestinationViewId().getInt());
        destination != ownerByView.end())
      ++movementCounts[destination->second];
  }

  DenseMap<Value, TaskCreateOp> producer;
  for (Operation &op : graph.getBody().front()) {
    auto task = dyn_cast<TaskCreateOp>(&op);
    if (!task)
      continue;
    for (Value output : task.getOutputs())
      producer.try_emplace(output, task);
  }

  SmallVector<TileMemoryOwnerAttr> orderedOwners(*owners);
  llvm::sort(orderedOwners,
             [](TileMemoryOwnerAttr left, TileMemoryOwnerAttr right) {
               return left.getId().getInt() < right.getId().getInt();
             });
  SmallVector<Candidate> candidates;
  for (TileMemoryOwnerAttr owner : orderedOwners) {
    auto lifetime = lifetimeByOwner.find(owner.getId().getInt());
    if (lifetime == lifetimeByOwner.end() ||
        lifetime->second.getStorage() != MemoryLifetimeStorage::Workspace ||
        (owner.getKind() != MemoryOwnerKind::Intermediate &&
         owner.getKind() != MemoryOwnerKind::Assembly &&
         owner.getKind() != MemoryOwnerKind::RouteInput &&
         owner.getKind() != MemoryOwnerKind::RouteOutput) ||
        owner.getByteSize().getInt() <= 0)
      continue;
    FailureOr<std::optional<Value>> resource =
        findOwnerResource(graph, owner);
    if (failed(resource))
      return failure();
    if (!*resource)
      continue;
    Value resourceValue = **resource;
    if (!isSupportedResource(resourceValue))
      continue;
    FailureOr<int64_t> resourceBytes = getTaskResourceByteSize(resourceValue);
    if (failed(resourceBytes) || *resourceBytes != owner.getByteSize().getInt())
      return resourceValue.getDefiningOp()->emitError(
          "scratchpad candidate size disagrees with its memory owner");

    Candidate candidate;
    candidate.id = candidates.size();
    candidate.value = resourceValue;
    candidate.owner = owner;
    candidate.lifetime = lifetime->second;
    candidate.byteSize = static_cast<uint64_t>(owner.getByteSize().getInt());
    candidate.alignment = std::max<uint64_t>(
        alignment,
        static_cast<uint64_t>(lifetime->second.getAlignment().getInt()));
    FailureOr<uint64_t> demand =
        alignValue(module, candidate.byteSize, candidate.alignment);
    if (failed(demand))
      return failure();
    candidate.capacityDemand = *demand;
    bool boundary = owner.getKind() == MemoryOwnerKind::RouteInput ||
                    owner.getKind() == MemoryOwnerKind::RouteOutput;
    if (doubleBufferBoundaries && boundary) {
      FailureOr<uint64_t> doubled = checkedMul(module, candidate.capacityDemand,
                                               2, "double-buffered capacity");
      if (failed(doubled))
        return failure();
      candidate.capacityDemand = *doubled;
    }
    std::tie(candidate.readCount, candidate.writeCount) =
        accessCounts[owner.getId().getInt()];
    candidate.movementCount = movementCounts[owner.getId().getInt()];
    FailureOr<uint64_t> accessesOr =
        checkedAdd(module, candidate.readCount, candidate.writeCount,
                   "candidate access count");
    if (failed(accessesOr))
      return failure();
    uint64_t accesses = *accessesOr;
    candidate.reuseCount = accesses > 0 ? accesses - 1 : 0;
    FailureOr<uint64_t> weightedAccesses =
        checkedAdd(module, std::max<uint64_t>(1, candidate.reuseCount),
                   candidate.movementCount, "candidate access weight");
    if (failed(weightedAccesses))
      return failure();
    FailureOr<uint64_t> savings =
        checkedMul(module, candidate.byteSize, *weightedAccesses,
                   "candidate transfer savings");
    if (failed(savings))
      return failure();
    candidate.estimatedSavings = *savings;

    if (auto produced = producer.find(resourceValue);
        produced != producer.end()) {
      if (auto taskId = produced->second->getAttrOfType<IntegerAttr>(
              deployment_attrs::kGlobalTaskIdAttrName)) {
        if (taskId.getInt() < 0 ||
            static_cast<uint64_t>(taskId.getInt()) > UINT32_MAX)
          return produced->second.emitError(
              "scratchpad producer task ID exceeds the ABI range");
        candidate.producerGlobalTaskId = static_cast<uint32_t>(taskId.getInt());
      }
    }
    if (boundary) {
      auto routeId = resourceValue.getDefiningOp()->getAttrOfType<IntegerAttr>(
          deployment_attrs::kRouteIdAttrName);
      if (!routeId || routeId.getInt() < 0 ||
          static_cast<uint64_t>(routeId.getInt()) > UINT32_MAX)
        return resourceValue.getDefiningOp()->emitError(
            "scratchpad route candidate has no 32-bit route ID");
      candidate.routeId = static_cast<uint32_t>(routeId.getInt());
    }
    candidates.push_back(candidate);
  }
  return candidates;
}

FailureOr<SmallVector<CandidateGroup>>
buildCandidateGroups(ModuleOp module, ArrayRef<Candidate> candidates) {
  auto aliases = getPlanArray<TileMemoryInPlaceAliasAttr>(
      module, tile_memory::kInPlaceAliasesAttrName);
  if (failed(aliases))
    return failure();

  SmallVector<unsigned> parent(candidates.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](unsigned value) {
    unsigned root = value;
    while (parent[root] != root)
      root = parent[root];
    while (parent[value] != value) {
      unsigned next = parent[value];
      parent[value] = root;
      value = next;
    }
    return root;
  };
  auto unite = [&](unsigned left, unsigned right) {
    left = find(left);
    right = find(right);
    if (left == right)
      return;
    if (left > right)
      std::swap(left, right);
    parent[right] = left;
  };

  std::map<int64_t, unsigned> candidateByOwner;
  for (auto indexed : llvm::enumerate(candidates))
    candidateByOwner[indexed.value().owner.getId().getInt()] = indexed.index();
  for (TileMemoryInPlaceAliasAttr alias : *aliases) {
    auto input = candidateByOwner.find(alias.getInputOwnerId().getInt());
    auto output = candidateByOwner.find(alias.getOutputOwnerId().getInt());
    if (input != candidateByOwner.end() && output != candidateByOwner.end())
      unite(input->second, output->second);
  }

  std::map<unsigned, CandidateGroup> groupByRoot;
  for (auto indexed : llvm::enumerate(candidates)) {
    const Candidate &candidate = indexed.value();
    CandidateGroup &group = groupByRoot[find(indexed.index())];
    group.members.push_back(indexed.index());
    group.byteSize = std::max(group.byteSize, candidate.byteSize);
    group.alignment = std::max(group.alignment, candidate.alignment);
    group.capacityDemand =
        std::max(group.capacityDemand, candidate.capacityDemand);
    FailureOr<uint64_t> benefit =
        checkedAdd(module, group.estimatedSavings, candidate.estimatedSavings,
                   "candidate-group benefit");
    if (failed(benefit))
      return failure();
    group.estimatedSavings = *benefit;
  }
  SmallVector<CandidateGroup> groups;
  for (auto &[root, group] : groupByRoot) {
    llvm::sort(group.members);
    group.key = candidates[group.members.front()].id;
    groups.push_back(group);
    (void)root;
  }
  return groups;
}

FailureOr<SmallVector<SelectedGroup>>
selectGroups(ModuleOp module, ArrayRef<Candidate> candidates,
             SmallVector<CandidateGroup> groups, uint64_t capacity) {
  auto relations = getPlanArray<TileMemoryInterferenceAttr>(
      module, tile_memory::kInterferencesAttrName);
  if (failed(relations))
    return failure();
  MemoryInterferenceRelation defaultRelation =
      MemoryInterferenceRelation::Interferes;
  if (auto value = module->getAttrOfType<StringAttr>(
          tile_memory::kInterferenceDefaultAttrName)) {
    auto parsed = symbolizeMemoryInterferenceRelation(value.getValue());
    if (!parsed)
      return module.emitError("invalid tile memory interference default");
    defaultRelation = *parsed;
  }
  std::map<std::pair<int64_t, int64_t>, MemoryInterferenceRelation> relation;
  for (TileMemoryInterferenceAttr item : *relations)
    relation[{item.getLeftLifetimeId().getInt(),
              item.getRightLifetimeId().getInt()}] = item.getRelation();
  if (auto exceptions = module->getAttrOfType<DenseI64ArrayAttr>(
          tile_memory::kInterferenceExceptionsAttrName)) {
    ArrayRef<int64_t> values = exceptions.asArrayRef();
    if (values.size() % 3 != 0)
      return module.emitError(
          "tile memory interference exceptions must contain triples");
    for (size_t index = 0; index < values.size(); index += 3) {
      auto parsed = symbolizeMemoryInterferenceRelation(
          static_cast<uint32_t>(values[index + 2]));
      if (!parsed)
        return module.emitError(
            "tile memory interference exception has an invalid relation");
      relation[{values[index], values[index + 1]}] = *parsed;
    }
  }

  auto canShare = [&](unsigned left, unsigned right) {
    int64_t leftLifetime = candidates[left].lifetime.getId().getInt();
    int64_t rightLifetime = candidates[right].lifetime.getId().getInt();
    if (leftLifetime == rightLifetime)
      return true;
    auto found = relation.find(std::minmax(leftLifetime, rightLifetime));
    MemoryInterferenceRelation resolved =
        found == relation.end() ? defaultRelation : found->second;
    return resolved == MemoryInterferenceRelation::Before ||
           resolved == MemoryInterferenceRelation::After ||
           resolved == MemoryInterferenceRelation::InPlaceAlias;
  };
  auto groupsConflict = [&](const CandidateGroup &left,
                            const CandidateGroup &right) {
    return llvm::any_of(left.members, [&](unsigned leftMember) {
      return llvm::any_of(right.members, [&](unsigned rightMember) {
        return !canShare(leftMember, rightMember);
      });
    });
  };

  llvm::sort(groups,
             [](const CandidateGroup &left, const CandidateGroup &right) {
               unsigned __int128 leftDensity =
                   static_cast<unsigned __int128>(left.estimatedSavings) *
                   right.capacityDemand;
               unsigned __int128 rightDensity =
                   static_cast<unsigned __int128>(right.estimatedSavings) *
                   left.capacityDemand;
               if (leftDensity != rightDensity)
                 return leftDensity > rightDensity;
               if (left.estimatedSavings != right.estimatedSavings)
                 return left.estimatedSavings > right.estimatedSavings;
               return left.key < right.key;
             });

  SmallVector<SelectedGroup> selected;
  for (const CandidateGroup &group : groups) {
    uint64_t offset = 0;
    bool placed = false;
    while (offset <= capacity) {
      FailureOr<uint64_t> aligned = alignValue(module, offset, group.alignment);
      if (failed(aligned))
        return failure();
      offset = *aligned;
      if (offset > capacity || group.capacityDemand > capacity - offset)
        break;
      bool conflict = false;
      uint64_t nextOffset = offset + group.alignment;
      for (const SelectedGroup &existing : selected) {
        if (!groupsConflict(group, existing.group) ||
            !rangesOverlap(offset, group.capacityDemand, existing.offset,
                           existing.group.capacityDemand))
          continue;
        conflict = true;
        nextOffset = std::max(nextOffset,
                              existing.offset + existing.group.capacityDemand);
      }
      if (!conflict) {
        selected.push_back(SelectedGroup{group, offset});
        placed = true;
        break;
      }
      offset = nextOffset;
    }
    (void)placed;
  }
  llvm::sort(selected,
             [](const SelectedGroup &left, const SelectedGroup &right) {
               return left.group.key < right.group.key;
             });
  return selected;
}

DictionaryAttr buildDMADescriptor(Builder &builder, const Candidate &candidate,
                                  uint32_t id, uint32_t completionToken) {
  Operation *resource = candidate.value.getDefiningOp();
  uint32_t direction = 0;
  uint32_t sourceStorage = 0;
  uint32_t destinationStorage = 0;
  uint32_t triggerKind = 0;
  uint32_t triggerId = scratchpad_attrs::kInvalidU32;
  if (isa<TaskGraphRouteInputOp>(resource)) {
    direction = scratchpad_attrs::kDirectionNicToScratchpad;
    sourceStorage = scratchpad_attrs::kStorageNic;
    destinationStorage = scratchpad_attrs::kStorageScratchpad;
    triggerKind = scratchpad_attrs::kTriggerRouteArrival;
    triggerId = *candidate.routeId;
  } else {
    direction = scratchpad_attrs::kDirectionScratchpadToNic;
    sourceStorage = scratchpad_attrs::kStorageScratchpad;
    destinationStorage = scratchpad_attrs::kStorageNic;
    triggerKind = scratchpad_attrs::kTriggerTaskComplete;
    triggerId =
        candidate.producerGlobalTaskId.value_or(scratchpad_attrs::kInvalidU32);
  }
  auto globalId = resource->getAttrOfType<IntegerAttr>(
      deployment_attrs::kGlobalResourceIdAttrName);
  return builder.getDictionaryAttr({
      builder.getNamedAttr(scratchpad_attrs::kDMAIdFieldName,
                           builder.getI32IntegerAttr(id)),
      builder.getNamedAttr(scratchpad_attrs::kDMADirectionFieldName,
                           builder.getI32IntegerAttr(direction)),
      builder.getNamedAttr(scratchpad_attrs::kDMAGlobalResourceIdFieldName,
                           globalId),
      builder.getNamedAttr(scratchpad_attrs::kDMARouteIdFieldName,
                           builder.getI64IntegerAttr(*candidate.routeId)),
      builder.getNamedAttr(
          scratchpad_attrs::kDMAScratchpadOffsetFieldName,
          builder.getI64IntegerAttr(candidate.lifetime.getOffset().getInt())),
      builder.getNamedAttr(scratchpad_attrs::kDMAByteSizeFieldName,
                           builder.getI64IntegerAttr(candidate.byteSize)),
      builder.getNamedAttr(scratchpad_attrs::kDMACompletionTokenFieldName,
                           builder.getI32IntegerAttr(completionToken)),
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

LogicalResult
updateMemoryPlan(ModuleOp module, ArrayRef<Candidate> candidates,
                 const std::map<int64_t, uint64_t> &offsetByOwner,
                 std::map<int64_t, uint32_t> &dmaCompletionByOwner) {
  auto completions = getPlanArray<TileMemoryCompletionEventAttr>(
      module, tile_memory::kCompletionEventsAttrName);
  auto edges = getPlanArray<TileMemoryEventEdgeAttr>(
      module, tile_memory::kEventEdgesAttrName);
  auto lifetimes = getPlanArray<TileMemoryLifetimeAttr>(
      module, tile_memory::kLifetimesAttrName);
  if (failed(completions) || failed(edges) || failed(lifetimes))
    return failure();

  Builder builder(module.getContext());
  int64_t nextCompletionId = 0;
  int64_t nextEdgeId = 0;
  std::map<int64_t, int64_t> releaseByOwner;
  for (TileMemoryCompletionEventAttr completion : *completions) {
    nextCompletionId =
        std::max(nextCompletionId, completion.getId().getInt() + 1);
    if (completion.getKind() == MemoryCompletionKind::OwnerRelease ||
        completion.getKind() == MemoryCompletionKind::ScratchpadRelease)
      releaseByOwner[completion.getOwnerId().getInt()] =
          completion.getId().getInt();
  }
  for (TileMemoryEventEdgeAttr edge : *edges)
    nextEdgeId = std::max(nextEdgeId, edge.getId().getInt() + 1);

  std::set<int64_t> selectedOwners;
  for (const auto &[owner, offset] : offsetByOwner) {
    selectedOwners.insert(owner);
    (void)offset;
  }

  SmallVector<Attribute> updatedCompletions;
  updatedCompletions.reserve(completions->size() + candidates.size());
  for (TileMemoryCompletionEventAttr completion : *completions) {
    auto kind = completion.getKind();
    if ((kind == MemoryCompletionKind::OwnerRelease ||
         kind == MemoryCompletionKind::ScratchpadRelease) &&
        completion.getOwnerId().getInt() >= 0)
      kind = selectedOwners.count(completion.getOwnerId().getInt())
                 ? MemoryCompletionKind::ScratchpadRelease
                 : MemoryCompletionKind::OwnerRelease;
    updatedCompletions.push_back(TileMemoryCompletionEventAttr::get(
        module.getContext(), completion.getId(), kind, completion.getTile(),
        completion.getRoutine(), completion.getRouteId(),
        completion.getOwnerId(), completion.getViewId()));
  }

  SmallVector<Attribute> updatedEdges;
  updatedEdges.reserve(edges->size() + candidates.size() * 2);
  for (TileMemoryEventEdgeAttr edge : *edges) {
    auto kind = edge.getKind();
    for (const auto &[owner, release] : releaseByOwner) {
      if ((edge.getSourceEventId().getInt() == release ||
           edge.getTargetEventId().getInt() == release) &&
          (kind == MemoryEventEdgeKind::LifetimeRelease ||
           kind == MemoryEventEdgeKind::ScratchpadRelease)) {
        kind = selectedOwners.count(owner)
                   ? MemoryEventEdgeKind::ScratchpadRelease
                   : MemoryEventEdgeKind::LifetimeRelease;
        break;
      }
    }
    updatedEdges.push_back(TileMemoryEventEdgeAttr::get(
        module.getContext(), edge.getId(), edge.getSourceEventId(),
        edge.getTargetEventId(), kind));
  }

  std::map<int64_t, SmallVector<int64_t>> boundaryEventsByOwner;
  for (TileMemoryCompletionEventAttr completion : *completions) {
    int64_t owner = completion.getOwnerId().getInt();
    if (!selectedOwners.count(owner))
      continue;
    if (completion.getKind() == MemoryCompletionKind::RouteArrival ||
        completion.getKind() == MemoryCompletionKind::RouteSendComplete)
      boundaryEventsByOwner[owner].push_back(completion.getId().getInt());
  }
  for (const Candidate &candidate : candidates) {
    int64_t owner = candidate.owner.getId().getInt();
    if (!selectedOwners.count(owner) || !candidate.routeId)
      continue;
    auto boundary = boundaryEventsByOwner.find(owner);
    if (boundary == boundaryEventsByOwner.end() || boundary->second.empty())
      return module.emitError("scratchpad route owner has no completion event");
    auto release = releaseByOwner.find(owner);
    if (release == releaseByOwner.end())
      return module.emitError("scratchpad owner has no release event");

    int64_t dmaEvent = nextCompletionId++;
    if (static_cast<uint64_t>(dmaEvent) > UINT32_MAX)
      return module.emitError(
          "scratchpad DMA completion event exceeds the ABI range");
    updatedCompletions.push_back(TileMemoryCompletionEventAttr::get(
        module.getContext(), builder.getI64IntegerAttr(dmaEvent),
        MemoryCompletionKind::DMAComplete, candidate.owner.getTile(),
        candidate.owner.getRoutine(),
        builder.getI64IntegerAttr(*candidate.routeId), candidate.owner.getId(),
        builder.getI64IntegerAttr(-1)));
    for (int64_t predecessor : boundary->second) {
      updatedEdges.push_back(TileMemoryEventEdgeAttr::get(
          module.getContext(), builder.getI64IntegerAttr(nextEdgeId++),
          builder.getI64IntegerAttr(predecessor),
          builder.getI64IntegerAttr(dmaEvent),
          MemoryEventEdgeKind::DMACompletion));
    }
    updatedEdges.push_back(TileMemoryEventEdgeAttr::get(
        module.getContext(), builder.getI64IntegerAttr(nextEdgeId++),
        builder.getI64IntegerAttr(dmaEvent),
        builder.getI64IntegerAttr(release->second),
        MemoryEventEdgeKind::ScratchpadRelease));
    dmaCompletionByOwner[owner] = static_cast<uint32_t>(dmaEvent);
  }

  std::map<int64_t, int64_t> lifetimeByOwner;
  SmallVector<Attribute> updatedLifetimes;
  updatedLifetimes.reserve(lifetimes->size());
  for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
    if (lifetime.getSubjectKind() != MemoryLifetimeSubjectKind::Owner) {
      updatedLifetimes.push_back(lifetime);
      continue;
    }
    int64_t owner = lifetime.getOwnerId().getInt();
    lifetimeByOwner[owner] = lifetime.getId().getInt();
    auto selected = offsetByOwner.find(owner);
    MemoryLifetimeStorage storage = lifetime.getStorage();
    int64_t offset = lifetime.getOffset().getInt();
    if (selected != offsetByOwner.end()) {
      storage = MemoryLifetimeStorage::Scratchpad;
      offset = static_cast<int64_t>(selected->second);
    } else if (storage == MemoryLifetimeStorage::Scratchpad) {
      storage = MemoryLifetimeStorage::Workspace;
      offset = -1;
    }
    SmallVector<int64_t> accessEvents;
    for (Attribute item : lifetime.getAccessEventIds())
      accessEvents.push_back(cast<IntegerAttr>(item).getInt());
    if (auto dma = dmaCompletionByOwner.find(owner);
        dma != dmaCompletionByOwner.end())
      accessEvents.push_back(dma->second);
    llvm::sort(accessEvents);
    accessEvents.erase(std::unique(accessEvents.begin(), accessEvents.end()),
                       accessEvents.end());
    SmallVector<Attribute> accessAttrs;
    for (int64_t event : accessEvents)
      accessAttrs.push_back(builder.getI64IntegerAttr(event));
    updatedLifetimes.push_back(TileMemoryLifetimeAttr::get(
        module.getContext(), lifetime.getId(), lifetime.getSubjectKind(),
        storage, lifetime.getOwnerId(), lifetime.getRoutine(),
        lifetime.getAllocationOrdinal(), lifetime.getTile(),
        lifetime.getByteSize(), lifetime.getAlignment(),
        builder.getI64IntegerAttr(offset), lifetime.getViewIds(),
        builder.getArrayAttr(accessAttrs)));
  }

  module->setAttr(tile_memory::kCompletionEventsAttrName,
                  builder.getArrayAttr(updatedCompletions));
  module->setAttr(tile_memory::kEventEdgesAttrName,
                  builder.getArrayAttr(updatedEdges));
  module->setAttr(tile_memory::kLifetimesAttrName,
                  builder.getArrayAttr(updatedLifetimes));
  return success();
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
  if (bytes <= 0 || bytes > kPlatformMaximumScratchpadBytes) {
    module.emitError(
        "scratchpad capacity must be between one byte and the Platform v0.2 "
        "16 MiB window");
    return signalPassFailure();
  }
  if (alignment <= 0 || !llvm::isPowerOf2_64(alignment) || alignment > bytes) {
    module.emitError(
        "scratchpad alignment must be a positive power of two within capacity");
    return signalPassFailure();
  }
  if (!module->hasAttr(tile_runtime_attrs::kTaskCoreIdAttrName)) {
    module.emitError("scratchpad planning requires an extracted core module");
    return signalPassFailure();
  }
  if (failed(tile_memory::verifyTileMemoryPlan(module)))
    return signalPassFailure();

  auto graphOr = findTaskGraphFunction(module);
  if (failed(graphOr))
    return signalPassFailure();
  func::FuncOp graph = *graphOr;
  for (Operation &op : graph.getBody().front()) {
    if (op.hasAttr(tile_runtime_attrs::kResourceSlotAttrName)) {
      op.emitError("scratchpad planning must run before task graph resource "
                   "finalization");
      return signalPassFailure();
    }
    op.removeAttr(scratchpad_attrs::kStorageClassAttrName);
    op.removeAttr(scratchpad_attrs::kScratchpadOffsetAttrName);
  }

  FailureOr<SmallVector<Candidate>> candidates =
      buildCandidates(module, graph, alignment, doubleBufferBoundaries);
  if (failed(candidates))
    return signalPassFailure();
  FailureOr<SmallVector<CandidateGroup>> groups =
      buildCandidateGroups(module, *candidates);
  if (failed(groups))
    return signalPassFailure();
  FailureOr<SmallVector<SelectedGroup>> selected = selectGroups(
      module, *candidates, std::move(*groups), static_cast<uint64_t>(bytes));
  if (failed(selected))
    return signalPassFailure();

  std::map<int64_t, uint64_t> offsetByOwner;
  uint64_t requiredBytes = 0;
  uint64_t estimatedSavings = 0;
  for (const SelectedGroup &selection : *selected) {
    requiredBytes = std::max(requiredBytes,
                             selection.offset + selection.group.capacityDemand);
    FailureOr<uint64_t> benefit =
        checkedAdd(module, estimatedSavings, selection.group.estimatedSavings,
                   "selected scratchpad benefit");
    if (failed(benefit))
      return signalPassFailure();
    estimatedSavings = *benefit;
    for (unsigned member : selection.group.members)
      offsetByOwner[(*candidates)[member].owner.getId().getInt()] =
          selection.offset;
  }

  std::map<int64_t, uint32_t> dmaCompletionByOwner;
  if (failed(updateMemoryPlan(module, *candidates, offsetByOwner,
                              dmaCompletionByOwner)))
    return signalPassFailure();

  Builder builder(module.getContext());
  SmallVector<Attribute> candidateAttrs;
  SmallVector<Attribute> allocationAttrs;
  SmallVector<Attribute> descriptors;
  uint32_t descriptorId = 0;
  for (Candidate &candidate : *candidates) {
    if (candidate.byteSize > INT64_MAX || candidate.alignment > INT64_MAX ||
        candidate.capacityDemand > INT64_MAX ||
        candidate.readCount > INT64_MAX || candidate.writeCount > INT64_MAX ||
        candidate.movementCount > INT64_MAX ||
        candidate.reuseCount > INT64_MAX ||
        candidate.estimatedSavings > INT64_MAX) {
      module.emitError(
          "scratchpad candidate metadata exceeds the signed IR range");
      return signalPassFailure();
    }
    int64_t ownerId = candidate.owner.getId().getInt();
    auto selectedOffset = offsetByOwner.find(ownerId);
    bool isSelected = selectedOffset != offsetByOwner.end();
    candidateAttrs.push_back(TileScratchpadCandidateAttr::get(
        module.getContext(), builder.getI64IntegerAttr(candidate.id),
        candidate.owner.getId(), candidate.lifetime.getId(),
        builder.getI64IntegerAttr(candidate.byteSize),
        builder.getI64IntegerAttr(candidate.alignment),
        builder.getI64IntegerAttr(candidate.capacityDemand),
        builder.getI64IntegerAttr(candidate.readCount),
        builder.getI64IntegerAttr(candidate.writeCount),
        builder.getI64IntegerAttr(candidate.movementCount),
        builder.getI64IntegerAttr(candidate.reuseCount),
        builder.getI64IntegerAttr(candidate.estimatedSavings),
        builder.getBoolAttr(isSelected)));
    if (!isSelected)
      continue;
    annotateOwnerResourcesAsScratchpad(graph, candidate.owner,
                                       selectedOffset->second, builder);
    candidate.lifetime = TileMemoryLifetimeAttr::get(
        module.getContext(), candidate.lifetime.getId(),
        candidate.lifetime.getSubjectKind(), MemoryLifetimeStorage::Scratchpad,
        candidate.lifetime.getOwnerId(), candidate.lifetime.getRoutine(),
        candidate.lifetime.getAllocationOrdinal(), candidate.lifetime.getTile(),
        candidate.lifetime.getByteSize(), candidate.lifetime.getAlignment(),
        builder.getI64IntegerAttr(selectedOffset->second),
        candidate.lifetime.getViewIds(),
        candidate.lifetime.getAccessEventIds());
    allocationAttrs.push_back(TileScratchpadAllocationAttr::get(
        module.getContext(), builder.getI64IntegerAttr(candidate.id),
        candidate.owner.getId(), candidate.lifetime.getId(),
        builder.getI64IntegerAttr(selectedOffset->second),
        builder.getI64IntegerAttr(candidate.byteSize),
        builder.getI64IntegerAttr(candidate.capacityDemand),
        builder.getI64IntegerAttr(candidate.alignment),
        builder.getI64IntegerAttr(candidate.estimatedSavings)));
    if (candidate.routeId) {
      if (candidate.owner.getKind() == MemoryOwnerKind::RouteOutput &&
          !candidate.producerGlobalTaskId) {
        candidate.value.getDefiningOp()->emitError(
            "scratchpad route output requires a producer global task ID");
        return signalPassFailure();
      }
      auto dma = dmaCompletionByOwner.find(ownerId);
      if (dma == dmaCompletionByOwner.end()) {
        module.emitError("scratchpad route has no DMA completion event");
        return signalPassFailure();
      }
      descriptors.push_back(
          buildDMADescriptor(builder, candidate, descriptorId++, dma->second));
    }
  }

  module->setAttr(scratchpad_attrs::kScratchpadCandidatesAttrName,
                  builder.getArrayAttr(candidateAttrs));
  module->setAttr(scratchpad_attrs::kScratchpadAllocationsAttrName,
                  builder.getArrayAttr(allocationAttrs));
  module->setAttr(scratchpad_attrs::kScratchpadEstimatedSavingsAttrName,
                  builder.getI64IntegerAttr(estimatedSavings));
  module->setAttr(scratchpad_attrs::kScratchpadCapacityAttrName,
                  builder.getI64IntegerAttr(bytes));
  graph->setAttr(scratchpad_attrs::kScratchpadRequiredBytesAttrName,
                 builder.getI64IntegerAttr(requiredBytes));
  graph->setAttr(scratchpad_attrs::kScratchpadAlignmentAttrName,
                 builder.getI64IntegerAttr(alignment));
  graph->setAttr(scratchpad_attrs::kScratchpadDMADescriptorsAttrName,
                 builder.getArrayAttr(descriptors));
  graph->setAttr(scratchpad_attrs::kScratchpadABIVersionAttrName,
                 builder.getI32IntegerAttr(2));
  graph->setAttr(scratchpad_attrs::kScratchpadFeatureBitsAttrName,
                 builder.getI32IntegerAttr(1));

  if (failed(tile_memory::rebuildTileMemoryInterference(module)) ||
      failed(tile_memory::rebuildTileMemoryCapacitySummary(module)) ||
      failed(tile_memory::verifyTileMemoryPlan(module)))
    return signalPassFailure();
}

void registerPlanTileScratchpadPass() {
  PassRegistration<PlanTileScratchpadPass>();
}

} // namespace mlir::sculptor
