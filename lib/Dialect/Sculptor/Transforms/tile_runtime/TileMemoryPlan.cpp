#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::tile_memory;

constexpr StringLiteral kPhysicalTileIdAttr =
    "sculptor.deployment.physical_tile_id";
constexpr StringLiteral kGlobalRoutineIdAttr =
    "sculptor.deployment.global_routine_id";
constexpr StringLiteral kRoutineKindAttr = "sculptor.deployment.routine_kind";
constexpr StringLiteral kLocalRoutineIndexAttr =
    "sculptor.deployment.local_routine_index";
constexpr StringLiteral kInputResourceIdsAttr =
    "sculptor.deployment.input_resource_ids";
constexpr StringLiteral kOutputResourceIdsAttr =
    "sculptor.deployment.output_resource_ids";
constexpr StringLiteral kControlDependencyIdsAttr =
    "sculptor.deployment.control_dependency_ids";
constexpr StringLiteral kRoutesAttr = "sculptor.deployment.routes";
constexpr StringLiteral kLocalBindingsAttr =
    "sculptor.deployment.local_bindings";
constexpr StringLiteral kModelInputsAttr = "sculptor.deployment.model_inputs";
constexpr StringLiteral kModelOutputsAttr = "sculptor.deployment.model_outputs";
constexpr int64_t kMaxStaticSegmentCount = 4096;
constexpr size_t kMaxExplicitInterferencePairCount = 4096;
constexpr size_t kMaxExactInterferencePairCount = 1'000'000;

using OwnerKey = std::pair<int64_t, int64_t>;
using PortKey = std::pair<int64_t, int64_t>;

struct RoutineInfo {
  int64_t id = -1;
  int64_t tile = -1;
  func::FuncOp function;
  SmallVector<int64_t> inputResources;
  SmallVector<int64_t> outputResources;
  SmallVector<int64_t> controlDependencies;
  bool boot = false;
  int64_t localIndex = -1;
};

struct OwnerRecord {
  int64_t id = -1;
  int64_t resourceId = -1;
  int64_t tensorId = -1;
  int64_t tile = -1;
  int64_t routine = -1;
  int64_t port = -1;
  MemoryOwnerKind kind = MemoryOwnerKind::Intermediate;
  int64_t byteSize = -1;
  Type type;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
};

struct ViewRecord {
  int64_t id = -1;
  int64_t ownerId = -1;
  int64_t byteOffset = 0;
  int64_t byteSize = 0;
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  MemoryContiguity contiguity = MemoryContiguity::Contiguous;
};

struct BindingRecord {
  int64_t id = -1;
  int64_t routine = -1;
  int64_t port = -1;
  bool input = false;
  int64_t ownerId = -1;
  int64_t viewId = -1;
  MemoryAccessEffect effect = MemoryAccessEffect::Read;
};

struct MovementRecord {
  int64_t id = -1;
  MemoryMovementMode mode = MemoryMovementMode::LocalAlias;
  int64_t sourceViewId = -1;
  int64_t destinationViewId = -1;
  int64_t routeId = -1;
  int64_t sourceCompletionEventId = -1;
  int64_t destinationCompletionEventId = -1;
  int64_t assemblyContributionEventId = -1;
  int64_t assemblyId = -1;
  int64_t byteSize = 0;
};

struct SegmentRecord {
  int64_t id = -1;
  int64_t movementId = -1;
  int64_t ordinal = -1;
  int64_t sourceByteOffset = 0;
  int64_t destinationByteOffset = 0;
  int64_t byteSize = 0;
};

struct AssemblyRecord {
  int64_t id = -1;
  int64_t ownerId = -1;
  SmallVector<int64_t> contributingViewIds;
  SmallVector<int64_t> destinationViewIds;
  SmallVector<int64_t> completionEventIds;
  int64_t readinessEventId = -1;
  int64_t localCopyBytes = 0;
  int64_t routedBytes = 0;
};

struct CompletionRecord {
  int64_t id = -1;
  MemoryCompletionKind kind = MemoryCompletionKind::RoutineComplete;
  int64_t tile = -1;
  int64_t routine = -1;
  int64_t routeId = -1;
  int64_t ownerId = -1;
  int64_t viewId = -1;
};

struct EventEdgeRecord {
  int64_t id = -1;
  int64_t sourceEventId = -1;
  int64_t targetEventId = -1;
  MemoryEventEdgeKind kind = MemoryEventEdgeKind::RoutineExecution;
};

struct LifetimeRecord {
  int64_t id = -1;
  MemoryLifetimeSubjectKind subjectKind = MemoryLifetimeSubjectKind::Owner;
  MemoryLifetimeStorage storage = MemoryLifetimeStorage::Workspace;
  int64_t ownerId = -1;
  int64_t routine = -1;
  int64_t allocationOrdinal = -1;
  int64_t tile = -1;
  int64_t byteSize = 0;
  int64_t alignment = 16;
  int64_t offset = -1;
  SmallVector<int64_t> viewIds;
  SmallVector<int64_t> accessEventIds;
};

struct InPlaceAliasRecord {
  int64_t id = -1;
  int64_t routine = -1;
  int64_t inputPort = -1;
  int64_t outputPort = -1;
  int64_t inputOwnerId = -1;
  int64_t outputOwnerId = -1;
  int64_t inputViewId = -1;
  int64_t outputViewId = -1;
};

struct PlanRecords {
  std::map<int64_t, RoutineInfo> routines;
  SmallVector<TileRoutineRouteAttr> routes;
  SmallVector<TileRoutineBindingAttr> localBindings;
  SmallVector<TileRoutineModelIOAttr> modelInputs;
  SmallVector<TileRoutineModelIOAttr> modelOutputs;
  SmallVector<OwnerRecord> owners;
  SmallVector<ViewRecord> views;
  SmallVector<BindingRecord> bindings;
  SmallVector<MovementRecord> movements;
  SmallVector<SegmentRecord> segments;
  SmallVector<AssemblyRecord> assemblies;
  SmallVector<CompletionRecord> completions;
  SmallVector<EventEdgeRecord> eventEdges;
  SmallVector<LifetimeRecord> lifetimes;
  SmallVector<InPlaceAliasRecord> inPlaceAliases;
  std::map<OwnerKey, int64_t> ownerByKey;
  std::map<int64_t, int64_t> fullViewByOwner;
  std::map<PortKey, int64_t> inputBindingByPort;
  std::map<PortKey, int64_t> outputBindingByPort;
  std::map<int64_t, int64_t> routineStartEvent;
  std::map<int64_t, int64_t> routineCompleteEvent;
  std::map<int64_t, int64_t> routeSendEvent;
  std::map<int64_t, int64_t> routeArrivalEvent;
  std::map<int64_t, int64_t> finalFanOutEvent;
  std::map<int64_t, int64_t> finalConsumerEvent;
  std::map<int64_t, int64_t> ownerReleaseEvent;
  DenseMap<Value, int64_t> localTemporaryView;
  std::set<std::tuple<int64_t, int64_t, MemoryEventEdgeKind>> eventEdgeKeys;
};

/// Dense transitive-closure index for the tile-memory event DAG.
///
/// Event IDs are part of the serialized ABI and are not assumed to be dense.
/// Internally, each ID is assigned a dense index so reachability is represented
/// by compact bit vectors instead of individually allocated std::set nodes.
class EventReachability {
public:
  LogicalResult initialize(ArrayRef<int64_t> eventIds,
                           ArrayRef<std::pair<int64_t, int64_t>> edges,
                           Operation *anchor) {
    ids.clear();
    indexById.clear();
    successors.clear();
    predecessors.clear();
    reachable.clear();
    topologicalRanks.clear();

    ids.reserve(eventIds.size());
    successors.resize(eventIds.size());
    predecessors.resize(eventIds.size());
    for (int64_t eventId : eventIds) {
      if (eventId < 0 || indexById.count(eventId))
        return anchor->emitError(
            "cannot analyze duplicate or invalid memory event IDs");
      unsigned index = ids.size();
      ids.push_back(eventId);
      indexById[eventId] = index;
    }

    SmallVector<unsigned> indegrees(eventIds.size(), 0);
    for (auto [sourceId, targetId] : edges) {
      auto source = indexById.find(sourceId);
      auto target = indexById.find(targetId);
      if (source == indexById.end() || target == indexById.end())
        return anchor->emitError(
            "cannot analyze interference with an unknown event edge");
      successors[source->second].push_back(target->second);
      predecessors[target->second].push_back(source->second);
      ++indegrees[target->second];
    }

    SmallVector<unsigned> ready;
    ready.reserve(eventIds.size());
    for (unsigned index = 0; index < indegrees.size(); ++index)
      if (indegrees[index] == 0)
        ready.push_back(index);

    SmallVector<unsigned> topologicalOrder;
    topologicalOrder.reserve(eventIds.size());
    while (!ready.empty()) {
      unsigned event = ready.pop_back_val();
      topologicalOrder.push_back(event);
      for (unsigned successor : successors[event])
        if (--indegrees[successor] == 0)
          ready.push_back(successor);
    }
    if (topologicalOrder.size() != eventIds.size())
      return anchor->emitError("tile memory event graph contains a cycle");

    topologicalRanks.resize(eventIds.size());
    for (auto [rank, event] : llvm::enumerate(topologicalOrder))
      topologicalRanks[event] = rank;

    reachable.reserve(eventIds.size());
    for (size_t index = 0; index < eventIds.size(); ++index)
      reachable.emplace_back(eventIds.size(), false);
    for (unsigned source : llvm::reverse(topologicalOrder)) {
      for (unsigned successor : successors[source]) {
        reachable[source].set(successor);
        reachable[source] |= reachable[successor];
      }
    }
    return success();
  }

  bool contains(int64_t eventId) const { return indexById.count(eventId); }

  std::optional<unsigned> topologicalRank(int64_t eventId) const {
    auto event = indexById.find(eventId);
    if (event == indexById.end())
      return std::nullopt;
    return topologicalRanks[event->second];
  }

  bool reaches(int64_t sourceId, int64_t targetId) const {
    auto source = indexById.find(sourceId);
    auto target = indexById.find(targetId);
    return source != indexById.end() && target != indexById.end() &&
           reachable[source->second].test(target->second);
  }

  llvm::BitVector reachableFromAll(ArrayRef<int64_t> sourceIds) const {
    if (sourceIds.empty())
      return llvm::BitVector(ids.size(), false);
    auto first = indexById.find(sourceIds.front());
    if (first == indexById.end())
      return llvm::BitVector(ids.size(), false);
    llvm::BitVector common = reachable[first->second];
    for (int64_t sourceId : sourceIds.drop_front()) {
      auto source = indexById.find(sourceId);
      if (source == indexById.end()) {
        common.reset();
        break;
      }
      common &= reachable[source->second];
    }
    return common;
  }

  bool maskContains(const llvm::BitVector &mask, int64_t eventId) const {
    auto event = indexById.find(eventId);
    return event != indexById.end() && mask.test(event->second);
  }

  SmallVector<int64_t>
  findMinimalCommonSuccessors(ArrayRef<int64_t> predecessorIds) const {
    llvm::BitVector common = reachableFromAll(predecessorIds);
    SmallVector<int64_t> minimal;
    for (unsigned candidate = 0; candidate < ids.size(); ++candidate) {
      if (!common.test(candidate))
        continue;
      // If any direct predecessor is common, the candidate cannot be a
      // minimal common successor. Conversely, a common ancestor implies that
      // the final edge on its path to the candidate is also common.
      bool hasEarlierCommon =
          llvm::any_of(predecessors[candidate], [&](unsigned predecessor) {
            return common.test(predecessor);
          });
      if (!hasEarlierCommon)
        minimal.push_back(ids[candidate]);
    }
    llvm::sort(minimal);
    return minimal;
  }

  /// Incrementally insert a join between predecessor and successor events.
  /// Existing event-to-event reachability does not change because every
  /// successor is already reachable from every predecessor.
  LogicalResult insertJoin(int64_t eventId, ArrayRef<int64_t> predecessorIds,
                           ArrayRef<int64_t> successorIds, Operation *anchor) {
    if (eventId < 0 || indexById.count(eventId))
      return anchor->emitError("cannot insert a duplicate memory join event");

    SmallVector<unsigned> predecessorIndices;
    SmallVector<unsigned> successorIndices;
    predecessorIndices.reserve(predecessorIds.size());
    successorIndices.reserve(successorIds.size());
    for (int64_t predecessorId : predecessorIds) {
      auto predecessor = indexById.find(predecessorId);
      if (predecessor == indexById.end())
        return anchor->emitError(
            "memory join references an unknown predecessor event");
      predecessorIndices.push_back(predecessor->second);
    }
    for (int64_t successorId : successorIds) {
      auto successor = indexById.find(successorId);
      if (successor == indexById.end())
        return anchor->emitError(
            "memory join references an unknown successor event");
      successorIndices.push_back(successor->second);
    }

    unsigned newIndex = ids.size();
    for (llvm::BitVector &row : reachable)
      row.resize(newIndex + 1);

    llvm::BitVector newReachable(newIndex + 1, false);
    for (unsigned successor : successorIndices) {
      newReachable.set(successor);
      newReachable |= reachable[successor];
    }
    for (unsigned source = 0; source < newIndex; ++source) {
      if (llvm::any_of(predecessorIndices, [&](unsigned predecessor) {
            return source == predecessor || reachable[source].test(predecessor);
          }))
        reachable[source].set(newIndex);
    }

    ids.push_back(eventId);
    indexById[eventId] = newIndex;
    reachable.push_back(std::move(newReachable));
    successors.emplace_back(successorIndices.begin(), successorIndices.end());
    predecessors.emplace_back(predecessorIndices.begin(),
                              predecessorIndices.end());
    for (unsigned predecessor : predecessorIndices)
      successors[predecessor].push_back(newIndex);
    for (unsigned successor : successorIndices)
      predecessors[successor].push_back(newIndex);
    return success();
  }

private:
  SmallVector<int64_t> ids;
  DenseMap<int64_t, unsigned> indexById;
  SmallVector<SmallVector<unsigned>> successors;
  SmallVector<SmallVector<unsigned>> predecessors;
  SmallVector<llvm::BitVector> reachable;
  SmallVector<unsigned> topologicalRanks;
};

FailureOr<EventReachability>
computeEventReachability(ArrayRef<int64_t> eventIds,
                         ArrayRef<std::pair<int64_t, int64_t>> edges,
                         Operation *anchor) {
  EventReachability analysis;
  if (failed(analysis.initialize(eventIds, edges, anchor)))
    return failure();
  return analysis;
}

FailureOr<EventReachability> computeEventReachability(const PlanRecords &plan,
                                                      Operation *anchor) {
  SmallVector<int64_t> eventIds;
  SmallVector<std::pair<int64_t, int64_t>> edges;
  eventIds.reserve(plan.completions.size());
  edges.reserve(plan.eventEdges.size());
  for (const CompletionRecord &completion : plan.completions)
    eventIds.push_back(completion.id);
  for (const EventEdgeRecord &edge : plan.eventEdges)
    edges.emplace_back(edge.sourceEventId, edge.targetEventId);
  return computeEventReachability(eventIds, edges, anchor);
}

FailureOr<SmallVector<int64_t>> parseIntegerArray(Operation *anchor,
                                                  Attribute attribute,
                                                  StringRef description) {
  auto array = dyn_cast_or_null<ArrayAttr>(attribute);
  if (!array) {
    anchor->emitError("expected integer array for ") << description;
    return failure();
  }
  SmallVector<int64_t> values;
  values.reserve(array.size());
  for (Attribute item : array) {
    auto integer = dyn_cast<IntegerAttr>(item);
    if (!integer) {
      anchor->emitError("expected integer element in ") << description;
      return failure();
    }
    values.push_back(integer.getInt());
  }
  return values;
}

template <typename AttrTy>
FailureOr<SmallVector<AttrTy>> parseTypedArray(Operation *anchor,
                                               StringRef name) {
  auto array = anchor->getAttrOfType<ArrayAttr>(name);
  if (!array) {
    anchor->emitError("expected array attribute '") << name << "'";
    return failure();
  }
  SmallVector<AttrTy> values;
  values.reserve(array.size());
  for (Attribute item : array) {
    auto typed = dyn_cast<AttrTy>(item);
    if (!typed) {
      anchor->emitError("attribute '")
          << name << "' contains an entry of the wrong type";
      return failure();
    }
    values.push_back(typed);
  }
  return values;
}

FailureOr<int64_t> getStaticByteSize(Type type, Operation *anchor) {
  if (isa<LogicalArrayType>(type))
    return int64_t{0};
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape()) {
    anchor->emitError("tile memory planning requires a static shaped type, "
                      "but found ")
        << type;
    return failure();
  }
  Type elementType = shaped.getElementType();
  if (!elementType.isIntOrFloat()) {
    anchor->emitError("tile memory planning requires an integer or floating "
                      "element type, but found ")
        << elementType;
    return failure();
  }
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0) {
    anchor->emitError("tile memory planning requires a byte-addressable "
                      "element type, but found ")
        << elementType;
    return failure();
  }
  std::optional<int64_t> bytes =
      llvm::checkedMul(shaped.getNumElements(), int64_t{bitWidth / 8});
  if (!bytes) {
    anchor->emitError("tile memory byte-size calculation overflowed");
    return failure();
  }
  return *bytes;
}

FailureOr<int64_t> getElementByteSize(Type type, Operation *anchor) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return int64_t{0};
  Type elementType = shaped.getElementType();
  if (!elementType.isIntOrFloat())
    return anchor->emitError("expected integer or floating element type");
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return anchor->emitError("expected byte-addressable element type");
  return static_cast<int64_t>(bitWidth / 8);
}

FailureOr<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getShapeAndStrides(Type type, Operation *anchor) {
  if (isa<LogicalArrayType>(type))
    return std::make_pair(SmallVector<int64_t>{}, SmallVector<int64_t>{});
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return anchor->emitError("expected a static shaped memory-plan type");
  SmallVector<int64_t> shape(shaped.getShape());
  SmallVector<int64_t> strides(shape.size(), 1);
  int64_t running = 1;
  for (int64_t index = static_cast<int64_t>(shape.size()) - 1; index >= 0;
       --index) {
    strides[index] = running;
    std::optional<int64_t> next = llvm::checkedMul(running, shape[index]);
    if (!next)
      return anchor->emitError("memory-plan stride calculation overflowed");
    running = *next;
  }
  return std::make_pair(std::move(shape), std::move(strides));
}

int ownerKindPriority(MemoryOwnerKind kind) {
  switch (kind) {
  case MemoryOwnerKind::ModelInput:
  case MemoryOwnerKind::ModelOutput:
    return 6;
  case MemoryOwnerKind::Assembly:
    return 5;
  case MemoryOwnerKind::RouteInput:
  case MemoryOwnerKind::RouteOutput:
    return 4;
  case MemoryOwnerKind::Persistent:
    return 3;
  case MemoryOwnerKind::Intermediate:
  case MemoryOwnerKind::LocalTemporary:
    return 2;
  }
  return 0;
}

struct TensorIdentity {
  int64_t tensorId = -1;
  bool ambiguous = false;
};

void mergeTensorIdentity(TensorIdentity &identity, int64_t tensorId) {
  if (tensorId < 0 || identity.ambiguous)
    return;
  if (identity.tensorId < 0) {
    identity.tensorId = tensorId;
    return;
  }
  if (identity.tensorId != tensorId) {
    identity.tensorId = -1;
    identity.ambiguous = true;
  }
}

std::map<OwnerKey, TensorIdentity>
collectOwnerTensorIdentities(const PlanRecords &plan) {
  std::map<OwnerKey, TensorIdentity> identities;
  for (TileRoutineRouteAttr route : plan.routes) {
    int64_t resourceId = route.getResourceId().getInt();
    int64_t tensorId = route.getTensorId().getInt();
    mergeTensorIdentity(
        identities[{resourceId, route.getSourceTile().getInt()}], tensorId);
    mergeTensorIdentity(
        identities[{resourceId, route.getDestinationTile().getInt()}],
        tensorId);
  }
  for (TileRoutineModelIOAttr modelInput : plan.modelInputs) {
    mergeTensorIdentity(identities[{modelInput.getResourceId().getInt(),
                                    modelInput.getTile().getInt()}],
                        modelInput.getTensorId().getInt());
  }
  for (TileRoutineModelIOAttr modelOutput : plan.modelOutputs) {
    mergeTensorIdentity(identities[{modelOutput.getResourceId().getInt(),
                                    modelOutput.getTile().getInt()}],
                        modelOutput.getTensorId().getInt());
  }
  return identities;
}

int64_t getOwnerTensorId(const std::map<OwnerKey, TensorIdentity> &identities,
                         OwnerKey key) {
  auto found = identities.find(key);
  return found == identities.end() ? -1 : found->second.tensorId;
}

LogicalResult mergeOwner(PlanRecords &plan, OwnerKey key,
                         const OwnerRecord &candidate, Operation *anchor) {
  auto found = plan.ownerByKey.find(key);
  if (found == plan.ownerByKey.end()) {
    plan.ownerByKey.emplace(key, plan.owners.size());
    plan.owners.push_back(candidate);
    return success();
  }
  OwnerRecord &owner = plan.owners[found->second];
  if (owner.byteSize != candidate.byteSize || owner.type != candidate.type) {
    return anchor->emitError(
        "one tile memory owner has inconsistent type or byte size");
  }
  if (owner.tensorId >= 0 && candidate.tensorId >= 0 &&
      owner.tensorId != candidate.tensorId) {
    return anchor->emitError(
        "one tile memory owner has inconsistent tensor identities");
  }
  if (owner.tensorId < 0)
    owner.tensorId = candidate.tensorId;
  if (ownerKindPriority(candidate.kind) > ownerKindPriority(owner.kind)) {
    owner.kind = candidate.kind;
    owner.routine = candidate.routine;
    owner.port = candidate.port;
  }
  return success();
}

template <typename AttrTy>
std::optional<AttrTy> findPortAttr(ArrayRef<AttrTy> values, int64_t routine,
                                   int64_t port, bool source) {
  for (AttrTy value : values) {
    int64_t candidateRoutine = source ? value.getSourceRoutine().getInt()
                                      : value.getDestinationRoutine().getInt();
    int64_t candidatePort = source ? value.getSourceOutput().getInt()
                                   : value.getDestinationInput().getInt();
    if (candidateRoutine == routine && candidatePort == port)
      return value;
  }
  return std::nullopt;
}

std::optional<TileRoutineModelIOAttr>
findModelIO(ArrayRef<TileRoutineModelIOAttr> values, int64_t routine,
            int64_t port) {
  for (TileRoutineModelIOAttr value : values) {
    if (value.getRoutine().getInt() == routine &&
        value.getPort().getInt() == port)
      return value;
  }
  return std::nullopt;
}

LogicalResult collectDeploymentRecords(ModuleOp deployment, PlanRecords &plan) {
  FailureOr<SmallVector<TileRoutineRouteAttr>> routes =
      parseTypedArray<TileRoutineRouteAttr>(deployment, kRoutesAttr);
  FailureOr<SmallVector<TileRoutineBindingAttr>> localBindings =
      parseTypedArray<TileRoutineBindingAttr>(deployment, kLocalBindingsAttr);
  FailureOr<SmallVector<TileRoutineModelIOAttr>> modelInputs =
      parseTypedArray<TileRoutineModelIOAttr>(deployment, kModelInputsAttr);
  FailureOr<SmallVector<TileRoutineModelIOAttr>> modelOutputs =
      parseTypedArray<TileRoutineModelIOAttr>(deployment, kModelOutputsAttr);
  if (failed(routes) || failed(localBindings) || failed(modelInputs) ||
      failed(modelOutputs))
    return failure();
  plan.routes = std::move(*routes);
  plan.localBindings = std::move(*localBindings);
  plan.modelInputs = std::move(*modelInputs);
  plan.modelOutputs = std::move(*modelOutputs);

  llvm::sort(plan.routes,
             [](TileRoutineRouteAttr left, TileRoutineRouteAttr right) {
               return left.getId().getInt() < right.getId().getInt();
             });

  for (ModuleOp tile : deployment.getOps<ModuleOp>()) {
    auto tileId = tile->getAttrOfType<IntegerAttr>(kPhysicalTileIdAttr);
    if (!tileId)
      return tile.emitError("outlined tile has no physical tile ID");
    for (func::FuncOp function : tile.getOps<func::FuncOp>()) {
      auto routineId =
          function->getAttrOfType<IntegerAttr>(kGlobalRoutineIdAttr);
      if (!routineId)
        continue;
      FailureOr<SmallVector<int64_t>> inputs =
          parseIntegerArray(function, function->getAttr(kInputResourceIdsAttr),
                            "routine input resource IDs");
      FailureOr<SmallVector<int64_t>> outputs =
          parseIntegerArray(function, function->getAttr(kOutputResourceIdsAttr),
                            "routine output resource IDs");
      SmallVector<int64_t> controlDependencies;
      if (Attribute dependencies =
              function->getAttr(kControlDependencyIdsAttr)) {
        FailureOr<SmallVector<int64_t>> parsed = parseIntegerArray(
            function, dependencies, "routine control dependency IDs");
        if (failed(parsed))
          return failure();
        controlDependencies = std::move(*parsed);
      }
      if (failed(inputs) || failed(outputs))
        return failure();
      if (inputs->size() != function.getNumArguments() ||
          outputs->size() != function.getNumResults()) {
        return function.emitError(
            "routine resource IDs do not match its function signature");
      }
      auto routineKind = function->getAttrOfType<StringAttr>(kRoutineKindAttr);
      auto localIndex =
          function->getAttrOfType<IntegerAttr>(kLocalRoutineIndexAttr);
      if (!routineKind || (routineKind.getValue() != "boot" &&
                           routineKind.getValue() != "compute"))
        return function.emitError("expected routine kind 'boot' or 'compute'");
      if (!localIndex || localIndex.getInt() < 0)
        return function.emitError("expected nonnegative local routine index");
      RoutineInfo info{routineId.getInt(),
                       tileId.getInt(),
                       function,
                       std::move(*inputs),
                       std::move(*outputs),
                       std::move(controlDependencies),
                       routineKind.getValue() == "boot",
                       localIndex.getInt()};
      if (!plan.routines.emplace(info.id, std::move(info)).second)
        return function.emitError("duplicate global routine ID in memory plan");
    }
  }
  if (plan.routines.empty())
    return deployment.emitError("memory planning found no outlined routines");
  return success();
}

OwnerRecord makeOwnerCandidate(int64_t resourceId, int64_t tensorId,
                               int64_t tile, int64_t routine, int64_t port,
                               MemoryOwnerKind kind, int64_t byteSize,
                               Type type, ArrayRef<int64_t> shape,
                               ArrayRef<int64_t> strides) {
  OwnerRecord owner;
  owner.resourceId = resourceId;
  owner.tensorId = tensorId;
  owner.tile = tile;
  owner.routine = routine;
  owner.port = port;
  owner.kind = kind;
  owner.byteSize = byteSize;
  owner.type = type;
  owner.shape.assign(shape.begin(), shape.end());
  owner.strides.assign(strides.begin(), strides.end());
  return owner;
}

LogicalResult collectOwners(ModuleOp deployment, PlanRecords &plan) {
  const std::map<OwnerKey, TensorIdentity> tensorIdentities =
      collectOwnerTensorIdentities(plan);
  for (auto &[routineId, routine] : plan.routines) {
    for (auto [port, resourceId] : llvm::enumerate(routine.outputResources)) {
      Type type = routine.function.getResultTypes()[port];
      FailureOr<int64_t> bytes = getStaticByteSize(type, deployment);
      auto geometry = getShapeAndStrides(type, deployment);
      if (failed(bytes) || failed(geometry))
        return failure();

      OwnerKey ownerKey{resourceId, routine.tile};
      int64_t tensorId = getOwnerTensorId(tensorIdentities, ownerKey);
      MemoryOwnerKind kind = MemoryOwnerKind::Intermediate;
      if (isa<LogicalArrayType>(type))
        kind = MemoryOwnerKind::Persistent;
      if (auto model = findModelIO(plan.modelOutputs, routineId, port)) {
        kind = MemoryOwnerKind::ModelOutput;
        if (model->getResourceId().getInt() != resourceId ||
            model->getByteSize().getInt() != *bytes)
          return deployment.emitError("model-output owner metadata mismatch");
      } else if (auto route = findPortAttr<TileRoutineRouteAttr>(
                     plan.routes, routineId, port, true)) {
        kind = MemoryOwnerKind::RouteOutput;
        if (route->getResourceId().getInt() != resourceId ||
            route->getByteSize().getInt() != *bytes)
          return deployment.emitError("route-output owner metadata mismatch");
      }

      OwnerRecord candidate = makeOwnerCandidate(
          resourceId, tensorId, routine.tile, routineId, port, kind, *bytes,
          type, geometry->first, geometry->second);
      if (failed(mergeOwner(plan, ownerKey, candidate, deployment)))
        return failure();
    }
  }

  for (auto &[routineId, routine] : plan.routines) {
    for (auto [port, resourceId] : llvm::enumerate(routine.inputResources)) {
      Type type = routine.function.getArgumentTypes()[port];
      FailureOr<int64_t> bytes = getStaticByteSize(type, deployment);
      auto geometry = getShapeAndStrides(type, deployment);
      if (failed(bytes) || failed(geometry))
        return failure();

      OwnerKey ownerKey{resourceId, routine.tile};
      int64_t tensorId = getOwnerTensorId(tensorIdentities, ownerKey);
      MemoryOwnerKind kind = isa<LogicalArrayType>(type)
                                 ? MemoryOwnerKind::Persistent
                                 : MemoryOwnerKind::Intermediate;
      if (auto model = findModelIO(plan.modelInputs, routineId, port)) {
        kind = MemoryOwnerKind::ModelInput;
        if (model->getResourceId().getInt() != resourceId ||
            model->getByteSize().getInt() != *bytes)
          return deployment.emitError("model-input owner metadata mismatch");
      } else if (auto route = findPortAttr<TileRoutineRouteAttr>(
                     plan.routes, routineId, port, false)) {
        kind = MemoryOwnerKind::RouteInput;
        if (route->getResourceId().getInt() != resourceId ||
            route->getByteSize().getInt() != *bytes)
          return deployment.emitError("route-input owner metadata mismatch");
      }

      OwnerRecord candidate = makeOwnerCandidate(
          resourceId, tensorId, routine.tile, routineId, port, kind, *bytes,
          type, geometry->first, geometry->second);
      if (failed(mergeOwner(plan, ownerKey, candidate, deployment)))
        return failure();
    }
  }

  llvm::sort(plan.owners,
             [](const OwnerRecord &left, const OwnerRecord &right) {
               return std::tie(left.tile, left.resourceId) <
                      std::tie(right.tile, right.resourceId);
             });
  plan.ownerByKey.clear();
  for (auto [index, owner] : llvm::enumerate(plan.owners)) {
    owner.id = index;
    plan.ownerByKey[{owner.resourceId, owner.tile}] = owner.id;
  }
  return success();
}

LogicalResult createFullViews(PlanRecords &plan) {
  for (const OwnerRecord &owner : plan.owners) {
    ViewRecord view;
    view.id = plan.views.size();
    view.ownerId = owner.id;
    view.byteSize = owner.byteSize;
    view.offsets.assign(owner.shape.size(), 0);
    view.sizes = owner.shape;
    view.strides = owner.strides;
    plan.fullViewByOwner[owner.id] = view.id;
    plan.views.push_back(std::move(view));
  }
  return success();
}

FailureOr<ViewRecord> createSubview(const OwnerRecord &owner, int64_t viewId,
                                    ArrayRef<int64_t> offsets,
                                    ArrayRef<int64_t> sizes,
                                    ArrayRef<int64_t> sliceStrides,
                                    Operation *anchor);

bool hasLocalInputSource(const PlanRecords &plan, int64_t routine,
                         int64_t port) {
  return llvm::any_of(plan.localBindings, [&](TileRoutineBindingAttr binding) {
    return binding.getDestinationRoutine().getInt() == routine &&
           binding.getDestinationInput().getInt() == port;
  });
}

bool isModelInput(const PlanRecords &plan, int64_t routine, int64_t port) {
  return llvm::any_of(plan.modelInputs, [&](TileRoutineModelIOAttr input) {
    return input.getRoutine().getInt() == routine &&
           input.getPort().getInt() == port;
  });
}

FailureOr<std::optional<int64_t>>
tryCreateZeroCopyInputView(PlanRecords &plan, RoutineInfo &routine,
                           int64_t port, int64_t ownerId) {
  if (!hasLocalInputSource(plan, routine.id, port) &&
      !isModelInput(plan, routine.id, port))
    return std::optional<int64_t>{};

  BlockArgument argument = routine.function.getArgument(port);
  if (!argument.hasOneUse())
    return std::optional<int64_t>{};
  OpOperand &use = *argument.getUses().begin();
  auto slice = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
  if (!slice || slice.getSource() != argument)
    return std::optional<int64_t>{};
  ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  ArrayRef<int64_t> sizes = slice.getStaticSizes();
  ArrayRef<int64_t> strides = slice.getStaticStrides();
  if (llvm::is_contained(offsets, ShapedType::kDynamic) ||
      llvm::is_contained(sizes, ShapedType::kDynamic) ||
      llvm::is_contained(strides, ShapedType::kDynamic) ||
      !llvm::all_of(strides, [](int64_t value) { return value == 1; }))
    return std::optional<int64_t>{};

  const OwnerRecord &owner = plan.owners[ownerId];
  FailureOr<ViewRecord> view =
      createSubview(owner, plan.views.size(), offsets, sizes, strides, slice);
  if (failed(view))
    return failure();
  if (view->contiguity != MemoryContiguity::Contiguous)
    return std::optional<int64_t>{};
  if (view->byteOffset == 0 && view->byteSize == owner.byteSize)
    return std::optional<int64_t>{};

  int64_t viewId = view->id;
  plan.views.push_back(*view);
  Builder builder(slice.getContext());
  slice->setAttr(kOwnerIdAttrName, builder.getI64IntegerAttr(ownerId));
  slice->setAttr(kViewIdAttrName, builder.getI64IntegerAttr(viewId));
  slice->setAttr(kZeroCopyViewAttrName, builder.getUnitAttr());
  return std::optional<int64_t>{viewId};
}

FailureOr<int64_t> requireOwner(PlanRecords &plan, int64_t resourceId,
                                int64_t tile, Operation *anchor) {
  auto found = plan.ownerByKey.find({resourceId, tile});
  if (found == plan.ownerByKey.end()) {
    anchor->emitError("routine port has no tile-local storage owner for "
                      "resource ")
        << resourceId << " on tile " << tile;
    return failure();
  }
  return found->second;
}

LogicalResult createRoutineBindings(PlanRecords &plan) {
  for (auto &[routineId, routine] : plan.routines) {
    for (auto [port, resourceId] : llvm::enumerate(routine.inputResources)) {
      FailureOr<int64_t> ownerId =
          requireOwner(plan, resourceId, routine.tile, routine.function);
      if (failed(ownerId))
        return failure();
      BindingRecord binding;
      binding.id = plan.bindings.size();
      binding.routine = routineId;
      binding.port = port;
      binding.input = true;
      binding.ownerId = *ownerId;
      binding.viewId = plan.fullViewByOwner.at(*ownerId);
      binding.effect = MemoryAccessEffect::Read;
      FailureOr<std::optional<int64_t>> zeroCopyView =
          tryCreateZeroCopyInputView(plan, routine, port, *ownerId);
      if (failed(zeroCopyView))
        return failure();
      if (*zeroCopyView)
        binding.viewId = **zeroCopyView;
      plan.inputBindingByPort[{routineId, port}] = binding.id;
      plan.bindings.push_back(binding);
    }
    for (auto [port, resourceId] : llvm::enumerate(routine.outputResources)) {
      FailureOr<int64_t> ownerId =
          requireOwner(plan, resourceId, routine.tile, routine.function);
      if (failed(ownerId))
        return failure();
      BindingRecord binding;
      binding.id = plan.bindings.size();
      binding.routine = routineId;
      binding.port = port;
      binding.input = false;
      binding.ownerId = *ownerId;
      binding.viewId = plan.fullViewByOwner.at(*ownerId);
      binding.effect = MemoryAccessEffect::Write;
      plan.outputBindingByPort[{routineId, port}] = binding.id;
      plan.bindings.push_back(binding);
    }
  }
  return success();
}

bool canUpdateOwnerInPlace(MemoryOwnerKind kind) {
  return kind == MemoryOwnerKind::Intermediate ||
         kind == MemoryOwnerKind::RouteInput ||
         kind == MemoryOwnerKind::Assembly;
}

bool canStoreInPlaceResult(MemoryOwnerKind kind) {
  return kind == MemoryOwnerKind::Intermediate ||
         kind == MemoryOwnerKind::RouteOutput;
}

LogicalResult discoverInPlaceAliases(ModuleOp deployment, PlanRecords &plan) {
  DenseSet<int64_t> claimedInputOwners;
  DenseSet<int64_t> claimedOutputOwners;

  for (auto &[routineId, routine] : plan.routines) {
    func::FuncOp function = routine.function;
    if (routine.boot || !function.getBody().hasOneBlock())
      continue;
    auto returnOp =
        dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
    if (!returnOp)
      continue;

    for (auto [outputPort, returned] :
         llvm::enumerate(returnOp.getOperands())) {
      auto result = dyn_cast<OpResult>(returned);
      if (!result || !result.hasOneUse())
        continue;
      auto linalgOp = dyn_cast<linalg::LinalgOp>(result.getOwner());
      if (!linalgOp || result.getResultNumber() >= linalgOp.getNumDpsInits())
        continue;

      OpOperand *outputOperand =
          linalgOp.getDpsInitOperand(result.getResultNumber());
      auto outputBinding =
          plan.outputBindingByPort.find({routineId, outputPort});
      if (outputBinding == plan.outputBindingByPort.end())
        return function.emitError(
            "destination-style result has no memory-plan output binding");
      BindingRecord &output = plan.bindings[outputBinding->second];
      OwnerRecord &outputOwner = plan.owners[output.ownerId];
      if (!canStoreInPlaceResult(outputOwner.kind) ||
          claimedOutputOwners.contains(output.ownerId))
        continue;

      auto tryAliasArgument = [&](BlockArgument argument) -> FailureOr<bool> {
        if (!argument || argument.getOwner() != &function.getBody().front() ||
            !argument.hasOneUse() || argument.getType() != returned.getType())
          return false;

        int64_t inputPort = argument.getArgNumber();
        auto inputBinding =
            plan.inputBindingByPort.find({routineId, inputPort});
        if (inputBinding == plan.inputBindingByPort.end())
          return function.emitError(
                     "in-place operand has no memory-plan input binding"),
                 failure();
        BindingRecord &input = plan.bindings[inputBinding->second];
        OwnerRecord &inputOwner = plan.owners[input.ownerId];
        if (input.ownerId == output.ownerId ||
            !canUpdateOwnerInPlace(inputOwner.kind) ||
            claimedInputOwners.contains(input.ownerId) ||
            inputOwner.type != outputOwner.type ||
            inputOwner.byteSize != outputOwner.byteSize ||
            inputOwner.shape != outputOwner.shape ||
            inputOwner.strides != outputOwner.strides)
          return false;

        unsigned inputReaders =
            llvm::count_if(plan.bindings, [&](const BindingRecord &binding) {
              return binding.input && binding.ownerId == input.ownerId;
            });
        if (inputReaders != 1)
          return false;

        bool pendingSend =
            llvm::any_of(plan.routes, [&](TileRoutineRouteAttr route) {
              return route.getSourceTile().getInt() == inputOwner.tile &&
                     route.getResourceId().getInt() == inputOwner.resourceId;
            });
        if (pendingSend)
          return false;

        input.effect = MemoryAccessEffect::ReadWrite;
        output.effect = MemoryAccessEffect::ReadWrite;
        InPlaceAliasRecord alias;
        alias.id = plan.inPlaceAliases.size();
        alias.routine = routineId;
        alias.inputPort = inputPort;
        alias.outputPort = outputPort;
        alias.inputOwnerId = input.ownerId;
        alias.outputOwnerId = output.ownerId;
        alias.inputViewId = input.viewId;
        alias.outputViewId = output.viewId;
        plan.inPlaceAliases.push_back(alias);
        claimedInputOwners.insert(input.ownerId);
        claimedOutputOwners.insert(output.ownerId);
        return true;
      };

      // A destination-style tensor result is the updated DPS init by
      // definition.  When that init is a uniquely consumed routine argument,
      // retain the same backing storage across the routine boundary.  This
      // applies to pooling, convolution, matmul, and other DPS operations—not
      // just elementwise generics.
      if (auto initArgument = dyn_cast<BlockArgument>(outputOperand->get())) {
        FailureOr<bool> aliased = tryAliasArgument(initArgument);
        if (failed(aliased))
          return failure();
        if (*aliased)
          continue;
      }

      if (!linalg::isElementwise(linalgOp))
        continue;
      AffineMap outputMap = linalgOp.getMatchingIndexingMap(outputOperand);

      // An output buffer may be used to materialize a derived DPS input before
      // the returned elementwise operation executes.  Aliasing that output to
      // another input would overwrite the latter before its final read.  Only
      // direct arguments and immutable constants are safe inputs for this
      // routine-boundary optimization.
      bool hasDerivedInput =
          llvm::any_of(linalgOp.getDpsInputOperands(), [](OpOperand *operand) {
            Value value = operand->get();
            if (isa<BlockArgument>(value))
              return false;
            Operation *definingOp = value.getDefiningOp();
            return definingOp == nullptr ||
                   !definingOp->hasTrait<OpTrait::ConstantLike>();
          });
      if (hasDerivedInput)
        continue;

      for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
        auto argument = dyn_cast<BlockArgument>(inputOperand->get());
        if (!argument || argument.getOwner() != &function.getBody().front() ||
            !argument.hasOneUse() ||
            inputOperand->get().getType() != returned.getType() ||
            linalgOp.getMatchingIndexingMap(inputOperand) != outputMap)
          continue;
        FailureOr<bool> aliased = tryAliasArgument(argument);
        if (failed(aliased))
          return failure();
        if (*aliased)
          break;
      }
    }
  }
  (void)deployment;
  return success();
}

void createRoutineCompletionEvents(PlanRecords &plan) {
  for (auto &[routineId, routine] : plan.routines) {
    CompletionRecord start;
    start.id = plan.completions.size();
    start.kind = MemoryCompletionKind::RoutineStart;
    start.tile = routine.tile;
    start.routine = routineId;
    plan.routineStartEvent[routineId] = start.id;
    plan.completions.push_back(start);

    CompletionRecord complete;
    complete.id = plan.completions.size();
    complete.kind = MemoryCompletionKind::RoutineComplete;
    complete.tile = routine.tile;
    complete.routine = routineId;
    plan.routineCompleteEvent[routineId] = complete.id;
    plan.completions.push_back(complete);
  }
}

LogicalResult addEventEdge(PlanRecords &plan, int64_t sourceEventId,
                           int64_t targetEventId, MemoryEventEdgeKind kind,
                           Operation *anchor) {
  if (sourceEventId < 0 || targetEventId < 0 ||
      sourceEventId >= static_cast<int64_t>(plan.completions.size()) ||
      targetEventId >= static_cast<int64_t>(plan.completions.size()))
    return anchor->emitError("memory event edge references an unknown event");
  if (sourceEventId == targetEventId)
    return anchor->emitError("memory event graph contains a self edge");
  auto key = std::make_tuple(sourceEventId, targetEventId, kind);
  if (!plan.eventEdgeKeys.insert(key).second)
    return success();
  plan.eventEdges.push_back(
      EventEdgeRecord{static_cast<int64_t>(plan.eventEdges.size()),
                      sourceEventId, targetEventId, kind});
  return success();
}

LogicalResult createMovements(ModuleOp deployment, PlanRecords &plan) {
  for (auto [index, binding] : llvm::enumerate(plan.localBindings)) {
    auto sourceRoutine =
        plan.routines.find(binding.getSourceRoutine().getInt());
    auto destinationRoutine =
        plan.routines.find(binding.getDestinationRoutine().getInt());
    if (sourceRoutine == plan.routines.end() ||
        destinationRoutine == plan.routines.end())
      return deployment.emitError("local binding references unknown routine");
    FailureOr<int64_t> sourceOwner =
        requireOwner(plan, binding.getResourceId().getInt(),
                     sourceRoutine->second.tile, deployment);
    FailureOr<int64_t> destinationOwner =
        requireOwner(plan, binding.getResourceId().getInt(),
                     destinationRoutine->second.tile, deployment);
    if (failed(sourceOwner) || failed(destinationOwner))
      return failure();
    if (*sourceOwner != *destinationOwner)
      return deployment.emitError("same-tile binding does not alias one owner");
    MovementRecord movement;
    movement.id = plan.movements.size();
    movement.mode = MemoryMovementMode::LocalAlias;
    auto sourceBinding = plan.outputBindingByPort.find(
        {sourceRoutine->second.id, binding.getSourceOutput().getInt()});
    auto destinationBinding =
        plan.inputBindingByPort.find({destinationRoutine->second.id,
                                      binding.getDestinationInput().getInt()});
    if (sourceBinding == plan.outputBindingByPort.end() ||
        destinationBinding == plan.inputBindingByPort.end())
      return deployment.emitError(
          "local binding has no matching routine memory ports");
    movement.sourceViewId = plan.bindings[sourceBinding->second].viewId;
    movement.destinationViewId =
        plan.bindings[destinationBinding->second].viewId;
    movement.sourceCompletionEventId =
        plan.routineCompleteEvent.at(binding.getSourceRoutine().getInt());
    movement.destinationCompletionEventId =
        plan.routineStartEvent.at(binding.getDestinationRoutine().getInt());
    movement.byteSize = plan.views[movement.destinationViewId].byteSize;
    plan.movements.push_back(movement);
    (void)index;
  }

  for (TileRoutineRouteAttr route : plan.routes) {
    FailureOr<int64_t> sourceOwner =
        requireOwner(plan, route.getResourceId().getInt(),
                     route.getSourceTile().getInt(), deployment);
    FailureOr<int64_t> destinationOwner =
        requireOwner(plan, route.getResourceId().getInt(),
                     route.getDestinationTile().getInt(), deployment);
    if (failed(sourceOwner) || failed(destinationOwner))
      return failure();

    auto sourceBinding = plan.outputBindingByPort.find(
        {route.getSourceRoutine().getInt(), route.getSourceOutput().getInt()});
    auto destinationBinding =
        plan.inputBindingByPort.find({route.getDestinationRoutine().getInt(),
                                      route.getDestinationInput().getInt()});
    if (sourceBinding == plan.outputBindingByPort.end() ||
        destinationBinding == plan.inputBindingByPort.end())
      return deployment.emitError("route has no matching routine port binding");
    BindingRecord asyncSource = plan.bindings[sourceBinding->second];
    asyncSource.id = plan.bindings.size();
    asyncSource.effect = MemoryAccessEffect::AsyncTransferSource;
    plan.bindings.push_back(asyncSource);
    BindingRecord asyncDestination = plan.bindings[destinationBinding->second];
    asyncDestination.id = plan.bindings.size();
    asyncDestination.effect = MemoryAccessEffect::AsyncTransferDestination;
    plan.bindings.push_back(asyncDestination);

    CompletionRecord send;
    send.id = plan.completions.size();
    send.kind = MemoryCompletionKind::RouteSendComplete;
    send.tile = route.getSourceTile().getInt();
    send.routine = route.getSourceRoutine().getInt();
    send.routeId = route.getId().getInt();
    send.ownerId = *sourceOwner;
    send.viewId = plan.bindings[sourceBinding->second].viewId;
    plan.routeSendEvent[send.routeId] = send.id;
    plan.completions.push_back(send);

    CompletionRecord arrival;
    arrival.id = plan.completions.size();
    arrival.kind = MemoryCompletionKind::RouteArrival;
    arrival.tile = route.getDestinationTile().getInt();
    arrival.routine = route.getDestinationRoutine().getInt();
    arrival.routeId = route.getId().getInt();
    arrival.ownerId = *destinationOwner;
    arrival.viewId = plan.bindings[destinationBinding->second].viewId;
    plan.routeArrivalEvent[arrival.routeId] = arrival.id;
    plan.completions.push_back(arrival);

    MovementRecord movement;
    movement.id = plan.movements.size();
    movement.mode = MemoryMovementMode::Contiguous;
    movement.sourceViewId = plan.bindings[sourceBinding->second].viewId;
    movement.destinationViewId =
        plan.bindings[destinationBinding->second].viewId;
    movement.routeId = route.getId().getInt();
    movement.sourceCompletionEventId = send.id;
    movement.destinationCompletionEventId = arrival.id;
    movement.byteSize = route.getByteSize().getInt();
    plan.movements.push_back(movement);
  }
  return success();
}

std::optional<int64_t> findBindingView(const PlanRecords &plan, int64_t routine,
                                       Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  func::FuncOp function = plan.routines.at(routine).function;
  if (!argument ||
      argument.getOwner()->getParentOp() != function.getOperation())
    return std::nullopt;
  auto found = plan.inputBindingByPort.find({routine, argument.getArgNumber()});
  if (found == plan.inputBindingByPort.end())
    return std::nullopt;
  return plan.bindings[found->second].viewId;
}

FailureOr<int64_t> getOrCreateLocalTemporaryView(PlanRecords &plan,
                                                 int64_t routineId,
                                                 int64_t tile, Value value,
                                                 Operation *anchor) {
  auto found = plan.localTemporaryView.find(value);
  if (found != plan.localTemporaryView.end())
    return found->second;
  FailureOr<int64_t> bytes = getStaticByteSize(value.getType(), anchor);
  auto geometry = getShapeAndStrides(value.getType(), anchor);
  if (failed(bytes) || failed(geometry))
    return failure();

  OwnerRecord owner = makeOwnerCandidate(
      -1, -1, tile, routineId, -1, MemoryOwnerKind::LocalTemporary, *bytes,
      value.getType(), geometry->first, geometry->second);
  owner.id = plan.owners.size();
  plan.owners.push_back(owner);

  ViewRecord view;
  view.id = plan.views.size();
  view.ownerId = owner.id;
  view.byteSize = owner.byteSize;
  view.offsets.assign(owner.shape.size(), 0);
  view.sizes = owner.shape;
  view.strides = owner.strides;
  plan.views.push_back(view);
  plan.fullViewByOwner[owner.id] = view.id;
  plan.localTemporaryView[value] = view.id;
  return view.id;
}

FailureOr<ViewRecord> createSubview(const OwnerRecord &owner, int64_t viewId,
                                    ArrayRef<int64_t> offsets,
                                    ArrayRef<int64_t> sizes,
                                    ArrayRef<int64_t> sliceStrides,
                                    Operation *anchor) {
  if (offsets.size() != owner.shape.size() ||
      sizes.size() != owner.shape.size() ||
      sliceStrides.size() != owner.shape.size())
    return anchor->emitError("static assembly view rank mismatch");
  FailureOr<int64_t> elementBytes = getElementByteSize(owner.type, anchor);
  if (failed(elementBytes))
    return failure();

  int64_t linearOffset = 0;
  int64_t elementCount = 1;
  int64_t spanElements = owner.shape.empty() ? 1 : 0;
  for (size_t index = 0; index < owner.shape.size(); ++index) {
    if (offsets[index] < 0 || sizes[index] < 0 || sliceStrides[index] <= 0 ||
        offsets[index] + (sizes[index] == 0
                              ? 0
                              : (sizes[index] - 1) * sliceStrides[index]) >=
            owner.shape[index])
      return anchor->emitError("static assembly view exceeds its owner");
    std::optional<int64_t> offsetTerm =
        llvm::checkedMul(offsets[index], owner.strides[index]);
    std::optional<int64_t> nextOffset =
        offsetTerm ? llvm::checkedAdd(linearOffset, *offsetTerm) : std::nullopt;
    std::optional<int64_t> nextCount =
        llvm::checkedMul(elementCount, sizes[index]);
    std::optional<int64_t> spanTerm =
        llvm::checkedMul(sizes[index] == 0 ? int64_t{0} : sizes[index] - 1,
                         owner.strides[index] * sliceStrides[index]);
    std::optional<int64_t> nextSpan =
        spanTerm ? llvm::checkedAdd(spanElements, *spanTerm) : std::nullopt;
    if (!nextOffset || !nextCount || !nextSpan)
      return anchor->emitError("static assembly view arithmetic overflowed");
    linearOffset = *nextOffset;
    elementCount = *nextCount;
    spanElements = *nextSpan;
  }
  if (!owner.shape.empty()) {
    std::optional<int64_t> plusOne = llvm::checkedAdd(spanElements, int64_t{1});
    if (!plusOne)
      return anchor->emitError("static assembly span overflowed");
    spanElements = *plusOne;
  }
  std::optional<int64_t> byteOffset =
      llvm::checkedMul(linearOffset, *elementBytes);
  std::optional<int64_t> byteSize =
      llvm::checkedMul(elementCount, *elementBytes);
  if (!byteOffset || !byteSize)
    return anchor->emitError("static assembly byte range overflowed");

  ViewRecord view;
  view.id = viewId;
  view.ownerId = owner.id;
  view.byteOffset = *byteOffset;
  view.byteSize = *byteSize;
  view.offsets.assign(offsets.begin(), offsets.end());
  view.sizes.assign(sizes.begin(), sizes.end());
  view.strides.reserve(sliceStrides.size());
  for (size_t index = 0; index < sliceStrides.size(); ++index) {
    std::optional<int64_t> physicalStride =
        llvm::checkedMul(owner.strides[index], sliceStrides[index]);
    if (!physicalStride)
      return anchor->emitError("static assembly stride overflowed");
    view.strides.push_back(*physicalStride);
  }
  view.contiguity =
      spanElements == elementCount &&
              llvm::all_of(sliceStrides, [](int64_t v) { return v == 1; })
          ? MemoryContiguity::Contiguous
          : MemoryContiguity::NonContiguous;
  return view;
}

LogicalResult createAssemblyRecords(ModuleOp deployment, PlanRecords &plan) {
  for (auto &[routineId, routine] : plan.routines) {
    auto returnOp = dyn_cast<func::ReturnOp>(
        routine.function.getBody().front().getTerminator());
    if (!returnOp)
      continue;
    for (auto [outputPort, returned] :
         llvm::enumerate(returnOp.getOperands())) {
      SmallVector<tensor::InsertSliceOp> inserts;
      Value cursor = returned;
      while (auto insert = cursor.getDefiningOp<tensor::InsertSliceOp>()) {
        inserts.push_back(insert);
        cursor = insert.getDest();
      }
      if (inserts.empty())
        continue;
      std::reverse(inserts.begin(), inserts.end());

      bool hasDynamicGeometry =
          llvm::any_of(inserts, [](tensor::InsertSliceOp insert) {
            return llvm::is_contained(insert.getStaticOffsets(),
                                      ShapedType::kDynamic) ||
                   llvm::is_contained(insert.getStaticSizes(),
                                      ShapedType::kDynamic) ||
                   llvm::is_contained(insert.getStaticStrides(),
                                      ShapedType::kDynamic);
          });
      if (hasDynamicGeometry) {
        bool crossesRoutineBoundary =
            llvm::any_of(inserts, [&](tensor::InsertSliceOp insert) {
              return findBindingView(plan, routineId, insert.getSource())
                  .has_value();
            });
        if (crossesRoutineBoundary)
          return routine.function.emitError(
              "cross-routine tile memory assembly requires static slice "
              "geometry");
        continue;
      }

      auto outputBinding =
          plan.outputBindingByPort.find({routineId, outputPort});
      if (outputBinding == plan.outputBindingByPort.end())
        return routine.function.emitError(
            "assembled result has no output memory binding");
      int64_t ownerId = plan.bindings[outputBinding->second].ownerId;

      AssemblyRecord assembly;
      assembly.id = plan.assemblies.size();
      assembly.ownerId = ownerId;
      for (tensor::InsertSliceOp insert : inserts) {
        std::optional<int64_t> boundaryView =
            findBindingView(plan, routineId, insert.getSource());
        int64_t sourceView = -1;
        if (boundaryView) {
          sourceView = *boundaryView;
        } else {
          FailureOr<int64_t> localView = getOrCreateLocalTemporaryView(
              plan, routineId, routine.tile, insert.getSource(), insert);
          if (failed(localView))
            return failure();
          sourceView = *localView;
        }
        ArrayRef<int64_t> offsets = insert.getStaticOffsets();
        ArrayRef<int64_t> sizes = insert.getStaticSizes();
        ArrayRef<int64_t> strides = insert.getStaticStrides();
        if (llvm::is_contained(offsets, ShapedType::kDynamic) ||
            llvm::is_contained(sizes, ShapedType::kDynamic) ||
            llvm::is_contained(strides, ShapedType::kDynamic)) {
          return insert.emitError(
              "tile memory assembly requires static slice geometry");
        }
        // Creating a local source view can append an owner and reallocate the
        // owner table. Reacquire the destination owner after that operation.
        const OwnerRecord &destinationOwner = plan.owners[ownerId];
        FailureOr<ViewRecord> destination =
            createSubview(destinationOwner, plan.views.size(), offsets, sizes,
                          strides, insert);
        if (failed(destination))
          return failure();
        plan.views.push_back(*destination);

        assembly.contributingViewIds.push_back(sourceView);
        assembly.destinationViewIds.push_back(destination->id);
        const OwnerRecord &sourceOwner =
            plan.owners[plan.views[sourceView].ownerId];
        int64_t sourceCompletionEvent = plan.routineStartEvent.at(routineId);
        int64_t routeId = -1;
        if (sourceOwner.kind == MemoryOwnerKind::RouteInput) {
          auto sourceArgument = dyn_cast<BlockArgument>(insert.getSource());
          std::optional<TileRoutineRouteAttr> route =
              sourceArgument ? findPortAttr<TileRoutineRouteAttr>(
                                   plan.routes, routineId,
                                   sourceArgument.getArgNumber(), false)
                             : std::nullopt;
          if (!route)
            return insert.emitError(
                "route-input assembly contribution has no incoming route");
          routeId = route->getId().getInt();
          auto arrival = plan.routeArrivalEvent.find(routeId);
          if (arrival == plan.routeArrivalEvent.end())
            return insert.emitError(
                "route-input assembly contribution has no arrival event");
          sourceCompletionEvent = arrival->second;
          std::optional<int64_t> total =
              llvm::checkedAdd(assembly.routedBytes, destination->byteSize);
          if (!total)
            return insert.emitError("assembly routed byte count overflowed");
          assembly.routedBytes = *total;
        } else {
          std::optional<int64_t> total =
              llvm::checkedAdd(assembly.localCopyBytes, destination->byteSize);
          if (!total)
            return insert.emitError("assembly local byte count overflowed");
          assembly.localCopyBytes = *total;
          if (auto sourceArgument =
                  dyn_cast<BlockArgument>(insert.getSource())) {
            if (auto binding = findPortAttr<TileRoutineBindingAttr>(
                    plan.localBindings, routineId,
                    sourceArgument.getArgNumber(), false)) {
              sourceCompletionEvent = plan.routineCompleteEvent.at(
                  binding->getSourceRoutine().getInt());
            }
          }
        }

        CompletionRecord dmaComplete;
        dmaComplete.id = plan.completions.size();
        dmaComplete.kind = MemoryCompletionKind::DMAComplete;
        dmaComplete.tile = routine.tile;
        dmaComplete.routine = routineId;
        dmaComplete.routeId = routeId;
        dmaComplete.ownerId = ownerId;
        dmaComplete.viewId = destination->id;
        plan.completions.push_back(dmaComplete);

        CompletionRecord contribution;
        contribution.id = plan.completions.size();
        contribution.kind = MemoryCompletionKind::AssemblyContribution;
        contribution.tile = routine.tile;
        contribution.routine = routineId;
        contribution.routeId = routeId;
        contribution.ownerId = ownerId;
        contribution.viewId = destination->id;
        plan.completions.push_back(contribution);
        assembly.completionEventIds.push_back(contribution.id);

        MovementRecord movement;
        movement.id = plan.movements.size();
        // Assembly describes the readiness join. The contribution itself is a
        // real pack into the destination owner and must remain visible as
        // memory work for vectorization, lifetime planning, and accounting.
        movement.mode = MemoryMovementMode::Packed;
        movement.sourceViewId = sourceView;
        movement.destinationViewId = destination->id;
        movement.routeId = routeId;
        movement.sourceCompletionEventId = sourceCompletionEvent;
        movement.destinationCompletionEventId = dmaComplete.id;
        movement.assemblyContributionEventId = contribution.id;
        movement.assemblyId = assembly.id;
        movement.byteSize = destination->byteSize;
        plan.movements.push_back(movement);
      }
      // Assembly is a dataflow property, not always a storage class. Preserve
      // model-I/O and route ownership so externally or runtime-owned buffers
      // are not incorrectly charged to tile workspace.
      if (plan.owners[ownerId].kind == MemoryOwnerKind::Intermediate)
        plan.owners[ownerId].kind = MemoryOwnerKind::Assembly;
      CompletionRecord ready;
      ready.id = plan.completions.size();
      ready.kind = MemoryCompletionKind::AssemblyReady;
      ready.tile = routine.tile;
      ready.routine = routineId;
      ready.ownerId = ownerId;
      ready.viewId = plan.fullViewByOwner.at(ownerId);
      plan.completions.push_back(ready);
      assembly.readinessEventId = ready.id;
      plan.assemblies.push_back(std::move(assembly));
    }
  }
  return success();
}

struct ByteRun {
  int64_t byteOffset = 0;
  int64_t byteSize = 0;
};

FailureOr<std::optional<SmallVector<ByteRun>>>
buildViewByteRuns(const ViewRecord &view, int64_t elementBytes,
                  int64_t maximumRuns, Operation *anchor) {
  if (elementBytes <= 0 || maximumRuns <= 0 ||
      view.sizes.size() != view.strides.size())
    return anchor->emitError("invalid static view geometry for segmentation");

  int64_t elementCount = 1;
  for (int64_t size : view.sizes) {
    if (size <= 0)
      return std::optional<SmallVector<ByteRun>>{};
    std::optional<int64_t> next = llvm::checkedMul(elementCount, size);
    if (!next)
      return anchor->emitError("segmented view element count overflowed");
    elementCount = *next;
  }
  std::optional<int64_t> logicalBytes =
      llvm::checkedMul(elementCount, elementBytes);
  if (!logicalBytes || *logicalBytes != view.byteSize)
    return anchor->emitError(
        "segmented view byte size disagrees with its static geometry");

  int64_t runElements = 1;
  int64_t suffixStart = view.sizes.size();
  for (int64_t dimension = static_cast<int64_t>(view.sizes.size()) - 1;
       dimension >= 0; --dimension) {
    if (view.sizes[dimension] == 1) {
      suffixStart = dimension;
      continue;
    }
    if (view.strides[dimension] != runElements)
      break;
    std::optional<int64_t> next =
        llvm::checkedMul(runElements, view.sizes[dimension]);
    if (!next)
      return anchor->emitError("segmented view run size overflowed");
    runElements = *next;
    suffixStart = dimension;
  }

  int64_t outerCount = 1;
  for (int64_t dimension = 0; dimension < suffixStart; ++dimension) {
    std::optional<int64_t> next =
        llvm::checkedMul(outerCount, view.sizes[dimension]);
    if (!next)
      return anchor->emitError("segmented view run count overflowed");
    outerCount = *next;
  }
  if (outerCount > maximumRuns)
    return std::optional<SmallVector<ByteRun>>{};

  std::optional<int64_t> runBytes = llvm::checkedMul(runElements, elementBytes);
  if (!runBytes)
    return anchor->emitError("segmented view byte run overflowed");
  SmallVector<ByteRun> runs;
  runs.reserve(outerCount);
  for (int64_t linear = 0; linear < outerCount; ++linear) {
    int64_t remaining = linear;
    int64_t physicalOffset = 0;
    for (int64_t dimension = suffixStart - 1; dimension >= 0; --dimension) {
      int64_t coordinate = remaining % view.sizes[dimension];
      remaining /= view.sizes[dimension];
      std::optional<int64_t> term =
          llvm::checkedMul(coordinate, view.strides[dimension]);
      std::optional<int64_t> next =
          term ? llvm::checkedAdd(physicalOffset, *term) : std::nullopt;
      if (!next)
        return anchor->emitError("segmented view offset overflowed");
      physicalOffset = *next;
    }
    std::optional<int64_t> physicalBytes =
        llvm::checkedMul(physicalOffset, elementBytes);
    std::optional<int64_t> byteOffset =
        physicalBytes ? llvm::checkedAdd(view.byteOffset, *physicalBytes)
                      : std::nullopt;
    if (!byteOffset)
      return anchor->emitError("segmented view byte offset overflowed");
    if (!runs.empty()) {
      std::optional<int64_t> previousEnd =
          llvm::checkedAdd(runs.back().byteOffset, runs.back().byteSize);
      if (previousEnd && *previousEnd == *byteOffset) {
        std::optional<int64_t> combined =
            llvm::checkedAdd(runs.back().byteSize, *runBytes);
        if (!combined)
          return anchor->emitError("segmented view run merge overflowed");
        runs.back().byteSize = *combined;
        continue;
      }
    }
    runs.push_back(ByteRun{*byteOffset, *runBytes});
  }
  return std::optional<SmallVector<ByteRun>>(std::move(runs));
}

LogicalResult selectMovementModesAndBuildSegments(ModuleOp deployment,
                                                  PlanRecords &plan) {
  for (MovementRecord &movement : plan.movements) {
    if (movement.mode == MemoryMovementMode::LocalAlias ||
        movement.mode == MemoryMovementMode::Assembly)
      continue;
    if (movement.sourceViewId < 0 || movement.destinationViewId < 0 ||
        movement.sourceViewId >= static_cast<int64_t>(plan.views.size()) ||
        movement.destinationViewId >= static_cast<int64_t>(plan.views.size()))
      return deployment.emitError(
          "materialized movement references an unknown view");
    const ViewRecord &source = plan.views[movement.sourceViewId];
    const ViewRecord &destination = plan.views[movement.destinationViewId];
    const OwnerRecord &sourceOwner = plan.owners[source.ownerId];
    const OwnerRecord &destinationOwner = plan.owners[destination.ownerId];
    if (isa<LogicalArrayType>(sourceOwner.type) ||
        isa<LogicalArrayType>(destinationOwner.type)) {
      movement.mode = MemoryMovementMode::Packed;
      continue;
    }
    FailureOr<int64_t> sourceElementBytes =
        getElementByteSize(sourceOwner.type, deployment);
    FailureOr<int64_t> destinationElementBytes =
        getElementByteSize(destinationOwner.type, deployment);
    if (failed(sourceElementBytes) || failed(destinationElementBytes))
      return failure();
    if (*sourceElementBytes != *destinationElementBytes ||
        source.byteSize != movement.byteSize ||
        destination.byteSize != movement.byteSize) {
      movement.mode = MemoryMovementMode::Packed;
      continue;
    }

    auto sourceRuns = buildViewByteRuns(source, *sourceElementBytes,
                                        kMaxStaticSegmentCount, deployment);
    auto destinationRuns =
        buildViewByteRuns(destination, *destinationElementBytes,
                          kMaxStaticSegmentCount, deployment);
    if (failed(sourceRuns) || failed(destinationRuns))
      return failure();
    if (!*sourceRuns || !*destinationRuns) {
      movement.mode = MemoryMovementMode::Packed;
      continue;
    }

    SmallVector<SegmentRecord> candidate;
    size_t sourceIndex = 0;
    size_t destinationIndex = 0;
    int64_t sourceConsumed = 0;
    int64_t destinationConsumed = 0;
    int64_t totalBytes = 0;
    while (sourceIndex < (**sourceRuns).size() &&
           destinationIndex < (**destinationRuns).size()) {
      const ByteRun &sourceRun = (**sourceRuns)[sourceIndex];
      const ByteRun &destinationRun = (**destinationRuns)[destinationIndex];
      int64_t sourceRemaining = sourceRun.byteSize - sourceConsumed;
      int64_t destinationRemaining =
          destinationRun.byteSize - destinationConsumed;
      int64_t bytes = std::min(sourceRemaining, destinationRemaining);
      if (bytes <= 0)
        return deployment.emitError("segmented movement contains an empty run");
      std::optional<int64_t> sourceOffset =
          llvm::checkedAdd(sourceRun.byteOffset, sourceConsumed);
      std::optional<int64_t> destinationOffset =
          llvm::checkedAdd(destinationRun.byteOffset, destinationConsumed);
      std::optional<int64_t> nextTotal = llvm::checkedAdd(totalBytes, bytes);
      if (!sourceOffset || !destinationOffset || !nextTotal)
        return deployment.emitError("segmented movement arithmetic overflowed");

      bool merged = false;
      if (!candidate.empty()) {
        std::optional<int64_t> previousSourceEnd = llvm::checkedAdd(
            candidate.back().sourceByteOffset, candidate.back().byteSize);
        std::optional<int64_t> previousDestinationEnd = llvm::checkedAdd(
            candidate.back().destinationByteOffset, candidate.back().byteSize);
        if (previousSourceEnd && previousDestinationEnd &&
            *previousSourceEnd == *sourceOffset &&
            *previousDestinationEnd == *destinationOffset) {
          std::optional<int64_t> combined =
              llvm::checkedAdd(candidate.back().byteSize, bytes);
          if (!combined)
            return deployment.emitError(
                "segmented movement coalescing overflowed");
          candidate.back().byteSize = *combined;
          merged = true;
        }
      }
      if (!merged) {
        if (candidate.size() >= kMaxStaticSegmentCount) {
          candidate.clear();
          break;
        }
        candidate.push_back(SegmentRecord{
            -1, movement.id, static_cast<int64_t>(candidate.size()),
            *sourceOffset, *destinationOffset, bytes});
      }
      totalBytes = *nextTotal;
      sourceConsumed += bytes;
      destinationConsumed += bytes;
      if (sourceConsumed == sourceRun.byteSize) {
        ++sourceIndex;
        sourceConsumed = 0;
      }
      if (destinationConsumed == destinationRun.byteSize) {
        ++destinationIndex;
        destinationConsumed = 0;
      }
    }
    if (candidate.empty() || sourceIndex != (**sourceRuns).size() ||
        destinationIndex != (**destinationRuns).size() ||
        totalBytes != movement.byteSize) {
      movement.mode = MemoryMovementMode::Packed;
      continue;
    }
    if (candidate.size() == 1) {
      movement.mode = MemoryMovementMode::Contiguous;
      continue;
    }
    movement.mode = MemoryMovementMode::Segmented;
    for (SegmentRecord &segment : candidate) {
      segment.id = plan.segments.size();
      plan.segments.push_back(segment);
    }
  }
  return success();
}

LogicalResult createEventGraph(ModuleOp deployment, PlanRecords &plan) {
  for (const auto &[routineId, routine] : plan.routines) {
    if (failed(addEventEdge(plan, plan.routineStartEvent.at(routineId),
                            plan.routineCompleteEvent.at(routineId),
                            MemoryEventEdgeKind::RoutineExecution,
                            routine.function)))
      return failure();
  }

  for (TileRoutineBindingAttr binding : plan.localBindings) {
    int64_t sourceRoutine = binding.getSourceRoutine().getInt();
    int64_t destinationRoutine = binding.getDestinationRoutine().getInt();
    if (failed(addEventEdge(plan, plan.routineCompleteEvent.at(sourceRoutine),
                            plan.routineStartEvent.at(destinationRoutine),
                            MemoryEventEdgeKind::LocalDependency, deployment)))
      return failure();
  }

  for (const auto &[routineId, routine] : plan.routines) {
    for (int64_t predecessor : routine.controlDependencies) {
      auto source = plan.routines.find(predecessor);
      if (source == plan.routines.end())
        return deployment.emitError(
                   "routine control dependency references unknown routine ")
               << predecessor;
      if (source->second.tile != routine.tile)
        return deployment.emitError("routine control dependency crosses "
                                    "physical tiles");
      if (failed(addEventEdge(plan, plan.routineCompleteEvent.at(predecessor),
                              plan.routineStartEvent.at(routineId),
                              MemoryEventEdgeKind::LocalDependency,
                              deployment)))
        return failure();
    }
  }

  for (TileRoutineRouteAttr route : plan.routes) {
    int64_t routeId = route.getId().getInt();
    int64_t sourceRoutine = route.getSourceRoutine().getInt();
    int64_t destinationRoutine = route.getDestinationRoutine().getInt();
    if (failed(addEventEdge(plan, plan.routineCompleteEvent.at(sourceRoutine),
                            plan.routeSendEvent.at(routeId),
                            MemoryEventEdgeKind::RouteSend, deployment)) ||
        failed(addEventEdge(plan, plan.routeSendEvent.at(routeId),
                            plan.routeArrivalEvent.at(routeId),
                            MemoryEventEdgeKind::NetworkTransfer,
                            deployment)) ||
        failed(addEventEdge(plan, plan.routeArrivalEvent.at(routeId),
                            plan.routineStartEvent.at(destinationRoutine),
                            MemoryEventEdgeKind::RouteReady, deployment)))
      return failure();
  }

  for (const MovementRecord &movement : plan.movements) {
    if (movement.sourceCompletionEventId < 0 ||
        movement.destinationCompletionEventId < 0)
      continue;
    std::optional<MemoryEventEdgeKind> kind;
    if (movement.mode == MemoryMovementMode::Assembly)
      kind = MemoryEventEdgeKind::AssemblyContribution;
    else if ((movement.mode == MemoryMovementMode::Contiguous &&
              movement.assemblyId >= 0) ||
             movement.mode == MemoryMovementMode::Packed ||
             movement.mode == MemoryMovementMode::Segmented)
      kind = MemoryEventEdgeKind::DMACompletion;
    if (kind && failed(addEventEdge(plan, movement.sourceCompletionEventId,
                                    movement.destinationCompletionEventId,
                                    *kind, deployment)))
      return failure();
    if (movement.assemblyContributionEventId >= 0 &&
        failed(addEventEdge(plan, movement.destinationCompletionEventId,
                            movement.assemblyContributionEventId,
                            MemoryEventEdgeKind::AssemblyContribution,
                            deployment)))
      return failure();
  }

  for (const AssemblyRecord &assembly : plan.assemblies) {
    for (int64_t contribution : assembly.completionEventIds) {
      if (failed(addEventEdge(plan, contribution, assembly.readinessEventId,
                              MemoryEventEdgeKind::AssemblyJoin, deployment)))
        return failure();
    }
    int64_t routine = plan.completions[assembly.readinessEventId].routine;
    if (failed(addEventEdge(plan, assembly.readinessEventId,
                            plan.routineCompleteEvent.at(routine),
                            MemoryEventEdgeKind::AssemblyJoin, deployment)))
      return failure();
  }

  std::map<int64_t, SmallVector<const RoutineInfo *>> bootByTile;
  std::map<int64_t, SmallVector<const RoutineInfo *>> computeByTile;
  for (const auto &[routineId, routine] : plan.routines) {
    (routine.boot ? bootByTile[routine.tile] : computeByTile[routine.tile])
        .push_back(&routine);
    (void)routineId;
  }
  for (auto &[tile, bootRoutines] : bootByTile) {
    llvm::sort(bootRoutines,
               [](const RoutineInfo *left, const RoutineInfo *right) {
                 return std::tie(left->localIndex, left->id) <
                        std::tie(right->localIndex, right->id);
               });
    for (size_t index = 1; index < bootRoutines.size(); ++index) {
      if (failed(addEventEdge(
              plan, plan.routineCompleteEvent.at(bootRoutines[index - 1]->id),
              plan.routineStartEvent.at(bootRoutines[index]->id),
              MemoryEventEdgeKind::BootBarrier, deployment)))
        return failure();
    }
    if (bootRoutines.empty())
      continue;
    int64_t barrier = plan.routineCompleteEvent.at(bootRoutines.back()->id);
    for (const RoutineInfo *compute : computeByTile[tile]) {
      if (failed(addEventEdge(plan, barrier,
                              plan.routineStartEvent.at(compute->id),
                              MemoryEventEdgeKind::BootBarrier, deployment)))
        return failure();
    }
  }
  return success();
}

FailureOr<int64_t> appendJoinEvent(PlanRecords &plan,
                                   EventReachability &reachable,
                                   MemoryCompletionKind completionKind,
                                   MemoryEventEdgeKind edgeKind, int64_t tile,
                                   int64_t routine, int64_t ownerId,
                                   ArrayRef<int64_t> predecessorEvents,
                                   bool updateReachability, Operation *anchor) {
  SmallVector<int64_t> predecessors(predecessorEvents);
  llvm::sort(predecessors);
  predecessors.erase(std::unique(predecessors.begin(), predecessors.end()),
                     predecessors.end());
  if (predecessors.empty())
    return anchor->emitError("memory final-use join has no predecessors");

  SmallVector<int64_t> successors =
      reachable.findMinimalCommonSuccessors(predecessors);

  CompletionRecord completion;
  completion.id = plan.completions.size();
  completion.kind = completionKind;
  completion.tile = tile;
  completion.routine = routine;
  completion.ownerId = ownerId;
  plan.completions.push_back(completion);

  for (int64_t predecessor : predecessors)
    if (failed(
            addEventEdge(plan, predecessor, completion.id, edgeKind, anchor)))
      return failure();
  for (int64_t successor : successors)
    if (failed(addEventEdge(plan, completion.id, successor, edgeKind, anchor)))
      return failure();
  if (updateReachability &&
      failed(reachable.insertJoin(completion.id, predecessors, successors,
                                  anchor)))
    return failure();
  return completion.id;
}

LogicalResult createFinalUseJoinEvents(ModuleOp deployment, PlanRecords &plan,
                                       EventReachability &reachable) {
  std::map<int64_t, SmallVector<int64_t>> sendsByOwner;
  for (const CompletionRecord &completion : plan.completions) {
    if (completion.kind == MemoryCompletionKind::RouteSendComplete &&
        completion.ownerId >= 0)
      sendsByOwner[completion.ownerId].push_back(completion.id);
  }
  for (auto &[ownerId, sends] : sendsByOwner) {
    const OwnerRecord &owner = plan.owners[ownerId];
    FailureOr<int64_t> event =
        appendJoinEvent(plan, reachable, MemoryCompletionKind::FinalFanOutSend,
                        MemoryEventEdgeKind::FanOutJoin, owner.tile,
                        owner.routine, ownerId, sends,
                        /*updateReachability=*/false, deployment);
    if (failed(event))
      return failure();
    plan.finalFanOutEvent[ownerId] = *event;
  }

  std::map<int64_t, SmallVector<int64_t>> consumersByOwner;
  for (const BindingRecord &binding : plan.bindings) {
    if (!binding.input ||
        binding.effect == MemoryAccessEffect::AsyncTransferDestination ||
        binding.effect == MemoryAccessEffect::AsyncTransferSource ||
        plan.owners[binding.ownerId].kind != MemoryOwnerKind::RouteInput)
      continue;
    consumersByOwner[binding.ownerId].push_back(
        plan.routineCompleteEvent.at(binding.routine));
  }
  for (auto &[ownerId, consumers] : consumersByOwner) {
    const OwnerRecord &owner = plan.owners[ownerId];
    FailureOr<int64_t> event = appendJoinEvent(
        plan, reachable, MemoryCompletionKind::FinalConsumerComplete,
        MemoryEventEdgeKind::FinalConsumer, owner.tile, owner.routine, ownerId,
        consumers, /*updateReachability=*/false, deployment);
    if (failed(event))
      return failure();
    plan.finalConsumerEvent[ownerId] = *event;
  }
  return success();
}

LogicalResult orderPendingSendsBeforeControlSuccessors(ModuleOp deployment,
                                                       PlanRecords &plan) {
  std::map<int64_t, SmallVector<int64_t>> finalSendsByRoutine;
  for (const auto &[ownerId, eventId] : plan.finalFanOutEvent)
    finalSendsByRoutine[plan.owners[ownerId].routine].push_back(eventId);

  // A task with pending output routes cannot be followed by a local task that
  // overwrites those output bytes. Blocking transmission drains every route
  // first; the overlap policy applies the same rule through its output-range
  // conflict check. Reflect that runtime invariant in the memory DAG whenever
  // an explicit local control edge orders the two tasks, allowing a bounded
  // sequence wave to reuse staging only after its terminal sends complete.
  for (const auto &[routineId, routine] : plan.routines) {
    int64_t targetStart = plan.routineStartEvent.at(routineId);
    for (int64_t predecessor : routine.controlDependencies) {
      auto source = plan.routines.find(predecessor);
      if (source == plan.routines.end())
        return deployment.emitError(
                   "routine control dependency references unknown routine ")
               << predecessor;
      if (source->second.tile != routine.tile)
        return deployment.emitError(
            "routine control dependency crosses physical tiles");
      for (int64_t finalSend : finalSendsByRoutine[predecessor])
        if (failed(addEventEdge(plan, finalSend, targetStart,
                                MemoryEventEdgeKind::LocalDependency,
                                deployment)))
          return failure();
    }
  }
  return success();
}

void collectOwnerAccessEvents(
    const PlanRecords &plan,
    std::map<int64_t, std::set<int64_t>> &accessEventsByOwner) {
  for (const BindingRecord &binding : plan.bindings) {
    if (binding.effect == MemoryAccessEffect::AsyncTransferSource ||
        binding.effect == MemoryAccessEffect::AsyncTransferDestination)
      continue;
    auto start = plan.routineStartEvent.find(binding.routine);
    auto complete = plan.routineCompleteEvent.find(binding.routine);
    if (start != plan.routineStartEvent.end())
      accessEventsByOwner[binding.ownerId].insert(start->second);
    if (complete != plan.routineCompleteEvent.end())
      accessEventsByOwner[binding.ownerId].insert(complete->second);
  }
  for (const MovementRecord &movement : plan.movements) {
    int64_t sourceOwner = plan.views[movement.sourceViewId].ownerId;
    int64_t destinationOwner = plan.views[movement.destinationViewId].ownerId;
    if (movement.sourceCompletionEventId >= 0)
      accessEventsByOwner[sourceOwner].insert(movement.sourceCompletionEventId);
    if (movement.destinationCompletionEventId >= 0)
      accessEventsByOwner[destinationOwner].insert(
          movement.destinationCompletionEventId);
    if (movement.assemblyContributionEventId >= 0)
      accessEventsByOwner[destinationOwner].insert(
          movement.assemblyContributionEventId);
  }
  for (const CompletionRecord &completion : plan.completions) {
    if (completion.ownerId >= 0)
      accessEventsByOwner[completion.ownerId].insert(completion.id);
  }
}

bool hasReleasableStorage(MemoryOwnerKind kind) {
  return kind != MemoryOwnerKind::ModelInput &&
         kind != MemoryOwnerKind::ModelOutput &&
         kind != MemoryOwnerKind::Persistent &&
         kind != MemoryOwnerKind::LocalTemporary;
}

LogicalResult createOwnerReleaseEvents(ModuleOp deployment, PlanRecords &plan,
                                       EventReachability &reachable) {
  std::map<int64_t, std::set<int64_t>> accesses;
  collectOwnerAccessEvents(plan, accesses);

  for (const OwnerRecord &owner : plan.owners) {
    if (!hasReleasableStorage(owner.kind))
      continue;
    auto ownerAccesses = accesses.find(owner.id);
    if (ownerAccesses == accesses.end() || ownerAccesses->second.empty())
      continue;
    SmallVector<int64_t> maximal;
    for (int64_t event : ownerAccesses->second) {
      bool precedesAnotherAccess =
          llvm::any_of(ownerAccesses->second, [&](int64_t other) {
            return event != other && reachable.reaches(event, other);
          });
      if (!precedesAnotherAccess)
        maximal.push_back(event);
    }
    if (auto fanOut = plan.finalFanOutEvent.find(owner.id);
        fanOut != plan.finalFanOutEvent.end())
      maximal.push_back(fanOut->second);
    if (auto consumer = plan.finalConsumerEvent.find(owner.id);
        consumer != plan.finalConsumerEvent.end())
      maximal.push_back(consumer->second);
    FailureOr<int64_t> release =
        appendJoinEvent(plan, reachable, MemoryCompletionKind::OwnerRelease,
                        MemoryEventEdgeKind::LifetimeRelease, owner.tile,
                        owner.routine, owner.id, maximal,
                        /*updateReachability=*/false, deployment);
    if (failed(release))
      return failure();
    plan.ownerReleaseEvent[owner.id] = *release;
  }
  return success();
}

void createOwnerLifetimes(PlanRecords &plan) {
  std::map<int64_t, std::set<int64_t>> accessEventsByOwner;
  std::map<int64_t, SmallVector<int64_t>> viewsByOwner;
  for (const ViewRecord &view : plan.views)
    viewsByOwner[view.ownerId].push_back(view.id);

  collectOwnerAccessEvents(plan, accessEventsByOwner);

  plan.lifetimes.clear();
  plan.lifetimes.reserve(plan.owners.size());
  for (const OwnerRecord &owner : plan.owners) {
    LifetimeRecord lifetime;
    lifetime.id = plan.lifetimes.size();
    lifetime.ownerId = owner.id;
    lifetime.routine = owner.routine;
    lifetime.tile = owner.tile;
    lifetime.byteSize = owner.byteSize;
    switch (owner.kind) {
    case MemoryOwnerKind::ModelInput:
    case MemoryOwnerKind::ModelOutput:
      lifetime.storage = MemoryLifetimeStorage::External;
      break;
    case MemoryOwnerKind::Persistent:
      lifetime.storage = MemoryLifetimeStorage::Persistent;
      break;
    case MemoryOwnerKind::LocalTemporary:
      lifetime.storage = MemoryLifetimeStorage::RoutineLocal;
      break;
    default:
      lifetime.storage = MemoryLifetimeStorage::Workspace;
      break;
    }
    lifetime.viewIds = viewsByOwner[owner.id];
    auto &accesses = accessEventsByOwner[owner.id];
    if (accesses.empty() && owner.routine >= 0) {
      if (auto start = plan.routineStartEvent.find(owner.routine);
          start != plan.routineStartEvent.end())
        accesses.insert(start->second);
      if (auto complete = plan.routineCompleteEvent.find(owner.routine);
          complete != plan.routineCompleteEvent.end())
        accesses.insert(complete->second);
    }
    lifetime.accessEventIds.assign(accesses.begin(), accesses.end());
    plan.lifetimes.push_back(std::move(lifetime));
  }
}

void attachConservativeGlobalInterference(ModuleOp module,
                                          const PlanRecords &plan) {
  DenseMap<int64_t, int64_t> lifetimeByOwner;
  for (const LifetimeRecord &lifetime : plan.lifetimes)
    if (lifetime.subjectKind == MemoryLifetimeSubjectKind::Owner)
      lifetimeByOwner[lifetime.ownerId] = lifetime.id;

  std::set<std::pair<int64_t, int64_t>> aliasPairs;
  for (const InPlaceAliasRecord &alias : plan.inPlaceAliases) {
    auto input = lifetimeByOwner.find(alias.inputOwnerId);
    auto output = lifetimeByOwner.find(alias.outputOwnerId);
    if (input == lifetimeByOwner.end() || output == lifetimeByOwner.end())
      continue;
    aliasPairs.insert(std::minmax(input->second, output->second));
  }

  SmallVector<int64_t> exceptions;
  exceptions.reserve(aliasPairs.size() * 3);
  for (auto [left, right] : aliasPairs) {
    exceptions.push_back(left);
    exceptions.push_back(right);
    exceptions.push_back(
        static_cast<int64_t>(MemoryInterferenceRelation::InPlaceAlias));
  }
  Builder builder(module.getContext());
  module->setAttr(kInterferenceDefaultAttrName,
                  builder.getStringAttr(stringifyMemoryInterferenceRelation(
                      MemoryInterferenceRelation::Interferes)));
  module->setAttr(kInterferenceExceptionsAttrName,
                  builder.getDenseI64ArrayAttr(exceptions));
  module->setAttr(kInterferencesAttrName, builder.getArrayAttr({}));
}

ArrayAttr getI64Array(MLIRContext *context, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attributes;
  attributes.reserve(values.size());
  Builder builder(context);
  for (int64_t value : values)
    attributes.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildOwnerAttrs(MLIRContext *context, ArrayRef<OwnerRecord> owners,
                          std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const OwnerRecord &owner : owners) {
    if (tile && owner.tile != *tile)
      continue;
    attributes.push_back(TileMemoryOwnerAttr::get(
        context, builder.getI64IntegerAttr(owner.id),
        builder.getI64IntegerAttr(owner.resourceId),
        builder.getI64IntegerAttr(owner.tensorId),
        builder.getI64IntegerAttr(owner.tile),
        builder.getI64IntegerAttr(owner.routine),
        builder.getI64IntegerAttr(owner.port), owner.kind,
        builder.getI64IntegerAttr(owner.byteSize),
        getI64Array(context, owner.shape),
        getI64Array(context, owner.strides)));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildViewAttrs(MLIRContext *context, ArrayRef<ViewRecord> views,
                         ArrayRef<OwnerRecord> owners,
                         std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const ViewRecord &view : views) {
    if (tile && owners[view.ownerId].tile != *tile)
      continue;
    attributes.push_back(TileMemoryViewAttr::get(
        context, builder.getI64IntegerAttr(view.id),
        builder.getI64IntegerAttr(view.ownerId),
        builder.getI64IntegerAttr(view.byteOffset),
        builder.getI64IntegerAttr(view.byteSize),
        getI64Array(context, view.offsets), getI64Array(context, view.sizes),
        getI64Array(context, view.strides), view.contiguity));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildBindingAttrs(MLIRContext *context,
                            ArrayRef<BindingRecord> bindings,
                            const std::map<int64_t, RoutineInfo> &routines,
                            std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const BindingRecord &binding : bindings) {
    if (tile && routines.at(binding.routine).tile != *tile)
      continue;
    attributes.push_back(TileMemoryBindingAttr::get(
        context, builder.getI64IntegerAttr(binding.id),
        builder.getI64IntegerAttr(binding.routine),
        builder.getI64IntegerAttr(binding.port),
        builder.getBoolAttr(binding.input),
        builder.getI64IntegerAttr(binding.ownerId),
        builder.getI64IntegerAttr(binding.viewId), binding.effect));
  }
  return builder.getArrayAttr(attributes);
}

bool movementTouchesTile(const MovementRecord &movement,
                         ArrayRef<ViewRecord> views,
                         ArrayRef<OwnerRecord> owners, int64_t tile) {
  return owners[views[movement.sourceViewId].ownerId].tile == tile ||
         owners[views[movement.destinationViewId].ownerId].tile == tile;
}

ArrayAttr buildMovementAttrs(MLIRContext *context,
                             ArrayRef<MovementRecord> movements,
                             ArrayRef<ViewRecord> views,
                             ArrayRef<OwnerRecord> owners,
                             std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const MovementRecord &movement : movements) {
    if (tile && !movementTouchesTile(movement, views, owners, *tile))
      continue;
    attributes.push_back(TileMemoryMovementAttr::get(
        context, builder.getI64IntegerAttr(movement.id), movement.mode,
        builder.getI64IntegerAttr(movement.sourceViewId),
        builder.getI64IntegerAttr(movement.destinationViewId),
        builder.getI64IntegerAttr(movement.routeId),
        builder.getI64IntegerAttr(movement.sourceCompletionEventId),
        builder.getI64IntegerAttr(movement.destinationCompletionEventId),
        builder.getI64IntegerAttr(movement.assemblyId),
        builder.getI64IntegerAttr(movement.byteSize)));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildSegmentAttrs(MLIRContext *context,
                            ArrayRef<SegmentRecord> segments,
                            ArrayRef<MovementRecord> movements,
                            ArrayRef<ViewRecord> views,
                            ArrayRef<OwnerRecord> owners,
                            std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const SegmentRecord &segment : segments) {
    auto movement = llvm::find_if(movements, [&](const MovementRecord &item) {
      return item.id == segment.movementId;
    });
    if (movement == movements.end())
      continue;
    if (tile && !movementTouchesTile(*movement, views, owners, *tile))
      continue;
    attributes.push_back(TileMemorySegmentAttr::get(
        context, builder.getI64IntegerAttr(segment.id),
        builder.getI64IntegerAttr(segment.movementId),
        builder.getI64IntegerAttr(segment.ordinal),
        builder.getI64IntegerAttr(segment.sourceByteOffset),
        builder.getI64IntegerAttr(segment.destinationByteOffset),
        builder.getI64IntegerAttr(segment.byteSize)));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildAssemblyAttrs(MLIRContext *context,
                             ArrayRef<AssemblyRecord> assemblies,
                             ArrayRef<OwnerRecord> owners,
                             std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const AssemblyRecord &assembly : assemblies) {
    if (tile && owners[assembly.ownerId].tile != *tile)
      continue;
    attributes.push_back(TileMemoryAssemblyAttr::get(
        context, builder.getI64IntegerAttr(assembly.id),
        builder.getI64IntegerAttr(assembly.ownerId),
        getI64Array(context, assembly.contributingViewIds),
        getI64Array(context, assembly.destinationViewIds),
        getI64Array(context, assembly.completionEventIds),
        builder.getI64IntegerAttr(assembly.readinessEventId),
        builder.getI64IntegerAttr(assembly.localCopyBytes),
        builder.getI64IntegerAttr(assembly.routedBytes)));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildCompletionAttrs(MLIRContext *context,
                               ArrayRef<CompletionRecord> completions,
                               std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const CompletionRecord &completion : completions) {
    if (tile && completion.tile != *tile)
      continue;
    attributes.push_back(TileMemoryCompletionEventAttr::get(
        context, builder.getI64IntegerAttr(completion.id), completion.kind,
        builder.getI64IntegerAttr(completion.tile),
        builder.getI64IntegerAttr(completion.routine),
        builder.getI64IntegerAttr(completion.routeId),
        builder.getI64IntegerAttr(completion.ownerId),
        builder.getI64IntegerAttr(completion.viewId)));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildEventEdgeAttrs(MLIRContext *context,
                              ArrayRef<EventEdgeRecord> edges,
                              ArrayRef<CompletionRecord> completions,
                              std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const EventEdgeRecord &edge : edges) {
    if (tile && (completions[edge.sourceEventId].tile != *tile ||
                 completions[edge.targetEventId].tile != *tile))
      continue;
    attributes.push_back(TileMemoryEventEdgeAttr::get(
        context, builder.getI64IntegerAttr(edge.id),
        builder.getI64IntegerAttr(edge.sourceEventId),
        builder.getI64IntegerAttr(edge.targetEventId), edge.kind));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildLifetimeAttrs(MLIRContext *context,
                             ArrayRef<LifetimeRecord> lifetimes,
                             std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const LifetimeRecord &lifetime : lifetimes) {
    if (tile && lifetime.tile != *tile)
      continue;
    attributes.push_back(TileMemoryLifetimeAttr::get(
        context, builder.getI64IntegerAttr(lifetime.id), lifetime.subjectKind,
        lifetime.storage, builder.getI64IntegerAttr(lifetime.ownerId),
        builder.getI64IntegerAttr(lifetime.routine),
        builder.getI64IntegerAttr(lifetime.allocationOrdinal),
        builder.getI64IntegerAttr(lifetime.tile),
        builder.getI64IntegerAttr(lifetime.byteSize),
        builder.getI64IntegerAttr(lifetime.alignment),
        builder.getI64IntegerAttr(lifetime.offset),
        getI64Array(context, lifetime.viewIds),
        getI64Array(context, lifetime.accessEventIds)));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr buildInPlaceAliasAttrs(MLIRContext *context,
                                 ArrayRef<InPlaceAliasRecord> aliases,
                                 const std::map<int64_t, RoutineInfo> &routines,
                                 std::optional<int64_t> tile = std::nullopt) {
  Builder builder(context);
  SmallVector<Attribute> attributes;
  for (const InPlaceAliasRecord &alias : aliases) {
    auto routine = routines.find(alias.routine);
    if (routine == routines.end() || (tile && routine->second.tile != *tile))
      continue;
    attributes.push_back(TileMemoryInPlaceAliasAttr::get(
        context, builder.getI64IntegerAttr(alias.id),
        builder.getI64IntegerAttr(alias.routine),
        builder.getI64IntegerAttr(alias.inputPort),
        builder.getI64IntegerAttr(alias.outputPort),
        builder.getI64IntegerAttr(alias.inputOwnerId),
        builder.getI64IntegerAttr(alias.outputOwnerId),
        builder.getI64IntegerAttr(alias.inputViewId),
        builder.getI64IntegerAttr(alias.outputViewId)));
  }
  return builder.getArrayAttr(attributes);
}

void attachPlan(ModuleOp module, const PlanRecords &plan,
                std::optional<int64_t> tile = std::nullopt) {
  MLIRContext *context = module.getContext();
  Builder builder(context);
  module->setAttr(kPlanVersionAttrName, builder.getI64IntegerAttr(3));
  module->setAttr(kOwnersAttrName, buildOwnerAttrs(context, plan.owners, tile));
  module->setAttr(kViewsAttrName,
                  buildViewAttrs(context, plan.views, plan.owners, tile));
  module->setAttr(kBindingsAttrName, buildBindingAttrs(context, plan.bindings,
                                                       plan.routines, tile));
  module->setAttr(kMovementsAttrName,
                  buildMovementAttrs(context, plan.movements, plan.views,
                                     plan.owners, tile));
  module->setAttr(kSegmentsAttrName,
                  buildSegmentAttrs(context, plan.segments, plan.movements,
                                    plan.views, plan.owners, tile));
  module->setAttr(
      kAssembliesAttrName,
      buildAssemblyAttrs(context, plan.assemblies, plan.owners, tile));
  module->setAttr(kCompletionEventsAttrName,
                  buildCompletionAttrs(context, plan.completions, tile));
  module->setAttr(
      kEventEdgesAttrName,
      buildEventEdgeAttrs(context, plan.eventEdges, plan.completions, tile));
  module->setAttr(kLifetimesAttrName,
                  buildLifetimeAttrs(context, plan.lifetimes, tile));
  module->setAttr(kInPlaceAliasesAttrName,
                  buildInPlaceAliasAttrs(context, plan.inPlaceAliases,
                                         plan.routines, tile));
}

std::map<int64_t, PlanRecords>
bucketPlanRecordsByTile(const PlanRecords &plan,
                        const EventReachability &globalReachability) {
  std::map<int64_t, PlanRecords> result;
  for (const auto &[routineId, routine] : plan.routines)
    result[routine.tile].routines.emplace(routineId, routine);
  for (const OwnerRecord &owner : plan.owners)
    result[owner.tile].owners.push_back(owner);
  for (const ViewRecord &view : plan.views)
    result[plan.owners[view.ownerId].tile].views.push_back(view);
  for (const BindingRecord &binding : plan.bindings)
    result[plan.routines.at(binding.routine).tile].bindings.push_back(binding);
  for (const MovementRecord &movement : plan.movements) {
    int64_t sourceTile =
        plan.owners[plan.views[movement.sourceViewId].ownerId].tile;
    int64_t destinationTile =
        plan.owners[plan.views[movement.destinationViewId].ownerId].tile;
    result[sourceTile].movements.push_back(movement);
    if (destinationTile != sourceTile)
      result[destinationTile].movements.push_back(movement);
  }
  DenseMap<int64_t, const MovementRecord *> movementById;
  for (const MovementRecord &movement : plan.movements)
    movementById[movement.id] = &movement;
  for (const SegmentRecord &segment : plan.segments) {
    auto found = movementById.find(segment.movementId);
    if (found == movementById.end())
      continue;
    const MovementRecord &movement = *found->second;
    int64_t sourceTile =
        plan.owners[plan.views[movement.sourceViewId].ownerId].tile;
    int64_t destinationTile =
        plan.owners[plan.views[movement.destinationViewId].ownerId].tile;
    result[sourceTile].segments.push_back(segment);
    if (destinationTile != sourceTile)
      result[destinationTile].segments.push_back(segment);
  }
  for (const AssemblyRecord &assembly : plan.assemblies)
    result[plan.owners[assembly.ownerId].tile].assemblies.push_back(assembly);
  for (const CompletionRecord &completion : plan.completions)
    result[completion.tile].completions.push_back(completion);
  for (const EventEdgeRecord &edge : plan.eventEdges) {
    int64_t sourceTile = plan.completions[edge.sourceEventId].tile;
    if (plan.completions[edge.targetEventId].tile == sourceTile)
      result[sourceTile].eventEdges.push_back(edge);
  }

  // A tile-local memory plan must retain happens-before paths that leave the
  // tile and later return.  Dropping every cross-tile edge makes sequential
  // model stages appear concurrent after extraction, so route buffers that
  // can safely reuse storage are incorrectly forced to coexist.  Project the
  // deployment-wide DAG onto each tile by adding only the first reachable
  // local successors.  These analysis-only local_dependency edges summarize
  // already-proven paths; they do not impose new runtime ordering.
  std::map<int64_t, SmallVector<int64_t>> eventIdsByTile;
  for (const CompletionRecord &completion : plan.completions)
    eventIdsByTile[completion.tile].push_back(completion.id);

  int64_t nextEventEdgeId = 0;
  for (const EventEdgeRecord &edge : plan.eventEdges)
    nextEventEdgeId = std::max(nextEventEdgeId, edge.id + 1);
  for (auto &[tile, eventIds] : eventIdsByTile) {
    llvm::sort(eventIds, [&](int64_t left, int64_t right) {
      return std::pair{*globalReachability.topologicalRank(left), left} <
             std::pair{*globalReachability.topologicalRank(right), right};
    });
    std::set<std::pair<int64_t, int64_t>> directEdges;
    for (const EventEdgeRecord &edge : result[tile].eventEdges)
      directEdges.emplace(edge.sourceEventId, edge.targetEventId);

    for (auto [sourceIndex, source] : llvm::enumerate(eventIds)) {
      SmallVector<int64_t> firstLocalSuccessors;
      for (int64_t target :
           ArrayRef<int64_t>(eventIds).drop_front(sourceIndex + 1)) {
        if (!globalReachability.reaches(source, target))
          continue;
        bool reachedThroughEarlierLocal =
            llvm::any_of(firstLocalSuccessors, [&](int64_t intermediate) {
              return globalReachability.reaches(intermediate, target);
            });
        if (reachedThroughEarlierLocal)
          continue;
        firstLocalSuccessors.push_back(target);
        if (directEdges.emplace(source, target).second) {
          result[tile].eventEdges.push_back(
              {nextEventEdgeId++, source, target,
               MemoryEventEdgeKind::LocalDependency});
        }
      }
    }
  }
  for (const LifetimeRecord &lifetime : plan.lifetimes)
    result[lifetime.tile].lifetimes.push_back(lifetime);
  for (const InPlaceAliasRecord &alias : plan.inPlaceAliases)
    result[plan.routines.at(alias.routine).tile].inPlaceAliases.push_back(
        alias);
  return result;
}

bool hasSeparateStorage(TileMemoryLifetimeAttr left,
                        TileMemoryLifetimeAttr right) {
  auto isNonWorkspace = [](MemoryLifetimeStorage storage) {
    return storage == MemoryLifetimeStorage::External ||
           storage == MemoryLifetimeStorage::Persistent;
  };
  if (isNonWorkspace(left.getStorage()) || isNonWorkspace(right.getStorage()))
    return true;
  return (left.getStorage() == MemoryLifetimeStorage::Scratchpad) !=
         (right.getStorage() == MemoryLifetimeStorage::Scratchpad);
}

LogicalResult computeAndAttachInterferences(ModuleOp module) {
  auto lifetimes =
      parseTypedArray<TileMemoryLifetimeAttr>(module, kLifetimesAttrName);
  auto completions = parseTypedArray<TileMemoryCompletionEventAttr>(
      module, kCompletionEventsAttrName);
  auto edges =
      parseTypedArray<TileMemoryEventEdgeAttr>(module, kEventEdgesAttrName);
  auto aliases = parseTypedArray<TileMemoryInPlaceAliasAttr>(
      module, kInPlaceAliasesAttrName);
  if (failed(lifetimes) || failed(completions) || failed(edges) ||
      failed(aliases))
    return failure();

  std::map<int64_t, int64_t> lifetimeByOwner;
  for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
    if (lifetime.getSubjectKind() == MemoryLifetimeSubjectKind::Owner)
      lifetimeByOwner[lifetime.getOwnerId().getInt()] =
          lifetime.getId().getInt();
  }
  std::set<std::pair<int64_t, int64_t>> aliasLifetimePairs;
  for (TileMemoryInPlaceAliasAttr alias : *aliases) {
    auto input = lifetimeByOwner.find(alias.getInputOwnerId().getInt());
    auto output = lifetimeByOwner.find(alias.getOutputOwnerId().getInt());
    if (input == lifetimeByOwner.end() || output == lifetimeByOwner.end())
      return module.emitError("in-place alias references an unknown lifetime");
    aliasLifetimePairs.insert(std::minmax(input->second, output->second));
  }

  std::set<int64_t> eventIds;
  SmallVector<int64_t> eventIdList;
  SmallVector<std::pair<int64_t, int64_t>> eventGraphEdges;
  eventIdList.reserve(completions->size());
  eventGraphEdges.reserve(edges->size());
  for (TileMemoryCompletionEventAttr completion : *completions) {
    int64_t eventId = completion.getId().getInt();
    eventIds.insert(eventId);
    eventIdList.push_back(eventId);
  }
  for (TileMemoryEventEdgeAttr edge : *edges) {
    int64_t source = edge.getSourceEventId().getInt();
    int64_t target = edge.getTargetEventId().getInt();
    if (!eventIds.count(source) || !eventIds.count(target))
      return module.emitError(
          "cannot analyze interference with an unknown event edge");
    eventGraphEdges.emplace_back(source, target);
  }
  FailureOr<EventReachability> reachable =
      computeEventReachability(eventIdList, eventGraphEdges, module);
  if (failed(reachable))
    return failure();

  std::map<int64_t, SmallVector<int64_t>> accesses;
  std::map<int64_t, llvm::BitVector> reachableFromAllAccesses;
  for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
    auto values = parseIntegerArray(module, lifetime.getAccessEventIds(),
                                    "lifetime access events");
    if (failed(values))
      return failure();
    for (int64_t event : *values)
      if (!eventIds.count(event))
        return module.emitError(
            "cannot analyze a lifetime with an unknown access event");
    reachableFromAllAccesses[lifetime.getId().getInt()] =
        reachable->reachableFromAll(*values);
    accesses[lifetime.getId().getInt()] = std::move(*values);
  }

  auto allBefore = [&](TileMemoryLifetimeAttr left,
                       TileMemoryLifetimeAttr right) {
    ArrayRef<int64_t> leftAccesses = accesses[left.getId().getInt()];
    ArrayRef<int64_t> rightAccesses = accesses[right.getId().getInt()];
    if (leftAccesses.empty() || rightAccesses.empty())
      return false;
    const llvm::BitVector &common =
        reachableFromAllAccesses.at(left.getId().getInt());
    return llvm::all_of(rightAccesses, [&](int64_t target) {
      return reachable->maskContains(common, target);
    });
  };

  llvm::sort(*lifetimes,
             [](TileMemoryLifetimeAttr left, TileMemoryLifetimeAttr right) {
               return left.getId().getInt() < right.getId().getInt();
             });
  std::map<int64_t, TileMemoryLifetimeAttr> lifetimeById;
  std::map<int64_t, SmallVector<int64_t>> lifetimeIdsByTile;
  for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
    int64_t lifetimeId = lifetime.getId().getInt();
    lifetimeById[lifetimeId] = lifetime;
    lifetimeIdsByTile[lifetime.getTile().getInt()].push_back(lifetimeId);
  }
  size_t localPairCount = 0;
  for (const auto &[tile, tileLifetimeIds] : lifetimeIdsByTile) {
    size_t count = tileLifetimeIds.size();
    if (count > 1) {
      if (count > std::numeric_limits<size_t>::max() / (count - 1))
        return module.emitError("tile memory interference pair count overflow");
      size_t pairs = count * (count - 1) / 2;
      if (localPairCount > std::numeric_limits<size_t>::max() - pairs)
        return module.emitError("tile memory interference pair count overflow");
      localPairCount += pairs;
    }
    (void)tile;
  }
  if (localPairCount > kMaxExactInterferencePairCount) {
    // Serializing every exact pair is quadratic in local lifetimes. Keep a
    // conservative descriptive summary for large task graphs. Executable
    // workspace allocation remains exact: buildExactWorkspaceLayout queries
    // this same event DAG lazily without materializing the pair table.
    Builder builder(module.getContext());
    SmallVector<int64_t> exceptions;
    exceptions.reserve(aliasLifetimePairs.size() * 3);
    for (auto [left, right] : aliasLifetimePairs) {
      exceptions.push_back(left);
      exceptions.push_back(right);
      exceptions.push_back(
          static_cast<int64_t>(MemoryInterferenceRelation::InPlaceAlias));
    }
    module->setAttr(kInterferenceDefaultAttrName,
                    builder.getStringAttr(stringifyMemoryInterferenceRelation(
                        MemoryInterferenceRelation::Interferes)));
    module->setAttr(kInterferenceExceptionsAttrName,
                    builder.getDenseI64ArrayAttr(exceptions));
    module->setAttr(kInterferencesAttrName, builder.getArrayAttr({}));
    return success();
  }
  SmallVector<std::pair<int64_t, int64_t>> localLifetimePairs;
  localLifetimePairs.reserve(localPairCount);
  for (auto &[tile, tileLifetimeIds] : lifetimeIdsByTile) {
    for (size_t leftIndex = 0; leftIndex < tileLifetimeIds.size(); ++leftIndex)
      for (size_t rightIndex = leftIndex + 1;
           rightIndex < tileLifetimeIds.size(); ++rightIndex)
        localLifetimePairs.emplace_back(tileLifetimeIds[leftIndex],
                                        tileLifetimeIds[rightIndex]);
    (void)tile;
  }
  llvm::sort(localLifetimePairs);

  SmallVector<MemoryInterferenceRelation> relationValues;
  relationValues.reserve(localLifetimePairs.size());
  size_t beforeCount = 0;
  size_t interferesCount = 0;
  for (auto [leftId, rightId] : localLifetimePairs) {
    TileMemoryLifetimeAttr left = lifetimeById.at(leftId);
    TileMemoryLifetimeAttr right = lifetimeById.at(rightId);
    MemoryInterferenceRelation relation =
        MemoryInterferenceRelation::Interferes;
    if (hasSeparateStorage(left, right))
      relation = MemoryInterferenceRelation::SeparateStorage;
    else if (aliasLifetimePairs.count(
                 {left.getId().getInt(), right.getId().getInt()}))
      relation = MemoryInterferenceRelation::InPlaceAlias;
    else if (allBefore(left, right))
      relation = MemoryInterferenceRelation::Before;
    else if (allBefore(right, left))
      relation = MemoryInterferenceRelation::After;
    relationValues.push_back(relation);
    beforeCount += relation == MemoryInterferenceRelation::Before;
    interferesCount += relation == MemoryInterferenceRelation::Interferes;
  }

  // A complete pair table is quadratic and dominated deployment size for
  // model-scale graphs (BERT produced hundreds of megabytes of attributes).
  // Store the more common ordinary relation as a module default and emit only
  // exceptions. Alias and separate-storage relations always remain explicit.
  MemoryInterferenceRelation defaultRelation =
      beforeCount > interferesCount ? MemoryInterferenceRelation::Before
                                    : MemoryInterferenceRelation::Interferes;
  Builder builder(module.getContext());
  if (localLifetimePairs.size() <= kMaxExplicitInterferencePairCount) {
    SmallVector<Attribute> relations;
    relations.reserve(localLifetimePairs.size());
    for (auto [index, pair] : llvm::enumerate(localLifetimePairs)) {
      TileMemoryLifetimeAttr left = lifetimeById.at(pair.first);
      TileMemoryLifetimeAttr right = lifetimeById.at(pair.second);
      relations.push_back(TileMemoryInterferenceAttr::get(
          module.getContext(), builder.getI64IntegerAttr(index), left.getId(),
          right.getId(), relationValues[index]));
    }
    module->removeAttr(kInterferenceDefaultAttrName);
    module->removeAttr(kInterferenceExceptionsAttrName);
    module->setAttr(kInterferencesAttrName, builder.getArrayAttr(relations));
    return success();
  }

  SmallVector<int64_t> relationExceptions;
  relationExceptions.reserve(
      3 * (localLifetimePairs.size() - std::max(beforeCount, interferesCount)));
  for (auto [index, pair] : llvm::enumerate(localLifetimePairs)) {
    MemoryInterferenceRelation relation = relationValues[index];
    if (relation == defaultRelation)
      continue;
    relationExceptions.push_back(pair.first);
    relationExceptions.push_back(pair.second);
    relationExceptions.push_back(static_cast<int64_t>(relation));
  }
  module->setAttr(kInterferenceDefaultAttrName,
                  builder.getStringAttr(
                      stringifyMemoryInterferenceRelation(defaultRelation)));
  module->setAttr(kInterferenceExceptionsAttrName,
                  builder.getDenseI64ArrayAttr(relationExceptions));
  // Keep the legacy typed table present but empty. Readers accept old complete
  // tables and merge this compact exception encoding for new plans.
  module->setAttr(kInterferencesAttrName, builder.getArrayAttr({}));
  return success();
}

FailureOr<ExactWorkspaceLayout>
buildExactWorkspaceLayoutImpl(ModuleOp module,
                              ArrayRef<WorkspaceAllocationRequest> requests) {
  auto lifetimes =
      parseTypedArray<TileMemoryLifetimeAttr>(module, kLifetimesAttrName);
  auto completions = parseTypedArray<TileMemoryCompletionEventAttr>(
      module, kCompletionEventsAttrName);
  auto edges =
      parseTypedArray<TileMemoryEventEdgeAttr>(module, kEventEdgesAttrName);
  auto aliases = parseTypedArray<TileMemoryInPlaceAliasAttr>(
      module, kInPlaceAliasesAttrName);
  if (failed(lifetimes) || failed(completions) || failed(edges) ||
      failed(aliases))
    return failure();

  std::map<int64_t, TileMemoryLifetimeAttr> lifetimeById;
  std::map<int64_t, int64_t> lifetimeByOwner;
  for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
    int64_t lifetimeId = lifetime.getId().getInt();
    if (!lifetimeById.emplace(lifetimeId, lifetime).second)
      return module.emitError(
          "exact workspace allocation found a duplicate lifetime ID");
    if (lifetime.getSubjectKind() == MemoryLifetimeSubjectKind::Owner)
      lifetimeByOwner[lifetime.getOwnerId().getInt()] = lifetimeId;
  }

  SmallVector<int64_t> eventIds;
  SmallVector<std::pair<int64_t, int64_t>> eventGraphEdges;
  eventIds.reserve(completions->size());
  eventGraphEdges.reserve(edges->size());
  for (TileMemoryCompletionEventAttr completion : *completions)
    eventIds.push_back(completion.getId().getInt());
  for (TileMemoryEventEdgeAttr edge : *edges)
    eventGraphEdges.emplace_back(edge.getSourceEventId().getInt(),
                                 edge.getTargetEventId().getInt());
  FailureOr<EventReachability> reachable =
      computeEventReachability(eventIds, eventGraphEdges, module);
  if (failed(reachable))
    return failure();

  struct Candidate {
    int64_t lifetimeId = -1;
    int64_t byteSize = 0;
    int64_t alignment = 1;
    SmallVector<int64_t> accesses;
    llvm::BitVector reachableFromAllAccesses;
  };
  std::map<int64_t, Candidate> candidateByLifetime;
  for (const WorkspaceAllocationRequest &request : requests) {
    auto lifetime = lifetimeById.find(request.lifetimeId);
    if (lifetime == lifetimeById.end())
      return module.emitError(
                 "workspace allocation references unknown lifetime ")
             << request.lifetimeId;
    if (lifetime->second.getStorage() != MemoryLifetimeStorage::Workspace ||
        request.byteSize != lifetime->second.getByteSize().getInt() ||
        request.alignment != lifetime->second.getAlignment().getInt() ||
        request.byteSize < 0 || request.alignment <= 0 ||
        !llvm::isPowerOf2_64(request.alignment))
      return module.emitError(
                 "workspace allocation request disagrees with lifetime ")
             << request.lifetimeId;
    FailureOr<SmallVector<int64_t>> accesses =
        parseIntegerArray(module, lifetime->second.getAccessEventIds(),
                          "exact workspace lifetime access events");
    if (failed(accesses) || accesses->empty())
      return module.emitError(
                 "workspace allocation requires nonempty access events for "
                 "lifetime ")
             << request.lifetimeId;
    Candidate candidate;
    candidate.lifetimeId = request.lifetimeId;
    candidate.byteSize = std::max<int64_t>(request.byteSize, 1);
    candidate.alignment = request.alignment;
    candidate.accesses = std::move(*accesses);
    candidate.reachableFromAllAccesses =
        reachable->reachableFromAll(candidate.accesses);
    auto existing = candidateByLifetime.find(candidate.lifetimeId);
    if (existing != candidateByLifetime.end()) {
      // Fan-out routes can materialize several runtime resource slots backed
      // by the same global owner. They intentionally share one lifetime and
      // therefore one allocation; reject only inconsistent duplicate views.
      if (existing->second.byteSize != candidate.byteSize ||
          existing->second.alignment != candidate.alignment ||
          existing->second.accesses != candidate.accesses)
        return module.emitError(
                   "workspace allocation has inconsistent requests for "
                   "lifetime ")
               << request.lifetimeId;
      continue;
    }
    candidateByLifetime.emplace(candidate.lifetimeId, std::move(candidate));
  }

  // Every workspace owner must participate. Omitting one would make the
  // resulting capacity look smaller than the executable resource table.
  size_t expectedWorkspaceLifetimes =
      llvm::count_if(*lifetimes, [](TileMemoryLifetimeAttr lifetime) {
        return lifetime.getSubjectKind() == MemoryLifetimeSubjectKind::Owner &&
               lifetime.getStorage() == MemoryLifetimeStorage::Workspace;
      });
  if (candidateByLifetime.size() != expectedWorkspaceLifetimes)
    return module.emitError(
        "exact workspace allocation does not cover every workspace owner");

  std::map<int64_t, SmallVector<int64_t>> aliasNeighbors;
  for (TileMemoryInPlaceAliasAttr alias : *aliases) {
    auto input = lifetimeByOwner.find(alias.getInputOwnerId().getInt());
    auto output = lifetimeByOwner.find(alias.getOutputOwnerId().getInt());
    if (input == lifetimeByOwner.end() || output == lifetimeByOwner.end())
      return module.emitError("in-place alias has no exact workspace lifetime");
    bool inputWorkspace = candidateByLifetime.count(input->second);
    bool outputWorkspace = candidateByLifetime.count(output->second);
    if (inputWorkspace != outputWorkspace)
      return module.emitError(
          "in-place alias crosses workspace storage classes");
    if (!inputWorkspace)
      continue;
    aliasNeighbors[input->second].push_back(output->second);
    aliasNeighbors[output->second].push_back(input->second);
  }

  struct Group {
    int64_t key = -1;
    SmallVector<int64_t> lifetimeIds;
    int64_t byteSize = 0;
    int64_t alignment = 1;
    int64_t offset = -1;
  };
  llvm::SmallDenseSet<int64_t> grouped;
  SmallVector<Group> groups;
  for (const auto &[lifetimeId, candidate] : candidateByLifetime) {
    if (!grouped.insert(lifetimeId).second)
      continue;
    Group group;
    SmallVector<int64_t> worklist{lifetimeId};
    while (!worklist.empty()) {
      int64_t memberId = worklist.pop_back_val();
      auto member = candidateByLifetime.find(memberId);
      if (member == candidateByLifetime.end())
        return module.emitError(
            "in-place alias references non-workspace storage");
      group.lifetimeIds.push_back(memberId);
      group.byteSize = std::max(group.byteSize, member->second.byteSize);
      group.alignment = std::max(group.alignment, member->second.alignment);
      for (int64_t neighbor : aliasNeighbors[memberId])
        if (grouped.insert(neighbor).second)
          worklist.push_back(neighbor);
    }
    llvm::sort(group.lifetimeIds);
    group.key = group.lifetimeIds.front();
    groups.push_back(std::move(group));
  }
  llvm::sort(groups, [](const Group &left, const Group &right) {
    return left.key < right.key;
  });

  auto allBefore = [&](int64_t leftId, int64_t rightId) {
    const Candidate &left = candidateByLifetime.at(leftId);
    const Candidate &right = candidateByLifetime.at(rightId);
    return llvm::all_of(right.accesses, [&](int64_t target) {
      return reachable->maskContains(left.reachableFromAllAccesses, target);
    });
  };
  auto canShare = [&](int64_t leftId, int64_t rightId) {
    return allBefore(leftId, rightId) || allBefore(rightId, leftId);
  };
  auto rangesOverlap = [](int64_t leftOffset, int64_t leftSize,
                          int64_t rightOffset, int64_t rightSize) {
    return leftOffset < rightOffset + rightSize &&
           rightOffset < leftOffset + leftSize;
  };

  int64_t workspaceBytes = 0;
  SmallVector<Group> assigned;
  assigned.reserve(groups.size());
  // Keep the assigned groups ordered by address.  The previous first-fit
  // implementation restarted a full scan every time it encountered one
  // occupied interval.  When no lifetimes could share storage, group N was
  // therefore scanned N times across N prior groups, making allocation cubic
  // in the number of lifetimes.  A single address-ordered sweep finds the
  // same first aligned gap: non-interfering intervals are ignored, intervals
  // below the candidate are skipped, and an overlap advances the candidate
  // directly to that interval's aligned end.
  SmallVector<size_t> assignedByOffset;
  assignedByOffset.reserve(groups.size());
  for (Group group : groups) {
    int64_t offset = 0;
    for (size_t existingIndex : assignedByOffset) {
      const Group &existing = assigned[existingIndex];
      bool allShare = llvm::all_of(group.lifetimeIds, [&](int64_t current) {
        return llvm::all_of(existing.lifetimeIds, [&](int64_t prior) {
          return canShare(current, prior);
        });
      });
      if (allShare)
        continue;

      std::optional<int64_t> rounded =
          llvm::checkedAdd(offset, group.alignment - 1);
      if (!rounded)
        return module.emitError(
            "exact workspace allocation alignment overflowed");
      offset = (*rounded / group.alignment) * group.alignment;

      std::optional<int64_t> candidateEnd =
          llvm::checkedAdd(offset, group.byteSize);
      std::optional<int64_t> existingEnd =
          llvm::checkedAdd(existing.offset, existing.byteSize);
      if (!candidateEnd || !existingEnd)
        return module.emitError("exact workspace allocation overflowed");
      if (*candidateEnd <= existing.offset)
        break;
      if (*existingEnd <= offset)
        continue;
      if (!rangesOverlap(offset, group.byteSize, existing.offset,
                         existing.byteSize))
        continue;
      offset = *existingEnd;
    }
    std::optional<int64_t> rounded =
        llvm::checkedAdd(offset, group.alignment - 1);
    if (!rounded)
      return module.emitError(
          "exact workspace allocation alignment overflowed");
    offset = (*rounded / group.alignment) * group.alignment;
    group.offset = offset;
    std::optional<int64_t> end = llvm::checkedAdd(offset, group.byteSize);
    if (!end)
      return module.emitError("exact workspace allocation overflowed");
    workspaceBytes = std::max(workspaceBytes, *end);
    assigned.push_back(std::move(group));
    size_t newIndex = assigned.size() - 1;
    auto insertion = llvm::lower_bound(
        assignedByOffset, newIndex, [&](size_t left, size_t right) {
          return std::tie(assigned[left].offset, assigned[left].key) <
                 std::tie(assigned[right].offset, assigned[right].key);
        });
    assignedByOffset.insert(insertion, newIndex);
  }

  ExactWorkspaceLayout result;
  result.workspaceBytes = workspaceBytes;
  result.allocations.reserve(candidateByLifetime.size());
  for (const Group &group : assigned)
    for (int64_t lifetimeId : group.lifetimeIds)
      result.allocations.push_back({lifetimeId, group.offset});
  llvm::sort(result.allocations, [](const WorkspaceAllocation &left,
                                    const WorkspaceAllocation &right) {
    return left.lifetimeId < right.lifetimeId;
  });
  return result;
}

LogicalResult checkedAccumulate(Operation *anchor, int64_t &target,
                                int64_t value, StringRef description) {
  std::optional<int64_t> sum = llvm::checkedAdd(target, value);
  if (!sum)
    return anchor->emitError("tile memory capacity overflow in ")
           << description;
  target = *sum;
  return success();
}

FailureOr<int64_t> alignedValue(Operation *anchor, int64_t value,
                                int64_t alignment) {
  if (value < 0 || alignment <= 0 || !llvm::isPowerOf2_64(alignment))
    return anchor->emitError("invalid tile memory capacity alignment");
  std::optional<int64_t> rounded = llvm::checkedAdd(value, alignment - 1);
  if (!rounded)
    return anchor->emitError("tile memory capacity alignment overflow");
  return (*rounded / alignment) * alignment;
}

LogicalResult computeAndAttachCapacity(ModuleOp module) {
  auto lifetimes =
      parseTypedArray<TileMemoryLifetimeAttr>(module, kLifetimesAttrName);
  auto owners = parseTypedArray<TileMemoryOwnerAttr>(module, kOwnersAttrName);
  auto assemblies =
      parseTypedArray<TileMemoryAssemblyAttr>(module, kAssembliesAttrName);
  if (failed(lifetimes) || failed(owners) || failed(assemblies))
    return failure();

  int64_t tile = -1;
  if (auto value = module->getAttrOfType<IntegerAttr>(kPhysicalTileIdAttr))
    tile = value.getInt();
  else if (!lifetimes->empty())
    tile = lifetimes->front().getTile().getInt();
  if (tile < 0)
    return module.emitError("capacity summary requires one physical tile");
  for (TileMemoryLifetimeAttr lifetime : *lifetimes)
    if (lifetime.getTile().getInt() != tile)
      return module.emitError(
          "capacity summary cannot combine multiple physical tiles");

  std::map<int64_t, MemoryOwnerKind> ownerKind;
  std::map<int64_t, TileMemoryOwnerAttr> ownerById;
  for (TileMemoryOwnerAttr owner : *owners) {
    ownerKind[owner.getId().getInt()] = owner.getKind();
    ownerById[owner.getId().getInt()] = owner;
  }

  int64_t externalBytes = 0;
  int64_t persistentBytes = 0;
  int64_t scratchpadBytes = 0;
  int64_t workspaceLogicalBytes = 0;
  int64_t workspaceBytes = 0;
  int64_t routeInputBytes = 0;
  int64_t routeOutputBytes = 0;
  int64_t assemblyBytes = 0;
  int64_t intermediateBytes = 0;
  int64_t routineTemporaryTotalBytes = 0;
  std::map<int64_t, int64_t> routineTemporaryBytes;
  bool workspaceAssigned = true;
  bool scratchpadAssigned = true;
  bool complete = true;

  llvm::SmallDenseSet<int64_t> assemblyOwnerIds;
  for (TileMemoryAssemblyAttr assembly : *assemblies) {
    int64_t ownerId = assembly.getOwnerId().getInt();
    auto owner = ownerById.find(ownerId);
    if (owner == ownerById.end())
      return module.emitError("capacity assembly names an unknown owner");
    if (assemblyOwnerIds.insert(ownerId).second &&
        failed(checkedAccumulate(module, assemblyBytes,
                                 owner->second.getByteSize().getInt(),
                                 "assembly destination bytes")))
      return failure();
  }

  for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
    int64_t bytes = lifetime.getByteSize().getInt();
    int64_t offset = lifetime.getOffset().getInt();
    int64_t alignment = lifetime.getAlignment().getInt();
    auto storage = lifetime.getStorage();
    if (storage == MemoryLifetimeStorage::External) {
      if (failed(checkedAccumulate(module, externalBytes, bytes,
                                   "external storage")))
        return failure();
    } else if (storage == MemoryLifetimeStorage::Persistent) {
      if (failed(checkedAccumulate(module, persistentBytes, bytes,
                                   "persistent storage")))
        return failure();
    } else if (storage == MemoryLifetimeStorage::Workspace) {
      if (failed(checkedAccumulate(module, workspaceLogicalBytes, bytes,
                                   "workspace logical bytes")))
        return failure();
      if (offset < 0) {
        workspaceAssigned = false;
      } else {
        std::optional<int64_t> end = llvm::checkedAdd(offset, bytes);
        if (!end)
          return module.emitError("workspace capacity range overflow");
        workspaceBytes = std::max(workspaceBytes, *end);
      }
    } else if (storage == MemoryLifetimeStorage::Scratchpad) {
      if (offset < 0) {
        scratchpadAssigned = false;
        FailureOr<int64_t> aligned =
            alignedValue(module, scratchpadBytes, alignment);
        if (failed(aligned) ||
            failed(checkedAccumulate(module, scratchpadBytes,
                                     *aligned - scratchpadBytes + bytes,
                                     "scratchpad storage")))
          return failure();
      } else {
        std::optional<int64_t> end = llvm::checkedAdd(offset, bytes);
        if (!end)
          return module.emitError("scratchpad capacity range overflow");
        scratchpadBytes = std::max(scratchpadBytes, *end);
      }
    } else {
      if (failed(checkedAccumulate(module, routineTemporaryTotalBytes, bytes,
                                   "routine temporary storage")) ||
          failed(checkedAccumulate(
              module, routineTemporaryBytes[lifetime.getRoutine().getInt()],
              bytes, "per-routine temporary storage")))
        return failure();
    }

    if (lifetime.getSubjectKind() != MemoryLifetimeSubjectKind::Owner)
      continue;
    auto kind = ownerKind.find(lifetime.getOwnerId().getInt());
    if (kind == ownerKind.end())
      return module.emitError("capacity lifetime names an unknown owner");
    int64_t *category = nullptr;
    switch (kind->second) {
    case MemoryOwnerKind::RouteInput:
      category = &routeInputBytes;
      break;
    case MemoryOwnerKind::RouteOutput:
      category = &routeOutputBytes;
      break;
    case MemoryOwnerKind::Assembly:
      break;
    case MemoryOwnerKind::Intermediate:
      category = &intermediateBytes;
      break;
    case MemoryOwnerKind::LocalTemporary:
      if (storage != MemoryLifetimeStorage::RoutineLocal)
        return module.emitError(
            "local-temporary owner is not routine-local storage");
      break;
    default:
      break;
    }
    if (category && failed(checkedAccumulate(module, *category, bytes,
                                             "owner storage category")))
      return failure();
  }

  if (!workspaceAssigned) {
    workspaceBytes = 0;
    for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
      if (lifetime.getStorage() != MemoryLifetimeStorage::Workspace)
        continue;
      FailureOr<int64_t> aligned = alignedValue(
          module, workspaceBytes, lifetime.getAlignment().getInt());
      if (failed(aligned))
        return failure();
      workspaceBytes = *aligned;
      if (failed(checkedAccumulate(module, workspaceBytes,
                                   lifetime.getByteSize().getInt(),
                                   "unassigned workspace storage")))
        return failure();
    }
  }
  (void)scratchpadAssigned;

  int64_t routineTemporaryPeakBytes = 0;
  for (auto [routine, bytes] : routineTemporaryBytes) {
    routineTemporaryPeakBytes = std::max(routineTemporaryPeakBytes, bytes);
    (void)routine;
  }
  int64_t peakLiveBytes = 0;
  for (auto [bytes, description] :
       {std::pair{externalBytes, StringRef("peak external bytes")},
        std::pair{persistentBytes, StringRef("peak persistent bytes")},
        std::pair{workspaceBytes, StringRef("peak workspace bytes")},
        std::pair{scratchpadBytes, StringRef("peak scratchpad bytes")},
        std::pair{routineTemporaryPeakBytes,
                  StringRef("peak routine temporary bytes")}}) {
    if (failed(checkedAccumulate(module, peakLiveBytes, bytes, description)))
      return failure();
  }
  int64_t reusableBytes =
      std::max<int64_t>(0, workspaceLogicalBytes - workspaceBytes);

  if (auto audit = module->getAttrOfType<DictionaryAttr>(
          "sculptor.memory.bufferization_audit")) {
    auto allocationCount = audit.getAs<IntegerAttr>("allocation_count");
    int64_t recordedAllocations =
        llvm::count_if(*lifetimes, [](TileMemoryLifetimeAttr lifetime) {
          return lifetime.getSubjectKind() ==
                 MemoryLifetimeSubjectKind::RoutineAllocation;
        });
    complete =
        allocationCount && allocationCount.getInt() == recordedAllocations;
  }

  Builder builder(module.getContext());
  module->setAttr(kCapacityAttrName,
                  TileMemoryCapacityAttr::get(
                      module.getContext(), builder.getI64IntegerAttr(tile),
                      builder.getI64IntegerAttr(externalBytes),
                      builder.getI64IntegerAttr(persistentBytes),
                      builder.getI64IntegerAttr(workspaceBytes),
                      builder.getI64IntegerAttr(scratchpadBytes),
                      builder.getI64IntegerAttr(routeInputBytes),
                      builder.getI64IntegerAttr(routeOutputBytes),
                      builder.getI64IntegerAttr(assemblyBytes),
                      builder.getI64IntegerAttr(intermediateBytes),
                      builder.getI64IntegerAttr(routineTemporaryPeakBytes),
                      builder.getI64IntegerAttr(routineTemporaryTotalBytes),
                      builder.getI64IntegerAttr(peakLiveBytes),
                      builder.getI64IntegerAttr(peakLiveBytes),
                      builder.getI64IntegerAttr(reusableBytes),
                      builder.getBoolAttr(complete)));
  return success();
}

FailureOr<int64_t> inferOwnerElementBytes(Operation *anchor,
                                          TileMemoryOwnerAttr owner) {
  FailureOr<SmallVector<int64_t>> shape =
      parseIntegerArray(anchor, owner.getShape(), "owner element shape");
  if (failed(shape))
    return failure();
  int64_t elementCount = 1;
  for (int64_t size : *shape) {
    if (size <= 0)
      return anchor->emitError(
          "segmented movement requires positive static owner dimensions");
    std::optional<int64_t> next = llvm::checkedMul(elementCount, size);
    if (!next)
      return anchor->emitError("owner element count overflowed");
    elementCount = *next;
  }
  int64_t byteSize = owner.getByteSize().getInt();
  if (byteSize <= 0 || byteSize % elementCount != 0)
    return anchor->emitError(
        "owner byte size does not define a static element size");
  int64_t elementBytes = byteSize / elementCount;
  if (elementBytes <= 0)
    return anchor->emitError("owner has a nonpositive element size");
  return elementBytes;
}

FailureOr<SmallVector<ByteRun>> getVerifiedViewRuns(Operation *anchor,
                                                    TileMemoryViewAttr view,
                                                    TileMemoryOwnerAttr owner,
                                                    int64_t elementBytes) {
  FailureOr<SmallVector<int64_t>> offsets =
      parseIntegerArray(anchor, view.getOffsets(), "segment view offsets");
  FailureOr<SmallVector<int64_t>> sizes =
      parseIntegerArray(anchor, view.getSizes(), "segment view sizes");
  FailureOr<SmallVector<int64_t>> strides =
      parseIntegerArray(anchor, view.getStrides(), "segment view strides");
  if (failed(offsets) || failed(sizes) || failed(strides))
    return failure();
  ViewRecord record;
  record.id = view.getId().getInt();
  record.ownerId = owner.getId().getInt();
  record.byteOffset = view.getByteOffset().getInt();
  record.byteSize = view.getByteSize().getInt();
  record.offsets = std::move(*offsets);
  record.sizes = std::move(*sizes);
  record.strides = std::move(*strides);
  record.contiguity = view.getContiguity();
  auto runs =
      buildViewByteRuns(record, elementBytes, kMaxStaticSegmentCount, anchor);
  if (failed(runs))
    return failure();
  if (!*runs)
    return anchor->emitError(
        "segmented movement exceeds the supported static segment count");
  return std::move(**runs);
}

bool byteRangeContainedInRuns(ArrayRef<ByteRun> runs, int64_t offset,
                              int64_t byteSize) {
  std::optional<int64_t> end = llvm::checkedAdd(offset, byteSize);
  if (!end)
    return false;
  return llvm::any_of(runs, [&](const ByteRun &run) {
    std::optional<int64_t> runEnd =
        llvm::checkedAdd(run.byteOffset, run.byteSize);
    return runEnd && offset >= run.byteOffset && *end <= *runEnd;
  });
}

bool byteRangesOverlap(int64_t leftOffset, int64_t leftBytes,
                       int64_t rightOffset, int64_t rightBytes) {
  std::optional<int64_t> leftEnd = llvm::checkedAdd(leftOffset, leftBytes);
  std::optional<int64_t> rightEnd = llvm::checkedAdd(rightOffset, rightBytes);
  return !leftEnd || !rightEnd ||
         (leftOffset < *rightEnd && rightOffset < *leftEnd);
}

LogicalResult verifyPlan(ModuleOp module, bool local) {
  auto version = module->getAttrOfType<IntegerAttr>(kPlanVersionAttrName);
  if (!version || version.getInt() != 3)
    return module.emitError("expected tile memory plan version 3");
  auto owners = parseTypedArray<TileMemoryOwnerAttr>(module, kOwnersAttrName);
  auto views = parseTypedArray<TileMemoryViewAttr>(module, kViewsAttrName);
  auto bindings =
      parseTypedArray<TileMemoryBindingAttr>(module, kBindingsAttrName);
  auto movements =
      parseTypedArray<TileMemoryMovementAttr>(module, kMovementsAttrName);
  auto segments =
      parseTypedArray<TileMemorySegmentAttr>(module, kSegmentsAttrName);
  auto assemblies =
      parseTypedArray<TileMemoryAssemblyAttr>(module, kAssembliesAttrName);
  auto completions = parseTypedArray<TileMemoryCompletionEventAttr>(
      module, kCompletionEventsAttrName);
  auto eventEdges =
      parseTypedArray<TileMemoryEventEdgeAttr>(module, kEventEdgesAttrName);
  auto lifetimes =
      parseTypedArray<TileMemoryLifetimeAttr>(module, kLifetimesAttrName);
  auto interferences = parseTypedArray<TileMemoryInterferenceAttr>(
      module, kInterferencesAttrName);
  auto interferenceExceptions =
      module->getAttrOfType<DenseI64ArrayAttr>(kInterferenceExceptionsAttrName);
  std::optional<MemoryInterferenceRelation> defaultInterference;
  if (auto value =
          module->getAttrOfType<StringAttr>(kInterferenceDefaultAttrName))
    defaultInterference = symbolizeMemoryInterferenceRelation(value.getValue());
  auto aliases = parseTypedArray<TileMemoryInPlaceAliasAttr>(
      module, kInPlaceAliasesAttrName);
  if (failed(owners) || failed(views) || failed(bindings) ||
      failed(movements) || failed(segments) || failed(assemblies) ||
      failed(completions) || failed(eventEdges) || failed(lifetimes) ||
      failed(interferences) || failed(aliases))
    return failure();
  if (module->hasAttr(kInterferenceDefaultAttrName) &&
      (!defaultInterference ||
       (*defaultInterference != MemoryInterferenceRelation::Before &&
        *defaultInterference != MemoryInterferenceRelation::Interferes)))
    return module.emitError("invalid tile memory interference default");

  std::map<int64_t, TileMemoryOwnerAttr> ownerById;
  std::set<std::pair<int64_t, int64_t>> ownerKeys;
  for (TileMemoryOwnerAttr owner : *owners) {
    bool localTemporary = owner.getKind() == MemoryOwnerKind::LocalTemporary;
    if (owner.getId().getInt() < 0 ||
        (owner.getResourceId().getInt() < 0 && !localTemporary) ||
        (owner.getResourceId().getInt() >= 0 && localTemporary) ||
        owner.getTile().getInt() < 0 || owner.getByteSize().getInt() < 0 ||
        !ownerById.emplace(owner.getId().getInt(), owner).second ||
        (!localTemporary &&
         !ownerKeys
              .emplace(owner.getResourceId().getInt(), owner.getTile().getInt())
              .second))
      return module.emitError("invalid or duplicate tile memory owner");
    auto shape = parseIntegerArray(module, owner.getShape(), "owner shape");
    auto strides =
        parseIntegerArray(module, owner.getStrides(), "owner strides");
    if (failed(shape) || failed(strides) || shape->size() != strides->size())
      return module.emitError("invalid owner shape or stride metadata");
  }

  std::map<int64_t, TileMemoryViewAttr> viewById;
  for (TileMemoryViewAttr view : *views) {
    auto owner = ownerById.find(view.getOwnerId().getInt());
    if (view.getId().getInt() < 0 || owner == ownerById.end() ||
        !viewById.emplace(view.getId().getInt(), view).second)
      return module.emitError("memory view names an invalid owner or ID");
    auto offsets = parseIntegerArray(module, view.getOffsets(), "view offsets");
    auto sizes = parseIntegerArray(module, view.getSizes(), "view sizes");
    auto strides = parseIntegerArray(module, view.getStrides(), "view strides");
    auto shape =
        parseIntegerArray(module, owner->second.getShape(), "owner shape");
    if (failed(offsets) || failed(sizes) || failed(strides) || failed(shape) ||
        offsets->size() != shape->size() || sizes->size() != shape->size() ||
        strides->size() != shape->size())
      return module.emitError("memory view rank does not match its owner");
    for (size_t index = 0; index < shape->size(); ++index) {
      std::optional<int64_t> end =
          llvm::checkedAdd((*offsets)[index], (*sizes)[index]);
      if ((*offsets)[index] < 0 || (*sizes)[index] < 0 ||
          (*strides)[index] <= 0 || !end || *end > (*shape)[index]) {
        return module.emitError("memory view ")
               << view.getId().getInt() << " lies outside owner "
               << owner->second.getId().getInt() << " at dimension " << index;
      }
    }
    if (view.getByteOffset().getInt() < 0 || view.getByteSize().getInt() < 0)
      return module.emitError("memory view has a negative byte range");
    std::optional<int64_t> end = llvm::checkedAdd(view.getByteOffset().getInt(),
                                                  view.getByteSize().getInt());
    if (!end || *end > owner->second.getByteSize().getInt())
      return module.emitError("memory view ")
             << view.getId().getInt() << " lies outside owner "
             << owner->second.getId().getInt() << " byte range";
  }

  auto viewsOverlap = [&](TileMemoryViewAttr left,
                          TileMemoryViewAttr right) -> FailureOr<bool> {
    if (left.getOwnerId() != right.getOwnerId())
      return false;
    auto owner = ownerById.find(left.getOwnerId().getInt());
    if (owner == ownerById.end())
      return module.emitError(
          "memory overlap check references an unknown owner");

    FailureOr<SmallVector<int64_t>> ownerStrides = parseIntegerArray(
        module, owner->second.getStrides(), "overlap owner strides");
    FailureOr<SmallVector<int64_t>> leftOffsets =
        parseIntegerArray(module, left.getOffsets(), "overlap view offsets");
    FailureOr<SmallVector<int64_t>> leftSizes =
        parseIntegerArray(module, left.getSizes(), "overlap view sizes");
    FailureOr<SmallVector<int64_t>> leftStrides =
        parseIntegerArray(module, left.getStrides(), "overlap view strides");
    FailureOr<SmallVector<int64_t>> rightOffsets =
        parseIntegerArray(module, right.getOffsets(), "overlap view offsets");
    FailureOr<SmallVector<int64_t>> rightSizes =
        parseIntegerArray(module, right.getSizes(), "overlap view sizes");
    FailureOr<SmallVector<int64_t>> rightStrides =
        parseIntegerArray(module, right.getStrides(), "overlap view strides");
    if (failed(ownerStrides) || failed(leftOffsets) || failed(leftSizes) ||
        failed(leftStrides) || failed(rightOffsets) || failed(rightSizes) ||
        failed(rightStrides))
      return failure();

    bool unitSliceStrides = true;
    for (size_t dimension = 0; dimension < ownerStrides->size(); ++dimension) {
      if ((*leftSizes)[dimension] == 0 || (*rightSizes)[dimension] == 0)
        return false;
      int64_t ownerStride = (*ownerStrides)[dimension];
      int64_t leftStride = (*leftStrides)[dimension];
      int64_t rightStride = (*rightStrides)[dimension];
      if (ownerStride <= 0 || leftStride <= 0 || rightStride <= 0 ||
          leftStride % ownerStride != 0 || rightStride % ownerStride != 0)
        return module.emitError(
            "memory overlap check found incompatible view strides");
      int64_t leftStep = leftStride / ownerStride;
      int64_t rightStep = rightStride / ownerStride;
      unitSliceStrides &= leftStep == 1 && rightStep == 1;
      std::optional<int64_t> leftSpan = llvm::checkedMul(
          std::max<int64_t>((*leftSizes)[dimension] - 1, 0), leftStep);
      std::optional<int64_t> rightSpan = llvm::checkedMul(
          std::max<int64_t>((*rightSizes)[dimension] - 1, 0), rightStep);
      std::optional<int64_t> leftEnd =
          leftSpan ? llvm::checkedAdd((*leftOffsets)[dimension], *leftSpan)
                   : std::nullopt;
      std::optional<int64_t> rightEnd =
          rightSpan ? llvm::checkedAdd((*rightOffsets)[dimension], *rightSpan)
                    : std::nullopt;
      if (!leftEnd || !rightEnd)
        return module.emitError("memory overlap bound calculation overflowed");
      if (*leftEnd < (*rightOffsets)[dimension] ||
          *rightEnd < (*leftOffsets)[dimension])
        return false;
    }
    if (unitSliceStrides)
      return true;

    FailureOr<int64_t> elementBytes =
        inferOwnerElementBytes(module, owner->second);
    if (failed(elementBytes))
      return failure();
    FailureOr<SmallVector<ByteRun>> leftRuns =
        getVerifiedViewRuns(module, left, owner->second, *elementBytes);
    FailureOr<SmallVector<ByteRun>> rightRuns =
        getVerifiedViewRuns(module, right, owner->second, *elementBytes);
    if (failed(leftRuns) || failed(rightRuns))
      return failure();
    for (const ByteRun &leftRun : *leftRuns)
      for (const ByteRun &rightRun : *rightRuns)
        if (byteRangesOverlap(leftRun.byteOffset, leftRun.byteSize,
                              rightRun.byteOffset, rightRun.byteSize))
          return true;
    return false;
  };

  std::map<int64_t, TileMemoryCompletionEventAttr> completionById;
  std::optional<int64_t> localTile;
  if (local) {
    auto tile = module->getAttrOfType<IntegerAttr>(kPhysicalTileIdAttr);
    if (!tile)
      return module.emitError("tile-local memory plan has no physical tile ID");
    localTile = tile.getInt();
  }
  for (TileMemoryCompletionEventAttr completion : *completions) {
    if (completion.getId().getInt() < 0 ||
        (localTile && completion.getTile().getInt() != *localTile) ||
        !completionById.emplace(completion.getId().getInt(), completion).second)
      return module.emitError("invalid or duplicate memory completion event");
  }

  llvm::SmallDenseSet<int64_t> eventEdgeIds;
  std::set<std::tuple<int64_t, int64_t, MemoryEventEdgeKind>> eventEdgeKeys;
  SmallVector<std::pair<int64_t, int64_t>> eventGraphEdges;
  eventGraphEdges.reserve(eventEdges->size());
  for (TileMemoryEventEdgeAttr edge : *eventEdges) {
    int64_t edgeId = edge.getId().getInt();
    int64_t source = edge.getSourceEventId().getInt();
    int64_t target = edge.getTargetEventId().getInt();
    auto key = std::make_tuple(source, target, edge.getKind());
    if (edgeId < 0 || source == target || !completionById.count(source) ||
        !completionById.count(target) || !eventEdgeIds.insert(edgeId).second ||
        !eventEdgeKeys.insert(key).second)
      return module.emitError("invalid or duplicate memory event edge");
    eventGraphEdges.emplace_back(source, target);
  }
  SmallVector<int64_t> eventIds;
  eventIds.reserve(completionById.size());
  for (const auto &[eventId, completion] : completionById) {
    eventIds.push_back(eventId);
    (void)completion;
  }
  FailureOr<EventReachability> eventReachability =
      computeEventReachability(eventIds, eventGraphEdges, module);
  if (failed(eventReachability))
    return failure();

  std::map<int64_t, int64_t> routineStarts;
  std::map<int64_t, int64_t> routineCompletions;
  for (const auto &[eventId, completion] : completionById) {
    if (completion.getKind() == MemoryCompletionKind::RoutineStart) {
      if (!routineStarts.emplace(completion.getRoutine().getInt(), eventId)
               .second)
        return module.emitError("routine has duplicate start events");
    } else if (completion.getKind() == MemoryCompletionKind::RoutineComplete) {
      if (!routineCompletions.emplace(completion.getRoutine().getInt(), eventId)
               .second)
        return module.emitError("routine has duplicate completion events");
    }
  }
  for (const auto &[routine, start] : routineStarts) {
    auto complete = routineCompletions.find(routine);
    if (complete == routineCompletions.end() ||
        !eventEdgeKeys.count(std::make_tuple(
            start, complete->second, MemoryEventEdgeKind::RoutineExecution)))
      return module.emitError("routine has no execution event edge");
  }
  if (routineStarts.size() != routineCompletions.size())
    return module.emitError("routine completion event set is incomplete");

  std::map<int64_t, SmallVector<int64_t>> sendsByOwner;
  std::map<int64_t, int64_t> finalFanOutByOwner;
  std::map<int64_t, int64_t> finalConsumerByOwner;
  std::map<int64_t, int64_t> releaseByOwner;
  for (const auto &[eventId, completion] : completionById) {
    int64_t ownerId = completion.getOwnerId().getInt();
    if (ownerId < 0)
      continue;
    if (!ownerById.count(ownerId))
      return module.emitError("memory completion references an unknown owner");
    if (completion.getKind() == MemoryCompletionKind::RouteSendComplete) {
      sendsByOwner[ownerId].push_back(eventId);
    } else if (completion.getKind() == MemoryCompletionKind::FinalFanOutSend) {
      if (!finalFanOutByOwner.emplace(ownerId, eventId).second)
        return module.emitError("memory owner has multiple fan-out joins");
    } else if (completion.getKind() ==
               MemoryCompletionKind::FinalConsumerComplete) {
      if (!finalConsumerByOwner.emplace(ownerId, eventId).second)
        return module.emitError("memory owner has multiple consumer joins");
    } else if (completion.getKind() == MemoryCompletionKind::OwnerRelease ||
               completion.getKind() ==
                   MemoryCompletionKind::ScratchpadRelease) {
      if (!releaseByOwner.emplace(ownerId, eventId).second)
        return module.emitError("memory owner has multiple release events");
    }
  }

  for (auto &[ownerId, sends] : sendsByOwner) {
    auto join = finalFanOutByOwner.find(ownerId);
    if (join == finalFanOutByOwner.end())
      return module.emitError("route-output owner has no final fan-out event");
    for (int64_t send : sends)
      if (!eventEdgeKeys.count(std::make_tuple(
              send, join->second, MemoryEventEdgeKind::FanOutJoin)))
        return module.emitError(
            "final fan-out event does not join every route send");
  }

  std::map<int64_t, std::set<int64_t>> consumersByOwner;
  for (TileMemoryBindingAttr binding : *bindings) {
    if (!binding.getInput() ||
        binding.getEffect() == MemoryAccessEffect::AsyncTransferDestination ||
        binding.getEffect() == MemoryAccessEffect::AsyncTransferSource)
      continue;
    auto owner = ownerById.find(binding.getOwnerId().getInt());
    if (owner != ownerById.end() &&
        owner->second.getKind() == MemoryOwnerKind::RouteInput)
      consumersByOwner[owner->first].insert(binding.getRoutine().getInt());
  }
  for (auto &[ownerId, consumers] : consumersByOwner) {
    auto join = finalConsumerByOwner.find(ownerId);
    if (join == finalConsumerByOwner.end())
      return module.emitError("route-input owner has no final-consumer event");
    for (int64_t routine : consumers) {
      auto complete = routineCompletions.find(routine);
      if (complete == routineCompletions.end() ||
          !eventEdgeKeys.count(
              std::make_tuple(complete->second, join->second,
                              MemoryEventEdgeKind::FinalConsumer)))
        return module.emitError(
            "route-input final-consumer event omits a local consumer");
    }
  }

  for (const auto &[ownerId, owner] : ownerById) {
    if (!hasReleasableStorage(owner.getKind()))
      continue;
    auto release = releaseByOwner.find(ownerId);
    if (release == releaseByOwner.end())
      return module.emitError("workspace owner has no final release event");
    bool hasReleasePredecessor =
        llvm::any_of(*eventEdges, [&](TileMemoryEventEdgeAttr edge) {
          return edge.getTargetEventId().getInt() == release->second &&
                 (edge.getKind() == MemoryEventEdgeKind::LifetimeRelease ||
                  edge.getKind() == MemoryEventEdgeKind::ScratchpadRelease);
        });
    if (!hasReleasePredecessor)
      return module.emitError("memory release has no final-use predecessor");
    if (auto fanOut = finalFanOutByOwner.find(ownerId);
        fanOut != finalFanOutByOwner.end() &&
        !eventReachability->reaches(fanOut->second, release->second))
      return module.emitError("route-output owner ")
             << ownerId << " release event " << release->second
             << " does not follow final fan-out event " << fanOut->second;
    if (auto consumer = finalConsumerByOwner.find(ownerId);
        consumer != finalConsumerByOwner.end() &&
        !eventReachability->reaches(consumer->second, release->second))
      return module.emitError("route-input owner ")
             << ownerId << " release event " << release->second
             << " does not follow final-consumer event " << consumer->second;
  }

  llvm::SmallDenseSet<int64_t> lifetimeIds;
  llvm::SmallDenseSet<int64_t> lifetimeOwnerIds;
  std::map<int64_t, TileMemoryLifetimeAttr> lifetimeById;
  for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
    int64_t id = lifetime.getId().getInt();
    int64_t ownerId = lifetime.getOwnerId().getInt();
    int64_t tile = lifetime.getTile().getInt();
    if (id < 0 || tile < 0 || lifetime.getByteSize().getInt() < 0 ||
        lifetime.getAlignment().getInt() <= 0 ||
        lifetime.getOffset().getInt() < -1 ||
        !llvm::isPowerOf2_64(lifetime.getAlignment().getInt()) ||
        !lifetimeIds.insert(id).second || (localTile && tile != *localTile))
      return module.emitError("invalid tile memory lifetime");
    lifetimeById[id] = lifetime;
    auto viewIds =
        parseIntegerArray(module, lifetime.getViewIds(), "lifetime views");
    auto accessEvents = parseIntegerArray(module, lifetime.getAccessEventIds(),
                                          "lifetime access events");
    if (failed(viewIds) || failed(accessEvents))
      return failure();
    llvm::SmallDenseSet<int64_t> uniqueViews;
    llvm::SmallDenseSet<int64_t> uniqueAccesses;
    for (int64_t event : *accessEvents) {
      if (!completionById.count(event) || !uniqueAccesses.insert(event).second)
        return module.emitError("lifetime has an invalid access event");
    }
    if (lifetime.getSubjectKind() == MemoryLifetimeSubjectKind::Owner) {
      auto owner = ownerById.find(ownerId);
      if (owner == ownerById.end() ||
          lifetime.getAllocationOrdinal().getInt() != -1 ||
          owner->second.getTile().getInt() != tile ||
          owner->second.getByteSize().getInt() !=
              lifetime.getByteSize().getInt() ||
          !lifetimeOwnerIds.insert(ownerId).second)
        return module.emitError("owner lifetime does not match its owner");
      if (hasReleasableStorage(owner->second.getKind())) {
        auto release = releaseByOwner.find(ownerId);
        if (release == releaseByOwner.end() ||
            !uniqueAccesses.count(release->second))
          return module.emitError(
              "workspace lifetime omits its final release event");
        MemoryCompletionKind releaseKind =
            completionById.at(release->second).getKind();
        if ((lifetime.getStorage() == MemoryLifetimeStorage::Scratchpad) !=
            (releaseKind == MemoryCompletionKind::ScratchpadRelease))
          return module.emitError(
              "scratchpad lifetime and release event do not agree");
      }
      for (int64_t viewId : *viewIds) {
        auto view = viewById.find(viewId);
        if (view == viewById.end() ||
            view->second.getOwnerId().getInt() != ownerId ||
            !uniqueViews.insert(viewId).second)
          return module.emitError("owner lifetime has an invalid view");
      }
    } else {
      if (lifetime.getSubjectKind() !=
              MemoryLifetimeSubjectKind::RoutineAllocation ||
          ownerId != -1 || lifetime.getRoutine().getInt() < 0 ||
          lifetime.getAllocationOrdinal().getInt() < 0 || !viewIds->empty() ||
          lifetime.getStorage() != MemoryLifetimeStorage::RoutineLocal)
        return module.emitError("invalid routine-allocation lifetime");
    }
  }
  if (lifetimeOwnerIds.size() != ownerById.size())
    return module.emitError("not every tile memory owner has one lifetime");

  std::map<int64_t, int64_t> lifetimeIdByOwner;
  for (const auto &[lifetimeId, lifetime] : lifetimeById) {
    if (lifetime.getSubjectKind() == MemoryLifetimeSubjectKind::Owner)
      lifetimeIdByOwner[lifetime.getOwnerId().getInt()] = lifetimeId;
  }
  std::set<std::pair<int64_t, int64_t>> inPlaceLifetimePairs;
  for (TileMemoryInPlaceAliasAttr alias : *aliases) {
    auto input = lifetimeIdByOwner.find(alias.getInputOwnerId().getInt());
    auto output = lifetimeIdByOwner.find(alias.getOutputOwnerId().getInt());
    if (input == lifetimeIdByOwner.end() || output == lifetimeIdByOwner.end() ||
        input->second == output->second ||
        !inPlaceLifetimePairs.insert(std::minmax(input->second, output->second))
             .second)
      return module.emitError("invalid or duplicate in-place lifetime pair");
  }

  llvm::SmallDenseSet<int64_t> interferenceIds;
  std::set<std::pair<int64_t, int64_t>> interferencePairs;
  for (TileMemoryInterferenceAttr interference : *interferences) {
    int64_t id = interference.getId().getInt();
    int64_t left = interference.getLeftLifetimeId().getInt();
    int64_t right = interference.getRightLifetimeId().getInt();
    auto leftLifetime = lifetimeById.find(left);
    auto rightLifetime = lifetimeById.find(right);
    if (id < 0 || left >= right || leftLifetime == lifetimeById.end() ||
        rightLifetime == lifetimeById.end() ||
        leftLifetime->second.getTile().getInt() !=
            rightLifetime->second.getTile().getInt() ||
        !interferenceIds.insert(id).second ||
        !interferencePairs.emplace(left, right).second)
      return module.emitError("invalid or duplicate lifetime interference");
    if (defaultInterference &&
        interference.getRelation() == *defaultInterference)
      return module.emitError(
          "explicit lifetime relation duplicates the interference default");
    if (interference.getRelation() ==
            MemoryInterferenceRelation::SeparateStorage &&
        !hasSeparateStorage(leftLifetime->second, rightLifetime->second))
      return module.emitError(
          "lifetime relation claims separate storage for one storage pool");
    bool expectedAlias = inPlaceLifetimePairs.count({left, right});
    if ((interference.getRelation() ==
         MemoryInterferenceRelation::InPlaceAlias) != expectedAlias)
      return module.emitError(
          "in-place alias and lifetime relation do not agree");
  }
  if (module->hasAttr(kInterferenceExceptionsAttrName) &&
      !interferenceExceptions)
    return module.emitError(
        "tile memory interference exceptions must be a dense i64 array");
  if (interferenceExceptions) {
    ArrayRef<int64_t> values = interferenceExceptions.asArrayRef();
    if (values.size() % 3 != 0)
      return module.emitError(
          "tile memory interference exceptions must contain triples");
    for (size_t index = 0; index < values.size(); index += 3) {
      int64_t left = values[index];
      int64_t right = values[index + 1];
      auto relation = symbolizeMemoryInterferenceRelation(
          static_cast<uint32_t>(values[index + 2]));
      auto leftLifetime = lifetimeById.find(left);
      auto rightLifetime = lifetimeById.find(right);
      if (!relation || left >= right || leftLifetime == lifetimeById.end() ||
          rightLifetime == lifetimeById.end() ||
          leftLifetime->second.getTile().getInt() !=
              rightLifetime->second.getTile().getInt() ||
          !interferencePairs.emplace(left, right).second)
        return module.emitError(
            "invalid or duplicate compact lifetime interference");
      if (defaultInterference && *relation == *defaultInterference)
        return module.emitError(
            "compact lifetime relation duplicates the interference default");
      if (*relation == MemoryInterferenceRelation::SeparateStorage &&
          !hasSeparateStorage(leftLifetime->second, rightLifetime->second))
        return module.emitError(
            "lifetime relation claims separate storage for one storage pool");
      bool expectedAlias = inPlaceLifetimePairs.count({left, right});
      if ((*relation == MemoryInterferenceRelation::InPlaceAlias) !=
          expectedAlias)
        return module.emitError(
            "in-place alias and compact lifetime relation do not agree");
    }
  }
  if (!defaultInterference) {
    size_t expectedInterferenceCount = 0;
    for (auto left = lifetimeById.begin(); left != lifetimeById.end(); ++left) {
      for (auto right = std::next(left); right != lifetimeById.end(); ++right) {
        if (left->second.getTile().getInt() == right->second.getTile().getInt())
          ++expectedInterferenceCount;
      }
    }
    if (interferencePairs.size() != expectedInterferenceCount)
      return module.emitError(
          "lifetime interference table does not cover every local pair");
  } else {
    for (const auto &pair : inPlaceLifetimePairs) {
      if (!interferencePairs.count(pair))
        return module.emitError(
            "in-place alias was omitted from sparse interference metadata");
    }
  }

  auto capacity =
      module->getAttrOfType<TileMemoryCapacityAttr>(kCapacityAttrName);
  if (local && !capacity)
    return module.emitError("tile-local memory plan has no capacity summary");
  if (capacity) {
    if (capacity.getTile().getInt() < 0 ||
        (localTile && capacity.getTile().getInt() != *localTile))
      return module.emitError("memory capacity summary has the wrong tile");
    for (IntegerAttr value :
         {capacity.getExternalBytes(), capacity.getPersistentBytes(),
          capacity.getWorkspaceBytes(), capacity.getScratchpadBytes(),
          capacity.getRouteInputBytes(), capacity.getRouteOutputBytes(),
          capacity.getAssemblyBytes(), capacity.getIntermediateBytes(),
          capacity.getRoutineTemporaryPeakBytes(),
          capacity.getRoutineTemporaryTotalBytes(), capacity.getPeakLiveBytes(),
          capacity.getRequiredLocalBytes(), capacity.getReusableBytes()}) {
      if (value.getInt() < 0)
        return module.emitError("memory capacity summary has negative bytes");
    }
    if (capacity.getRequiredLocalBytes().getInt() !=
        capacity.getPeakLiveBytes().getInt())
      return module.emitError(
          "memory capacity required bytes disagree with peak live bytes");
    for (TileMemoryLifetimeAttr lifetime : *lifetimes) {
      if (lifetime.getOffset().getInt() < 0)
        continue;
      std::optional<int64_t> end = llvm::checkedAdd(
          lifetime.getOffset().getInt(), lifetime.getByteSize().getInt());
      int64_t available =
          lifetime.getStorage() == MemoryLifetimeStorage::Workspace
              ? capacity.getWorkspaceBytes().getInt()
          : lifetime.getStorage() == MemoryLifetimeStorage::Scratchpad
              ? capacity.getScratchpadBytes().getInt()
              : std::numeric_limits<int64_t>::max();
      if (!end || *end > available)
        return module.emitError(
            "assigned lifetime exceeds its capacity summary");
    }
  }

  std::map<int64_t, TileMemoryBindingAttr> bindingById;
  std::map<int64_t, SmallVector<TileMemoryBindingAttr>> writersByOwner;
  for (TileMemoryBindingAttr binding : *bindings) {
    auto owner = ownerById.find(binding.getOwnerId().getInt());
    auto view = viewById.find(binding.getViewId().getInt());
    if (binding.getId().getInt() < 0 || owner == ownerById.end() ||
        view == viewById.end() ||
        view->second.getOwnerId().getInt() != binding.getOwnerId().getInt() ||
        !bindingById.emplace(binding.getId().getInt(), binding).second)
      return module.emitError("invalid tile memory binding");
    if (binding.getEffect() == MemoryAccessEffect::Write ||
        binding.getEffect() == MemoryAccessEffect::ReadWrite)
      writersByOwner[binding.getOwnerId().getInt()].push_back(binding);
  }

  llvm::SmallDenseSet<int64_t> aliasIds;
  llvm::SmallDenseSet<int64_t> aliasedInputOwners;
  llvm::SmallDenseSet<int64_t> aliasedOutputOwners;
  for (TileMemoryInPlaceAliasAttr alias : *aliases) {
    int64_t inputOwnerId = alias.getInputOwnerId().getInt();
    int64_t outputOwnerId = alias.getOutputOwnerId().getInt();
    auto inputOwner = ownerById.find(inputOwnerId);
    auto outputOwner = ownerById.find(outputOwnerId);
    auto inputView = viewById.find(alias.getInputViewId().getInt());
    auto outputView = viewById.find(alias.getOutputViewId().getInt());
    if (alias.getId().getInt() < 0 || alias.getRoutine().getInt() < 0 ||
        alias.getInputPort().getInt() < 0 ||
        alias.getOutputPort().getInt() < 0 ||
        !aliasIds.insert(alias.getId().getInt()).second ||
        inputOwner == ownerById.end() || outputOwner == ownerById.end() ||
        inputOwnerId == outputOwnerId || inputView == viewById.end() ||
        outputView == viewById.end() ||
        inputView->second.getOwnerId().getInt() != inputOwnerId ||
        outputView->second.getOwnerId().getInt() != outputOwnerId ||
        !aliasedInputOwners.insert(inputOwnerId).second ||
        !aliasedOutputOwners.insert(outputOwnerId).second)
      return module.emitError("invalid or duplicate in-place memory alias");
    if (!canUpdateOwnerInPlace(inputOwner->second.getKind()) ||
        !canStoreInPlaceResult(outputOwner->second.getKind()) ||
        inputOwner->second.getTile() != outputOwner->second.getTile() ||
        inputOwner->second.getByteSize() != outputOwner->second.getByteSize() ||
        inputOwner->second.getShape() != outputOwner->second.getShape() ||
        inputOwner->second.getStrides() != outputOwner->second.getStrides())
      return module.emitError(
          "in-place alias owners have incompatible storage or geometry");

    auto inputLifetime = lifetimeById.find(lifetimeIdByOwner.at(inputOwnerId));
    auto outputLifetime =
        lifetimeById.find(lifetimeIdByOwner.at(outputOwnerId));
    MemoryLifetimeStorage inputStorage = inputLifetime->second.getStorage();
    MemoryLifetimeStorage outputStorage = outputLifetime->second.getStorage();
    if (inputStorage != outputStorage ||
        (inputStorage != MemoryLifetimeStorage::Workspace &&
         inputStorage != MemoryLifetimeStorage::Scratchpad))
      return module.emitError(
          "in-place alias requires matching workspace or scratchpad storage");
    if (inputStorage == MemoryLifetimeStorage::Scratchpad &&
        (inputLifetime->second.getOffset().getInt() < 0 ||
         inputLifetime->second.getOffset() !=
             outputLifetime->second.getOffset()))
      return module.emitError(
          "in-place scratchpad alias requires one assigned offset");

    TileMemoryBindingAttr inputBinding;
    TileMemoryBindingAttr outputBinding;
    unsigned inputReaderCount = 0;
    bool pendingSend = false;
    for (TileMemoryBindingAttr binding : *bindings) {
      if (binding.getOwnerId().getInt() == inputOwnerId &&
          binding.getInput().getValue() &&
          (binding.getEffect() == MemoryAccessEffect::Read ||
           binding.getEffect() == MemoryAccessEffect::ReadWrite)) {
        ++inputReaderCount;
        if (binding.getRoutine().getInt() == alias.getRoutine().getInt() &&
            binding.getPort().getInt() == alias.getInputPort().getInt() &&
            binding.getEffect() == MemoryAccessEffect::ReadWrite)
          inputBinding = binding;
      }
      if (binding.getOwnerId().getInt() == outputOwnerId &&
          !binding.getInput().getValue() &&
          binding.getRoutine().getInt() == alias.getRoutine().getInt() &&
          binding.getPort().getInt() == alias.getOutputPort().getInt() &&
          binding.getEffect() == MemoryAccessEffect::ReadWrite)
        outputBinding = binding;
      if (binding.getOwnerId().getInt() == inputOwnerId &&
          binding.getEffect() == MemoryAccessEffect::AsyncTransferSource)
        pendingSend = true;
    }
    if (!inputBinding || !outputBinding || inputReaderCount != 1 ||
        pendingSend ||
        inputBinding.getViewId().getInt() != alias.getInputViewId().getInt() ||
        outputBinding.getViewId().getInt() !=
            alias.getOutputViewId().getInt() ||
        inputBinding.getEffect() != MemoryAccessEffect::ReadWrite ||
        outputBinding.getEffect() != MemoryAccessEffect::ReadWrite)
      return module.emitError(
          "in-place alias lacks exclusive read-write routine bindings");

    auto isFullView = [&](TileMemoryViewAttr view, TileMemoryOwnerAttr owner) {
      auto offsets =
          parseIntegerArray(module, view.getOffsets(), "in-place view offsets");
      if (failed(offsets) ||
          !llvm::all_of(*offsets, [](int64_t value) { return value == 0; }))
        return false;
      return view.getByteOffset().getInt() == 0 &&
             view.getByteSize() == owner.getByteSize() &&
             view.getSizes() == owner.getShape() &&
             view.getStrides() == owner.getStrides();
    };
    if (!isFullView(inputView->second, inputOwner->second) ||
        !isFullView(outputView->second, outputOwner->second))
      return module.emitError("in-place alias requires exact full views");
  }

  for (auto &[ownerId, writers] : writersByOwner) {
    for (size_t left = 0; left < writers.size(); ++left) {
      for (size_t right = left + 1; right < writers.size(); ++right) {
        if (writers[left].getRoutine().getInt() ==
            writers[right].getRoutine().getInt())
          continue;
        TileMemoryViewAttr leftView =
            viewById.at(writers[left].getViewId().getInt());
        TileMemoryViewAttr rightView =
            viewById.at(writers[right].getViewId().getInt());
        int64_t leftRoutine = writers[left].getRoutine().getInt();
        int64_t rightRoutine = writers[right].getRoutine().getInt();
        if (!routineCompletions.count(leftRoutine) ||
            !routineCompletions.count(rightRoutine) ||
            !routineStarts.count(leftRoutine) ||
            !routineStarts.count(rightRoutine))
          return module.emitError(
              "memory writer references a routine without completion events");
        FailureOr<bool> overlap = viewsOverlap(leftView, rightView);
        if (failed(overlap))
          return failure();
        if (!*overlap)
          continue;
        bool ordered =
            eventReachability->reaches(routineCompletions.at(leftRoutine),
                                       routineStarts.at(rightRoutine)) ||
            eventReachability->reaches(routineCompletions.at(rightRoutine),
                                       routineStarts.at(leftRoutine));
        if (!ordered)
          return module.emitError(
              "overlapping memory writers require explicit ordering");
      }
    }
    (void)ownerId;
  }

  std::map<int64_t, TileMemoryAssemblyAttr> assemblyById;
  for (TileMemoryAssemblyAttr assembly : *assemblies) {
    if (assembly.getId().getInt() < 0 ||
        !assemblyById.emplace(assembly.getId().getInt(), assembly).second)
      return module.emitError("invalid or duplicate memory assembly ID");
  }

  std::map<int64_t, TileMemoryMovementAttr> movementById;
  for (TileMemoryMovementAttr movement : *movements) {
    if (movement.getId().getInt() < 0 ||
        !movementById.emplace(movement.getId().getInt(), movement).second)
      return module.emitError("invalid or duplicate memory movement ID");
    bool sourceKnown = viewById.count(movement.getSourceViewId().getInt());
    bool destinationKnown =
        viewById.count(movement.getDestinationViewId().getInt());
    if ((!local && (!sourceKnown || !destinationKnown)) ||
        (local && !sourceKnown && !destinationKnown))
      return module.emitError("memory movement has no valid endpoint view");
    if (movement.getMode() == MemoryMovementMode::LocalAlias) {
      if (!sourceKnown || !destinationKnown ||
          viewById.at(movement.getSourceViewId().getInt()).getOwnerId() !=
              viewById.at(movement.getDestinationViewId().getInt())
                  .getOwnerId() ||
          movement.getRouteId().getInt() != -1)
        return module.emitError("invalid same-tile alias movement");
    }
    for (int64_t event :
         {movement.getSourceCompletionEventId().getInt(),
          movement.getDestinationCompletionEventId().getInt()}) {
      if (event >= 0 && !completionById.count(event) && !local)
        return module.emitError("memory movement has an unknown completion");
    }
    int64_t sourceEvent = movement.getSourceCompletionEventId().getInt();
    int64_t destinationEvent =
        movement.getDestinationCompletionEventId().getInt();
    int64_t assemblyId = movement.getAssemblyId().getInt();
    if (assemblyId >= 0 && !assemblyById.count(assemblyId))
      return module.emitError("memory movement references an unknown assembly");
    if (completionById.count(sourceEvent) &&
        completionById.count(destinationEvent)) {
      std::optional<MemoryEventEdgeKind> requiredKind;
      if (movement.getMode() == MemoryMovementMode::LocalAlias)
        requiredKind = MemoryEventEdgeKind::LocalDependency;
      else if (movement.getMode() == MemoryMovementMode::Contiguous &&
               movement.getRouteId().getInt() >= 0 && assemblyId < 0)
        requiredKind = MemoryEventEdgeKind::NetworkTransfer;
      else if (movement.getMode() == MemoryMovementMode::Assembly)
        requiredKind = MemoryEventEdgeKind::AssemblyContribution;
      else if (movement.getMode() == MemoryMovementMode::Contiguous ||
               movement.getMode() == MemoryMovementMode::Packed ||
               movement.getMode() == MemoryMovementMode::Segmented)
        requiredKind = MemoryEventEdgeKind::DMACompletion;
      if (requiredKind && !eventEdgeKeys.count(std::make_tuple(
                              sourceEvent, destinationEvent, *requiredKind)))
        return module.emitError(
            "memory movement has no matching event dependency");
    }
    if (movement.getRouteId().getInt() >= 0) {
      const bool assemblyContribution = assemblyId >= 0;
      if (auto source = completionById.find(sourceEvent);
          source != completionById.end()) {
        if (assemblyContribution) {
          if (source->second.getKind() != MemoryCompletionKind::RouteArrival ||
              source->second.getRouteId().getInt() !=
                  movement.getRouteId().getInt())
            return module.emitError(
                "routed assembly movement does not start at route arrival");
        } else {
          if (source->second.getKind() !=
              MemoryCompletionKind::RouteSendComplete)
            return module.emitError(
                "route movement source is not send-complete");
          auto complete =
              routineCompletions.find(source->second.getRoutine().getInt());
          if (complete == routineCompletions.end() ||
              !eventEdgeKeys.count(
                  std::make_tuple(complete->second, sourceEvent,
                                  MemoryEventEdgeKind::RouteSend)))
            return module.emitError("route send has no producing routine edge");
        }
      }
      if (auto destination = completionById.find(destinationEvent);
          destination != completionById.end()) {
        if (assemblyContribution) {
          if (destination->second.getKind() !=
                  MemoryCompletionKind::DMAComplete ||
              destination->second.getRouteId().getInt() !=
                  movement.getRouteId().getInt())
            return module.emitError(
                "routed assembly movement has the wrong DMA completion event");
        } else {
          if (destination->second.getKind() !=
              MemoryCompletionKind::RouteArrival)
            return module.emitError("route movement target is not arrival");
          auto start =
              routineStarts.find(destination->second.getRoutine().getInt());
          if (start == routineStarts.end() ||
              !eventEdgeKeys.count(
                  std::make_tuple(destinationEvent, start->second,
                                  MemoryEventEdgeKind::RouteReady)))
            return module.emitError("route arrival has no consumer-ready edge");
        }
      }
    }
  }

  std::map<int64_t, SmallVector<TileMemorySegmentAttr>> segmentsByMovement;
  int64_t previousSegmentId = -1;
  int64_t previousMovementId = -1;
  for (TileMemorySegmentAttr segment : *segments) {
    int64_t id = segment.getId().getInt();
    int64_t movementId = segment.getMovementId().getInt();
    auto movement = movementById.find(movementId);
    if (id < 0 || id <= previousSegmentId || movement == movementById.end() ||
        movement->second.getMode() != MemoryMovementMode::Segmented ||
        segment.getOrdinal().getInt() < 0 ||
        segment.getSourceByteOffset().getInt() < 0 ||
        segment.getDestinationByteOffset().getInt() < 0 ||
        segment.getByteSize().getInt() <= 0 || movementId < previousMovementId)
      return module.emitError(
          "invalid or nondeterministically ordered memory segment");
    SmallVector<TileMemorySegmentAttr> &group = segmentsByMovement[movementId];
    if (segment.getOrdinal().getInt() != static_cast<int64_t>(group.size()))
      return module.emitError("memory segment ordinals are not contiguous");
    group.push_back(segment);
    previousSegmentId = id;
    previousMovementId = movementId;
  }

  for (const auto &[movementId, movement] : movementById) {
    auto group = segmentsByMovement.find(movementId);
    if (movement.getMode() != MemoryMovementMode::Segmented) {
      if (group != segmentsByMovement.end())
        return module.emitError(
            "non-segmented movement unexpectedly owns segment records");
      continue;
    }
    if (group == segmentsByMovement.end() || group->second.size() < 2)
      return module.emitError(
          "segmented movement requires at least two nonempty segments");

    bool sourceKnown = viewById.count(movement.getSourceViewId().getInt());
    bool destinationKnown =
        viewById.count(movement.getDestinationViewId().getInt());
    int64_t sourceElementBytes = -1;
    int64_t destinationElementBytes = -1;
    SmallVector<ByteRun> sourceRuns;
    SmallVector<ByteRun> destinationRuns;
    if (sourceKnown) {
      TileMemoryViewAttr view =
          viewById.at(movement.getSourceViewId().getInt());
      TileMemoryOwnerAttr owner = ownerById.at(view.getOwnerId().getInt());
      FailureOr<int64_t> elementBytes = inferOwnerElementBytes(module, owner);
      if (failed(elementBytes))
        return failure();
      sourceElementBytes = *elementBytes;
      auto runs = getVerifiedViewRuns(module, view, owner, *elementBytes);
      if (failed(runs))
        return failure();
      sourceRuns = std::move(*runs);
    }
    if (destinationKnown) {
      TileMemoryViewAttr view =
          viewById.at(movement.getDestinationViewId().getInt());
      TileMemoryOwnerAttr owner = ownerById.at(view.getOwnerId().getInt());
      FailureOr<int64_t> elementBytes = inferOwnerElementBytes(module, owner);
      if (failed(elementBytes))
        return failure();
      destinationElementBytes = *elementBytes;
      auto runs = getVerifiedViewRuns(module, view, owner, *elementBytes);
      if (failed(runs))
        return failure();
      destinationRuns = std::move(*runs);
    }
    if (sourceKnown && destinationKnown &&
        sourceElementBytes != destinationElementBytes)
      return module.emitError(
          "segmented movement has incompatible endpoint element sizes");

    int64_t totalBytes = 0;
    for (size_t index = 0; index < group->second.size(); ++index) {
      TileMemorySegmentAttr segment = group->second[index];
      int64_t bytes = segment.getByteSize().getInt();
      std::optional<int64_t> next = llvm::checkedAdd(totalBytes, bytes);
      if (!next)
        return module.emitError("segmented movement byte count overflowed");
      totalBytes = *next;
      if ((sourceKnown &&
           (segment.getSourceByteOffset().getInt() % sourceElementBytes != 0 ||
            bytes % sourceElementBytes != 0 ||
            !byteRangeContainedInRuns(
                sourceRuns, segment.getSourceByteOffset().getInt(), bytes))) ||
          (destinationKnown &&
           (segment.getDestinationByteOffset().getInt() %
                    destinationElementBytes !=
                0 ||
            bytes % destinationElementBytes != 0 ||
            !byteRangeContainedInRuns(
                destinationRuns, segment.getDestinationByteOffset().getInt(),
                bytes))))
        return module.emitError(
            "memory segment lies outside its endpoint view geometry");
      for (size_t prior = 0; prior < index; ++prior) {
        TileMemorySegmentAttr other = group->second[prior];
        if ((sourceKnown &&
             byteRangesOverlap(segment.getSourceByteOffset().getInt(), bytes,
                               other.getSourceByteOffset().getInt(),
                               other.getByteSize().getInt())) ||
            (destinationKnown &&
             byteRangesOverlap(segment.getDestinationByteOffset().getInt(),
                               bytes, other.getDestinationByteOffset().getInt(),
                               other.getByteSize().getInt())))
          return module.emitError(
              "segmented movement contains overlapping endpoint ranges");
      }
    }
    if (totalBytes != movement.getByteSize().getInt())
      return module.emitError(
          "segmented movement byte count does not match its movement");
  }

  for (TileMemoryAssemblyAttr assembly : *assemblies) {
    if (!ownerById.count(assembly.getOwnerId().getInt()))
      return module.emitError("invalid or duplicate memory assembly");
    auto sources = parseIntegerArray(module, assembly.getContributingViewIds(),
                                     "assembly source views");
    auto destinations = parseIntegerArray(
        module, assembly.getDestinationViewIds(), "assembly destination views");
    auto events = parseIntegerArray(module, assembly.getCompletionEventIds(),
                                    "assembly completion events");
    if (failed(sources) || failed(destinations) || failed(events) ||
        sources->empty() || sources->size() != destinations->size() ||
        sources->size() != events->size() ||
        !completionById.count(assembly.getReadinessEventId().getInt()))
      return module.emitError("incomplete memory assembly join metadata");
    int64_t readyEvent = assembly.getReadinessEventId().getInt();
    auto readyCompletion = completionById.find(readyEvent);
    if (readyCompletion == completionById.end())
      return module.emitError("assembly readiness event is unknown");
    auto routineComplete =
        routineCompletions.find(readyCompletion->second.getRoutine().getInt());
    if (routineComplete == routineCompletions.end() ||
        !eventEdgeKeys.count(
            std::make_tuple(readyEvent, routineComplete->second,
                            MemoryEventEdgeKind::AssemblyJoin)))
      return module.emitError(
          "assembly readiness does not precede routine completion");
    for (size_t index = 0; index < destinations->size(); ++index) {
      if (!viewById.count((*destinations)[index]) ||
          viewById.at((*destinations)[index]).getOwnerId().getInt() !=
              assembly.getOwnerId().getInt() ||
          !completionById.count((*events)[index]))
        return module.emitError("invalid memory assembly contribution");
      if (!local && !viewById.count((*sources)[index]))
        return module.emitError("assembly source view is unknown");
      if (!eventEdgeKeys.count(std::make_tuple(
              (*events)[index], readyEvent, MemoryEventEdgeKind::AssemblyJoin)))
        return module.emitError(
            "assembly contribution has no readiness-join edge");
      unsigned matchingMovements = 0;
      for (const auto &[movementId, movement] : movementById) {
        if (movement.getAssemblyId().getInt() == assembly.getId().getInt() &&
            movement.getSourceViewId().getInt() == (*sources)[index] &&
            movement.getDestinationViewId().getInt() ==
                (*destinations)[index]) {
          int64_t dmaEvent =
              movement.getDestinationCompletionEventId().getInt();
          if (eventEdgeKeys.count(
                  std::make_tuple(dmaEvent, (*events)[index],
                                  MemoryEventEdgeKind::AssemblyContribution)))
            ++matchingMovements;
        }
        (void)movementId;
      }
      if (matchingMovements != 1)
        return module.emitError(
            "assembly contribution has no unique materialized movement");
      for (size_t other = index + 1; other < destinations->size(); ++other) {
        FailureOr<bool> overlap =
            viewsOverlap(viewById.at((*destinations)[index]),
                         viewById.at((*destinations)[other]));
        if (failed(overlap))
          return failure();
        if (*overlap)
          return module.emitError("assembly destination regions overlap");
      }
    }
  }
  return success();
}

} // namespace

namespace mlir::sculptor::tile_memory {

LogicalResult buildAndAttachTileMemoryPlan(ModuleOp deployment) {
  PlanRecords plan;
  if (failed(collectDeploymentRecords(deployment, plan)) ||
      failed(collectOwners(deployment, plan)) ||
      failed(createFullViews(plan)) || failed(createRoutineBindings(plan)) ||
      failed(discoverInPlaceAliases(deployment, plan)))
    return failure();
  createRoutineCompletionEvents(plan);
  if (failed(createMovements(deployment, plan)) ||
      failed(createAssemblyRecords(deployment, plan)) ||
      failed(selectMovementModesAndBuildSegments(deployment, plan)) ||
      failed(createEventGraph(deployment, plan)))
    return failure();
  {
    FailureOr<EventReachability> reachable =
        computeEventReachability(plan, deployment);
    if (failed(reachable) ||
        failed(createFinalUseJoinEvents(deployment, plan, *reachable)) ||
        failed(orderPendingSendsBeforeControlSuccessors(deployment, plan)))
      return failure();
  }
  // Final-use joins are independent additions: they do not change
  // reachability among pre-existing events.  Insert all of them first and
  // rebuild the dense closure once instead of resizing every reachability row
  // for every owner.  Release joins are likewise batched; one final closure
  // below captures them for exact cross-tile projection.
  {
    FailureOr<EventReachability> withFinalUse =
        computeEventReachability(plan, deployment);
    if (failed(withFinalUse) ||
        failed(createOwnerReleaseEvents(deployment, plan, *withFinalUse)))
      return failure();
  }
  createOwnerLifetimes(plan);

  // Release joins are inserted into the middle of existing paths.  Rebuild
  // reachability once after all joins are present so tile-local projections
  // preserve their deployment-wide ordering exactly.
  FailureOr<EventReachability> completeReachability =
      computeEventReachability(plan, deployment);
  if (failed(completeReachability))
    return failure();

  attachPlan(deployment, plan);
  // The global deployment plan is descriptive; allocation and capacity are
  // decided by the exact tile-local plans below.  Avoid rebuilding one dense
  // global reachability closure merely to classify same-tile pairs.  A
  // conservative default cannot permit an illegal overlap, while proven
  // in-place aliases remain explicit.
  attachConservativeGlobalInterference(deployment, plan);
  // Exact allocation is tile-local.  Verifying the duplicated global memory
  // attributes repeats whole-program reachability analysis without adding an
  // allocation guarantee; each authoritative tile plan is verified below,
  // and the outliner separately verifies all cross-tile routes.
  std::map<int64_t, PlanRecords> plansByTile =
      bucketPlanRecordsByTile(plan, *completeReachability);
  for (ModuleOp tile : deployment.getOps<ModuleOp>()) {
    int64_t tileId =
        tile->getAttrOfType<IntegerAttr>(kPhysicalTileIdAttr).getInt();
    attachPlan(tile, plansByTile[tileId]);
    if (failed(rebuildTileMemoryInterference(tile)) ||
        failed(rebuildTileMemoryCapacitySummary(tile)) ||
        failed(verifyPlan(tile, true)))
      return failure();
  }
  return success();
}

LogicalResult verifyTileMemoryPlan(ModuleOp module) {
  bool local = module->hasAttr(kPhysicalTileIdAttr);
  return verifyPlan(module, local);
}

LogicalResult rebuildTileMemoryInterference(ModuleOp module) {
  return computeAndAttachInterferences(module);
}

LogicalResult rebuildTileMemoryCapacitySummary(ModuleOp module) {
  return computeAndAttachCapacity(module);
}

LogicalResult validateTileMemoryCapacity(ModuleOp module) {
  auto configured =
      module->getAttrOfType<IntegerAttr>(kConfiguredCapacityAttrName);
  if (!configured || configured.getInt() == 0)
    return success();
  if (configured.getInt() < 0)
    return module.emitError("tile memory capacity must be nonnegative");

  auto capacity =
      module->getAttrOfType<TileMemoryCapacityAttr>(kCapacityAttrName);
  if (!capacity)
    return module.emitError(
        "finalized tile is missing its memory-capacity summary");
  if (!capacity.getComplete().getValue())
    return module.emitError(
        "finalized tile memory-capacity summary is incomplete");
  int64_t requiredBytes = capacity.getRequiredLocalBytes().getInt();
  if (requiredBytes <= configured.getInt())
    return success();

  return module.emitError("physical tile ")
         << capacity.getTile().getInt() << " requires " << requiredBytes
         << " finalized local-memory bytes (external "
         << capacity.getExternalBytes().getInt() << ", persistent "
         << capacity.getPersistentBytes().getInt() << ", workspace "
         << capacity.getWorkspaceBytes().getInt() << ", scratchpad "
         << capacity.getScratchpadBytes().getInt()
         << ", peak routine temporary "
         << capacity.getRoutineTemporaryPeakBytes().getInt()
         << "), exceeding the configured capacity of " << configured.getInt()
         << " bytes";
}

FailureOr<ExactWorkspaceLayout>
buildExactWorkspaceLayout(ModuleOp module,
                          ArrayRef<WorkspaceAllocationRequest> requests) {
  return buildExactWorkspaceLayoutImpl(module, requests);
}

} // namespace mlir::sculptor::tile_memory
