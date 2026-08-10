#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingRealization.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

struct ResourceDemand {
  SmallVector<int64_t> bindingGroups;
  SmallVector<int64_t> requiredDigitalTileIds;
  int64_t anonymousAnalogLanes = 0;
  int64_t digitalLanes = 0;
};

struct ResourcePool {
  SmallVector<int64_t> digitalTileIds;
  SmallVector<MappingAnalogLaneRef> analogLanes;
};

using AnalogLaneKey = std::pair<int64_t, int64_t>;
using LeafEndpoint = std::pair<int64_t, int64_t>;

AnalogLaneKey getKey(const MappingAnalogLaneRef &lane) {
  return {lane.tileId, lane.laneIndex};
}

void sortAndUnique(SmallVectorImpl<int64_t> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

void sortAnalogLanes(SmallVectorImpl<MappingAnalogLaneRef> &lanes) {
  llvm::sort(lanes, [](const MappingAnalogLaneRef &left,
                       const MappingAnalogLaneRef &right) {
    return getKey(left) < getKey(right);
  });
  lanes.erase(std::unique(lanes.begin(), lanes.end()), lanes.end());
}

bool containsAnalogLane(ArrayRef<MappingAnalogLaneRef> lanes,
                        MappingAnalogLaneRef expected) {
  return llvm::is_contained(lanes, expected);
}

MappingRealization makeInfeasible(const MappingProblem &problem,
                                  StringRef reason) {
  MappingRealization result;
  result.feasible = false;
  result.infeasibilityReason = reason.str();
  result.logicalTileCount =
      problem.hardware.meshRows * problem.hardware.meshCols;
  result.analogLanesPerTile = problem.hardware.arraysPerCore;
  return result;
}

class ResourceRealizer {
public:
  ResourceRealizer(const MappingProblem &problem,
                   const ResourceAllocationTree &tree)
      : problem(problem), tree(tree) {
    for (const StructuralRATreeNode &node : tree.nodes)
      nodesById[node.id] = &node;
    for (const MappingWorkUnit &workUnit : tree.workUnits)
      workUnitsById[workUnit.id] = &workUnit;
    for (const MappingWorkUnitEdge &edge : tree.workUnitEdges) {
      workUnitEdgesByOperation[edge.sourceOperationId].push_back(&edge);
      if (edge.targetOperationId != edge.sourceOperationId)
        workUnitEdgesByOperation[edge.targetOperationId].push_back(&edge);
    }
    discoverConsumerBoundFills();
  }

  FailureOr<MappingRealization> run() {
    FailureOr<int64_t> tileCount =
        problem.hardware.getCoreCount(problem.anchor);
    if (failed(tileCount))
      return failure();

    realization.logicalTileCount = *tileCount;
    realization.analogLanesPerTile = problem.hardware.arraysPerCore;
    realization.digitalWorkPerTile.assign(*tileCount, 0);

    ResourcePool rootPool;
    for (int64_t tileId = 0; tileId < *tileCount; ++tileId) {
      rootPool.digitalTileIds.push_back(tileId);
      for (int64_t laneIndex = 0; laneIndex < problem.hardware.arraysPerCore;
           ++laneIndex)
        rootPool.analogLanes.push_back({tileId, laneIndex});
    }

    SmallVector<const LaneBindingGroup *> groups;
    groups.reserve(problem.graph.laneBindingGroups.size());
    for (const LaneBindingGroup &group : problem.graph.laneBindingGroups)
      groups.push_back(&group);
    llvm::sort(groups,
               [](const LaneBindingGroup *left, const LaneBindingGroup *right) {
                 return left->id < right->id;
               });
    if (groups.size() > rootPool.analogLanes.size()) {
      return makeInfeasible(
          problem,
          (Twine("mapping requires ") + Twine(groups.size()) +
           " persistent analog lane bindings but the logical tile pool "
           "provides " +
           Twine(rootPool.analogLanes.size()))
              .str());
    }
    FailureOr<bool> assignedBindings =
        assignPersistentBindings(groups, rootPool);
    if (failed(assignedBindings))
      return failure();
    if (!*assignedBindings)
      return realization;

    if (problem.setupBindingPolicy == SetupBindingPolicy::ConsumerAnchored &&
        failed(initializeConsumerAnchoredReservations(groups)))
      return failure();

    if (failed(discoverUniformSiblingTemplates(tree.rootId)))
      return failure();

    FailureOr<ResourceDemand> rootDemand = collectDemand(tree.rootId);
    if (failed(rootDemand))
      return failure();
    if (rootDemand->digitalLanes >
        static_cast<int64_t>(rootPool.digitalTileIds.size())) {
      return makeInfeasible(
          problem,
          (Twine("mapping requires ") + Twine(rootDemand->digitalLanes) +
           " concurrent digital lanes but the logical tile pool provides " +
           Twine(rootPool.digitalTileIds.size()))
              .str());
    }
    SmallVector<int64_t> rootPhaseWork(*tileCount, 0);
    FailureOr<bool> assigned = assignNode(tree.rootId, rootPool, rootPhaseWork);
    if (failed(assigned))
      return failure();
    if (!*assigned)
      return realization;

    if (problem.setupBindingPolicy == SetupBindingPolicy::ConsumerAnchored &&
        failed(canonicalizeConsumerAnchoredTileIds()))
      return failure();

    llvm::sort(realization.nodeAllocations,
               [](const MappingNodeResourceAllocation &left,
                  const MappingNodeResourceAllocation &right) {
                 return left.nodeId < right.nodeId;
               });
    llvm::sort(realization.leafAssignments,
               [](const MappingLeafAssignment &left,
                  const MappingLeafAssignment &right) {
                 return left.leafId < right.leafId;
               });
    if (failed(verifyMappingRealization(realization, problem, tree)))
      return failure();
    return realization;
  }

private:
  FailureOr<int64_t> multiplyWork(int64_t left, int64_t right,
                                  StringRef description) const {
    std::optional<int64_t> product = llvm::checkedMul(left, right);
    if (!product) {
      problem.anchor->emitError(description) << " overflows int64";
      return failure();
    }
    return *product;
  }

  FailureOr<int64_t> addWork(int64_t left, int64_t right,
                             StringRef description) const {
    std::optional<int64_t> sum = llvm::checkedAdd(left, right);
    if (!sum) {
      problem.anchor->emitError(description) << " overflows int64";
      return failure();
    }
    return *sum;
  }

  int64_t getScalarBodyWeight(Operation *operation) const {
    int64_t scalarOperations = 0;
    for (Region &region : operation->getRegions()) {
      region.walk([&](Operation *nested) {
        if (!nested->hasTrait<OpTrait::IsTerminator>())
          ++scalarOperations;
      });
    }
    return std::max<int64_t>(scalarOperations, 1);
  }

  FailureOr<int64_t>
  estimateDigitalLeafWork(const StructuralRATreeNode &node) const {
    const ComputeOperation &operation =
        problem.graph.operations[node.operationId];
    if (operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
        LogicalLaneKind::Digital)
      return int64_t{0};

    const MappingWorkUnit *workUnit = nullptr;
    if (node.workUnitId >= 0) {
      workUnit = workUnitsById.lookup(node.workUnitId);
      if (!workUnit) {
        problem.anchor->emitError(
            "digital balancing cannot resolve mapping work unit ")
            << node.workUnitId;
        return failure();
      }
      if (workUnit->iterationSizes.size() != operation.iterationDomain.size()) {
        operation.operation->emitError(
            "digital work-unit rank does not match its iteration domain");
        return failure();
      }
    }

    int64_t work = 1;
    for (auto [index, dimension] : llvm::enumerate(operation.iterationDomain)) {
      int64_t extent =
          workUnit ? workUnit->iterationSizes[index] : dimension.staticExtent;
      if (ShapedType::isDynamic(extent) || extent <= 0) {
        operation.operation->emitError(
            "digital balancing requires positive static iteration extents");
        return failure();
      }
      FailureOr<int64_t> product =
          multiplyWork(work, extent, "digital iteration work estimate");
      if (failed(product))
        return failure();
      work = *product;
    }
    FailureOr<int64_t> bodyWeighted =
        multiplyWork(work, getScalarBodyWeight(operation.operation),
                     "digital scalar work estimate");
    if (failed(bodyWeighted))
      return failure();
    return multiplyWork(*bodyWeighted, node.workGroupCount,
                        "digital grouped work estimate");
  }

  FailureOr<int64_t> collectDigitalWork(int64_t nodeId) {
    auto cached = subtreeDigitalWork.find(nodeId);
    if (cached != subtreeDigitalWork.end())
      return cached->second;
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    if (!node) {
      problem.anchor->emitError(
          "digital balancing cannot resolve RA-tree node ")
          << nodeId;
      return failure();
    }
    if (node->kind == RATreeNodeKind::Leaf) {
      FailureOr<int64_t> work = estimateDigitalLeafWork(*node);
      if (failed(work))
        return failure();
      subtreeDigitalWork[nodeId] = *work;
      return *work;
    }

    int64_t total = 0;
    for (int64_t childId : node->childIds) {
      FailureOr<int64_t> child = collectDigitalWork(childId);
      if (failed(child))
        return failure();
      FailureOr<int64_t> sum =
          addWork(total, *child, "digital subtree work estimate");
      if (failed(sum))
        return failure();
      total = *sum;
    }
    subtreeDigitalWork[nodeId] = total;
    return total;
  }

  auto digitalTileScore(int64_t tileId, int64_t affinity,
                        ArrayRef<int64_t> phaseWork) const {
    int64_t phaseLoad = problem.balanceDigitalWork ? phaseWork[tileId] : 0;
    int64_t cumulativeLoad =
        problem.balanceDigitalWork ? realization.digitalWorkPerTile[tileId] : 0;
    return std::tuple(phaseLoad, cumulativeLoad, -affinity, tileId);
  }

  bool hasDependency(int64_t producerId, int64_t consumerId) const {
    const ComputeOperation &producer = problem.graph.operations[producerId];
    for (int64_t tensorId : producer.outputTensors) {
      if (llvm::is_contained(problem.graph.tensors[tensorId].consumerOperations,
                             consumerId))
        return true;
    }
    return false;
  }

  void discoverConsumerBoundFills() {
    for (const StructuralRATreeNode &node : tree.nodes) {
      if (node.kind != RATreeNodeKind::TemporalCut || node.childIds.size() < 2)
        continue;

      const StructuralRATreeNode *consumerNode =
          nodesById.lookup(node.childIds.back());
      if (!consumerNode || consumerNode->kind != RATreeNodeKind::Leaf)
        continue;
      const ComputeOperation &consumer =
          problem.graph.operations[consumerNode->operationId];
      if (consumer.requiredLane.value_or(LogicalLaneKind::Digital) !=
          LogicalLaneKind::Digital)
        continue;

      SmallVector<LeafEndpoint> fillEndpoints;
      for (int64_t childId : ArrayRef(node.childIds).drop_back()) {
        const StructuralRATreeNode *fillNode = nodesById.lookup(childId);
        if (!fillNode || fillNode->kind != RATreeNodeKind::Leaf) {
          fillEndpoints.clear();
          break;
        }
        const ComputeOperation &fill =
            problem.graph.operations[fillNode->operationId];
        if (fill.operation->getName().getStringRef() != "linalg.fill" ||
            fill.requiredLane.value_or(LogicalLaneKind::Digital) !=
                LogicalLaneKind::Digital ||
            !hasDependency(fill.id, consumer.id)) {
          fillEndpoints.clear();
          break;
        }
        fillEndpoints.push_back({fill.id, fillNode->workUnitId});
      }
      if (fillEndpoints.empty())
        continue;

      int64_t groupId = consumerNode->id;
      LeafEndpoint consumerEndpoint{consumer.id, consumerNode->workUnitId};
      digitalColocationGroupByEndpoint[consumerEndpoint] = groupId;
      SmallVector<LeafEndpoint> &members =
          digitalColocationGroupMembers[groupId];
      members.push_back(consumerEndpoint);
      for (LeafEndpoint fillEndpoint : fillEndpoints) {
        digitalColocationGroupByEndpoint[fillEndpoint] = groupId;
        members.push_back(fillEndpoint);
      }
      llvm::sort(members);
      members.erase(std::unique(members.begin(), members.end()), members.end());
    }
  }

  std::optional<int64_t>
  getPinnedDigitalTile(const ComputeOperation &operation) const {
    if (!operation.mvmWaveId)
      return std::nullopt;
    if (operation.mvmWaveMember) {
      auto member = waveMemberTiles.find(
          {*operation.mvmWaveId, *operation.mvmWaveMember});
      if (member != waveMemberTiles.end())
        return member->second;
    }
    auto wave = waveTiles.find(*operation.mvmWaveId);
    return wave == waveTiles.end() ? std::nullopt
                                   : std::optional<int64_t>{wave->second};
  }

  std::optional<int64_t>
  getRequiredDigitalTile(const ComputeOperation &operation,
                         int64_t workUnitId) const {
    auto uniform = uniformSiblingDigitalTiles.find({operation.id, workUnitId});
    if (uniform != uniformSiblingDigitalTiles.end())
      return uniform->second;
    return getPinnedDigitalTile(operation);
  }

  LogicalResult initializeConsumerAnchoredReservations(
      ArrayRef<const LaneBindingGroup *> groups) {
    for (const LaneBindingGroup *group : groups) {
      auto binding = bindingLanes.find(group->id);
      if (binding == bindingLanes.end()) {
        problem.anchor->emitError(
            "consumer-anchored setup policy cannot resolve lane-binding "
            "group ")
            << group->id;
        return failure();
      }
      bool hasPhysicalMVM = llvm::any_of(group->operationIds, [&](int64_t id) {
        return id >= 0 &&
               id < static_cast<int64_t>(problem.graph.operations.size()) &&
               problem.graph.operations[id].kind ==
                   ComputeOperationKind::PhysicalMVM;
      });
      if (!hasPhysicalMVM) {
        problem.graph.operations[group->setupOperationId].operation->emitError(
            "consumer-anchored setup has no physical MVM consumer");
        return failure();
      }
      consumerAnchoredReservedTiles.insert(binding->second.tileId);
    }
    return success();
  }

  bool isReservedForConsumer(int64_t tileId) const {
    return problem.setupBindingPolicy ==
               SetupBindingPolicy::ConsumerAnchored &&
           consumerAnchoredReservedTiles.contains(tileId);
  }

  bool canUseUnpinnedDigitalTile(int64_t tileId) const {
    return !isReservedForConsumer(tileId);
  }

  void collectNonSetupLeafOrder(int64_t nodeId,
                                DenseMap<int64_t, int64_t> &leafOrder,
                                int64_t &nextOrder) const {
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    assert(node && "verified RA tree must contain every referenced node");
    if (node->kind == RATreeNodeKind::Leaf) {
      if (problem.graph.operations[node->operationId].kind !=
          ComputeOperationKind::MatrixSetup)
        leafOrder[node->id] = nextOrder++;
      return;
    }
    for (int64_t childId : node->childIds)
      collectNonSetupLeafOrder(childId, leafOrder, nextOrder);
  }

  LogicalResult canonicalizeConsumerAnchoredTileIds() {
    DenseMap<int64_t, int64_t> leafOrder;
    int64_t nextOrder = 0;
    collectNonSetupLeafOrder(tree.rootId, leafOrder, nextOrder);

    DenseSet<int64_t> activeTileSet;
    DenseMap<int64_t, int64_t> firstUse;
    for (const MappingLeafAssignment &assignment :
         realization.leafAssignments) {
      activeTileSet.insert(assignment.tileId);
      auto order = leafOrder.find(assignment.leafId);
      if (order == leafOrder.end())
        continue;
      auto current = firstUse.find(assignment.tileId);
      if (current == firstUse.end() || order->second < current->second)
        firstUse[assignment.tileId] = order->second;
    }

    SmallVector<int64_t> targetTileIds(activeTileSet.begin(),
                                       activeTileSet.end());
    llvm::sort(targetTileIds);
    SmallVector<int64_t> tilesByConsumer(targetTileIds.begin(),
                                         targetTileIds.end());
    llvm::sort(tilesByConsumer, [&](int64_t left, int64_t right) {
      int64_t leftUse = firstUse.contains(left)
                            ? firstUse.lookup(left)
                            : std::numeric_limits<int64_t>::max();
      int64_t rightUse = firstUse.contains(right)
                             ? firstUse.lookup(right)
                             : std::numeric_limits<int64_t>::max();
      return std::pair(leftUse, left) < std::pair(rightUse, right);
    });

    DenseMap<int64_t, int64_t> remap;
    for (auto [index, oldTileId] : llvm::enumerate(tilesByConsumer))
      remap[oldTileId] = targetTileIds[index];

    auto remapTile = [&](int64_t tileId) {
      auto mapped = remap.find(tileId);
      return mapped == remap.end() ? tileId : mapped->second;
    };
    for (MappingLeafAssignment &assignment : realization.leafAssignments)
      assignment.tileId = remapTile(assignment.tileId);
    for (MappingNodeResourceAllocation &allocation :
         realization.nodeAllocations) {
      for (int64_t &tileId : allocation.digitalTileIds)
        tileId = remapTile(tileId);
      for (MappingAnalogLaneRef &lane : allocation.analogLanes)
        lane.tileId = remapTile(lane.tileId);
      sortAndUnique(allocation.digitalTileIds);
      sortAnalogLanes(allocation.analogLanes);
    }

    SmallVector<int64_t> remappedWork = realization.digitalWorkPerTile;
    for (int64_t oldTileId : targetTileIds)
      remappedWork[remapTile(oldTileId)] =
          realization.digitalWorkPerTile[oldTileId];
    realization.digitalWorkPerTile = std::move(remappedWork);
    return success();
  }

  std::string getSubtreeShapeSignature(int64_t nodeId) {
    auto cached = subtreeShapeSignatures.find(nodeId);
    if (cached != subtreeShapeSignatures.end())
      return cached->second;

    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    std::string signature;
    llvm::raw_string_ostream stream(signature);
    if (node->kind == RATreeNodeKind::Leaf) {
      const ComputeOperation &operation =
          problem.graph.operations[node->operationId];
      stream << "L(" << static_cast<int>(operation.kind) << ","
             << static_cast<int>(operation.requiredLane.value_or(
                    LogicalLaneKind::Digital))
             << "," << operation.members.size() << ","
             << operation.inputTensors.size() << ","
             << operation.outputTensors.size() << ","
             << operation.mvmWaveId.has_value() << ","
             << operation.mvmWaveMember.value_or(-1) << ","
             << operation.mvmWaveSize.value_or(-1) << ","
             << node->workGroupCount;
      for (const ComputeIterationDimension &dimension :
           operation.iterationDomain) {
        stream << ":" << static_cast<int>(dimension.kind) << "x"
               << dimension.staticExtent;
      }
      stream << ")";
    } else {
      stream << (node->kind == RATreeNodeKind::TemporalCut ? "T[" : "S[");
      for (int64_t childId : node->childIds)
        stream << getSubtreeShapeSignature(childId) << ";";
      stream << "]";
    }
    subtreeShapeSignatures[nodeId] = signature;
    return signature;
  }

  std::optional<int64_t> getMVMBodyHomeTile(int64_t nodeId) {
    std::optional<int64_t> waveId;
    std::optional<std::pair<int64_t, int64_t>> home;
    for (LeafEndpoint endpoint : collectSubtreeEndpoints(nodeId)) {
      const ComputeOperation &operation =
          problem.graph.operations[endpoint.first];
      if (operation.kind != ComputeOperationKind::PhysicalMVM)
        continue;
      if (!operation.mvmWaveId || !operation.mvmWaveMember)
        return std::nullopt;
      if (waveId && *waveId != *operation.mvmWaveId)
        return std::nullopt;
      waveId = *operation.mvmWaveId;
      std::optional<int64_t> tile = getPinnedDigitalTile(operation);
      if (!tile)
        return std::nullopt;
      std::pair<int64_t, int64_t> candidate{*operation.mvmWaveMember, *tile};
      if (!home || candidate < *home)
        home = candidate;
    }
    return waveId && home ? std::optional<int64_t>{home->second}
                          : std::nullopt;
  }

  LogicalResult configureUniformSiblingGroup(ArrayRef<int64_t> childIds) {
    SmallVector<std::pair<LeafEndpoint, int64_t>> proposed;
    for (int64_t childId : childIds) {
      std::optional<int64_t> homeTile = getMVMBodyHomeTile(childId);
      if (!homeTile)
        return success();
      for (LeafEndpoint endpoint : collectSubtreeEndpoints(childId)) {
        const ComputeOperation &operation =
            problem.graph.operations[endpoint.first];
        if (operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
                LogicalLaneKind::Digital ||
            getPinnedDigitalTile(operation))
          continue;
        proposed.push_back({endpoint, *homeTile});
      }
    }

    for (auto [endpoint, tileId] : proposed) {
      auto existing = uniformSiblingDigitalTiles.find(endpoint);
      if (existing != uniformSiblingDigitalTiles.end() &&
          existing->second != tileId)
        return success();
    }
    for (auto [endpoint, tileId] : proposed)
      uniformSiblingDigitalTiles[endpoint] = tileId;
    return success();
  }

  LogicalResult discoverUniformSiblingTemplates(int64_t nodeId) {
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    if (!node) {
      problem.anchor->emitError(
          "cannot discover uniform sibling templates for unknown RA node ")
          << nodeId;
      return failure();
    }
    for (int64_t childId : node->childIds) {
      if (failed(discoverUniformSiblingTemplates(childId)))
        return failure();
    }
    if (node->kind != RATreeNodeKind::SpatialCut)
      return success();

    std::map<std::string, SmallVector<int64_t>> equivalentChildren;
    for (int64_t childId : node->childIds)
      equivalentChildren[getSubtreeShapeSignature(childId)].push_back(childId);
    for (auto &[signature, childIds] : equivalentChildren) {
      (void)signature;
      if (childIds.size() > 1 &&
          failed(configureUniformSiblingGroup(childIds)))
        return failure();
    }
    return success();
  }

  FailureOr<std::optional<int64_t>>
  selectDigitalColocationTile(int64_t groupId, const ResourcePool &pool,
                              ArrayRef<int64_t> phaseWork) {
    auto existing = digitalColocationTiles.find(groupId);
    if (existing != digitalColocationTiles.end()) {
      if (!llvm::is_contained(pool.digitalTileIds, existing->second)) {
        realization = makeInfeasible(
            problem, (Twine("digital co-location group ") + Twine(groupId) +
                      " cannot access its selected core")
                         .str());
        return std::optional<int64_t>{};
      }
      return std::optional<int64_t>{existing->second};
    }

    auto group = digitalColocationGroupMembers.find(groupId);
    if (group == digitalColocationGroupMembers.end()) {
      problem.anchor->emitError("cannot resolve a digital co-location group");
      return failure();
    }
    const SmallVector<LeafEndpoint> &members = group->second;
    for (LeafEndpoint endpoint : members) {
      const ComputeOperation &operation =
          problem.graph.operations[endpoint.first];
      std::optional<int64_t> pinnedTile =
          getRequiredDigitalTile(operation, endpoint.second);
      if (!pinnedTile)
        continue;
      if (!llvm::is_contained(pool.digitalTileIds, *pinnedTile)) {
        realization = makeInfeasible(
            problem, (Twine("digital co-location group ") + Twine(groupId) +
                      " cannot access its MVM body core")
                         .str());
        return std::optional<int64_t>{};
      }
      digitalColocationTiles[groupId] = *pinnedTile;
      return pinnedTile;
    }

    if (pool.digitalTileIds.empty()) {
      realization =
          makeInfeasible(problem, (Twine("digital co-location group ") +
                                   Twine(groupId) + " has no available core")
                                      .str());
      return std::optional<int64_t>{};
    }

    auto affinity = [&](int64_t tileId) {
      int64_t result = 0;
      for (LeafEndpoint endpoint : members)
        result += tileAffinity(endpoint.first, tileId, endpoint.second);
      return result;
    };
    int64_t selected = pool.digitalTileIds.front();
    auto firstCandidate = llvm::find_if(pool.digitalTileIds, [&](int64_t id) {
      return canUseUnpinnedDigitalTile(id);
    });
    if (firstCandidate == pool.digitalTileIds.end()) {
      realization = makeInfeasible(
          problem, (Twine("digital co-location group ") + Twine(groupId) +
                    " has no consumer-unreserved core")
                       .str());
      return std::optional<int64_t>{};
    }
    selected = *firstCandidate;
    auto selectedScore =
        digitalTileScore(selected, affinity(selected), phaseWork);
    for (int64_t tileId : pool.digitalTileIds) {
      if (!canUseUnpinnedDigitalTile(tileId))
        continue;
      auto score = digitalTileScore(tileId, affinity(tileId), phaseWork);
      if (score < selectedScore) {
        selected = tileId;
        selectedScore = score;
      }
    }
    digitalColocationTiles[groupId] = selected;
    return std::optional<int64_t>{selected};
  }

  LogicalResult recordWaveBindingTiles(bool requireDistinctMemberTiles) {
    for (const MVMWave &wave : problem.graph.mvmWaves) {
      std::optional<std::pair<int64_t, int64_t>> home;
      DenseSet<int64_t> memberTiles;
      for (int64_t operationId : wave.physicalMVMOperationIds) {
        const ComputeOperation &operation =
            problem.graph.operations[operationId];
        if (!operation.laneBindingGroup || !operation.mvmWaveMember) {
          operation.operation->emitError(
              "MVM wave contains an unbound or unidentified physical MVM");
          return failure();
        }
        auto binding = bindingLanes.find(*operation.laneBindingGroup);
        if (binding == bindingLanes.end()) {
          operation.operation->emitError(
              "MVM wave cannot resolve its persistent analog binding");
          return failure();
        }
        int64_t tileId = binding->second.tileId;
        if (requireDistinctMemberTiles && !memberTiles.insert(tileId).second) {
          operation.operation->emitError(
              "spread MVM-body policy assigned multiple physical MVMs to "
              "one logical tile");
          return failure();
        }
        auto [member, inserted] = waveMemberTiles.try_emplace(
            std::make_pair(wave.id, *operation.mvmWaveMember), tileId);
        if (!inserted && member->second != tileId) {
          operation.operation->emitError(
              "one MVM wave member was assigned to multiple logical tiles");
          return failure();
        }
        std::pair<int64_t, int64_t> candidate{*operation.mvmWaveMember, tileId};
        if (!home || candidate < *home)
          home = candidate;
      }
      if (!home) {
        problem.anchor->emitError("MVM wave has no physical MVM members");
        return failure();
      }
      waveTiles[wave.id] = home->second;
    }
    return success();
  }

  FailureOr<bool>
  assignPackedBindings(ArrayRef<const LaneBindingGroup *> groups,
                       const ResourcePool &rootPool) {
    DenseSet<int64_t> knownGroups;
    for (const LaneBindingGroup *group : groups) {
      if (group->id < 0 || !knownGroups.insert(group->id).second) {
        problem.anchor->emitError(
            "mapping realization found an invalid or duplicate analog "
            "lane-binding group");
        return failure();
      }
    }

    DenseMap<int64_t, SmallVector<int64_t>> relatedGroups;
    DenseMap<int64_t, SmallVector<int64_t>> wavesByGroup;
    for (const MVMWave &wave : problem.graph.mvmWaves) {
      SmallVector<int64_t> waveBindings;
      for (int64_t operationId : wave.physicalMVMOperationIds) {
        const ComputeOperation &operation =
            problem.graph.operations[operationId];
        if (!operation.laneBindingGroup) {
          operation.operation->emitError(
              "packed MVM wave contains an unbound physical MVM");
          return failure();
        }
        if (!knownGroups.contains(*operation.laneBindingGroup)) {
          operation.operation->emitError(
              "MVM wave references an unknown analog lane-binding group ")
              << *operation.laneBindingGroup;
          return failure();
        }
        if (!llvm::is_contained(waveBindings, *operation.laneBindingGroup))
          waveBindings.push_back(*operation.laneBindingGroup);
      }
      llvm::sort(waveBindings);
      for (int64_t groupId : waveBindings)
        wavesByGroup[groupId].push_back(wave.id);
      if (waveBindings.size() > 1) {
        int64_t first = waveBindings.front();
        for (int64_t groupId : ArrayRef(waveBindings).drop_front()) {
          relatedGroups[first].push_back(groupId);
          relatedGroups[groupId].push_back(first);
        }
      }
    }

    struct BindingComponent {
      SmallVector<int64_t> groupIds;
      SmallVector<int64_t> waveIds;
    };

    SmallVector<BindingComponent> components;
    DenseSet<int64_t> visitedGroups;
    for (const LaneBindingGroup *group : groups) {
      if (!visitedGroups.insert(group->id).second)
        continue;

      BindingComponent component;
      SmallVector<int64_t> pending{group->id};
      for (size_t index = 0; index < pending.size(); ++index) {
        int64_t groupId = pending[index];
        component.groupIds.push_back(groupId);
        component.waveIds.append(wavesByGroup[groupId].begin(),
                                 wavesByGroup[groupId].end());
        for (int64_t neighbor : relatedGroups[groupId]) {
          if (visitedGroups.insert(neighbor).second)
            pending.push_back(neighbor);
        }
      }
      sortAndUnique(component.groupIds);
      sortAndUnique(component.waveIds);
      components.push_back(std::move(component));
    }

    llvm::sort(components,
               [](const BindingComponent &left, const BindingComponent &right) {
                 bool leftIsWave = !left.waveIds.empty();
                 bool rightIsWave = !right.waveIds.empty();
                 if (leftIsWave != rightIsWave)
                   return leftIsWave;
                 if (left.groupIds.size() != right.groupIds.size())
                   return left.groupIds.size() > right.groupIds.size();
                 if (left.waveIds != right.waveIds)
                   return left.waveIds < right.waveIds;
                 return left.groupIds < right.groupIds;
               });

    SmallVector<int64_t> tileIds;
    for (MappingAnalogLaneRef lane : rootPool.analogLanes)
      tileIds.push_back(lane.tileId);
    sortAndUnique(tileIds);

    std::set<AnalogLaneKey> usedLanes;
    DenseMap<int64_t, int64_t> tileLaneLoads;
    for (const BindingComponent &component : components) {
      for (size_t offset = 0; offset < component.groupIds.size();) {
        int64_t chunkSize = std::min<int64_t>(
            problem.hardware.arraysPerCore,
            static_cast<int64_t>(component.groupIds.size() - offset));
        std::optional<int64_t> selectedTile;
        for (int64_t tileId : tileIds) {
          // Keep separately derived wave components on separate logical
          // tiles. They can be concurrent siblings in the RA tree and cannot
          // share the tile's one digital lane even when analog lanes remain.
          if (tileLaneLoads.lookup(tileId) != 0)
            continue;
          if (!selectedTile || tileId < *selectedTile)
            selectedTile = tileId;
        }
        if (!selectedTile) {
          realization = makeInfeasible(
              problem,
              (Twine("cannot compactly place MVM binding chunk containing ") +
               Twine(chunkSize) + " analog arrays")
                  .str());
          return false;
        }

        for (int64_t index = 0; index < chunkSize; ++index) {
          int64_t groupId = component.groupIds[offset + index];
          auto available = llvm::find_if(
              rootPool.analogLanes, [&](MappingAnalogLaneRef lane) {
                return lane.tileId == *selectedTile &&
                       !usedLanes.contains(getKey(lane));
              });
          if (available == rootPool.analogLanes.end()) {
            problem.anchor->emitError(
                "packed MVM assignment exhausted a core unexpectedly");
            return failure();
          }
          bindingLanes[groupId] = *available;
          usedLanes.insert(getKey(*available));
        }
        tileLaneLoads[*selectedTile] += chunkSize;
        offset += chunkSize;
      }
    }
    if (failed(recordWaveBindingTiles(/*requireDistinctMemberTiles=*/false)))
      return failure();
    return true;
  }

  FailureOr<bool>
  assignSpreadBindings(ArrayRef<const LaneBindingGroup *> groups,
                       const ResourcePool &rootPool) {
    DenseSet<int64_t> knownGroups;
    for (const LaneBindingGroup *group : groups) {
      if (group->id < 0 || !knownGroups.insert(group->id).second) {
        problem.anchor->emitError(
            "mapping realization found an invalid or duplicate analog "
            "lane-binding group");
        return failure();
      }
    }

    DenseMap<int64_t, SmallVector<int64_t>> conflicts;
    DenseMap<int64_t, int64_t> waveCounts;
    for (const MVMWave &wave : problem.graph.mvmWaves) {
      SmallVector<int64_t> waveBindings;
      for (int64_t operationId : wave.physicalMVMOperationIds) {
        const ComputeOperation &operation =
            problem.graph.operations[operationId];
        if (!operation.laneBindingGroup || !operation.mvmWaveMember) {
          operation.operation->emitError(
              "spread MVM body contains an unbound physical MVM");
          return failure();
        }
        int64_t groupId = *operation.laneBindingGroup;
        if (!knownGroups.contains(groupId)) {
          operation.operation->emitError(
              "MVM body references an unknown analog lane-binding group ")
              << groupId;
          return failure();
        }
        if (!llvm::is_contained(waveBindings, groupId))
          waveBindings.push_back(groupId);
      }
      llvm::sort(waveBindings);
      for (int64_t groupId : waveBindings)
        ++waveCounts[groupId];
      for (auto [leftIndex, left] : llvm::enumerate(waveBindings)) {
        for (int64_t right : ArrayRef(waveBindings).drop_front(leftIndex + 1)) {
          conflicts[left].push_back(right);
          conflicts[right].push_back(left);
        }
      }
    }
    for (auto &[groupId, neighbors] : conflicts) {
      (void)groupId;
      sortAndUnique(neighbors);
    }

    SmallVector<const LaneBindingGroup *> orderedGroups(groups.begin(),
                                                        groups.end());
    auto conflictCount = [&](int64_t groupId) {
      auto found = conflicts.find(groupId);
      return found == conflicts.end() ? size_t{0} : found->second.size();
    };
    llvm::sort(orderedGroups, [&](const LaneBindingGroup *left,
                                  const LaneBindingGroup *right) {
      size_t leftDegree = conflictCount(left->id);
      size_t rightDegree = conflictCount(right->id);
      if (leftDegree != rightDegree)
        return leftDegree > rightDegree;
      if (waveCounts.lookup(left->id) != waveCounts.lookup(right->id))
        return waveCounts.lookup(left->id) > waveCounts.lookup(right->id);
      return left->id < right->id;
    });

    SmallVector<int64_t> tileIds;
    for (MappingAnalogLaneRef lane : rootPool.analogLanes)
      tileIds.push_back(lane.tileId);
    sortAndUnique(tileIds);

    std::set<AnalogLaneKey> usedLanes;
    DenseMap<int64_t, int64_t> groupTiles;
    DenseMap<int64_t, int64_t> tileLaneLoads;
    for (const LaneBindingGroup *group : orderedGroups) {
      std::optional<int64_t> selectedTile;
      for (int64_t tileId : tileIds) {
        bool hasFreeLane = llvm::any_of(
            rootPool.analogLanes, [&](MappingAnalogLaneRef lane) {
              return lane.tileId == tileId &&
                     !usedLanes.contains(getKey(lane));
            });
        if (!hasFreeLane)
          continue;
        bool conflictsOnTile = llvm::any_of(
            conflicts[group->id], [&](int64_t neighborId) {
              auto assigned = groupTiles.find(neighborId);
              return assigned != groupTiles.end() &&
                     assigned->second == tileId;
            });
        if (conflictsOnTile)
          continue;
        if (!selectedTile ||
            std::pair(tileLaneLoads.lookup(tileId), tileId) <
                std::pair(tileLaneLoads.lookup(*selectedTile), *selectedTile))
          selectedTile = tileId;
      }
      if (!selectedTile) {
        realization = makeInfeasible(
            problem,
            (Twine("cannot spread analog lane-binding group ") +
             Twine(group->id) + " across the available logical cores")
                .str());
        return false;
      }

      auto available =
          llvm::find_if(rootPool.analogLanes, [&](MappingAnalogLaneRef lane) {
            return lane.tileId == *selectedTile &&
                   !usedLanes.contains(getKey(lane));
          });
      if (available == rootPool.analogLanes.end()) {
        problem.anchor->emitError(
            "spread MVM assignment exhausted a core unexpectedly");
        return failure();
      }
      bindingLanes[group->id] = *available;
      groupTiles[group->id] = *selectedTile;
      usedLanes.insert(getKey(*available));
      ++tileLaneLoads[*selectedTile];
    }

    if (failed(recordWaveBindingTiles(/*requireDistinctMemberTiles=*/true)))
      return failure();
    return true;
  }

  FailureOr<bool>
  assignPersistentBindings(ArrayRef<const LaneBindingGroup *> groups,
                           const ResourcePool &rootPool) {
    switch (problem.mvmBodyPolicy) {
    case MVMBodyPolicy::Packed:
      return assignPackedBindings(groups, rootPool);
    case MVMBodyPolicy::Spread:
      return assignSpreadBindings(groups, rootPool);
    }
    llvm_unreachable("unknown MVM body policy");
  }

  FailureOr<ResourceDemand> collectDemand(int64_t nodeId) {
    auto cached = demands.find(nodeId);
    if (cached != demands.end())
      return cached->second;

    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    if (!node) {
      problem.anchor->emitError(
          "mapping realization cannot resolve RA-tree node ")
          << nodeId;
      return failure();
    }

    ResourceDemand result;
    if (node->kind == RATreeNodeKind::Leaf) {
      const ComputeOperation &operation =
          problem.graph.operations[node->operationId];
      LogicalLaneKind required =
          operation.requiredLane.value_or(LogicalLaneKind::Digital);
      if (required == LogicalLaneKind::Digital) {
        result.digitalLanes = 1;
        if (std::optional<int64_t> pinnedTile =
                getRequiredDigitalTile(operation, node->workUnitId))
          result.requiredDigitalTileIds.push_back(*pinnedTile);
      } else if (operation.laneBindingGroup) {
        if (!bindingLanes.contains(*operation.laneBindingGroup)) {
          operation.operation->emitError(
              "operation references an unknown analog lane-binding group ")
              << *operation.laneBindingGroup;
          return failure();
        }
        result.bindingGroups.push_back(*operation.laneBindingGroup);
      } else {
        result.anonymousAnalogLanes = 1;
      }
      demands[nodeId] = result;
      return result;
    }

    for (int64_t childId : node->childIds) {
      FailureOr<ResourceDemand> child = collectDemand(childId);
      if (failed(child))
        return failure();
      result.bindingGroups.append(child->bindingGroups.begin(),
                                  child->bindingGroups.end());
      result.requiredDigitalTileIds.append(
          child->requiredDigitalTileIds.begin(),
          child->requiredDigitalTileIds.end());
      if (node->kind == RATreeNodeKind::TemporalCut) {
        result.digitalLanes =
            std::max(result.digitalLanes, child->digitalLanes);
        result.anonymousAnalogLanes =
            std::max(result.anonymousAnalogLanes, child->anonymousAnalogLanes);
      } else {
        result.digitalLanes += child->digitalLanes;
        result.anonymousAnalogLanes += child->anonymousAnalogLanes;
      }
    }
    sortAndUnique(result.bindingGroups);
    sortAndUnique(result.requiredDigitalTileIds);
    demands[nodeId] = result;
    return result;
  }

  ArrayRef<LeafEndpoint> collectSubtreeEndpoints(int64_t nodeId) {
    auto cached = subtreeEndpoints.find(nodeId);
    if (cached != subtreeEndpoints.end())
      return cached->second;
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    SmallVector<LeafEndpoint> endpoints;
    if (node->kind == RATreeNodeKind::Leaf) {
      endpoints.push_back({node->operationId, node->workUnitId});
    } else {
      for (int64_t childId : node->childIds) {
        ArrayRef<LeafEndpoint> childEndpoints =
            collectSubtreeEndpoints(childId);
        endpoints.append(childEndpoints.begin(), childEndpoints.end());
      }
    }
    llvm::sort(endpoints);
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                    endpoints.end());
    auto [inserted, wasInserted] =
        subtreeEndpoints.try_emplace(nodeId, std::move(endpoints));
    (void)wasInserted;
    return inserted->second;
  }

  std::string describeSpatialChildren(int64_t parentId) {
    std::string description;
    llvm::raw_string_ostream stream(description);
    const StructuralRATreeNode *parent = nodesById.lookup(parentId);
    if (!parent)
      return description;
    stream << "; children=";
    for (auto [index, childId] : llvm::enumerate(parent->childIds)) {
      if (index != 0)
        stream << ",";
      stream << childId << "{ops=";
      ArrayRef<LeafEndpoint> endpoints = collectSubtreeEndpoints(childId);
      for (auto [endpointIndex, endpoint] : llvm::enumerate(endpoints)) {
        if (endpointIndex != 0)
          stream << ":";
        stream << endpoint.first;
      }
      auto demand = demands.find(childId);
      if (demand != demands.end() &&
          !demand->second.requiredDigitalTileIds.empty()) {
        stream << ",pinned=";
        for (auto [tileIndex, tileId] :
             llvm::enumerate(demand->second.requiredDigitalTileIds)) {
          if (tileIndex != 0)
            stream << ":";
          stream << tileId;
        }
      }
      stream << "}";
    }
    return description;
  }

  int64_t tileAffinity(int64_t operationId, int64_t tileId,
                       int64_t workUnitId = -1) const {
    int64_t affinity = 0;
    const ComputeOperation &operation = problem.graph.operations[operationId];
    llvm::SmallDenseSet<int64_t> tensors;
    tensors.insert(operation.inputTensors.begin(),
                   operation.inputTensors.end());
    tensors.insert(operation.outputTensors.begin(),
                   operation.outputTensors.end());
    for (int64_t tensorId : tensors) {
      const ComputeTensor &tensor = problem.graph.tensors[tensorId];
      int64_t weight = std::max<int64_t>(tensor.byteSize, 1);
      llvm::SmallDenseSet<int64_t> neighbors;
      neighbors.insert(tensor.producerOperations.begin(),
                       tensor.producerOperations.end());
      neighbors.insert(tensor.consumerOperations.begin(),
                       tensor.consumerOperations.end());
      neighbors.erase(operationId);
      for (int64_t neighborId : neighbors) {
        const ComputeOperation &neighbor = problem.graph.operations[neighborId];
        if (neighbor.laneBindingGroup) {
          auto binding = bindingLanes.find(*neighbor.laneBindingGroup);
          if (binding != bindingLanes.end() && binding->second.tileId == tileId)
            affinity += weight;
        }
        auto assigned = operationTiles.find(neighborId);
        if (assigned != operationTiles.end() &&
            llvm::is_contained(assigned->second, tileId))
          affinity += weight;
      }
    }

    auto addEndpointAffinity = [&](int64_t neighborOperationId,
                                   int64_t neighborWorkUnitId, int64_t weight) {
      if (neighborWorkUnitId >= 0) {
        auto assigned =
            endpointTiles.find({neighborOperationId, neighborWorkUnitId});
        if (assigned != endpointTiles.end() &&
            llvm::is_contained(assigned->second, tileId))
          affinity += weight;
        return;
      }
      auto assigned = operationTiles.find(neighborOperationId);
      if (assigned != operationTiles.end() &&
          llvm::is_contained(assigned->second, tileId))
        affinity += weight;
    };
    auto incidentEdges = workUnitEdgesByOperation.find(operationId);
    if (incidentEdges == workUnitEdgesByOperation.end())
      return affinity;
    for (const MappingWorkUnitEdge *edge : incidentEdges->second) {
      bool isTarget =
          edge->targetOperationId == operationId &&
          (edge->targetWorkUnitId < 0 || edge->targetWorkUnitId == workUnitId);
      if (isTarget) {
        addEndpointAffinity(edge->sourceOperationId, edge->sourceWorkUnitId,
                            std::max<int64_t>(edge->byteSize, 1));
      }
      bool isSource =
          edge->sourceOperationId == operationId &&
          (edge->sourceWorkUnitId < 0 || edge->sourceWorkUnitId == workUnitId);
      if (isSource) {
        addEndpointAffinity(edge->targetOperationId, edge->targetWorkUnitId,
                            std::max<int64_t>(edge->byteSize, 1));
      }
    }
    return affinity;
  }

  int64_t subtreeAffinity(int64_t nodeId, int64_t tileId) {
    int64_t affinity = 0;
    for (LeafEndpoint endpoint : collectSubtreeEndpoints(nodeId)) {
      int64_t operationId = endpoint.first;
      const ComputeOperation &operation = problem.graph.operations[operationId];
      if (operation.laneBindingGroup) {
        auto binding = bindingLanes.find(*operation.laneBindingGroup);
        if (binding != bindingLanes.end() && binding->second.tileId == tileId)
          ++affinity;
      }
      affinity += tileAffinity(operationId, tileId, endpoint.second);
    }
    return affinity;
  }

  void recordNodeAllocation(int64_t nodeId, const ResourcePool &pool) {
    MappingNodeResourceAllocation allocation;
    allocation.nodeId = nodeId;
    allocation.digitalTileIds = pool.digitalTileIds;
    allocation.analogLanes = pool.analogLanes;
    sortAndUnique(allocation.digitalTileIds);
    sortAnalogLanes(allocation.analogLanes);
    realization.nodeAllocations.push_back(std::move(allocation));
  }

  FailureOr<std::optional<ResourcePool>>
  allocateSpatialChild(int64_t parentId, int64_t childId,
                       const ResourcePool &parentPool,
                       std::set<int64_t> &usedDigitalTiles,
                       std::set<AnalogLaneKey> &usedAnalogLanes,
                       ArrayRef<int64_t> phaseWork) {
    FailureOr<ResourceDemand> demand = collectDemand(childId);
    if (failed(demand))
      return failure();

    ResourcePool childPool;
    for (int64_t groupId : demand->bindingGroups) {
      MappingAnalogLaneRef lane = bindingLanes.lookup(groupId);
      if (!containsAnalogLane(parentPool.analogLanes, lane)) {
        realization = makeInfeasible(
            problem,
            (Twine("spatial child ") + Twine(childId) +
             " requires an analog binding outside its parent resource pool")
                .str());
        return std::optional<ResourcePool>{};
      }
      if (!usedAnalogLanes.insert(getKey(lane)).second) {
        realization = makeInfeasible(
            problem, (Twine("spatial cut assigns analog lane binding ") +
                      Twine(groupId) + " to multiple concurrent children")
                         .str());
        return std::optional<ResourcePool>{};
      }
      childPool.analogLanes.push_back(lane);
    }

    for (int64_t index = 0; index < demand->anonymousAnalogLanes; ++index) {
      auto available =
          llvm::find_if(parentPool.analogLanes, [&](MappingAnalogLaneRef lane) {
            return !usedAnalogLanes.contains(getKey(lane));
          });
      if (available == parentPool.analogLanes.end()) {
        realization =
            makeInfeasible(problem, (Twine("spatial child ") + Twine(childId) +
                                     " exceeds the available analog lanes")
                                        .str());
        return std::optional<ResourcePool>{};
      }
      usedAnalogLanes.insert(getKey(*available));
      childPool.analogLanes.push_back(*available);
    }

    for (int64_t tileId : demand->requiredDigitalTileIds) {
      if (!llvm::is_contained(parentPool.digitalTileIds, tileId)) {
        realization = makeInfeasible(
            problem,
            (Twine("spatial child ") + Twine(childId) +
             " requires a digital lane outside its parent resource pool")
                .str());
        return std::optional<ResourcePool>{};
      }
      if (llvm::is_contained(childPool.digitalTileIds, tileId))
        continue;
      if (!usedDigitalTiles.insert(tileId).second) {
        realization = makeInfeasible(
            problem, (Twine("spatial cut assigns logical core ") +
                      Twine(tileId) + " to multiple concurrent children "
                      "under RA node " + Twine(parentId) +
                      " while allocating child " + Twine(childId) +
                      describeSpatialChildren(parentId))
                         .str());
        return std::optional<ResourcePool>{};
      }
      childPool.digitalTileIds.push_back(tileId);
    }

    SmallVector<int64_t> digitalCandidates = parentPool.digitalTileIds;
    llvm::erase_if(digitalCandidates, [&](int64_t tileId) {
      return isReservedForConsumer(tileId) &&
             !llvm::is_contained(demand->requiredDigitalTileIds, tileId);
    });
    DenseMap<int64_t, int64_t> candidateAffinities;
    candidateAffinities.reserve(digitalCandidates.size());
    for (int64_t tileId : digitalCandidates)
      candidateAffinities[tileId] = subtreeAffinity(childId, tileId);
    llvm::sort(digitalCandidates, [&](int64_t left, int64_t right) {
      return digitalTileScore(left, candidateAffinities.lookup(left),
                              phaseWork) <
             digitalTileScore(right, candidateAffinities.lookup(right),
                              phaseWork);
    });
    for (int64_t index = childPool.digitalTileIds.size();
         index < demand->digitalLanes; ++index) {
      auto available = llvm::find_if(digitalCandidates, [&](int64_t tileId) {
        return !usedDigitalTiles.contains(tileId);
      });
      if (available == digitalCandidates.end()) {
        realization =
            makeInfeasible(problem, (Twine("spatial child ") + Twine(childId) +
                                     " exceeds the available digital lanes")
                                        .str());
        return std::optional<ResourcePool>{};
      }
      usedDigitalTiles.insert(*available);
      childPool.digitalTileIds.push_back(*available);
      digitalCandidates.erase(available);
    }
    return std::optional<ResourcePool>{std::move(childPool)};
  }

  FailureOr<bool> assignLeaf(const StructuralRATreeNode &node,
                             const ResourcePool &pool,
                             SmallVectorImpl<int64_t> &phaseWork) {
    const ComputeOperation &operation =
        problem.graph.operations[node.operationId];
    LogicalLaneKind required =
        operation.requiredLane.value_or(LogicalLaneKind::Digital);

    MappingLeafAssignment assignment;
    assignment.leafId = node.id;
    assignment.operationId = node.operationId;
    assignment.laneKind = required;

    if (required == LogicalLaneKind::Analog) {
      MappingAnalogLaneRef lane;
      if (operation.laneBindingGroup) {
        lane = bindingLanes.lookup(*operation.laneBindingGroup);
        if (!containsAnalogLane(pool.analogLanes, lane)) {
          realization = makeInfeasible(
              problem, (Twine("leaf ") + Twine(node.id) +
                        " cannot access its persistent analog lane binding")
                           .str());
          return false;
        }
      } else {
        if (pool.analogLanes.empty()) {
          realization =
              makeInfeasible(problem, (Twine("leaf ") + Twine(node.id) +
                                       " has no available analog lane")
                                          .str());
          return false;
        }
        lane = pool.analogLanes.front();
      }
      assignment.tileId = lane.tileId;
      assignment.laneIndex = lane.laneIndex;
    } else {
      if (pool.digitalTileIds.empty()) {
        realization = makeInfeasible(problem, (Twine("leaf ") + Twine(node.id) +
                                               " has no available digital lane")
                                                  .str());
        return false;
      }
      auto colocation = digitalColocationGroupByEndpoint.find(
          {operation.id, node.workUnitId});
      if (colocation != digitalColocationGroupByEndpoint.end()) {
        FailureOr<std::optional<int64_t>> selected =
            selectDigitalColocationTile(colocation->second, pool, phaseWork);
        if (failed(selected))
          return failure();
        if (!*selected)
          return false;
        assignment.tileId = **selected;
      } else if (std::optional<int64_t> pinnedTile =
                     getRequiredDigitalTile(operation, node.workUnitId)) {
        if (!llvm::is_contained(pool.digitalTileIds, *pinnedTile)) {
          realization = makeInfeasible(
            problem, (Twine("digital operation ") + Twine(operation.id) +
                      " cannot access its required logical core")
                           .str());
          return false;
        }
        assignment.tileId = *pinnedTile;
      } else {
        auto firstCandidate = llvm::find_if(
            pool.digitalTileIds,
            [&](int64_t tileId) { return canUseUnpinnedDigitalTile(tileId); });
        if (firstCandidate == pool.digitalTileIds.end()) {
          realization = makeInfeasible(
              problem, (Twine("digital leaf ") + Twine(node.id) +
                        " has no consumer-unreserved core")
                           .str());
          return false;
        }
        assignment.tileId = *firstCandidate;
        auto selectedScore = digitalTileScore(
            assignment.tileId,
            tileAffinity(node.operationId, assignment.tileId, node.workUnitId),
            phaseWork);
        for (int64_t tileId : pool.digitalTileIds) {
          if (!canUseUnpinnedDigitalTile(tileId))
            continue;
          auto score = digitalTileScore(
              tileId, tileAffinity(node.operationId, tileId, node.workUnitId),
              phaseWork);
          if (score < selectedScore) {
            assignment.tileId = tileId;
            selectedScore = score;
          }
        }
      }
      assignment.laneIndex = 0;

      FailureOr<int64_t> leafWork = estimateDigitalLeafWork(node);
      if (failed(leafWork))
        return failure();
      FailureOr<int64_t> phaseTotal =
          addWork(phaseWork[assignment.tileId], *leafWork,
                  "per-phase digital core work");
      FailureOr<int64_t> cumulativeTotal =
          addWork(realization.digitalWorkPerTile[assignment.tileId], *leafWork,
                  "cumulative digital core work");
      if (failed(phaseTotal) || failed(cumulativeTotal))
        return failure();
      phaseWork[assignment.tileId] = *phaseTotal;
      realization.digitalWorkPerTile[assignment.tileId] = *cumulativeTotal;
    }

    realization.leafAssignments.push_back(assignment);
    SmallVector<int64_t> &tiles = operationTiles[node.operationId];
    if (!llvm::is_contained(tiles, assignment.tileId))
      tiles.push_back(assignment.tileId);
    SmallVector<int64_t> &endpointAssignments =
        endpointTiles[{node.operationId, node.workUnitId}];
    if (!llvm::is_contained(endpointAssignments, assignment.tileId))
      endpointAssignments.push_back(assignment.tileId);
    return true;
  }

  FailureOr<bool> assignNode(int64_t nodeId, const ResourcePool &pool,
                             SmallVectorImpl<int64_t> &phaseWork) {
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    if (!node) {
      problem.anchor->emitError(
          "mapping realization cannot resolve RA-tree node ")
          << nodeId;
      return failure();
    }
    recordNodeAllocation(nodeId, pool);

    if (node->kind == RATreeNodeKind::Leaf)
      return assignLeaf(*node, pool, phaseWork);

    if (node->kind == RATreeNodeKind::TemporalCut) {
      for (int64_t childId : node->childIds) {
        SmallVector<int64_t> childPhaseWork(realization.logicalTileCount, 0);
        FailureOr<bool> assigned = assignNode(childId, pool, childPhaseWork);
        if (failed(assigned) || !*assigned)
          return assigned;
      }
      return true;
    }

    SmallVector<std::pair<int64_t, int64_t>> spatialChildren;
    spatialChildren.reserve(node->childIds.size());
    for (int64_t childId : node->childIds) {
      FailureOr<int64_t> work = collectDigitalWork(childId);
      if (failed(work))
        return failure();
      spatialChildren.push_back({childId, *work});
    }
    if (problem.balanceDigitalWork) {
      llvm::stable_sort(spatialChildren,
                        [](const auto &left, const auto &right) {
                          return left.second > right.second;
                        });
    }

    std::set<int64_t> usedDigitalTiles;
    std::set<AnalogLaneKey> usedAnalogLanes;
    for (auto [childId, childWork] : spatialChildren) {
      (void)childWork;
      FailureOr<std::optional<ResourcePool>> childPool = allocateSpatialChild(
          nodeId, childId, pool, usedDigitalTiles, usedAnalogLanes,
          phaseWork);
      if (failed(childPool))
        return failure();
      if (!*childPool)
        return false;
      FailureOr<bool> assigned = assignNode(childId, **childPool, phaseWork);
      if (failed(assigned) || !*assigned)
        return assigned;
    }
    return true;
  }

  const MappingProblem &problem;
  const ResourceAllocationTree &tree;
  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  DenseMap<int64_t, const MappingWorkUnit *> workUnitsById;
  DenseMap<int64_t, SmallVector<const MappingWorkUnitEdge *>>
      workUnitEdgesByOperation;
  DenseMap<int64_t, ResourceDemand> demands;
  DenseMap<int64_t, int64_t> subtreeDigitalWork;
  DenseMap<int64_t, SmallVector<LeafEndpoint>> subtreeEndpoints;
  DenseMap<int64_t, std::string> subtreeShapeSignatures;
  DenseMap<int64_t, MappingAnalogLaneRef> bindingLanes;
  DenseSet<int64_t> consumerAnchoredReservedTiles;
  DenseMap<int64_t, int64_t> waveTiles;
  std::map<std::pair<int64_t, int64_t>, int64_t> waveMemberTiles;
  std::map<LeafEndpoint, int64_t> digitalColocationGroupByEndpoint;
  std::map<LeafEndpoint, int64_t> uniformSiblingDigitalTiles;
  DenseMap<int64_t, SmallVector<LeafEndpoint>> digitalColocationGroupMembers;
  DenseMap<int64_t, int64_t> digitalColocationTiles;
  DenseMap<int64_t, SmallVector<int64_t>> operationTiles;
  std::map<LeafEndpoint, SmallVector<int64_t>> endpointTiles;
  MappingRealization realization;
};

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<MappingRealization>
realizeResourceAllocationTree(const MappingProblem &problem,
                              const ResourceAllocationTree &tree) {
  if (!problem.anchor)
    return failure();
  ResourceRealizer realizer(problem, tree);
  return realizer.run();
}

LogicalResult verifyMappingRealization(const MappingRealization &realization,
                                       const MappingProblem &problem,
                                       const ResourceAllocationTree &tree) {
  if (!realization.feasible)
    return success();
  if (realization.logicalTileCount <= 0 ||
      realization.analogLanesPerTile <= 0) {
    problem.anchor->emitError(
        "mapping realization requires positive logical tile and analog lane "
        "counts");
    return failure();
  }
  if (realization.digitalWorkPerTile.size() !=
          static_cast<size_t>(realization.logicalTileCount) ||
      llvm::any_of(realization.digitalWorkPerTile,
                   [](int64_t work) { return work < 0; })) {
    problem.anchor->emitError(
        "mapping realization requires one nonnegative digital-work value per "
        "logical tile");
    return failure();
  }

  DenseMap<int64_t, const MappingNodeResourceAllocation *> allocations;
  for (const MappingNodeResourceAllocation &allocation :
       realization.nodeAllocations) {
    if (!allocations.try_emplace(allocation.nodeId, &allocation).second) {
      problem.anchor->emitError(
          "mapping realization contains duplicate allocation for RA-tree "
          "node ")
          << allocation.nodeId;
      return failure();
    }
    std::set<int64_t> digitalTiles;
    for (int64_t tileId : allocation.digitalTileIds) {
      if (tileId < 0 || tileId >= realization.logicalTileCount ||
          !digitalTiles.insert(tileId).second) {
        problem.anchor->emitError(
            "mapping node allocation has an invalid or duplicate digital "
            "tile");
        return failure();
      }
    }
    std::set<AnalogLaneKey> analogLanes;
    for (MappingAnalogLaneRef lane : allocation.analogLanes) {
      if (lane.tileId < 0 || lane.tileId >= realization.logicalTileCount ||
          lane.laneIndex < 0 ||
          lane.laneIndex >= realization.analogLanesPerTile ||
          !analogLanes.insert(getKey(lane)).second) {
        problem.anchor->emitError(
            "mapping node allocation has an invalid or duplicate analog "
            "lane");
        return failure();
      }
    }
  }
  if (allocations.size() != tree.nodes.size()) {
    problem.anchor->emitError(
        "mapping realization must allocate every RA-tree node exactly once");
    return failure();
  }

  DenseMap<int64_t, const MappingLeafAssignment *> assignments;
  DenseMap<int64_t, SmallVector<const MappingLeafAssignment *>>
      assignmentsByOperation;
  DenseMap<int64_t, MappingAnalogLaneRef> bindingAssignments;
  int64_t expectedLeaves = 0;
  for (const StructuralRATreeNode &node : tree.nodes)
    expectedLeaves += node.kind == RATreeNodeKind::Leaf;

  for (const MappingLeafAssignment &assignment : realization.leafAssignments) {
    if (!assignments.try_emplace(assignment.leafId, &assignment).second) {
      problem.anchor->emitError(
          "mapping realization contains duplicate assignment for RA-tree "
          "leaf ")
          << assignment.leafId;
      return failure();
    }
    if (assignment.tileId < 0 ||
        assignment.tileId >= realization.logicalTileCount) {
      problem.anchor->emitError("mapping leaf assignment has invalid tile ID");
      return failure();
    }
    if (!std::isfinite(assignment.startNs) ||
        !std::isfinite(assignment.finishNs) || assignment.startNs < 0.0 ||
        assignment.finishNs < assignment.startNs) {
      problem.anchor->emitError(
          "mapping leaf assignment has an invalid execution interval");
      return failure();
    }
    if (assignment.operationId < 0 ||
        assignment.operationId >=
            static_cast<int64_t>(problem.graph.operations.size())) {
      problem.anchor->emitError(
          "mapping leaf assignment has invalid operation ID");
      return failure();
    }
    const ComputeOperation &operation =
        problem.graph.operations[assignment.operationId];
    assignmentsByOperation[assignment.operationId].push_back(&assignment);
    LogicalLaneKind expected =
        operation.requiredLane.value_or(LogicalLaneKind::Digital);
    if (assignment.laneKind != expected) {
      operation.operation->emitError(
          "mapping realization assigns the operation to the wrong logical "
          "lane kind");
      return failure();
    }
    if (assignment.laneKind == LogicalLaneKind::Digital) {
      if (assignment.laneIndex != 0) {
        operation.operation->emitError(
            "digital mapping assignment requires lane index zero");
        return failure();
      }
    } else if (assignment.laneIndex < 0 ||
               assignment.laneIndex >= realization.analogLanesPerTile) {
      operation.operation->emitError(
          "analog mapping assignment has invalid lane index");
      return failure();
    }
    if (operation.laneBindingGroup) {
      MappingAnalogLaneRef lane{assignment.tileId, assignment.laneIndex};
      auto [binding, inserted] =
          bindingAssignments.try_emplace(*operation.laneBindingGroup, lane);
      if (!inserted && !(binding->second == lane)) {
        operation.operation->emitError(
            "analog lane-binding group is assigned to multiple logical "
            "lanes");
        return failure();
      }
    }

    const MappingNodeResourceAllocation *allocation =
        allocations.lookup(assignment.leafId);
    if (!allocation) {
      problem.anchor->emitError(
          "mapping leaf assignment has no node resource allocation");
      return failure();
    }
    if (assignment.laneKind == LogicalLaneKind::Digital) {
      if (!llvm::is_contained(allocation->digitalTileIds, assignment.tileId)) {
        problem.anchor->emitError(
            "digital leaf assignment is outside its node resource pool");
        return failure();
      }
    } else if (!containsAnalogLane(allocation->analogLanes,
                                   {assignment.tileId, assignment.laneIndex})) {
      problem.anchor->emitError(
          "analog leaf assignment is outside its node resource pool");
      return failure();
    }
  }
  if (assignments.size() != static_cast<size_t>(expectedLeaves)) {
    problem.anchor->emitError(
        "mapping realization must assign every RA-tree leaf exactly once");
    return failure();
  }

  auto getOperationTile = [&](int64_t operationId)
      -> FailureOr<std::optional<int64_t>> {
    auto found = assignmentsByOperation.find(operationId);
    if (found == assignmentsByOperation.end() || found->second.empty()) {
      problem.graph.operations[operationId].operation->emitError(
          "MVM-body policy cannot resolve the operation's logical tile");
      return failure();
    }
    int64_t tileId = found->second.front()->tileId;
    if (llvm::any_of(found->second, [tileId](const auto *assignment) {
          return assignment->tileId != tileId;
        })) {
      problem.graph.operations[operationId].operation->emitError(
          "one MVM-body operation is assigned to multiple logical tiles");
      return failure();
    }
    return std::optional<int64_t>{tileId};
  };

  for (const MVMWave &wave : problem.graph.mvmWaves) {
    DenseMap<int64_t, int64_t> memberTiles;
    DenseMap<int64_t, int64_t> membersPerTile;
    DenseSet<int64_t> distinctTiles;
    std::optional<std::pair<int64_t, int64_t>> home;
    for (int64_t operationId : wave.physicalMVMOperationIds) {
      const ComputeOperation &operation = problem.graph.operations[operationId];
      if (!operation.mvmWaveMember) {
        operation.operation->emitError(
            "MVM-body policy requires a physical-MVM wave-member identity");
        return failure();
      }
      FailureOr<std::optional<int64_t>> tile = getOperationTile(operationId);
      if (failed(tile))
        return failure();
      memberTiles[*operation.mvmWaveMember] = **tile;
      distinctTiles.insert(**tile);
      int64_t count = ++membersPerTile[**tile];
      if (count > realization.analogLanesPerTile) {
        operation.operation->emitError(
            "MVM-body policy overfills a logical tile's analog lanes");
        return failure();
      }
      std::pair<int64_t, int64_t> candidate{*operation.mvmWaveMember, **tile};
      if (!home || candidate < *home)
        home = candidate;
    }
    if (!home) {
      problem.anchor->emitError("MVM wave has no realized physical members");
      return failure();
    }

    int64_t expectedTileCount =
        problem.mvmBodyPolicy == MVMBodyPolicy::Spread
            ? static_cast<int64_t>(wave.physicalMVMOperationIds.size())
            : llvm::divideCeil(
                  static_cast<int64_t>(wave.physicalMVMOperationIds.size()),
                  realization.analogLanesPerTile);
    if (static_cast<int64_t>(distinctTiles.size()) != expectedTileCount) {
      problem.anchor->emitError(
          problem.mvmBodyPolicy == MVMBodyPolicy::Spread
              ? "spread MVM-body policy must use one logical tile per array"
              : "packed MVM-body policy did not use the minimum logical-tile "
                "count");
      return failure();
    }

    for (int64_t operationId : wave.vectorTileOperationIds) {
      const ComputeOperation &operation = problem.graph.operations[operationId];
      FailureOr<std::optional<int64_t>> tile = getOperationTile(operationId);
      if (failed(tile))
        return failure();
      int64_t expectedTile = home->second;
      if (operation.mvmWaveMember) {
        auto member = memberTiles.find(*operation.mvmWaveMember);
        if (member == memberTiles.end()) {
          operation.operation->emitError(
              "vector-tile wave member does not identify a physical MVM");
          return failure();
        }
        expectedTile = member->second;
      }
      if (**tile != expectedTile) {
        operation.operation->emitError(
            "vector-tile work is separated from its MVM wave destination");
        return failure();
      }
    }

    for (std::optional<int64_t> operationId :
         {wave.recombineOperationId, wave.biasAddOperationId}) {
      if (!operationId)
        continue;
      FailureOr<std::optional<int64_t>> tile = getOperationTile(*operationId);
      if (failed(tile))
        return failure();
      if (**tile != home->second) {
        problem.graph.operations[*operationId].operation->emitError(
            "MVM wave result processing is separated from its home tile");
        return failure();
      }
    }
  }

  for (const StructuralRATreeNode &node : tree.nodes) {
    if (node.kind == RATreeNodeKind::Leaf)
      continue;
    const MappingNodeResourceAllocation *parent = allocations.lookup(node.id);
    if (!parent) {
      problem.anchor->emitError("mapping cut has no node resource allocation");
      return failure();
    }
    std::set<int64_t> parentDigital(parent->digitalTileIds.begin(),
                                    parent->digitalTileIds.end());
    std::set<AnalogLaneKey> parentAnalog;
    for (MappingAnalogLaneRef lane : parent->analogLanes)
      parentAnalog.insert(getKey(lane));

    std::set<int64_t> usedDigital;
    std::set<AnalogLaneKey> usedAnalog;
    for (int64_t childId : node.childIds) {
      const MappingNodeResourceAllocation *child = allocations.lookup(childId);
      if (!child) {
        problem.anchor->emitError(
            "mapping cut child has no node resource allocation");
        return failure();
      }
      std::set<int64_t> childDigital(child->digitalTileIds.begin(),
                                     child->digitalTileIds.end());
      std::set<AnalogLaneKey> childAnalog;
      for (MappingAnalogLaneRef lane : child->analogLanes)
        childAnalog.insert(getKey(lane));

      if (node.kind == RATreeNodeKind::TemporalCut) {
        if (childDigital != parentDigital || childAnalog != parentAnalog) {
          problem.anchor->emitError(
              "T-cut child must inherit its parent's complete logical "
              "resource pool");
          return failure();
        }
        continue;
      }

      for (int64_t tileId : childDigital) {
        if (!parentDigital.contains(tileId) ||
            !usedDigital.insert(tileId).second) {
          problem.anchor->emitError(
              "S-cut children must receive disjoint digital lanes from "
              "their parent");
          return failure();
        }
      }
      for (AnalogLaneKey lane : childAnalog) {
        if (!parentAnalog.contains(lane) || !usedAnalog.insert(lane).second) {
          problem.anchor->emitError(
              "S-cut children must receive disjoint analog lanes from their "
              "parent");
          return failure();
        }
      }
    }
  }
  return success();
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
