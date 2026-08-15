#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingRealization.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostModel.h"

#include "llvm/ADT/BitVector.h"
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

struct ScheduledEndpoint {
  int64_t workUnitId = -1;
  int64_t tileId = -1;
  double finishNs = 0.0;
};

struct IncomingArrival {
  double readyNs = 0.0;
  double communicationNs = 0.0;
  int64_t crossingBytes = 0;
  int64_t messages = 0;
};

using DigitalScheduleScore =
    std::tuple<double, double, int64_t, int64_t, int64_t, int64_t, int64_t>;

struct DigitalScheduleChoice {
  int64_t tileId = -1;
  IncomingArrival arrival;
  double finishNs = 0.0;
  DigitalScheduleScore score;
};

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
      workUnitEdgesByTargetOperation[edge.targetOperationId].push_back(&edge);
      if (edge.tensorId < 0)
        wildcardRefinedEdges.insert(
            {edge.sourceOperationId, edge.targetOperationId});
      else
        refinedTensorEdges.insert(
            {edge.sourceOperationId, edge.targetOperationId, edge.tensorId});
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
    digitalAvailableNs.assign(*tileCount, 0.0);
    analogAvailableNs.assign(
        *tileCount * problem.hardware.arraysPerCore, 0.0);

    ResourcePool rootPool;
    for (int64_t tileId = 0; tileId < *tileCount; ++tileId) {
      rootPool.digitalTileIds.push_back(tileId);
      for (int64_t laneIndex = 0; laneIndex < problem.hardware.arraysPerCore;
           ++laneIndex)
        rootPool.analogLanes.push_back({tileId, laneIndex});
    }

    FailureOr<bool> initializedLayerPools = initializeLayerTilePools(rootPool);
    if (failed(initializedLayerPools))
      return failure();
    if (!*initializedLayerPools)
      return realization;

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
    if (usesSlidingWindow() && failed(initializeSlidingWindow()))
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
    double rootFinishNs = 0.0;
    FailureOr<bool> assigned =
        assignNode(tree.rootId, rootPool, rootPhaseWork,
                   /*notBeforeNs=*/0.0, rootFinishNs);
    if (failed(assigned))
      return failure();
    if (!*assigned)
      return realization;
    if (usesScheduleAwareTiming())
      realization.estimatedMakespanNs = rootFinishNs;

    // Sliding-window tile IDs encode the permanent logical retirement
    // sequence and must survive into physical placement. Consumer-first ID
    // canonicalization is useful for the other policies, but would erase that
    // scheduling contract after realization.
    if (problem.setupBindingPolicy == SetupBindingPolicy::ConsumerAnchored &&
        !usesSlidingWindow() &&
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
  std::map<int64_t, int64_t> collectLayerDigitalDemand(int64_t nodeId) const {
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    assert(node && "RA-tree node must be indexed before layer pool planning");
    if (node->kind == RATreeNodeKind::Leaf) {
      const ComputeOperation &operation =
          problem.graph.operations[node->operationId];
      if (operation.requiredLane.value_or(LogicalLaneKind::Digital) ==
          LogicalLaneKind::Digital)
        return {{operation.layerRegionId, 1}};
      return {};
    }
    if (node->kind == RATreeNodeKind::Layer)
      return collectLayerDigitalDemand(node->childIds.front());

    std::map<int64_t, int64_t> result;
    for (int64_t childId : node->childIds) {
      std::map<int64_t, int64_t> child = collectLayerDigitalDemand(childId);
      for (auto [regionId, demand] : child) {
        if (node->kind == RATreeNodeKind::TemporalCut)
          result[regionId] = std::max(result[regionId], demand);
        else
          result[regionId] += demand;
      }
    }
    return result;
  }

  FailureOr<bool> initializeLayerTilePools(const ResourcePool &rootPool) {
    const int64_t tileCount = realization.logicalTileCount;
    std::map<int64_t, int64_t> digitalDemand =
        collectLayerDigitalDemand(tree.rootId);
    DenseMap<int64_t, SmallVector<int64_t>> concurrentLayerRegions;
    auto addConcurrentRegions = [&](int64_t left, int64_t right) {
      if (left == right)
        return;
      concurrentLayerRegions[left].push_back(right);
      concurrentLayerRegions[right].push_back(left);
    };
    // Layer-cut spatial siblings promise real concurrency. Their digital tile
    // pools must therefore be disjoint; the normal one-tile handoff overlap
    // is legal only between temporal layers. Matrix-setup-only branches are
    // analog work and remain shareable across distinct array lanes.
    for (const StructuralRATreeNode &node : tree.nodes) {
      if (node.kind != RATreeNodeKind::SpatialCut)
        continue;
      SmallVector<SmallVector<int64_t>> branchRegions;
      branchRegions.reserve(node.childIds.size());
      for (int64_t childId : node.childIds) {
        SmallVector<int64_t> childRegions;
        for (LeafEndpoint endpoint : collectSubtreeEndpoints(childId)) {
          const ComputeOperation &operation =
              problem.graph.operations[endpoint.first];
          if (operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
              LogicalLaneKind::Digital)
            continue;
          childRegions.push_back(operation.layerRegionId);
        }
        sortAndUnique(childRegions);
        branchRegions.push_back(std::move(childRegions));
      }
      for (size_t leftIndex = 0; leftIndex < branchRegions.size();
           ++leftIndex) {
        for (size_t rightIndex = leftIndex + 1;
             rightIndex < branchRegions.size(); ++rightIndex) {
          for (int64_t left : branchRegions[leftIndex])
            for (int64_t right : branchRegions[rightIndex])
              addConcurrentRegions(left, right);
        }
      }
    }
    for (auto &[regionId, conflicts] : concurrentLayerRegions) {
      (void)regionId;
      sortAndUnique(conflicts);
    }
    DenseMap<int64_t, int64_t> spreadWaveDemand;
    if (problem.mvmBodyPolicy != MVMBodyPolicy::Packed) {
      for (const MVMWave &wave : problem.graph.mvmWaves) {
        DenseMap<int64_t, int64_t> membersByRegion;
        for (int64_t operationId : wave.physicalMVMOperationIds)
          ++membersByRegion[
              problem.graph.operations[operationId].layerRegionId];
        for (auto [regionId, members] : membersByRegion)
          spreadWaveDemand[regionId] =
              std::max(spreadWaveDemand.lookup(regionId), members);
      }
    }

    int64_t nextPoolStart = 0;
    for (const LayerRegion &region : problem.graph.layerRegions) {
      MappingLayerTilePool pool;
      pool.layerRegionId = region.id;
      if (region.isSingletonFallback) {
        pool.tileIds = rootPool.digitalTileIds;
      } else {
        int64_t analogTiles = llvm::divideCeil(
            region.analogLaneDemand, problem.hardware.arraysPerCore);
        int64_t memoryTiles = 0;
        if (region.estimatedStaticMemoryBytes > 0) {
          memoryTiles = llvm::divideCeil(
              region.estimatedStaticMemoryBytes,
              problem.hardware.localMemoryBytesPerCore);
        }
        int64_t targetSize = std::max(
            {int64_t{1}, analogTiles, memoryTiles,
             digitalDemand[region.id], spreadWaveDemand.lookup(region.id)});
        if (targetSize > tileCount) {
          realization = makeInfeasible(
              problem,
              (Twine("semantic layer region ") + Twine(region.id) +
               " requires a tile pool of " + Twine(targetSize) +
               " cores but the mesh provides " + Twine(tileCount))
                  .str());
          return false;
        }
        std::optional<int64_t> selectedStart;
        for (int64_t startOffset = 0; startOffset < tileCount;
             ++startOffset) {
          int64_t candidateStart =
              (nextPoolStart + startOffset) % tileCount;
          SmallVector<int64_t> candidateTiles;
          candidateTiles.reserve(targetSize);
          for (int64_t offset = 0; offset < targetSize; ++offset)
            candidateTiles.push_back((candidateStart + offset) % tileCount);
          bool overlapsConcurrentRegion = llvm::any_of(
              concurrentLayerRegions[region.id], [&](int64_t otherRegionId) {
                auto assigned =
                    allowedTilesByLayerRegion.find(otherRegionId);
                if (assigned == allowedTilesByLayerRegion.end())
                  return false;
                return llvm::any_of(candidateTiles, [&](int64_t tileId) {
                  return llvm::is_contained(assigned->second, tileId);
                });
              });
          if (!overlapsConcurrentRegion) {
            selectedStart = candidateStart;
            pool.tileIds = std::move(candidateTiles);
            break;
          }
        }
        if (!selectedStart) {
          realization = makeInfeasible(
              problem,
              (Twine("semantic layer region ") + Twine(region.id) +
               " cannot obtain a tile pool disjoint from its concurrent "
               "layer-cut siblings")
                  .str());
          return false;
        }
        // Adjacent semantic layers share one handoff tile when possible while
        // still advancing through the logical mesh for persistent matrices.
        // The search above moves past that overlap for concurrent siblings.
        nextPoolStart =
            (*selectedStart + std::max<int64_t>(1, targetSize - 1)) %
            tileCount;
      }
      sortAndUnique(pool.tileIds);
      allowedTilesByLayerRegion[region.id] = pool.tileIds;
      realization.layerTilePools.push_back(std::move(pool));
    }

    // A tied-weight model may invoke one immutable matrix from several
    // semantic layers (RetinaNet's shared prediction heads are a canonical
    // example). Keep those layers distinct, but expose the setup region's
    // compact base pool to every layer using the matrix. This gives the shared
    // binding group a legal common anchor without duplicating matrix state.
    DenseMap<int64_t, SmallVector<int64_t>> baseTilesByLayerRegion =
        allowedTilesByLayerRegion;
    auto extendRegionPool = [&](int64_t regionId,
                                ArrayRef<int64_t> tileIds) {
      SmallVector<int64_t> &allowed = allowedTilesByLayerRegion[regionId];
      llvm::append_range(allowed, tileIds);
      sortAndUnique(allowed);
      MappingLayerTilePool &serializedPool =
          realization.layerTilePools[regionId];
      serializedPool.tileIds = allowed;
    };

    for (const LaneBindingGroup &group : problem.graph.laneBindingGroups) {
      if (group.setupOperationId < 0 ||
          group.setupOperationId >=
              static_cast<int64_t>(problem.graph.operations.size())) {
        problem.anchor->emitError(
            "layer tile-pool planning found an invalid binding-group setup");
        return failure();
      }
      int64_t setupRegionId =
          problem.graph.operations[group.setupOperationId].layerRegionId;
      SmallVector<int64_t> memberRegionIds{setupRegionId};
      for (int64_t operationId : group.operationIds) {
        int64_t regionId =
            problem.graph.operations[operationId].layerRegionId;
        if (!llvm::is_contained(memberRegionIds, regionId))
          memberRegionIds.push_back(regionId);
      }
      if (memberRegionIds.size() > 1) {
        auto anchors = baseTilesByLayerRegion.find(setupRegionId);
        assert(anchors != baseTilesByLayerRegion.end() &&
               "every semantic layer region must have a base tile pool");
        ArrayRef<int64_t> sharedAnchors = anchors->second;
        for (int64_t regionId : memberRegionIds)
          extendRegionPool(regionId, sharedAnchors);
      }

      SmallVector<int64_t> commonTiles =
          allowedTilesByLayerRegion.lookup(setupRegionId);
      llvm::erase_if(commonTiles, [&](int64_t tileId) {
        return llvm::any_of(memberRegionIds, [&](int64_t regionId) {
          return !isTileAllowedForRegion(regionId, tileId);
        });
      });
      if (commonTiles.empty()) {
        problem.anchor->emitError(
            "shared analog binding group has no common semantic-layer tile");
        return failure();
      }
      allowedTilesByBindingGroup[group.id] = std::move(commonTiles);
    }
    return true;
  }

  bool isTileAllowedForRegion(int64_t regionId, int64_t tileId) const {
    auto found = allowedTilesByLayerRegion.find(regionId);
    return found != allowedTilesByLayerRegion.end() &&
           llvm::is_contained(found->second, tileId);
  }

  bool isTileAllowedForOperation(const ComputeOperation &operation,
                                 int64_t tileId) const {
    return isTileAllowedForRegion(operation.layerRegionId, tileId);
  }

  bool isTileAllowedForBindingGroup(int64_t groupId, int64_t tileId) const {
    auto found = allowedTilesByBindingGroup.find(groupId);
    return found != allowedTilesByBindingGroup.end() &&
           llvm::is_contained(found->second, tileId);
  }

  bool isTileAllowedForSubtree(int64_t nodeId, int64_t tileId) {
    for (LeafEndpoint endpoint : collectSubtreeEndpoints(nodeId)) {
      const ComputeOperation &operation =
          problem.graph.operations[endpoint.first];
      if (operation.requiredLane.value_or(LogicalLaneKind::Digital) ==
              LogicalLaneKind::Digital &&
          isTileAllowedForOperation(operation, tileId))
        return true;
    }
    return false;
  }

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

  FailureOr<int64_t>
  estimateBindingTraffic(const LaneBindingGroup &group) const {
    llvm::SmallDenseSet<int64_t, 16> tensors;
    for (int64_t operationId : group.operationIds) {
      if (operationId < 0 ||
          operationId >= static_cast<int64_t>(problem.graph.operations.size()))
        return problem.anchor->emitError(
            "analog lane-binding group references an unknown operation");
      const ComputeOperation &operation = problem.graph.operations[operationId];
      if (operation.kind != ComputeOperationKind::PhysicalMVM)
        continue;
      tensors.insert(operation.inputTensors.begin(),
                     operation.inputTensors.end());
      tensors.insert(operation.outputTensors.begin(),
                     operation.outputTensors.end());
    }

    int64_t bytes = 0;
    for (int64_t tensorId : tensors) {
      if (tensorId < 0 ||
          tensorId >= static_cast<int64_t>(problem.graph.tensors.size()))
        return problem.anchor->emitError(
            "analog lane-binding group references an unknown tensor");
      const ComputeTensor &tensor = problem.graph.tensors[tensorId];
      // Logical arrays are persistent analog state, not tile-local activation
      // storage. Dynamic tensors remain neutral here and are rejected later by
      // the exact static-memory planner if they cannot be represented.
      if (tensor.isLogicalArray || tensor.byteSize <= 0)
        continue;
      FailureOr<int64_t> total =
          addWork(bytes, tensor.byteSize,
                  "analog lane-binding activation traffic estimate");
      if (failed(total))
        return failure();
      bytes = *total;
    }
    return bytes;
  }

  auto digitalTileScore(int64_t tileId, int64_t affinity,
                        ArrayRef<int64_t> phaseWork) const {
    bool balance = problem.digitalSchedulingPolicy ==
                   DigitalSchedulingPolicy::Balanced;
    int64_t phaseLoad = balance ? phaseWork[tileId] : 0;
    int64_t cumulativeLoad =
        balance ? realization.digitalWorkPerTile[tileId] : 0;
    return std::tuple(phaseLoad, cumulativeLoad, -affinity, tileId);
  }

  bool usesScheduleAwareTiming() const {
    return problem.digitalSchedulingPolicy ==
               DigitalSchedulingPolicy::EarliestFinish ||
           problem.digitalSchedulingPolicy ==
               DigitalSchedulingPolicy::Progressive ||
           problem.digitalSchedulingPolicy ==
               DigitalSchedulingPolicy::SlidingWindow;
  }

  bool usesProgressiveAdmission() const {
    return problem.digitalSchedulingPolicy ==
           DigitalSchedulingPolicy::Progressive;
  }

  bool usesSlidingWindow() const {
    return problem.digitalSchedulingPolicy ==
           DigitalSchedulingPolicy::SlidingWindow;
  }

  LogicalResult initializeSlidingWindow() {
    const int64_t tileCount = realization.logicalTileCount;
    if (problem.digitalWindowSize <= 0 ||
        problem.digitalWindowSize > tileCount) {
      return problem.anchor->emitError(
          "sliding-window mapping has an invalid digital window size");
    }

    slidingTileRanks.assign(tileCount, -1);
    // Logical IDs encode wavefront position. Physical placement remains an
    // independent later decision and must see these IDs without a hidden
    // permutation here.
    for (int64_t tileId = 0; tileId < tileCount; ++tileId)
      slidingTileRanks[tileId] = tileId;

    totalFlexibleDigitalWork = 0;
    for (const StructuralRATreeNode &node : tree.nodes) {
      if (node.kind != RATreeNodeKind::Leaf)
        continue;
      const ComputeOperation &operation =
          problem.graph.operations[node.operationId];
      if (operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
              LogicalLaneKind::Digital ||
          getPinnedDigitalTile(operation))
        continue;
      FailureOr<int64_t> work = estimateDigitalLeafWork(node);
      if (failed(work))
        return failure();
      FailureOr<int64_t> total = addWork(
          totalFlexibleDigitalWork, *work,
          "sliding-window total flexible digital work");
      if (failed(total))
        return failure();
      totalFlexibleDigitalWork = *total;
    }
    return success();
  }

  int64_t slidingWindowHead() const {
    if (totalFlexibleDigitalWork <= 0)
      return 0;
    const int64_t maximumHead =
        realization.logicalTileCount - problem.digitalWindowSize;
    const long double progress =
        static_cast<long double>(completedFlexibleDigitalWork) /
        static_cast<long double>(totalFlexibleDigitalWork);
    return std::clamp<int64_t>(
        static_cast<int64_t>(std::floor(progress * maximumHead)), 0,
        maximumHead);
  }

  bool isInSlidingWindow(int64_t tileId) const {
    if (!usesSlidingWindow())
      return true;
    assert(tileId >= 0 &&
           tileId < static_cast<int64_t>(slidingTileRanks.size()));
    const int64_t rank = slidingTileRanks[tileId];
    const int64_t head = slidingWindowHead();
    return rank >= head && rank < head + problem.digitalWindowSize;
  }

  double cyclesToNanoseconds(int64_t cycles) const {
    return static_cast<double>(cycles) * 1.0e9 /
           static_cast<double>(problem.hardware.clockFrequencyHz);
  }

  FailureOr<int64_t> sumTensorBytes(ArrayRef<int64_t> tensorIds,
                                    StringRef description) const {
    int64_t total = 0;
    for (int64_t tensorId : tensorIds) {
      if (tensorId < 0 ||
          tensorId >= static_cast<int64_t>(problem.graph.tensors.size())) {
        problem.anchor->emitError(description) << " references tensor "
                                               << tensorId;
        return failure();
      }
      int64_t bytes = problem.graph.tensors[tensorId].byteSize;
      if (bytes < 0) {
        problem.anchor->emitError(description)
            << " requires static tensor byte sizes";
        return failure();
      }
      FailureOr<int64_t> next = addWork(total, bytes, description);
      if (failed(next))
        return failure();
      total = *next;
    }
    return total;
  }

  FailureOr<double> estimateOneHopTransferNs(int64_t bytes) const {
    if (bytes < 0) {
      problem.anchor->emitError(
          "schedule-aware mapping requires static transfer sizes");
      return failure();
    }
    FailureOr<int64_t> bits =
        multiplyWork(bytes, int64_t{8}, "logical transfer bit count");
    if (failed(bits))
      return failure();
    if (!problem.costProfile.useLegacyFormula) {
      int64_t words =
          llvm::divideCeil(*bits, problem.costProfile.network.wordBits);
      double result = problem.costProfile.runtime.routeSetupNs +
                      problem.costProfile.network.injectFixedNs +
                      problem.costProfile.network.ejectFixedNs +
                      problem.costProfile.network.hopPipelineNs * words +
                      problem.costProfile.network.dmaNsPerByte * bytes;
      if (!std::isfinite(result) || result < 0.0) {
        problem.anchor->emitError("logical transfer cost is invalid");
        return failure();
      }
      return result;
    }
    int64_t words = llvm::divideCeil(*bits, problem.hardware.networkWordBits);
    FailureOr<int64_t> cycles = multiplyWork(
        words, problem.hardware.networkHopCycles,
        "logical one-hop transfer cycle count");
    if (failed(cycles))
      return failure();
    return cyclesToNanoseconds(*cycles);
  }

  FailureOr<double>
  estimateDigitalLeafDurationNs(const StructuralRATreeNode &node) const {
    FailureOr<int64_t> work = estimateDigitalLeafWork(node);
    if (failed(work))
      return failure();
    const ComputeOperation &operation =
        problem.graph.operations[node.operationId];
    if (!problem.costProfile.useLegacyFormula) {
      FailureOr<int64_t> inputBytes =
          sumTensorBytes(operation.inputTensors,
                         "digital input byte estimate");
      FailureOr<int64_t> outputBytes =
          sumTensorBytes(operation.outputTensors,
                         "digital output byte estimate");
      if (failed(inputBytes) || failed(outputBytes))
        return failure();
      TaskCostFeatures features;
      features.operationId = operation.id;
      features.workUnitId = node.workUnitId;
      features.semanticTaskKind = operation.semanticTaskKind;
      features.workItems = *work;
      features.inputBytes = *inputBytes;
      features.outputBytes = *outputBytes;
      FailureOr<TaskCostEstimate> estimate = estimateDigitalTaskCost(
          problem.costProfile, features, operation.operation);
      if (failed(estimate))
        return failure();
      return estimate->totalNs;
    }
    int64_t vectorWidth = problem.hardware.digitalVectorBitsPerCycle / 32;
    int64_t effectiveOpsPerCycle =
        std::max(problem.hardware.digitalIssueWidth, vectorWidth);
    return cyclesToNanoseconds(llvm::divideCeil(*work, effectiveOpsPerCycle));
  }

  FailureOr<double>
  estimateAnalogLeafDurationNs(const StructuralRATreeNode &node) const {
    const ComputeOperation &operation =
        problem.graph.operations[node.operationId];
    if (operation.kind == ComputeOperationKind::PhysicalMVM &&
        operation.analogMVM) {
      FailureOr<int64_t> ioElements = addWork(
          operation.analogMVM->inputColumns,
          operation.analogMVM->outputRows,
          "physical MVM I/O element count");
      if (failed(ioElements))
        return failure();
      if (!problem.costProfile.useLegacyFormula) {
        FailureOr<int64_t> loadBytes = multiplyWork(
            operation.analogMVM->inputColumns, int64_t{4},
            "physical MVM load byte count");
        FailureOr<int64_t> storeBytes = multiplyWork(
            operation.analogMVM->outputRows, int64_t{4},
            "physical MVM store byte count");
        if (failed(loadBytes) || failed(storeBytes))
          return failure();
        FailureOr<TaskCostEstimate> estimate = estimateAnalogTaskCost(
            problem.costProfile, *loadBytes, *storeBytes,
            node.workGroupCount, operation.operation);
        if (failed(estimate))
          return failure();
        return estimate->totalNs;
      }
      FailureOr<int64_t> ioBits = multiplyWork(
          *ioElements, int64_t{32}, "physical MVM I/O bit count");
      if (failed(ioBits))
        return failure();
      int64_t ioCycles =
          llvm::divideCeil(*ioBits, problem.hardware.analogIOBitsPerCycle);
      return (static_cast<double>(problem.hardware.analogMVMLatencyNs) +
              cyclesToNanoseconds(ioCycles)) *
             static_cast<double>(node.workGroupCount);
    }

    // Matrix setup leaves are analog-lane work but use the generic static
    // iteration estimate, matching the reference evaluator's treatment.
    int64_t work = 1;
    for (const ComputeIterationDimension &dimension :
         operation.iterationDomain) {
      if (ShapedType::isDynamic(dimension.staticExtent) ||
          dimension.staticExtent <= 0) {
        operation.operation->emitError(
            "schedule-aware mapping requires static analog work");
        return failure();
      }
      FailureOr<int64_t> product = multiplyWork(
          work, dimension.staticExtent, "analog leaf work estimate");
      if (failed(product))
        return failure();
      work = *product;
    }
    FailureOr<int64_t> grouped = multiplyWork(
        work, node.workGroupCount, "grouped analog leaf work estimate");
    if (failed(grouped))
      return failure();
    int64_t vectorWidth = problem.hardware.digitalVectorBitsPerCycle / 32;
    int64_t effectiveOpsPerCycle =
        std::max(problem.hardware.digitalIssueWidth, vectorWidth);
    return cyclesToNanoseconds(
        llvm::divideCeil(*grouped, effectiveOpsPerCycle));
  }

  FailureOr<IncomingArrival>
  estimateIncomingArrival(const StructuralRATreeNode &node,
                          int64_t candidateTile) const {
    IncomingArrival result;
    const ComputeOperation &target =
        problem.graph.operations[node.operationId];

    auto addSource = [&](const ScheduledEndpoint &source,
                         int64_t bytes) -> LogicalResult {
      double transferNs = 0.0;
      if (source.tileId != candidateTile) {
        FailureOr<double> transfer = estimateOneHopTransferNs(bytes);
        if (failed(transfer))
          return failure();
        transferNs = *transfer;
        result.communicationNs += transferNs;
        FailureOr<int64_t> crossing = addWork(
            result.crossingBytes, bytes,
            "schedule-aware crossing byte count");
        if (failed(crossing))
          return failure();
        result.crossingBytes = *crossing;
        ++result.messages;
      }
      result.readyNs =
          std::max(result.readyNs, source.finishNs + transferNs);
      return success();
    };

    auto addScheduledOperation = [&](int64_t sourceOperationId,
                                     int64_t sourceWorkUnitId,
                                     int64_t bytes) -> LogicalResult {
      auto found = scheduledEndpoints.find(sourceOperationId);
      if (found == scheduledEndpoints.end())
        return success();
      for (const ScheduledEndpoint &source : found->second) {
        if (sourceWorkUnitId >= 0 &&
            source.workUnitId != sourceWorkUnitId)
          continue;
        if (failed(addSource(source, bytes)))
          return failure();
      }
      return success();
    };

    auto refined = workUnitEdgesByTargetOperation.find(target.id);
    if (refined != workUnitEdgesByTargetOperation.end()) {
      for (const MappingWorkUnitEdge *edge : refined->second) {
        if (edge->targetWorkUnitId >= 0 &&
            edge->targetWorkUnitId != node.workUnitId)
          continue;
        if (failed(addScheduledOperation(edge->sourceOperationId,
                                         edge->sourceWorkUnitId,
                                         edge->byteSize)))
          return failure();
      }
    }

    for (int64_t tensorId : target.inputTensors) {
      if (tensorId < 0 ||
          tensorId >= static_cast<int64_t>(problem.graph.tensors.size())) {
        target.operation->emitError(
            "schedule-aware input references an unknown tensor");
        return failure();
      }
      const ComputeTensor &tensor = problem.graph.tensors[tensorId];
      for (int64_t producerId : tensor.producerOperations) {
        if (producerId == target.id ||
            wildcardRefinedEdges.contains({producerId, target.id}) ||
            refinedTensorEdges.contains({producerId, target.id, tensorId}))
          continue;
        int64_t bytes =
            getProducerContributionByteSize(tensor, producerId);
        if (failed(addScheduledOperation(producerId, -1, bytes)))
          return failure();
      }
    }
    return result;
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

  std::optional<int64_t>
  getSchedulingRequiredDigitalTile(const ComputeOperation &operation,
                                   int64_t workUnitId) const {
    // Uniform-sibling placement is a locality heuristic, not a semantic
    // requirement. Schedule-aware policies may move that work when queueing
    // or input arrival makes another legal tile faster. MVM-body work remains
    // pinned because it genuinely accesses a resident analog array.
    if (usesScheduleAwareTiming())
      return getPinnedDigitalTile(operation);
    return getRequiredDigitalTile(operation, workUnitId);
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

  bool hasConsumerUnreservedDigitalTile(ArrayRef<int64_t> tileIds) const {
    return llvm::any_of(tileIds, [&](int64_t tileId) {
      return canUseUnpinnedDigitalTile(tileId);
    });
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
    for (MappingLayerTilePool &pool : realization.layerTilePools) {
      for (int64_t &tileId : pool.tileIds)
        tileId = remapTile(tileId);
      sortAndUnique(pool.tileIds);
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
      if (node->kind == RATreeNodeKind::Layer)
        stream << "R[";
      else
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
    const bool preferUnreserved =
        hasConsumerUnreservedDigitalTile(pool.digitalTileIds);
    int64_t selected = pool.digitalTileIds.front();
    auto firstCandidate = llvm::find_if(pool.digitalTileIds, [&](int64_t id) {
      return !preferUnreserved || canUseUnpinnedDigitalTile(id);
    });
    assert(firstCandidate != pool.digitalTileIds.end() &&
           "a nonempty digital pool must provide a fallback core");
    selected = *firstCandidate;
    auto selectedScore =
        digitalTileScore(selected, affinity(selected), phaseWork);
    for (int64_t tileId : pool.digitalTileIds) {
      if (preferUnreserved && !canUseUnpinnedDigitalTile(tileId))
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

    DenseMap<int64_t, int64_t> componentByGroup;
    DenseMap<int64_t, int64_t> componentByWave;
    for (auto [componentId, component] : llvm::enumerate(components)) {
      int64_t componentIndex = static_cast<int64_t>(componentId);
      for (int64_t groupId : component.groupIds)
        componentByGroup[groupId] = componentIndex;
      for (int64_t waveId : component.waveIds) {
        auto [entry, inserted] =
            componentByWave.try_emplace(waveId, componentIndex);
        if (!inserted && entry->second != componentIndex) {
          problem.anchor->emitError(
              "one MVM wave spans disconnected binding components");
          return failure();
        }
      }
    }

    // Different MVM components may share unused arrays only when the RA tree
    // does not require their pinned analog or digital wave work concurrently.
    // Record components found in distinct branches of every spatial cut.
    std::set<std::pair<int64_t, int64_t>> incompatibleComponents;
    for (const StructuralRATreeNode &node : tree.nodes) {
      if (node.kind != RATreeNodeKind::SpatialCut)
        continue;
      SmallVector<SmallVector<int64_t>> branchComponents;
      branchComponents.reserve(node.childIds.size());
      for (int64_t childId : node.childIds) {
        SmallVector<int64_t> childComponents;
        for (LeafEndpoint endpoint : collectSubtreeEndpoints(childId)) {
          const ComputeOperation &operation =
              problem.graph.operations[endpoint.first];
          // Distinct analog lanes on one tile are explicitly concurrent.
          // Sharing is forbidden here only when separately pinned wave work
          // competes for the tile's single digital lane.
          if (operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
              LogicalLaneKind::Digital)
            continue;
          if (operation.laneBindingGroup) {
            auto component = componentByGroup.find(*operation.laneBindingGroup);
            if (component != componentByGroup.end())
              childComponents.push_back(component->second);
          }
          if (operation.mvmWaveId) {
            auto component = componentByWave.find(*operation.mvmWaveId);
            if (component != componentByWave.end())
              childComponents.push_back(component->second);
          }
        }
        sortAndUnique(childComponents);
        branchComponents.push_back(std::move(childComponents));
      }
      for (size_t leftIndex = 0; leftIndex < branchComponents.size();
           ++leftIndex) {
        for (size_t rightIndex = leftIndex + 1;
             rightIndex < branchComponents.size(); ++rightIndex) {
          for (int64_t left : branchComponents[leftIndex]) {
            for (int64_t right : branchComponents[rightIndex]) {
              if (left != right)
                incompatibleComponents.insert(std::minmax(left, right));
            }
          }
        }
      }
    }

    SmallVector<int64_t> tileIds;
    for (MappingAnalogLaneRef lane : rootPool.analogLanes)
      tileIds.push_back(lane.tileId);
    sortAndUnique(tileIds);

    std::set<AnalogLaneKey> usedLanes;
    DenseMap<int64_t, int64_t> tileLaneLoads;
    DenseMap<int64_t, SmallVector<int64_t>> tileComponents;
    for (auto [componentId, component] : llvm::enumerate(components)) {
      int64_t componentIndex = static_cast<int64_t>(componentId);
      for (size_t offset = 0; offset < component.groupIds.size();) {
        int64_t chunkSize = std::min<int64_t>(
            problem.hardware.arraysPerCore,
            static_cast<int64_t>(component.groupIds.size() - offset));
        std::optional<int64_t> selectedTile;
        for (int64_t tileId : tileIds) {
          if (llvm::any_of(
                  ArrayRef(component.groupIds).slice(offset, chunkSize),
                  [&](int64_t groupId) {
                    return !isTileAllowedForBindingGroup(groupId, tileId);
                  }))
            continue;
          int64_t laneLoad = tileLaneLoads.lookup(tileId);
          if (laneLoad + chunkSize > problem.hardware.arraysPerCore)
            continue;
          bool concurrentConflict = llvm::any_of(
              tileComponents[tileId], [&](int64_t assignedComponent) {
                if (assignedComponent == componentIndex)
                  return false;
                return incompatibleComponents.contains(std::minmax(
                    assignedComponent, componentIndex));
              });
          if (concurrentConflict)
            continue;
          // Persistent bindings from unrelated MVM waves may occupy different
          // arrays on the same tile.  Prefer the fullest tile that can hold
          // the complete chunk so that every individual wave still occupies
          // its minimum ceil(array-count / arrays-per-core) tile count.  The
          // RA-tree realization below remains responsible for rejecting an
          // actual concurrent digital-lane conflict.
          if (!selectedTile ||
              laneLoad > tileLaneLoads.lookup(*selectedTile) ||
              (laneLoad == tileLaneLoads.lookup(*selectedTile) &&
               tileId < *selectedTile))
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
        if (!llvm::is_contained(tileComponents[*selectedTile],
                                componentIndex))
          tileComponents[*selectedTile].push_back(componentIndex);
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
    std::map<std::pair<int64_t, int64_t>, int64_t> groupByWaveMember;
    DenseMap<int64_t, int64_t> homeGroupByWave;
    for (const MVMWave &wave : problem.graph.mvmWaves) {
      SmallVector<int64_t> waveBindings;
      std::optional<std::pair<int64_t, int64_t>> homeGroup;
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
        auto [member, inserted] = groupByWaveMember.try_emplace(
            std::make_pair(wave.id, *operation.mvmWaveMember), groupId);
        if (!inserted && member->second != groupId) {
          operation.operation->emitError(
              "one MVM wave member references multiple matrix groups");
          return failure();
        }
        std::pair<int64_t, int64_t> candidate{*operation.mvmWaveMember,
                                              groupId};
        if (!homeGroup || candidate < *homeGroup)
          homeGroup = candidate;
      }
      if (!homeGroup) {
        problem.anchor->emitError("spread MVM wave has no matrix group");
        return failure();
      }
      homeGroupByWave[wave.id] = homeGroup->second;
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

    auto addConflict = [&](int64_t left, int64_t right) {
      if (left == right)
        return;
      conflicts[left].push_back(right);
      conflicts[right].push_back(left);
    };
    // Spread places members of one MVM wave on distinct tiles, but separate
    // waves may also be concurrent under a layer/fork spatial cut. Account
    // for the digital tile pinned by each vector/recombine member before
    // coloring persistent matrix homes; otherwise two spatial children can
    // receive distinct analog lanes on one tile while both require its single
    // digital lane.
    for (const StructuralRATreeNode &node : tree.nodes) {
      if (node.kind != RATreeNodeKind::SpatialCut)
        continue;
      SmallVector<SmallVector<int64_t>> branchGroups;
      branchGroups.reserve(node.childIds.size());
      for (int64_t childId : node.childIds) {
        SmallVector<int64_t> childGroups;
        for (LeafEndpoint endpoint : collectSubtreeEndpoints(childId)) {
          const ComputeOperation &operation =
              problem.graph.operations[endpoint.first];
          if (operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
              LogicalLaneKind::Digital)
            continue;
          std::optional<int64_t> pinnedGroup;
          if (operation.mvmWaveId && operation.mvmWaveMember) {
            auto member = groupByWaveMember.find(
                {*operation.mvmWaveId, *operation.mvmWaveMember});
            if (member != groupByWaveMember.end())
              pinnedGroup = member->second;
          }
          if (!pinnedGroup && operation.mvmWaveId) {
            auto home = homeGroupByWave.find(*operation.mvmWaveId);
            if (home != homeGroupByWave.end())
              pinnedGroup = home->second;
          }
          if (!pinnedGroup && operation.laneBindingGroup)
            pinnedGroup = *operation.laneBindingGroup;
          if (pinnedGroup && knownGroups.contains(*pinnedGroup))
            childGroups.push_back(*pinnedGroup);
        }
        sortAndUnique(childGroups);
        branchGroups.push_back(std::move(childGroups));
      }
      for (size_t leftIndex = 0; leftIndex < branchGroups.size();
           ++leftIndex) {
        for (size_t rightIndex = leftIndex + 1;
             rightIndex < branchGroups.size(); ++rightIndex) {
          for (int64_t left : branchGroups[leftIndex])
            for (int64_t right : branchGroups[rightIndex])
              addConflict(left, right);
        }
      }
    }
    for (auto &[groupId, neighbors] : conflicts) {
      (void)groupId;
      sortAndUnique(neighbors);
    }

    SmallVector<const LaneBindingGroup *> orderedGroups(groups.begin(),
                                                        groups.end());
    DenseMap<int64_t, int64_t> bindingTrafficBytes;
    for (const LaneBindingGroup *group : orderedGroups) {
      FailureOr<int64_t> traffic = estimateBindingTraffic(*group);
      if (failed(traffic))
        return failure();
      bindingTrafficBytes[group->id] = *traffic;
    }
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
      if (bindingTrafficBytes.lookup(left->id) !=
          bindingTrafficBytes.lookup(right->id))
        return bindingTrafficBytes.lookup(left->id) >
               bindingTrafficBytes.lookup(right->id);
      return left->id < right->id;
    });

    SmallVector<int64_t> tileIds;
    for (MappingAnalogLaneRef lane : rootPool.analogLanes)
      tileIds.push_back(lane.tileId);
    sortAndUnique(tileIds);

    std::set<AnalogLaneKey> usedLanes;
    DenseMap<int64_t, int64_t> groupTiles;
    DenseMap<int64_t, int64_t> tileLaneLoads;
    DenseMap<int64_t, int64_t> tileTrafficBytes;
    for (const LaneBindingGroup *group : orderedGroups) {
      std::optional<int64_t> selectedTile;
      for (int64_t tileId : tileIds) {
        if (!isTileAllowedForBindingGroup(group->id, tileId))
          continue;
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
        auto score = std::tuple(tileLaneLoads.lookup(tileId),
                                tileTrafficBytes.lookup(tileId), tileId);
        if (!selectedTile ||
            score < std::tuple(tileLaneLoads.lookup(*selectedTile),
                               tileTrafficBytes.lookup(*selectedTile),
                               *selectedTile))
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
      FailureOr<int64_t> traffic = addWork(
          tileTrafficBytes.lookup(*selectedTile),
          bindingTrafficBytes.lookup(group->id),
          "per-tile analog activation traffic estimate");
      if (failed(traffic))
        return failure();
      tileTrafficBytes[*selectedTile] = *traffic;
    }

    if (failed(recordWaveBindingTiles(/*requireDistinctMemberTiles=*/true)))
      return failure();
    return true;
  }

  FailureOr<bool>
  assignFirstUseWindowBindings(ArrayRef<const LaneBindingGroup *> groups,
                               const ResourcePool &rootPool,
                               bool preserveSpatialParallelism) {
    if (!usesSlidingWindow() || problem.digitalWindowSize <= 0) {
      problem.anchor->emitError(
          "first-use-window MVM placement requires a sliding window");
      return failure();
    }

    DenseSet<int64_t> knownGroups;
    DenseMap<int64_t, unsigned> groupIndex;
    for (auto [index, group] : llvm::enumerate(groups)) {
      if (group->id < 0 || !knownGroups.insert(group->id).second) {
        problem.anchor->emitError(
            "mapping realization found an invalid or duplicate analog "
            "lane-binding group");
        return failure();
      }
      groupIndex[group->id] = static_cast<unsigned>(index);
    }

    SmallVector<llvm::BitVector> conflicts(
        groups.size(), llvm::BitVector(groups.size()));
    auto addConflict = [&](int64_t leftId, int64_t rightId) {
      if (leftId == rightId)
        return;
      unsigned left = groupIndex.lookup(leftId);
      unsigned right = groupIndex.lookup(rightId);
      conflicts[left].set(right);
      conflicts[right].set(left);
    };
    std::map<std::pair<int64_t, int64_t>, int64_t> groupByWaveMember;
    DenseMap<int64_t, int64_t> homeGroupByWave;
    DenseMap<int64_t, int64_t> waveCounts;
    for (const MVMWave &wave : problem.graph.mvmWaves) {
      SmallVector<int64_t> waveBindings;
      std::optional<std::pair<int64_t, int64_t>> homeGroup;
      for (int64_t operationId : wave.physicalMVMOperationIds) {
        const ComputeOperation &operation =
            problem.graph.operations[operationId];
        if (!operation.laneBindingGroup || !operation.mvmWaveMember) {
          operation.operation->emitError(
              "first-use-window MVM body contains an unbound physical MVM");
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
        auto [member, inserted] = groupByWaveMember.try_emplace(
            std::make_pair(wave.id, *operation.mvmWaveMember), groupId);
        if (!inserted && member->second != groupId) {
          operation.operation->emitError(
              "one MVM wave member references multiple matrix groups");
          return failure();
        }
        std::pair<int64_t, int64_t> candidate{*operation.mvmWaveMember,
                                              groupId};
        if (!homeGroup || candidate < *homeGroup)
          homeGroup = candidate;
      }
      if (!homeGroup) {
        problem.anchor->emitError("first-use MVM wave has no matrix group");
        return failure();
      }
      homeGroupByWave[wave.id] = homeGroup->second;
      llvm::sort(waveBindings);
      for (int64_t groupId : waveBindings)
        ++waveCounts[groupId];
      for (auto [leftIndex, left] : llvm::enumerate(waveBindings)) {
        for (int64_t right : ArrayRef(waveBindings).drop_front(leftIndex + 1))
          addConflict(left, right);
      }
    }

    // A spatial RA cut is a promise that its children may execute at the
    // same time. The adaptive policy must therefore keep every matrix group
    // used by one child off the digital home tiles used by its siblings.
    // Bit vectors make this exact relation compact and avoid repeatedly
    // materializing the same pair at nested cuts.
    if (preserveSpatialParallelism) {
      for (const StructuralRATreeNode &node : tree.nodes) {
        if (node.kind != RATreeNodeKind::SpatialCut)
          continue;
        SmallVector<llvm::BitVector> branchGroups;
        branchGroups.reserve(node.childIds.size());
        for (int64_t childId : node.childIds) {
          llvm::BitVector childGroups(groups.size());
          for (LeafEndpoint endpoint : collectSubtreeEndpoints(childId)) {
            const ComputeOperation &operation =
                problem.graph.operations[endpoint.first];
            // Matrix setup and physical MVM leaves occupy analog lanes and
            // may coexist on distinct arrays of one tile. Only lane-bound
            // digital control/vector work can contend for the tile's single
            // digital lane, so only that work creates a home-tile conflict.
            if (operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
                LogicalLaneKind::Digital)
              continue;
            std::optional<int64_t> pinnedGroup;
            if (operation.mvmWaveId && operation.mvmWaveMember) {
              auto member = groupByWaveMember.find(
                  {*operation.mvmWaveId, *operation.mvmWaveMember});
              if (member != groupByWaveMember.end())
                pinnedGroup = member->second;
            }
            if (!pinnedGroup && operation.mvmWaveId) {
              auto home = homeGroupByWave.find(*operation.mvmWaveId);
              if (home != homeGroupByWave.end())
                pinnedGroup = home->second;
            }
            if (!pinnedGroup && operation.laneBindingGroup)
              pinnedGroup = *operation.laneBindingGroup;
            if (pinnedGroup)
              childGroups.set(groupIndex.lookup(*pinnedGroup));
          }
          branchGroups.push_back(std::move(childGroups));
        }
        for (size_t leftIndex = 0; leftIndex < branchGroups.size();
             ++leftIndex) {
          for (size_t rightIndex = leftIndex + 1;
               rightIndex < branchGroups.size(); ++rightIndex) {
            for (int left = branchGroups[leftIndex].find_first(); left >= 0;
                 left = branchGroups[leftIndex].find_next(left))
              conflicts[left] |= branchGroups[rightIndex];
            for (int right = branchGroups[rightIndex].find_first(); right >= 0;
                 right = branchGroups[rightIndex].find_next(right))
              conflicts[right] |= branchGroups[leftIndex];
          }
        }
      }
    }
    for (auto [index, conflictSet] : llvm::enumerate(conflicts))
      conflictSet.reset(index);

    DenseMap<int64_t, int64_t> leafOrder;
    int64_t nextOrder = 0;
    collectNonSetupLeafOrder(tree.rootId, leafOrder, nextOrder);
    DenseMap<int64_t, int64_t> operationFirstOrder;
    for (const StructuralRATreeNode &node : tree.nodes) {
      if (node.kind != RATreeNodeKind::Leaf)
        continue;
      const ComputeOperation &operation =
          problem.graph.operations[node.operationId];
      if (operation.kind != ComputeOperationKind::PhysicalMVM)
        continue;
      auto order = leafOrder.find(node.id);
      if (order == leafOrder.end()) {
        operation.operation->emitError(
            "first-use-window cannot order a physical MVM leaf");
        return failure();
      }
      auto existing = operationFirstOrder.find(operation.id);
      if (existing == operationFirstOrder.end() ||
          order->second < existing->second)
        operationFirstOrder[operation.id] = order->second;
    }

    struct BindingEvent {
      int64_t firstOrder = std::numeric_limits<int64_t>::max();
      int64_t waveId = -1;
      SmallVector<const LaneBindingGroup *> groups;
    };
    SmallVector<BindingEvent> events;
    DenseMap<int64_t, size_t> eventByWave;
    for (const LaneBindingGroup *group : groups) {
      std::optional<std::pair<int64_t, int64_t>> firstUse;
      for (int64_t operationId : group->operationIds) {
        if (operationId < 0 ||
            operationId >=
                static_cast<int64_t>(problem.graph.operations.size())) {
          problem.anchor->emitError(
              "first-use-window binding references an unknown operation");
          return failure();
        }
        const ComputeOperation &operation =
            problem.graph.operations[operationId];
        if (operation.kind != ComputeOperationKind::PhysicalMVM)
          continue;
        auto order = operationFirstOrder.find(operationId);
        if (order == operationFirstOrder.end() || !operation.mvmWaveId) {
          operation.operation->emitError(
              "first-use-window cannot resolve a physical MVM's scheduled "
              "first use");
          return failure();
        }
        std::pair<int64_t, int64_t> candidate{order->second,
                                              *operation.mvmWaveId};
        if (!firstUse || candidate < *firstUse)
          firstUse = candidate;
      }
      if (!firstUse) {
        problem.graph.operations[group->setupOperationId].operation->emitError(
            "first-use-window matrix binding has no physical MVM use");
        return failure();
      }

      auto found = eventByWave.find(firstUse->second);
      if (found == eventByWave.end()) {
        size_t index = events.size();
        eventByWave[firstUse->second] = index;
        events.push_back(
            BindingEvent{firstUse->first, firstUse->second, {group}});
      } else {
        BindingEvent &event = events[found->second];
        event.firstOrder = std::min(event.firstOrder, firstUse->first);
        event.groups.push_back(group);
      }
    }
    llvm::sort(events, [](const BindingEvent &left,
                          const BindingEvent &right) {
      return std::pair(left.firstOrder, left.waveId) <
             std::pair(right.firstOrder, right.waveId);
    });

    DenseMap<int64_t, int64_t> bindingTrafficBytes;
    for (const LaneBindingGroup *group : groups) {
      FailureOr<int64_t> traffic = estimateBindingTraffic(*group);
      if (failed(traffic))
        return failure();
      bindingTrafficBytes[group->id] = *traffic;
    }
    auto conflictCount = [&](int64_t groupId) {
      return conflicts[groupIndex.lookup(groupId)].count();
    };
    for (BindingEvent &event : events) {
      llvm::sort(event.groups, [&](const LaneBindingGroup *left,
                                   const LaneBindingGroup *right) {
        size_t leftDegree = conflictCount(left->id);
        size_t rightDegree = conflictCount(right->id);
        if (leftDegree != rightDegree)
          return leftDegree > rightDegree;
        if (waveCounts.lookup(left->id) != waveCounts.lookup(right->id))
          return waveCounts.lookup(left->id) > waveCounts.lookup(right->id);
        if (bindingTrafficBytes.lookup(left->id) !=
            bindingTrafficBytes.lookup(right->id))
          return bindingTrafficBytes.lookup(left->id) >
                 bindingTrafficBytes.lookup(right->id);
        return left->id < right->id;
      });
    }

    const int64_t tileCount = realization.logicalTileCount;
    const int64_t windowSize = problem.digitalWindowSize;
    const int64_t maximumHead = tileCount - windowSize;
    int64_t boundGroups = 0;
    std::set<AnalogLaneKey> usedLanes;
    DenseMap<int64_t, int64_t> groupTiles;
    DenseMap<int64_t, SmallVector<int64_t>> tileGroups;
    DenseMap<int64_t, int64_t> tileLaneLoads;
    DenseMap<int64_t, int64_t> tileTrafficBytes;
    int64_t previousWindowHead = 0;
    for (const BindingEvent &event : events) {
      // New matrix homes flow monotonically through logical tile IDs. All
      // matrices first encountered in one parallel MVM wave see the same
      // window, so the window never moves through the middle of a wave. If
      // persistent homes or RA concurrency conflicts fill the nominal
      // window, advance to the earliest later window that can hold the whole
      // first-use event; never widen the window or move it backward.
      FailureOr<int64_t> scaledProgress = multiplyWork(
          boundGroups, maximumHead,
          "first-use-window matrix binding progress");
      if (failed(scaledProgress))
        return failure();
      int64_t nominalWindowHead = groups.empty()
                                      ? 0
                                      : *scaledProgress /
                                            static_cast<int64_t>(groups.size());
      int64_t firstWindowHead =
          std::max(previousWindowHead, nominalWindowHead);
      bool eventAssigned = false;
      for (int64_t windowHead = firstWindowHead;
           windowHead <= maximumHead; ++windowHead) {
        std::set<AnalogLaneKey> trialUsedLanes = usedLanes;
        DenseMap<int64_t, int64_t> trialGroupTiles = groupTiles;
        DenseMap<int64_t, SmallVector<int64_t>> trialTileGroups = tileGroups;
        DenseMap<int64_t, int64_t> trialTileLaneLoads = tileLaneLoads;
        DenseMap<int64_t, int64_t> trialTileTrafficBytes = tileTrafficBytes;
        SmallVector<std::pair<int64_t, MappingAnalogLaneRef>>
            eventAssignments;
        bool windowFeasible = true;
        for (const LaneBindingGroup *group : event.groups) {
          std::optional<MappingAnalogLaneRef> selectedLane;
          const int64_t candidateBegin = preserveSpatialParallelism ? 0
                                                                    : windowHead;
          const int64_t candidateEnd = preserveSpatialParallelism
                                           ? tileCount
                                           : windowHead + windowSize;
          for (int64_t tileId = candidateBegin; tileId < candidateEnd;
               ++tileId) {
            if (!isTileAllowedForBindingGroup(group->id, tileId))
              continue;
            const llvm::BitVector &groupConflicts =
                conflicts[groupIndex.lookup(group->id)];
            bool conflictsOnTile = llvm::any_of(
                trialTileGroups[tileId], [&](int64_t neighborId) {
                  return groupConflicts.test(groupIndex.lookup(neighborId));
                });
            if (conflictsOnTile)
              continue;
            auto available = llvm::find_if(
                rootPool.analogLanes, [&](MappingAnalogLaneRef lane) {
                  return lane.tileId == tileId &&
                         !trialUsedLanes.contains(getKey(lane));
                });
            if (available == rootPool.analogLanes.end())
              continue;
            // Preserve future window capacity by filling an already-used
            // compatible tile before consuming a new tile. RA and wave
            // conflicts above still separate concurrently controlled MVMs.
            auto candidateScore = [&](int64_t candidateTile) {
              const bool inside = candidateTile >= windowHead &&
                                  candidateTile < windowHead + windowSize;
              const bool behind = candidateTile < windowHead;
              const int64_t distance =
                  inside ? 0
                         : behind
                               ? windowHead - candidateTile
                               : candidateTile -
                                     (windowHead + windowSize - 1);
              // In-window homes are best. Forward spills retain the desired
              // flow direction; backward spills are a final feasibility
              // escape hatch and remain visible in the resulting plan.
              return std::tuple(
                  inside ? 0 : behind ? 2 : 1, distance,
                  -trialTileLaneLoads.lookup(candidateTile),
                  trialTileTrafficBytes.lookup(candidateTile), candidateTile);
            };
            auto score = candidateScore(tileId);
            if (!selectedLane || score < candidateScore(selectedLane->tileId))
              selectedLane = *available;
          }
          if (!selectedLane) {
            windowFeasible = false;
            break;
          }

          eventAssignments.push_back({group->id, *selectedLane});
          trialGroupTiles[group->id] = selectedLane->tileId;
          trialTileGroups[selectedLane->tileId].push_back(group->id);
          trialUsedLanes.insert(getKey(*selectedLane));
          ++trialTileLaneLoads[selectedLane->tileId];
          FailureOr<int64_t> traffic = addWork(
              trialTileTrafficBytes.lookup(selectedLane->tileId),
              bindingTrafficBytes.lookup(group->id),
              "first-use-window per-tile analog activation traffic estimate");
          if (failed(traffic))
            return failure();
          trialTileTrafficBytes[selectedLane->tileId] = *traffic;
        }
        if (!windowFeasible)
          continue;

        for (auto [groupId, lane] : eventAssignments)
          bindingLanes[groupId] = lane;
        usedLanes = std::move(trialUsedLanes);
        groupTiles = std::move(trialGroupTiles);
        tileGroups = std::move(trialTileGroups);
        tileLaneLoads = std::move(trialTileLaneLoads);
        tileTrafficBytes = std::move(trialTileTrafficBytes);
        previousWindowHead = windowHead;
        eventAssigned = true;
        break;
      }
      if (!eventAssigned) {
        std::string reason;
        if (preserveSpatialParallelism) {
          reason = (Twine("first-use MVM wave ") + Twine(event.waveId) +
                    " cannot bind its new matrices without violating an RA "
                    "spatial-concurrency constraint")
                       .str();
        } else {
          reason = (Twine("first-use MVM wave ") + Twine(event.waveId) +
                    " cannot bind its new matrices in any monotonic logical "
                    "tile window from head " + Twine(firstWindowHead) +
                    " through " + Twine(maximumHead))
                       .str();
        }
        realization = makeInfeasible(problem, reason);
        return false;
      }
      boundGroups += static_cast<int64_t>(event.groups.size());
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
    case MVMBodyPolicy::FirstUseWindow:
      return assignFirstUseWindowBindings(
          groups, rootPool, /*preserveSpatialParallelism=*/false);
    case MVMBodyPolicy::FirstUseAdaptive:
      return assignFirstUseWindowBindings(
          groups, rootPool, /*preserveSpatialParallelism=*/true);
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
                getSchedulingRequiredDigitalTile(operation,
                                                 node->workUnitId))
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

    if (node->kind == RATreeNodeKind::Layer) {
      FailureOr<ResourceDemand> child = collectDemand(node->childIds.front());
      if (failed(child))
        return failure();
      demands[nodeId] = *child;
      return *child;
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
                       ArrayRef<int64_t> phaseWork, double notBeforeNs) {
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
      if (!isTileAllowedForSubtree(childId, tileId)) {
        realization = makeInfeasible(
            problem,
            (Twine("spatial child ") + Twine(childId) +
             " requires a digital lane outside its semantic layer tile pool")
                .str());
        return std::optional<ResourcePool>{};
      }
      if (llvm::is_contained(childPool.digitalTileIds, tileId))
        continue;
      if (!usedDigitalTiles.insert(tileId).second &&
          problem.mvmBodyPolicy != MVMBodyPolicy::FirstUseWindow) {
        realization = makeInfeasible(
            problem, (Twine("spatial cut assigns logical core ") +
                      Twine(tileId) + " to multiple concurrent children "
                      "under RA node " + Twine(parentId) +
                      " while allocating child " + Twine(childId) +
                      describeSpatialChildren(parentId))
                         .str());
        return std::optional<ResourcePool>{};
      }
      // First-use matrix homes deliberately constrain a wider RA spatial cut
      // to the active logical window. Independent branches may therefore
      // share one pinned tile. The schedule-aware digital availability clock
      // serializes their control work honestly, while distinct analog lanes
      // remain independently available.
      childPool.digitalTileIds.push_back(tileId);
    }

    SmallVector<int64_t> digitalCandidates = parentPool.digitalTileIds;
    llvm::erase_if(digitalCandidates, [&](int64_t tileId) {
      return !isTileAllowedForSubtree(childId, tileId);
    });
    const int64_t additionalDigitalLanes =
        demand->digitalLanes -
        static_cast<int64_t>(childPool.digitalTileIds.size());
    const int64_t availableUnreserved = llvm::count_if(
        digitalCandidates, [&](int64_t tileId) {
          return !usedDigitalTiles.contains(tileId) &&
                 canUseUnpinnedDigitalTile(tileId);
        });
    if (availableUnreserved >= additionalDigitalLanes) {
      llvm::erase_if(digitalCandidates, [&](int64_t tileId) {
        return isReservedForConsumer(tileId) &&
               !llvm::is_contained(demand->requiredDigitalTileIds, tileId) &&
               // The consumer reservation is a placement preference, while
               // the sliding window is the explicit scheduling contract.
               // Keep active-window homes legal; the RA cut and lane timing
               // still reject actual concurrent use.
               (!usesSlidingWindow() || !isInSlidingWindow(tileId));
      });
    }
    DenseMap<int64_t, int64_t> candidateAffinities;
    candidateAffinities.reserve(digitalCandidates.size());
    for (int64_t tileId : digitalCandidates)
      candidateAffinities[tileId] = subtreeAffinity(childId, tileId);
    llvm::sort(digitalCandidates, [&](int64_t left, int64_t right) {
      if (usesScheduleAwareTiming()) {
        auto score = [&](int64_t tileId) {
          return std::tuple(usesSlidingWindow() &&
                                    !isInSlidingWindow(tileId),
                            std::max(notBeforeNs, digitalAvailableNs[tileId]),
                            -candidateAffinities.lookup(tileId),
                            realization.digitalWorkPerTile[tileId], tileId);
        };
        return score(left) < score(right);
      }
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
                             SmallVectorImpl<int64_t> &phaseWork,
                             double notBeforeNs, double &finishNs) {
    const ComputeOperation &operation =
        problem.graph.operations[node.operationId];
    LogicalLaneKind required =
        operation.requiredLane.value_or(LogicalLaneKind::Digital);
    ResourcePool legalPool = pool;
    llvm::erase_if(legalPool.digitalTileIds, [&](int64_t tileId) {
      return !isTileAllowedForOperation(operation, tileId);
    });
    llvm::erase_if(legalPool.analogLanes, [&](MappingAnalogLaneRef lane) {
      return !isTileAllowedForOperation(operation, lane.tileId);
    });

    MappingLeafAssignment assignment;
    assignment.leafId = node.id;
    assignment.operationId = node.operationId;
    assignment.laneKind = required;

    if (required == LogicalLaneKind::Analog) {
      MappingAnalogLaneRef lane;
      if (operation.laneBindingGroup) {
        lane = bindingLanes.lookup(*operation.laneBindingGroup);
        if (!containsAnalogLane(legalPool.analogLanes, lane)) {
          realization = makeInfeasible(
              problem, (Twine("leaf ") + Twine(node.id) +
                        " cannot access its persistent analog lane binding")
                           .str());
          return false;
        }
      } else {
        if (legalPool.analogLanes.empty()) {
          realization =
              makeInfeasible(problem, (Twine("leaf ") + Twine(node.id) +
                                       " has no available analog lane")
                                          .str());
          return false;
        }
        lane = legalPool.analogLanes.front();
      }
      assignment.tileId = lane.tileId;
      assignment.laneIndex = lane.laneIndex;
      if (usesScheduleAwareTiming()) {
        FailureOr<IncomingArrival> arrival =
            estimateIncomingArrival(node, assignment.tileId);
        FailureOr<double> duration = estimateAnalogLeafDurationNs(node);
        if (failed(arrival) || failed(duration))
          return failure();
        int64_t laneOffset =
            assignment.tileId * realization.analogLanesPerTile +
            assignment.laneIndex;
        assignment.startNs =
            std::max({notBeforeNs, analogAvailableNs[laneOffset],
                      arrival->readyNs});
        assignment.finishNs = assignment.startNs + *duration;
        analogAvailableNs[laneOffset] = assignment.finishNs;
        realization.estimatedCommunicationNs += arrival->communicationNs;
      }
    } else {
      if (legalPool.digitalTileIds.empty()) {
        realization = makeInfeasible(problem, (Twine("leaf ") + Twine(node.id) +
                                               " has no available digital lane")
                                                  .str());
        return false;
      }
      auto colocation = digitalColocationGroupByEndpoint.find(
          {operation.id, node.workUnitId});
      if (!usesScheduleAwareTiming() &&
          colocation != digitalColocationGroupByEndpoint.end()) {
        FailureOr<std::optional<int64_t>> selected =
            selectDigitalColocationTile(colocation->second, legalPool,
                                         phaseWork);
        if (failed(selected))
          return failure();
        if (!*selected)
          return false;
        assignment.tileId = **selected;
      } else if (std::optional<int64_t> pinnedTile =
                     getSchedulingRequiredDigitalTile(operation,
                                                      node.workUnitId)) {
        if (!llvm::is_contained(legalPool.digitalTileIds, *pinnedTile)) {
          realization = makeInfeasible(
            problem, (Twine("digital operation ") + Twine(operation.id) +
                      " cannot access its required logical core")
                           .str());
          return false;
        }
        assignment.tileId = *pinnedTile;
      } else if (usesScheduleAwareTiming()) {
        FailureOr<double> duration = estimateDigitalLeafDurationNs(node);
        if (failed(duration))
          return failure();
        std::optional<DigitalScheduleChoice> admittedChoice;
        std::optional<DigitalScheduleChoice> maskedChoice;
        for (int64_t tileId : legalPool.digitalTileIds) {
          if (usesSlidingWindow() && !isInSlidingWindow(tileId))
            continue;
          FailureOr<IncomingArrival> arrival =
              estimateIncomingArrival(node, tileId);
          if (failed(arrival))
            return failure();
          double start = std::max(
              {notBeforeNs, digitalAvailableNs[tileId], arrival->readyNs});
          double candidateFinish = start + *duration;
          auto score = std::tuple(
              candidateFinish, arrival->communicationNs,
              arrival->crossingBytes, arrival->messages,
              -tileAffinity(node.operationId, tileId, node.workUnitId),
              realization.digitalWorkPerTile[tileId], tileId);
          DigitalScheduleChoice candidate{tileId, *arrival, candidateFinish,
                                          score};
          std::optional<DigitalScheduleChoice> &best =
              !usesProgressiveAdmission() || usesSlidingWindow() ||
                      admittedDigitalTiles.contains(tileId)
                  ? admittedChoice
                  : maskedChoice;
          if (!best || candidate.score < best->score)
            best = std::move(candidate);
        }

        const DigitalScheduleChoice *selected = nullptr;
        if (!usesProgressiveAdmission()) {
          if (!admittedChoice) {
            realization = makeInfeasible(
                problem,
                (Twine("sliding digital window [") +
                 Twine(slidingWindowHead()) + ", " +
                 Twine(slidingWindowHead() + problem.digitalWindowSize) +
                 ") does not intersect the legal resource pool for leaf " +
                 Twine(node.id))
                    .str());
            return false;
          }
          selected = &*admittedChoice;
        } else if (!admittedChoice) {
          // Seed this legal resource pool with its best lane. Admission is
          // global and monotonic, but disjoint S-cut pools may each need an
          // initial lane to preserve the RA tree's required parallelism.
          assert(maskedChoice && "a nonempty digital pool must select one tile");
          admittedDigitalTiles.insert(maskedChoice->tileId);
          selected = &*maskedChoice;
        } else if (!maskedChoice) {
          selected = &*admittedChoice;
        } else {
          const double finishImprovement =
              admittedChoice->finishNs - maskedChoice->finishNs;
          const double addedCommunication = std::max(
              0.0, maskedChoice->arrival.communicationNs -
                       admittedChoice->arrival.communicationNs);
          // Candidate finish already includes dependency arrival. Requiring
          // the residual finish improvement to exceed incremental total
          // communication deliberately regularizes against opening a lane
          // for a marginal local win that increases global network work.
          constexpr double kAdmissionEpsilonNs = 1.0e-9;
          if (finishImprovement >
              addedCommunication + kAdmissionEpsilonNs) {
            admittedDigitalTiles.insert(maskedChoice->tileId);
            selected = &*maskedChoice;
          } else {
            selected = &*admittedChoice;
          }
        }
        assert(selected && "a nonempty digital pool must select one tile");
        assignment.tileId = selected->tileId;
        assignment.startNs =
            std::max({notBeforeNs, digitalAvailableNs[assignment.tileId],
                      selected->arrival.readyNs});
        assignment.finishNs = assignment.startNs + *duration;
        realization.estimatedCommunicationNs +=
            selected->arrival.communicationNs;
      } else {
        const bool preferUnreserved =
            hasConsumerUnreservedDigitalTile(legalPool.digitalTileIds);
        auto firstCandidate = llvm::find_if(
            legalPool.digitalTileIds,
            [&](int64_t tileId) {
              return !preferUnreserved || canUseUnpinnedDigitalTile(tileId);
            });
        assert(firstCandidate != legalPool.digitalTileIds.end() &&
               "a nonempty digital pool must provide a fallback core");
        assignment.tileId = *firstCandidate;
        auto selectedScore = digitalTileScore(
            assignment.tileId,
            tileAffinity(node.operationId, assignment.tileId, node.workUnitId),
            phaseWork);
        for (int64_t tileId : legalPool.digitalTileIds) {
          if (preferUnreserved && !canUseUnpinnedDigitalTile(tileId))
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

      if (usesScheduleAwareTiming() && assignment.finishNs == 0.0) {
        // Pinned MVM-body digital work still participates in the same queue
        // and dependency-aware timing model.
        FailureOr<IncomingArrival> arrival =
            estimateIncomingArrival(node, assignment.tileId);
        FailureOr<double> duration = estimateDigitalLeafDurationNs(node);
        if (failed(arrival) || failed(duration))
          return failure();
        assignment.startNs =
            std::max({notBeforeNs, digitalAvailableNs[assignment.tileId],
                      arrival->readyNs});
        assignment.finishNs = assignment.startNs + *duration;
        realization.estimatedCommunicationNs += arrival->communicationNs;
      }

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
      if (usesScheduleAwareTiming())
        digitalAvailableNs[assignment.tileId] = assignment.finishNs;
      if (usesSlidingWindow() && !getPinnedDigitalTile(operation)) {
        FailureOr<int64_t> completed = addWork(
            completedFlexibleDigitalWork, *leafWork,
            "sliding-window completed flexible digital work");
        if (failed(completed))
          return failure();
        completedFlexibleDigitalWork = *completed;
      }
    }

    realization.leafAssignments.push_back(assignment);
    if (usesScheduleAwareTiming()) {
      scheduledEndpoints[node.operationId].push_back(
          {node.workUnitId, assignment.tileId, assignment.finishNs});
      finishNs = assignment.finishNs;
    } else {
      finishNs = 0.0;
    }
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
                             SmallVectorImpl<int64_t> &phaseWork,
                             double notBeforeNs, double &finishNs) {
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    if (!node) {
      problem.anchor->emitError(
          "mapping realization cannot resolve RA-tree node ")
          << nodeId;
      return failure();
    }
    recordNodeAllocation(nodeId, pool);

    if (node->kind == RATreeNodeKind::Leaf)
      return assignLeaf(*node, pool, phaseWork, notBeforeNs, finishNs);

    if (node->kind == RATreeNodeKind::Layer)
      return assignNode(node->childIds.front(), pool, phaseWork, notBeforeNs,
                        finishNs);

    if (node->kind == RATreeNodeKind::TemporalCut) {
      double phaseFinishNs = notBeforeNs;
      for (int64_t childId : node->childIds) {
        SmallVector<int64_t> childPhaseWork(realization.logicalTileCount, 0);
        double childFinishNs = phaseFinishNs;
        FailureOr<bool> assigned =
            assignNode(childId, pool, childPhaseWork, phaseFinishNs,
                       childFinishNs);
        if (failed(assigned) || !*assigned)
          return assigned;
        if (usesScheduleAwareTiming())
          phaseFinishNs = childFinishNs;
      }
      finishNs = usesScheduleAwareTiming() ? phaseFinishNs : 0.0;
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
    if (problem.digitalSchedulingPolicy ==
        DigitalSchedulingPolicy::Balanced) {
      llvm::stable_sort(spatialChildren,
                        [](const auto &left, const auto &right) {
                          return left.second > right.second;
                        });
    }

    std::set<int64_t> usedDigitalTiles;
    std::set<AnalogLaneKey> usedAnalogLanes;
    double spatialFinishNs = notBeforeNs;
    for (auto [childId, childWork] : spatialChildren) {
      (void)childWork;
      FailureOr<std::optional<ResourcePool>> childPool = allocateSpatialChild(
          nodeId, childId, pool, usedDigitalTiles, usedAnalogLanes,
          phaseWork, notBeforeNs);
      if (failed(childPool))
        return failure();
      if (!*childPool)
        return false;
      double childFinishNs = notBeforeNs;
      FailureOr<bool> assigned =
          assignNode(childId, **childPool, phaseWork, notBeforeNs,
                     childFinishNs);
      if (failed(assigned) || !*assigned)
        return assigned;
      if (usesScheduleAwareTiming())
        spatialFinishNs = std::max(spatialFinishNs, childFinishNs);
    }
    finishNs = usesScheduleAwareTiming() ? spatialFinishNs : 0.0;
    return true;
  }

  const MappingProblem &problem;
  const ResourceAllocationTree &tree;
  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  DenseMap<int64_t, const MappingWorkUnit *> workUnitsById;
  DenseMap<int64_t, SmallVector<const MappingWorkUnitEdge *>>
      workUnitEdgesByOperation;
  DenseMap<int64_t, SmallVector<const MappingWorkUnitEdge *>>
      workUnitEdgesByTargetOperation;
  std::set<std::pair<int64_t, int64_t>> wildcardRefinedEdges;
  std::set<std::tuple<int64_t, int64_t, int64_t>> refinedTensorEdges;
  DenseMap<int64_t, ResourceDemand> demands;
  DenseMap<int64_t, SmallVector<int64_t>> allowedTilesByLayerRegion;
  DenseMap<int64_t, SmallVector<int64_t>> allowedTilesByBindingGroup;
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
  DenseMap<int64_t, SmallVector<ScheduledEndpoint>> scheduledEndpoints;
  DenseSet<int64_t> admittedDigitalTiles;
  SmallVector<int64_t> slidingTileRanks;
  int64_t totalFlexibleDigitalWork = 0;
  int64_t completedFlexibleDigitalWork = 0;
  SmallVector<double> digitalAvailableNs;
  SmallVector<double> analogAvailableNs;
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
  if (!std::isfinite(realization.estimatedMakespanNs) ||
      realization.estimatedMakespanNs < 0.0 ||
      !std::isfinite(realization.estimatedCommunicationNs) ||
      realization.estimatedCommunicationNs < 0.0) {
    problem.anchor->emitError(
        "mapping realization requires finite nonnegative timing estimates");
    return failure();
  }

  DenseMap<int64_t, const MappingLayerTilePool *> layerPools;
  for (const MappingLayerTilePool &pool : realization.layerTilePools) {
    if (pool.layerRegionId < 0 ||
        pool.layerRegionId >=
            static_cast<int64_t>(problem.graph.layerRegions.size()) ||
        pool.tileIds.empty() ||
        !layerPools.try_emplace(pool.layerRegionId, &pool).second) {
      problem.anchor->emitError(
          "mapping realization has an invalid layer tile pool");
      return failure();
    }
    std::set<int64_t> uniqueTiles;
    for (int64_t tileId : pool.tileIds) {
      if (tileId < 0 || tileId >= realization.logicalTileCount ||
          !uniqueTiles.insert(tileId).second) {
        problem.anchor->emitError(
            "mapping layer tile pool has an invalid or duplicate tile ID");
        return failure();
      }
    }
  }
  if (layerPools.size() != problem.graph.layerRegions.size()) {
    problem.anchor->emitError(
        "mapping realization must allocate every layer region exactly once");
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
    const MappingLayerTilePool *layerPool =
        layerPools.lookup(operation.layerRegionId);
    if (!layerPool ||
        !llvm::is_contained(layerPool->tileIds, assignment.tileId)) {
      operation.operation->emitError(
          "mapping leaf assignment is outside its semantic layer tile pool");
      return failure();
    }
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
        (problem.mvmBodyPolicy == MVMBodyPolicy::Spread ||
         problem.mvmBodyPolicy == MVMBodyPolicy::FirstUseWindow ||
         problem.mvmBodyPolicy == MVMBodyPolicy::FirstUseAdaptive)
            ? static_cast<int64_t>(wave.physicalMVMOperationIds.size())
            : llvm::divideCeil(
                  static_cast<int64_t>(wave.physicalMVMOperationIds.size()),
                  realization.analogLanesPerTile);
    if (static_cast<int64_t>(distinctTiles.size()) != expectedTileCount) {
      problem.anchor->emitError(
          (problem.mvmBodyPolicy == MVMBodyPolicy::Spread ||
           problem.mvmBodyPolicy == MVMBodyPolicy::FirstUseWindow ||
           problem.mvmBodyPolicy == MVMBodyPolicy::FirstUseAdaptive)
              ? "spread-style MVM-body policy must use one logical tile per "
                "array"
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

      if (node.kind == RATreeNodeKind::TemporalCut ||
          node.kind == RATreeNodeKind::Layer) {
        if (childDigital != parentDigital || childAnalog != parentAnalog) {
          problem.anchor->emitError()
              << (node.kind == RATreeNodeKind::Layer ? "Layer" : "T-cut")
              << " child must inherit its parent's complete logical "
                 "resource pool";
          return failure();
        }
        continue;
      }

      for (int64_t tileId : childDigital) {
        if (!parentDigital.contains(tileId)) {
          problem.anchor->emitError(
              "S-cut child digital lane must belong to its parent");
          return failure();
        }
        if (!usedDigital.insert(tileId).second &&
            problem.mvmBodyPolicy != MVMBodyPolicy::FirstUseWindow) {
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
