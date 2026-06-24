#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <utility>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace task_schedulers = mlir::sculptor::task_schedulers;

using TaskGraphDAG = task_schedulers::TaskGraphDAG;
using TaskGraphNode = task_schedulers::TaskGraphNode;

constexpr int64_t kHeavyEdgeCutoffPercent = 80;
constexpr int64_t kCellSearchRadius = 1;
constexpr int64_t kUnplaced = -1;

struct PlacementGroup {
  unsigned index = 0;
  unsigned matrixSetupTaskIndex = 0;
  const TaskGraphNode *matrixSetupTask = nullptr;
  llvm::SmallVector<const TaskGraphNode *, 4> analogTasks;
  llvm::StringRef sourceLayer;
  int64_t physicalArrayId = kUnplaced;
};

struct WeightedGroupEdge {
  unsigned groupA = 0;
  unsigned groupB = 0;
  int64_t totalBytes = 0;
};

struct GroupAdjacency {
  unsigned neighborGroup = 0;
  int64_t totalBytes = 0;
};

struct HeavyComponent {
  llvm::SmallVector<unsigned, 4> groups;
  int64_t totalBytes = 0;
  unsigned minGroup = 0;
};

struct PhysicalArraySlot {
  int64_t physicalArrayId = 0;
  int64_t coreId = 0;
  int64_t localArrayId = 0;
  int64_t row = 0;
  int64_t col = 0;
};

struct GreedyPlacementState {
  explicit GreedyPlacementState(llvm::SmallVector<PlacementGroup, 8> &groups)
      : groups(groups) {}

  llvm::SmallVector<PlacementGroup, 8> &groups;
  llvm::SmallVector<PhysicalArraySlot, 8> slots;
  llvm::SmallVector<int64_t, 8> slotUseCounts;
  llvm::DenseMap<int64_t, unsigned> slotIndexByPhysicalArrayId;
  llvm::SmallVector<llvm::SmallVector<GroupAdjacency, 4>, 8> adjacency;
  llvm::SmallVector<int64_t, 8> incidentWeights;
  int64_t placedGroupCount = 0;

  bool isPlaced(unsigned groupIndex) const {
    return groups[groupIndex].physicalArrayId != kUnplaced;
  }
};

static bool isAnalogComputeTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getTaskKind() == task_graph_names::kMVMTaskKind ||
         taskOp.getTaskKind() == task_graph_names::kConvTileMVMTaskKind;
}

static mlir::LogicalResult
assignTaskGroup(llvm::SmallVectorImpl<int64_t> &groupByTaskIndex,
                const TaskGraphNode &node, unsigned groupIndex) {
  int64_t &assignedGroup = groupByTaskIndex[node.index];
  if (assignedGroup != kUnplaced &&
      assignedGroup != static_cast<int64_t>(groupIndex)) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    taskOp.emitError("expected task to belong to one matrix setup group");
    return mlir::failure();
  }

  assignedGroup = groupIndex;
  return mlir::success();
}

static llvm::SmallVector<PlacementGroup, 8>
collectMatrixSetupGroups(const TaskGraphDAG &dag) {
  llvm::SmallVector<PlacementGroup, 8> groups;

  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (taskOp.getTaskKind() != task_graph_names::kMatrixSetupTaskKind)
      continue;

    PlacementGroup group;
    group.index = static_cast<unsigned>(groups.size());
    group.matrixSetupTaskIndex = node.index;
    group.matrixSetupTask = &node;
    group.sourceLayer = taskOp.getSourceLayer();
    groups.push_back(std::move(group));
  }

  for (PlacementGroup &group : groups) {
    for (unsigned successorIndex : group.matrixSetupTask->successors) {
      const TaskGraphNode &successorNode = dag.nodes[successorIndex];
      if (isAnalogComputeTask(successorNode.op))
        group.analogTasks.push_back(&successorNode);
    }
  }

  return groups;
}

static void recordSourceLayerGroup(llvm::StringRef sourceLayer,
                                   unsigned groupIndex,
                                   llvm::StringMap<int64_t> &groupBySourceLayer,
                                   llvm::StringSet<> &ambiguousSourceLayers) {
  auto inserted = groupBySourceLayer.try_emplace(sourceLayer, groupIndex);
  if (!inserted.second &&
      inserted.first->second != static_cast<int64_t>(groupIndex))
    ambiguousSourceLayers.insert(sourceLayer);
}

static int64_t
findMostRecentOwnedProducer(const TaskGraphNode &node, const TaskGraphDAG &dag,
                            llvm::ArrayRef<int64_t> groupByTaskIndex) {
  int64_t selectedGroup = kUnplaced;
  unsigned selectedProducerIndex = 0;

  mlir::sculptor::TaskCreateOp taskOp = node.op;
  for (mlir::Value dependency : taskOp.getDependencies()) {
    auto producerIndexIt = dag.nodeIndexByTaskResult.find(dependency);
    if (producerIndexIt == dag.nodeIndexByTaskResult.end())
      continue;

    unsigned producerIndex = producerIndexIt->second;
    int64_t producerGroup = groupByTaskIndex[producerIndex];
    if (producerGroup == kUnplaced)
      continue;

    if (selectedGroup == kUnplaced || producerIndex > selectedProducerIndex) {
      selectedGroup = producerGroup;
      selectedProducerIndex = producerIndex;
    }
  }

  return selectedGroup;
}

static int64_t
findEarliestOwnedConsumer(const TaskGraphNode &node, const TaskGraphDAG &dag,
                          llvm::ArrayRef<int64_t> groupByTaskIndex) {
  int64_t selectedGroup = kUnplaced;
  unsigned selectedConsumerIndex = std::numeric_limits<unsigned>::max();

  for (unsigned successorIndex : node.successors) {
    int64_t consumerGroup = groupByTaskIndex[successorIndex];
    if (consumerGroup == kUnplaced)
      continue;

    if (selectedGroup == kUnplaced || successorIndex < selectedConsumerIndex) {
      selectedGroup = consumerGroup;
      selectedConsumerIndex = successorIndex;
    }
  }

  return selectedGroup;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 16>>
buildTaskGroupOwnership(const TaskGraphDAG &dag,
                        llvm::ArrayRef<PlacementGroup> groups) {
  llvm::SmallVector<int64_t, 16> groupByTaskIndex(dag.nodes.size(), kUnplaced);

  llvm::StringMap<int64_t> groupBySourceLayer;
  llvm::StringSet<> ambiguousSourceLayers;
  for (const PlacementGroup &group : groups) {
    if (mlir::failed(assignTaskGroup(groupByTaskIndex, *group.matrixSetupTask,
                                     group.index)))
      return mlir::failure();

    recordSourceLayerGroup(group.sourceLayer, group.index, groupBySourceLayer,
                           ambiguousSourceLayers);

    for (const TaskGraphNode *analogTask : group.analogTasks) {
      if (mlir::failed(
              assignTaskGroup(groupByTaskIndex, *analogTask, group.index)))
        return mlir::failure();
      mlir::sculptor::TaskCreateOp analogTaskOp = analogTask->op;
      recordSourceLayerGroup(analogTaskOp.getSourceLayer(), group.index,
                             groupBySourceLayer, ambiguousSourceLayers);
    }
  }

  for (const TaskGraphNode &node : dag.nodes) {
    if (groupByTaskIndex[node.index] != kUnplaced)
      continue;

    mlir::sculptor::TaskCreateOp taskOp = node.op;
    llvm::StringRef sourceLayer = taskOp.getSourceLayer();
    if (ambiguousSourceLayers.contains(sourceLayer))
      continue;

    auto groupIt = groupBySourceLayer.find(sourceLayer);
    if (groupIt == groupBySourceLayer.end())
      continue;

    groupByTaskIndex[node.index] = groupIt->second;
  }

  for (const TaskGraphNode &node : dag.nodes) {
    if (groupByTaskIndex[node.index] != kUnplaced)
      continue;

    int64_t producerGroup =
        findMostRecentOwnedProducer(node, dag, groupByTaskIndex);
    if (producerGroup != kUnplaced)
      groupByTaskIndex[node.index] = producerGroup;
  }

  for (const TaskGraphNode &node : llvm::reverse(dag.nodes)) {
    if (groupByTaskIndex[node.index] != kUnplaced)
      continue;

    int64_t consumerGroup =
        findEarliestOwnedConsumer(node, dag, groupByTaskIndex);
    if (consumerGroup != kUnplaced)
      groupByTaskIndex[node.index] = consumerGroup;
  }

  return groupByTaskIndex;
}

static int64_t getResourceByteSize(mlir::Value resource) {
  mlir::Operation *resourceOp = resource.getDefiningOp();
  if (!resourceOp)
    return 0;

  auto byteSizeAttr = resourceOp->getAttrOfType<mlir::IntegerAttr>(
      runtime_attrs::kResourceByteSizeAttrName);
  if (!byteSizeAttr)
    return 0;
  return byteSizeAttr.getInt();
}

static uint64_t getUndirectedEdgeKey(unsigned groupA, unsigned groupB) {
  if (groupA > groupB)
    std::swap(groupA, groupB);

  return (static_cast<uint64_t>(groupA) << 32) | groupB;
}

static mlir::LogicalResult collectResourceProducers(
    const TaskGraphDAG &dag,
    llvm::DenseMap<mlir::Value, unsigned> &producerByResource) {
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    for (mlir::Value output : taskOp.getOutputs()) {
      if (!producerByResource.try_emplace(output, node.index).second) {
        taskOp.emitError("expected task graph resource to have one producer");
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<WeightedGroupEdge, 16>>
buildWeightedGroupGraph(const TaskGraphDAG &dag,
                        llvm::ArrayRef<int64_t> groupByTaskIndex) {
  llvm::DenseMap<mlir::Value, unsigned> producerByResource;
  if (mlir::failed(collectResourceProducers(dag, producerByResource)))
    return mlir::failure();

  llvm::SmallVector<WeightedGroupEdge, 16> edges;
  llvm::DenseMap<uint64_t, unsigned> edgeIndexByKey;

  for (const TaskGraphNode &consumer : dag.nodes) {
    int64_t consumerGroup = groupByTaskIndex[consumer.index];
    if (consumerGroup == kUnplaced)
      continue;

    mlir::sculptor::TaskCreateOp consumerTask = consumer.op;
    for (mlir::Value input : consumerTask.getInputs()) {
      auto producerIt = producerByResource.find(input);
      if (producerIt == producerByResource.end())
        continue;

      int64_t producerGroup = groupByTaskIndex[producerIt->second];
      if (producerGroup == kUnplaced || producerGroup == consumerGroup)
        continue;

      int64_t byteSize = getResourceByteSize(input);
      if (byteSize <= 0)
        continue;

      unsigned groupA = static_cast<unsigned>(producerGroup);
      unsigned groupB = static_cast<unsigned>(consumerGroup);
      if (groupA > groupB)
        std::swap(groupA, groupB);

      uint64_t edgeKey = getUndirectedEdgeKey(groupA, groupB);
      auto edgeIt = edgeIndexByKey.find(edgeKey);
      if (edgeIt == edgeIndexByKey.end()) {
        WeightedGroupEdge edge;
        edge.groupA = groupA;
        edge.groupB = groupB;
        edge.totalBytes = byteSize;
        edgeIndexByKey.try_emplace(edgeKey,
                                   static_cast<unsigned>(edges.size()));
        edges.push_back(edge);
        continue;
      }

      edges[edgeIt->second].totalBytes += byteSize;
    }
  }

  llvm::sort(edges,
             [](const WeightedGroupEdge &lhs, const WeightedGroupEdge &rhs) {
               if (lhs.totalBytes != rhs.totalBytes)
                 return lhs.totalBytes > rhs.totalBytes;
               if (lhs.groupA != rhs.groupA)
                 return lhs.groupA < rhs.groupA;
               return lhs.groupB < rhs.groupB;
             });

  return edges;
}

static llvm::SmallVector<llvm::SmallVector<GroupAdjacency, 4>, 8>
buildGroupAdjacency(unsigned numGroups,
                    llvm::ArrayRef<WeightedGroupEdge> edges) {
  llvm::SmallVector<llvm::SmallVector<GroupAdjacency, 4>, 8> adjacency;
  adjacency.resize(numGroups);

  for (const WeightedGroupEdge &edge : edges) {
    adjacency[edge.groupA].push_back(
        GroupAdjacency{edge.groupB, edge.totalBytes});
    adjacency[edge.groupB].push_back(
        GroupAdjacency{edge.groupA, edge.totalBytes});
  }

  return adjacency;
}

static llvm::SmallVector<int64_t, 8> computeIncidentWeights(
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency) {
  llvm::SmallVector<int64_t, 8> incidentWeights(adjacency.size(), 0);
  for (auto indexedAdjacency : llvm::enumerate(adjacency)) {
    for (const GroupAdjacency &edge : indexedAdjacency.value())
      incidentWeights[indexedAdjacency.index()] += edge.totalBytes;
  }
  return incidentWeights;
}

class UnionFind {
public:
  explicit UnionFind(unsigned size) : parents(size), ranks(size, 0) {
    for (unsigned index = 0; index < size; ++index)
      parents[index] = index;
  }

  unsigned find(unsigned value) {
    if (parents[value] == value)
      return value;
    parents[value] = find(parents[value]);
    return parents[value];
  }

  void unite(unsigned lhs, unsigned rhs) {
    unsigned lhsRoot = find(lhs);
    unsigned rhsRoot = find(rhs);
    if (lhsRoot == rhsRoot)
      return;

    if (ranks[lhsRoot] < ranks[rhsRoot]) {
      parents[lhsRoot] = rhsRoot;
      return;
    }

    parents[rhsRoot] = lhsRoot;
    if (ranks[lhsRoot] == ranks[rhsRoot])
      ++ranks[lhsRoot];
  }

private:
  llvm::SmallVector<unsigned, 8> parents;
  llvm::SmallVector<unsigned, 8> ranks;
};

static llvm::SmallVector<WeightedGroupEdge, 16>
selectHeavyEdges(llvm::ArrayRef<WeightedGroupEdge> edges) {
  int64_t totalEdgeWeight = 0;
  for (const WeightedGroupEdge &edge : edges)
    totalEdgeWeight += edge.totalBytes;

  llvm::SmallVector<WeightedGroupEdge, 16> heavyEdges;
  if (totalEdgeWeight <= 0)
    return heavyEdges;

  int64_t selectedWeight = 0;
  int64_t targetWeight = (totalEdgeWeight * kHeavyEdgeCutoffPercent + 99) / 100;
  for (const WeightedGroupEdge &edge : edges) {
    heavyEdges.push_back(edge);
    selectedWeight += edge.totalBytes;
    if (selectedWeight >= targetWeight)
      break;
  }

  return heavyEdges;
}

static llvm::SmallVector<HeavyComponent, 8>
buildHeavyComponents(unsigned numGroups,
                     llvm::ArrayRef<WeightedGroupEdge> heavyEdges) {
  UnionFind components(numGroups);
  llvm::SmallBitVector touchedGroups(numGroups, false);
  for (const WeightedGroupEdge &edge : heavyEdges) {
    components.unite(edge.groupA, edge.groupB);
    touchedGroups.set(edge.groupA);
    touchedGroups.set(edge.groupB);
  }

  llvm::DenseMap<unsigned, unsigned> componentIndexByRoot;
  llvm::SmallVector<HeavyComponent, 8> result;
  for (unsigned groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
    if (!touchedGroups.test(groupIndex))
      continue;

    unsigned root = components.find(groupIndex);
    auto inserted = componentIndexByRoot.try_emplace(
        root, static_cast<unsigned>(result.size()));
    if (inserted.second) {
      HeavyComponent component;
      component.minGroup = groupIndex;
      result.push_back(std::move(component));
    }

    HeavyComponent &component = result[inserted.first->second];
    component.groups.push_back(groupIndex);
    component.minGroup = std::min(component.minGroup, groupIndex);
  }

  for (const WeightedGroupEdge &edge : heavyEdges) {
    unsigned root = components.find(edge.groupA);
    auto componentIt = componentIndexByRoot.find(root);
    if (componentIt != componentIndexByRoot.end())
      result[componentIt->second].totalBytes += edge.totalBytes;
  }

  for (HeavyComponent &component : result)
    llvm::sort(component.groups);

  llvm::sort(result, [](const HeavyComponent &lhs, const HeavyComponent &rhs) {
    if (lhs.totalBytes != rhs.totalBytes)
      return lhs.totalBytes > rhs.totalBytes;
    return lhs.minGroup < rhs.minGroup;
  });

  return result;
}

static mlir::FailureOr<llvm::SmallVector<PhysicalArraySlot, 8>>
buildPhysicalArraySlots(mlir::Operation *diagnosticOp,
                        const task_schedulers::HardwareBudget &budget) {
  llvm::SmallVector<PhysicalArraySlot, 8> slots;
  slots.reserve(budget.analogArrays.size());

  for (int64_t physicalArrayId : budget.analogArrays) {
    auto placement = task_schedulers::resolvePhysicalArrayPlacement(
        diagnosticOp, budget, physicalArrayId);
    if (mlir::failed(placement))
      return mlir::failure();

    PhysicalArraySlot slot;
    slot.physicalArrayId = placement->physicalArrayId;
    slot.coreId = placement->coreId;
    slot.localArrayId = placement->localArrayId;
    slot.row = placement->coreId / budget.meshCols;
    slot.col = placement->coreId % budget.meshCols;
    slots.push_back(slot);
  }

  return slots;
}

static int64_t getMeshDistance(const PhysicalArraySlot &lhs,
                               const PhysicalArraySlot &rhs) {
  int64_t rowDistance =
      lhs.row > rhs.row ? lhs.row - rhs.row : rhs.row - lhs.row;
  int64_t colDistance =
      lhs.col > rhs.col ? lhs.col - rhs.col : rhs.col - lhs.col;
  return rowDistance + colDistance;
}

static const PhysicalArraySlot *
getPlacedGroupSlot(const GreedyPlacementState &state, unsigned groupIndex) {
  int64_t physicalArrayId = state.groups[groupIndex].physicalArrayId;
  auto slotIt = state.slotIndexByPhysicalArrayId.find(physicalArrayId);
  if (slotIt == state.slotIndexByPhysicalArrayId.end())
    return nullptr;
  return &state.slots[slotIt->second];
}

static int64_t computePlacementCost(const GreedyPlacementState &state,
                                    unsigned groupIndex,
                                    const PhysicalArraySlot &candidateSlot) {
  int64_t cost = 0;
  for (const GroupAdjacency &edge : state.adjacency[groupIndex]) {
    if (!state.isPlaced(edge.neighborGroup))
      continue;

    const PhysicalArraySlot *neighborSlot =
        getPlacedGroupSlot(state, edge.neighborGroup);
    if (!neighborSlot)
      continue;

    cost += edge.totalBytes * getMeshDistance(candidateSlot, *neighborSlot);
  }
  return cost;
}

static bool isNearPlacedNeighbor(const GreedyPlacementState &state,
                                 unsigned groupIndex,
                                 const PhysicalArraySlot &candidateSlot) {
  bool hasPlacedNeighbor = false;
  int64_t bestDistance = std::numeric_limits<int64_t>::max();

  for (const GroupAdjacency &edge : state.adjacency[groupIndex]) {
    if (!state.isPlaced(edge.neighborGroup))
      continue;

    const PhysicalArraySlot *neighborSlot =
        getPlacedGroupSlot(state, edge.neighborGroup);
    if (!neighborSlot)
      continue;

    hasPlacedNeighbor = true;
    bestDistance =
        std::min(bestDistance, getMeshDistance(candidateSlot, *neighborSlot));
  }

  return !hasPlacedNeighbor || bestDistance <= kCellSearchRadius;
}

static bool isBetterSlotCandidate(const PhysicalArraySlot &candidateSlot,
                                  int64_t candidateCost,
                                  const PhysicalArraySlot &bestSlot,
                                  int64_t bestCost) {
  if (candidateCost != bestCost)
    return candidateCost < bestCost;
  if (candidateSlot.coreId != bestSlot.coreId)
    return candidateSlot.coreId < bestSlot.coreId;
  if (candidateSlot.localArrayId != bestSlot.localArrayId)
    return candidateSlot.localArrayId < bestSlot.localArrayId;
  return candidateSlot.physicalArrayId < bestSlot.physicalArrayId;
}

static mlir::FailureOr<unsigned>
findBestSlotForGroup(mlir::Operation *diagnosticOp,
                     const GreedyPlacementState &state, unsigned groupIndex,
                     bool localOnly) {
  if (state.slots.empty()) {
    diagnosticOp->emitError("expected greedy-heavy-edge scheduler to have at "
                            "least one physical analog array");
    return mlir::failure();
  }

  int64_t minUseCount =
      *std::min_element(state.slotUseCounts.begin(), state.slotUseCounts.end());
  bool foundCandidate = false;
  unsigned bestSlotIndex = 0;
  int64_t bestCost = std::numeric_limits<int64_t>::max();

  for (auto indexedSlot : llvm::enumerate(state.slots)) {
    unsigned slotIndex = static_cast<unsigned>(indexedSlot.index());
    if (state.slotUseCounts[slotIndex] != minUseCount)
      continue;

    const PhysicalArraySlot &candidateSlot = indexedSlot.value();
    if (localOnly && !isNearPlacedNeighbor(state, groupIndex, candidateSlot))
      continue;

    int64_t candidateCost =
        computePlacementCost(state, groupIndex, candidateSlot);
    if (!foundCandidate ||
        isBetterSlotCandidate(candidateSlot, candidateCost,
                              state.slots[bestSlotIndex], bestCost)) {
      foundCandidate = true;
      bestSlotIndex = slotIndex;
      bestCost = candidateCost;
    }
  }

  if (!foundCandidate)
    return mlir::failure();
  return bestSlotIndex;
}

static mlir::LogicalResult placeGroup(mlir::Operation *diagnosticOp,
                                      GreedyPlacementState &state,
                                      unsigned groupIndex) {
  if (state.isPlaced(groupIndex))
    return mlir::success();

  mlir::FailureOr<unsigned> slotIndex =
      findBestSlotForGroup(diagnosticOp, state, groupIndex,
                           /*localOnly=*/true);
  if (mlir::failed(slotIndex)) {
    slotIndex = findBestSlotForGroup(diagnosticOp, state, groupIndex,
                                     /*localOnly=*/false);
    if (mlir::failed(slotIndex))
      return mlir::failure();
  }

  PhysicalArraySlot &slot = state.slots[*slotIndex];
  state.groups[groupIndex].physicalArrayId = slot.physicalArrayId;
  ++state.slotUseCounts[*slotIndex];
  ++state.placedGroupCount;
  return mlir::success();
}

static unsigned selectAnchorGroup(const GreedyPlacementState &state,
                                  const HeavyComponent &component) {
  unsigned selectedGroup = component.groups.front();
  int64_t selectedWeight = std::numeric_limits<int64_t>::min();

  for (unsigned groupIndex : component.groups) {
    if (state.isPlaced(groupIndex))
      continue;

    int64_t incidentWeight = state.incidentWeights[groupIndex];
    if (incidentWeight > selectedWeight ||
        (incidentWeight == selectedWeight && groupIndex < selectedGroup)) {
      selectedGroup = groupIndex;
      selectedWeight = incidentWeight;
    }
  }

  return selectedGroup;
}

static unsigned
selectUnplacedComponentFallback(const GreedyPlacementState &state,
                                const HeavyComponent &component) {
  unsigned selectedGroup = component.groups.front();
  int64_t selectedWeight = std::numeric_limits<int64_t>::min();

  for (unsigned groupIndex : component.groups) {
    if (state.isPlaced(groupIndex))
      continue;

    int64_t incidentWeight = state.incidentWeights[groupIndex];
    if (incidentWeight > selectedWeight ||
        (incidentWeight == selectedWeight && groupIndex < selectedGroup)) {
      selectedGroup = groupIndex;
      selectedWeight = incidentWeight;
    }
  }

  return selectedGroup;
}

static unsigned
selectNextComponentFrontierGroup(const GreedyPlacementState &state,
                                 const HeavyComponent &component,
                                 const llvm::SmallBitVector &inComponent) {
  unsigned selectedGroup = std::numeric_limits<unsigned>::max();
  int64_t selectedEdgeWeight = std::numeric_limits<int64_t>::min();
  int64_t selectedIncidentWeight = std::numeric_limits<int64_t>::min();

  for (unsigned groupIndex : component.groups) {
    if (!state.isPlaced(groupIndex))
      continue;

    for (const GroupAdjacency &edge : state.adjacency[groupIndex]) {
      unsigned candidateGroup = edge.neighborGroup;
      if (!inComponent.test(candidateGroup) || state.isPlaced(candidateGroup))
        continue;

      int64_t candidateIncidentWeight = state.incidentWeights[candidateGroup];
      if (selectedGroup == std::numeric_limits<unsigned>::max() ||
          edge.totalBytes > selectedEdgeWeight ||
          (edge.totalBytes == selectedEdgeWeight &&
           candidateIncidentWeight > selectedIncidentWeight) ||
          (edge.totalBytes == selectedEdgeWeight &&
           candidateIncidentWeight == selectedIncidentWeight &&
           candidateGroup < selectedGroup)) {
        selectedGroup = candidateGroup;
        selectedEdgeWeight = edge.totalBytes;
        selectedIncidentWeight = candidateIncidentWeight;
      }
    }
  }

  return selectedGroup;
}

static mlir::LogicalResult
placeHeavyComponent(mlir::Operation *diagnosticOp, GreedyPlacementState &state,
                    const HeavyComponent &component) {
  if (component.groups.empty())
    return mlir::success();

  llvm::SmallBitVector inComponent(state.groups.size(), false);
  for (unsigned groupIndex : component.groups)
    inComponent.set(groupIndex);

  bool hasUnplacedGroup = llvm::any_of(
      component.groups, [&](unsigned group) { return !state.isPlaced(group); });
  if (!hasUnplacedGroup)
    return mlir::success();

  unsigned anchorGroup = selectAnchorGroup(state, component);
  if (mlir::failed(placeGroup(diagnosticOp, state, anchorGroup)))
    return mlir::failure();

  while (llvm::any_of(component.groups,
                      [&](unsigned group) { return !state.isPlaced(group); })) {
    unsigned nextGroup =
        selectNextComponentFrontierGroup(state, component, inComponent);
    if (nextGroup == std::numeric_limits<unsigned>::max())
      nextGroup = selectUnplacedComponentFallback(state, component);

    if (mlir::failed(placeGroup(diagnosticOp, state, nextGroup)))
      return mlir::failure();
  }

  return mlir::success();
}

struct FrontierEntry {
  int64_t placedNeighborWeight = 0;
  int64_t incidentWeight = 0;
  unsigned groupIndex = 0;
};

struct FrontierEntryLess {
  bool operator()(const FrontierEntry &lhs, const FrontierEntry &rhs) const {
    if (lhs.placedNeighborWeight != rhs.placedNeighborWeight)
      return lhs.placedNeighborWeight < rhs.placedNeighborWeight;
    if (lhs.incidentWeight != rhs.incidentWeight)
      return lhs.incidentWeight < rhs.incidentWeight;
    return lhs.groupIndex > rhs.groupIndex;
  }
};

static unsigned selectUnplacedFallback(const GreedyPlacementState &state) {
  unsigned selectedGroup = std::numeric_limits<unsigned>::max();
  int64_t selectedWeight = std::numeric_limits<int64_t>::min();

  for (const PlacementGroup &group : state.groups) {
    if (state.isPlaced(group.index))
      continue;

    int64_t incidentWeight = state.incidentWeights[group.index];
    if (selectedGroup == std::numeric_limits<unsigned>::max() ||
        incidentWeight > selectedWeight ||
        (incidentWeight == selectedWeight && group.index < selectedGroup)) {
      selectedGroup = group.index;
      selectedWeight = incidentWeight;
    }
  }

  return selectedGroup;
}

static void initializeFrontier(
    const GreedyPlacementState &state,
    llvm::SmallVectorImpl<int64_t> &placedNeighborWeights,
    std::priority_queue<FrontierEntry, std::vector<FrontierEntry>,
                        FrontierEntryLess> &frontier) {
  for (const PlacementGroup &group : state.groups) {
    if (!state.isPlaced(group.index))
      continue;

    for (const GroupAdjacency &edge : state.adjacency[group.index]) {
      if (state.isPlaced(edge.neighborGroup))
        continue;
      placedNeighborWeights[edge.neighborGroup] += edge.totalBytes;
    }
  }

  for (const PlacementGroup &group : state.groups) {
    if (state.isPlaced(group.index) || placedNeighborWeights[group.index] <= 0)
      continue;

    frontier.push(FrontierEntry{placedNeighborWeights[group.index],
                                state.incidentWeights[group.index],
                                group.index});
  }
}

static unsigned popNextFrontierGroup(
    const GreedyPlacementState &state,
    llvm::ArrayRef<int64_t> placedNeighborWeights,
    std::priority_queue<FrontierEntry, std::vector<FrontierEntry>,
                        FrontierEntryLess> &frontier) {
  while (!frontier.empty()) {
    FrontierEntry entry = frontier.top();
    frontier.pop();

    if (state.isPlaced(entry.groupIndex))
      continue;
    if (entry.placedNeighborWeight != placedNeighborWeights[entry.groupIndex])
      continue;
    if (entry.placedNeighborWeight <= 0)
      continue;

    return entry.groupIndex;
  }

  return std::numeric_limits<unsigned>::max();
}

static mlir::LogicalResult placeRemainingGroups(mlir::Operation *diagnosticOp,
                                                GreedyPlacementState &state) {
  llvm::SmallVector<int64_t, 8> placedNeighborWeights(state.groups.size(), 0);
  std::priority_queue<FrontierEntry, std::vector<FrontierEntry>,
                      FrontierEntryLess>
      frontier;
  initializeFrontier(state, placedNeighborWeights, frontier);

  while (state.placedGroupCount < static_cast<int64_t>(state.groups.size())) {
    unsigned nextGroup =
        popNextFrontierGroup(state, placedNeighborWeights, frontier);
    if (nextGroup == std::numeric_limits<unsigned>::max())
      nextGroup = selectUnplacedFallback(state);

    if (nextGroup == std::numeric_limits<unsigned>::max()) {
      diagnosticOp->emitError("failed to select next greedy placement group");
      return mlir::failure();
    }

    if (mlir::failed(placeGroup(diagnosticOp, state, nextGroup)))
      return mlir::failure();

    for (const GroupAdjacency &edge : state.adjacency[nextGroup]) {
      if (state.isPlaced(edge.neighborGroup))
        continue;

      placedNeighborWeights[edge.neighborGroup] += edge.totalBytes;
      frontier.push(FrontierEntry{placedNeighborWeights[edge.neighborGroup],
                                  state.incidentWeights[edge.neighborGroup],
                                  edge.neighborGroup});
    }
  }

  return mlir::success();
}

static mlir::FailureOr<GreedyPlacementState>
buildGreedyPlacementState(mlir::Operation *diagnosticOp,
                          llvm::SmallVector<PlacementGroup, 8> &groups,
                          const task_schedulers::HardwareBudget &budget,
                          llvm::ArrayRef<WeightedGroupEdge> edges) {
  auto slots = buildPhysicalArraySlots(diagnosticOp, budget);
  if (mlir::failed(slots))
    return mlir::failure();

  GreedyPlacementState state(groups);
  state.slots = std::move(*slots);
  state.slotUseCounts.assign(state.slots.size(), 0);
  for (auto indexedSlot : llvm::enumerate(state.slots))
    state.slotIndexByPhysicalArrayId.try_emplace(
        indexedSlot.value().physicalArrayId,
        static_cast<unsigned>(indexedSlot.index()));

  state.adjacency =
      buildGroupAdjacency(static_cast<unsigned>(groups.size()), edges);
  state.incidentWeights = computeIncidentWeights(state.adjacency);
  return state;
}

class GreedyHeavyEdgeTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "greedy-heavy-edge"; }

  mlir::LogicalResult schedule(
      mlir::ModuleOp module, mlir::func::FuncOp taskGraphFunc,
      const mlir::sculptor::task_schedulers::HardwareBudget &budget,
      const mlir::sculptor::task_schedulers::TaskGraphDAG &dag) const final {
    llvm::SmallVector<PlacementGroup, 8> groups = collectMatrixSetupGroups(dag);
    if (groups.empty()) {
      return task_schedulers::placeMatrixSetupGroupsAndSurroundingTasks(
          module, taskGraphFunc, budget, dag,
          llvm::ArrayRef<task_schedulers::MatrixSetupGroupPlacement>());
    }

    auto taskGroupOwnership = buildTaskGroupOwnership(dag, groups);
    if (mlir::failed(taskGroupOwnership))
      return mlir::failure();

    auto edges = buildWeightedGroupGraph(dag, *taskGroupOwnership);
    if (mlir::failed(edges))
      return mlir::failure();

    mlir::FailureOr<GreedyPlacementState> state = buildGreedyPlacementState(
        taskGraphFunc.getOperation(), groups, budget, *edges);
    if (mlir::failed(state))
      return mlir::failure();

    llvm::SmallVector<WeightedGroupEdge, 16> heavyEdges =
        selectHeavyEdges(*edges);
    llvm::SmallVector<HeavyComponent, 8> heavyComponents =
        buildHeavyComponents(static_cast<unsigned>(groups.size()), heavyEdges);

    for (const HeavyComponent &component : heavyComponents) {
      if (mlir::failed(placeHeavyComponent(taskGraphFunc.getOperation(), *state,
                                           component)))
        return mlir::failure();
    }

    if (mlir::failed(
            placeRemainingGroups(taskGraphFunc.getOperation(), *state)))
      return mlir::failure();

    llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>
        groupPlacements;
    groupPlacements.reserve(groups.size());
    for (const PlacementGroup &group : groups) {
      if (group.physicalArrayId == kUnplaced) {
        taskGraphFunc.emitError("expected greedy-heavy-edge scheduler to place "
                                "every matrix setup group");
        return mlir::failure();
      }

      groupPlacements.push_back(task_schedulers::MatrixSetupGroupPlacement{
          group.matrixSetupTaskIndex, group.physicalArrayId});
    }

    return task_schedulers::placeMatrixSetupGroupsAndSurroundingTasks(
        module, taskGraphFunc, budget, dag, groupPlacements);
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerGreedyHeavyEdgeTaskScheduler(
    TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(
      registry, std::make_unique<GreedyHeavyEdgeTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
