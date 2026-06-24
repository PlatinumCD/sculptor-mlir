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
#include <utility>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace task_schedulers = mlir::sculptor::task_schedulers;

using TaskGraphDAG = task_schedulers::TaskGraphDAG;
using TaskGraphNode = task_schedulers::TaskGraphNode;

constexpr int64_t kUnassigned = -1;
constexpr int64_t kInfiniteScore = std::numeric_limits<int64_t>::max() / 4;

struct PlacementGroup {
  unsigned index = 0;
  unsigned matrixSetupTaskIndex = 0;
  const TaskGraphNode *matrixSetupTask = nullptr;
  llvm::SmallVector<const TaskGraphNode *, 4> analogTasks;
  llvm::StringRef sourceLayer;
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

struct PhysicalArraySlot {
  int64_t physicalArrayId = 0;
  int64_t coreId = 0;
  int64_t localArrayId = 0;
  int64_t row = 0;
  int64_t col = 0;
};

struct MeshRegion {
  int64_t rowBegin = 0;
  int64_t rowEnd = 0;
  int64_t colBegin = 0;
  int64_t colEnd = 0;

  int64_t rows() const { return rowEnd - rowBegin; }
  int64_t cols() const { return colEnd - colBegin; }
  int64_t coreCount() const { return rows() * cols(); }
  bool isSingleCore() const { return rows() == 1 && cols() == 1; }
};

struct RegionSplit {
  MeshRegion first;
  MeshRegion second;
  bool vertical = false;
  int64_t cutLine = 0;
};

struct PartitionResult {
  llvm::SmallVector<unsigned, 8> firstGroups;
  llvm::SmallVector<unsigned, 8> secondGroups;
  int64_t cutWeight = 0;
  int64_t score = kInfiniteScore;
};

struct CandidateSplit {
  RegionSplit split;
  PartitionResult partition;
};

struct BoundaryAwarePlacementState {
  llvm::SmallVector<int64_t, 8> physicalArrayByGroup;
  llvm::SmallVector<PhysicalArraySlot, 8> slots;
  llvm::DenseMap<int64_t, unsigned> slotIndexByPhysicalArrayId;
};

enum class BoundaryCorridor { None, Top, Bottom, Left, Right };

struct EndpointBoundaryConstraint {
  BoundaryCorridor corridor = BoundaryCorridor::None;
  int64_t firstEndpointGroup = kUnassigned;
  int64_t lastEndpointGroup = kUnassigned;

  bool isActive() const {
    return corridor != BoundaryCorridor::None &&
           firstEndpointGroup != kUnassigned &&
           lastEndpointGroup != kUnassigned;
  }

  bool isEndpointGroup(unsigned group) const {
    if (!isActive())
      return false;
    return static_cast<int64_t>(group) == firstEndpointGroup ||
           static_cast<int64_t>(group) == lastEndpointGroup;
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
  if (assignedGroup != kUnassigned &&
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
  int64_t selectedGroup = kUnassigned;
  unsigned selectedProducerIndex = 0;

  mlir::sculptor::TaskCreateOp taskOp = node.op;
  for (mlir::Value dependency : taskOp.getDependencies()) {
    auto producerIndexIt = dag.nodeIndexByTaskResult.find(dependency);
    if (producerIndexIt == dag.nodeIndexByTaskResult.end())
      continue;

    unsigned producerIndex = producerIndexIt->second;
    int64_t producerGroup = groupByTaskIndex[producerIndex];
    if (producerGroup == kUnassigned)
      continue;

    if (selectedGroup == kUnassigned || producerIndex > selectedProducerIndex) {
      selectedGroup = producerGroup;
      selectedProducerIndex = producerIndex;
    }
  }

  return selectedGroup;
}

static int64_t
findEarliestOwnedConsumer(const TaskGraphNode &node, const TaskGraphDAG &dag,
                          llvm::ArrayRef<int64_t> groupByTaskIndex) {
  int64_t selectedGroup = kUnassigned;
  unsigned selectedConsumerIndex = std::numeric_limits<unsigned>::max();

  for (unsigned successorIndex : node.successors) {
    int64_t consumerGroup = groupByTaskIndex[successorIndex];
    if (consumerGroup == kUnassigned)
      continue;

    if (selectedGroup == kUnassigned ||
        successorIndex < selectedConsumerIndex) {
      selectedGroup = consumerGroup;
      selectedConsumerIndex = successorIndex;
    }
  }

  return selectedGroup;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 16>>
buildTaskGroupOwnership(const TaskGraphDAG &dag,
                        llvm::ArrayRef<PlacementGroup> groups) {
  llvm::SmallVector<int64_t, 16> groupByTaskIndex(dag.nodes.size(),
                                                  kUnassigned);

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
    if (groupByTaskIndex[node.index] != kUnassigned)
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
    if (groupByTaskIndex[node.index] != kUnassigned)
      continue;

    int64_t producerGroup =
        findMostRecentOwnedProducer(node, dag, groupByTaskIndex);
    if (producerGroup != kUnassigned)
      groupByTaskIndex[node.index] = producerGroup;
  }

  for (const TaskGraphNode &node : llvm::reverse(dag.nodes)) {
    if (groupByTaskIndex[node.index] != kUnassigned)
      continue;

    int64_t consumerGroup =
        findEarliestOwnedConsumer(node, dag, groupByTaskIndex);
    if (consumerGroup != kUnassigned)
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
    if (consumerGroup == kUnassigned)
      continue;

    mlir::sculptor::TaskCreateOp consumerTask = consumer.op;
    for (mlir::Value input : consumerTask.getInputs()) {
      auto producerIt = producerByResource.find(input);
      if (producerIt == producerByResource.end())
        continue;

      int64_t producerGroup = groupByTaskIndex[producerIt->second];
      if (producerGroup == kUnassigned || producerGroup == consumerGroup)
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

  llvm::sort(slots,
             [](const PhysicalArraySlot &lhs, const PhysicalArraySlot &rhs) {
               if (lhs.row != rhs.row)
                 return lhs.row < rhs.row;
               if (lhs.col != rhs.col)
                 return lhs.col < rhs.col;
               return lhs.localArrayId < rhs.localArrayId;
             });

  return slots;
}

static int64_t
getRegionCapacity(const MeshRegion &region,
                  const task_schedulers::HardwareBudget &budget) {
  return region.coreCount() * budget.arraysPerCore;
}

static bool containsSlot(const MeshRegion &region,
                         const PhysicalArraySlot &slot) {
  return slot.row >= region.rowBegin && slot.row < region.rowEnd &&
         slot.col >= region.colBegin && slot.col < region.colEnd;
}

static llvm::SmallVector<unsigned, 8>
collectRegionSlotIndices(llvm::ArrayRef<PhysicalArraySlot> slots,
                         const MeshRegion &region) {
  llvm::SmallVector<unsigned, 8> slotIndices;
  for (auto indexedSlot : llvm::enumerate(slots)) {
    if (containsSlot(region, indexedSlot.value()))
      slotIndices.push_back(static_cast<unsigned>(indexedSlot.index()));
  }
  return slotIndices;
}

static bool isSlotOnCorridor(const PhysicalArraySlot &slot,
                             const task_schedulers::HardwareBudget &budget,
                             BoundaryCorridor corridor) {
  switch (corridor) {
  case BoundaryCorridor::Top:
    return slot.row == 0;
  case BoundaryCorridor::Bottom:
    return slot.row == budget.meshRows - 1;
  case BoundaryCorridor::Left:
    return slot.col == 0;
  case BoundaryCorridor::Right:
    return slot.col == budget.meshCols - 1;
  case BoundaryCorridor::None:
    return true;
  }
  return true;
}

static int64_t
getCorridorCapacity(const MeshRegion &region,
                    const task_schedulers::HardwareBudget &budget,
                    BoundaryCorridor corridor) {
  switch (corridor) {
  case BoundaryCorridor::Top:
    return region.rowBegin <= 0 && region.rowEnd > 0
               ? region.cols() * budget.arraysPerCore
               : 0;
  case BoundaryCorridor::Bottom:
    return region.rowBegin <= budget.meshRows - 1 &&
                   region.rowEnd > budget.meshRows - 1
               ? region.cols() * budget.arraysPerCore
               : 0;
  case BoundaryCorridor::Left:
    return region.colBegin <= 0 && region.colEnd > 0
               ? region.rows() * budget.arraysPerCore
               : 0;
  case BoundaryCorridor::Right:
    return region.colBegin <= budget.meshCols - 1 &&
                   region.colEnd > budget.meshCols - 1
               ? region.rows() * budget.arraysPerCore
               : 0;
  case BoundaryCorridor::None:
    return getRegionCapacity(region, budget);
  }
  return getRegionCapacity(region, budget);
}

static unsigned
getEndpointDemandForGroup(unsigned group,
                          const EndpointBoundaryConstraint &constraint) {
  return constraint.isEndpointGroup(group) ? 1 : 0;
}

static unsigned
getEndpointDemandInSet(const llvm::SmallBitVector &groups,
                       const EndpointBoundaryConstraint &constraint) {
  if (!constraint.isActive())
    return 0;

  unsigned demand = 0;
  if (constraint.firstEndpointGroup != kUnassigned &&
      groups.test(static_cast<unsigned>(constraint.firstEndpointGroup)))
    ++demand;
  if (constraint.lastEndpointGroup != kUnassigned &&
      constraint.lastEndpointGroup != constraint.firstEndpointGroup &&
      groups.test(static_cast<unsigned>(constraint.lastEndpointGroup)))
    ++demand;
  return demand;
}

static unsigned
getEndpointDemandInGroups(llvm::ArrayRef<unsigned> groups,
                          const EndpointBoundaryConstraint &constraint) {
  if (!constraint.isActive())
    return 0;

  bool containsFirst = false;
  bool containsLast = false;
  for (unsigned group : groups) {
    containsFirst |=
        static_cast<int64_t>(group) == constraint.firstEndpointGroup;
    containsLast |= static_cast<int64_t>(group) == constraint.lastEndpointGroup;
  }

  unsigned demand = containsFirst ? 1 : 0;
  if (constraint.lastEndpointGroup != constraint.firstEndpointGroup &&
      containsLast)
    ++demand;
  return demand;
}

static bool canAddGroupToBoundaryRegion(
    unsigned group, const llvm::SmallBitVector &assignedGroups,
    const MeshRegion &region, const task_schedulers::HardwareBudget &budget,
    const EndpointBoundaryConstraint &constraint) {
  if (!constraint.isActive() || !constraint.isEndpointGroup(group))
    return true;

  unsigned currentDemand = getEndpointDemandInSet(assignedGroups, constraint);
  unsigned additionalDemand = getEndpointDemandForGroup(group, constraint);
  return currentDemand + additionalDemand <=
         static_cast<unsigned>(
             getCorridorCapacity(region, budget, constraint.corridor));
}

static bool
hasEndpointBoundaryCapacity(llvm::ArrayRef<unsigned> groups,
                            const MeshRegion &region,
                            const task_schedulers::HardwareBudget &budget,
                            const EndpointBoundaryConstraint &constraint) {
  if (!constraint.isActive())
    return true;

  unsigned demand = getEndpointDemandInGroups(groups, constraint);
  return demand <= static_cast<unsigned>(getCorridorCapacity(
                       region, budget, constraint.corridor));
}

static int64_t getAbsDiff(int64_t lhs, int64_t rhs) {
  return lhs > rhs ? lhs - rhs : rhs - lhs;
}

static int64_t getMeshDistance(const PhysicalArraySlot &lhs,
                               const PhysicalArraySlot &rhs) {
  return getAbsDiff(lhs.row, rhs.row) + getAbsDiff(lhs.col, rhs.col);
}

static unsigned
getSlotIndexForPhysicalArray(const BoundaryAwarePlacementState &state,
                             int64_t physicalArrayId) {
  auto slotIt = state.slotIndexByPhysicalArrayId.find(physicalArrayId);
  if (slotIt == state.slotIndexByPhysicalArrayId.end())
    return std::numeric_limits<unsigned>::max();
  return slotIt->second;
}

static unsigned getSlotIndexForGroup(const BoundaryAwarePlacementState &state,
                                     unsigned group) {
  if (group >= state.physicalArrayByGroup.size())
    return std::numeric_limits<unsigned>::max();

  int64_t physicalArrayId = state.physicalArrayByGroup[group];
  if (physicalArrayId == kUnassigned)
    return std::numeric_limits<unsigned>::max();

  return getSlotIndexForPhysicalArray(state, physicalArrayId);
}

static bool
endpointBoundarySatisfied(const BoundaryAwarePlacementState &state,
                          const task_schedulers::HardwareBudget &budget,
                          const EndpointBoundaryConstraint &constraint) {
  if (!constraint.isActive())
    return true;

  unsigned firstSlot = getSlotIndexForGroup(
      state, static_cast<unsigned>(constraint.firstEndpointGroup));
  unsigned lastSlot = getSlotIndexForGroup(
      state, static_cast<unsigned>(constraint.lastEndpointGroup));
  if (firstSlot == std::numeric_limits<unsigned>::max() ||
      lastSlot == std::numeric_limits<unsigned>::max())
    return false;

  return isSlotOnCorridor(state.slots[firstSlot], budget,
                          constraint.corridor) &&
         isSlotOnCorridor(state.slots[lastSlot], budget, constraint.corridor);
}

static int64_t
computeExactPlacementScore(const BoundaryAwarePlacementState &state,
                           llvm::ArrayRef<WeightedGroupEdge> edges) {
  int64_t score = 0;
  for (const WeightedGroupEdge &edge : edges) {
    unsigned slotA = getSlotIndexForGroup(state, edge.groupA);
    unsigned slotB = getSlotIndexForGroup(state, edge.groupB);
    if (slotA == std::numeric_limits<unsigned>::max() ||
        slotB == std::numeric_limits<unsigned>::max())
      return kInfiniteScore;

    score += edge.totalBytes *
             getMeshDistance(state.slots[slotA], state.slots[slotB]);
  }

  return score;
}

static llvm::SmallBitVector
collectUsedSlots(const BoundaryAwarePlacementState &state) {
  llvm::SmallBitVector usedSlots(state.slots.size(), false);
  for (int64_t physicalArrayId : state.physicalArrayByGroup) {
    if (physicalArrayId == kUnassigned)
      continue;

    unsigned slotIndex = getSlotIndexForPhysicalArray(state, physicalArrayId);
    if (slotIndex != std::numeric_limits<unsigned>::max())
      usedSlots.set(slotIndex);
  }
  return usedSlots;
}

static int64_t computeIncrementalPlacementCost(
    unsigned group, unsigned candidateSlotIndex,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
    const BoundaryAwarePlacementState &state) {
  int64_t cost = 0;
  const PhysicalArraySlot &candidateSlot = state.slots[candidateSlotIndex];
  for (const GroupAdjacency &edge : adjacency[group]) {
    unsigned neighborSlotIndex =
        getSlotIndexForGroup(state, edge.neighborGroup);
    if (neighborSlotIndex == std::numeric_limits<unsigned>::max())
      continue;

    cost += edge.totalBytes *
            getMeshDistance(candidateSlot, state.slots[neighborSlotIndex]);
  }
  return cost;
}

static bool isBetterSlotCandidate(const BoundaryAwarePlacementState &state,
                                  unsigned candidateSlotIndex,
                                  int64_t candidateCost, unsigned bestSlotIndex,
                                  int64_t bestCost) {
  if (bestSlotIndex == std::numeric_limits<unsigned>::max())
    return true;
  if (candidateCost != bestCost)
    return candidateCost < bestCost;

  const PhysicalArraySlot &candidateSlot = state.slots[candidateSlotIndex];
  const PhysicalArraySlot &bestSlot = state.slots[bestSlotIndex];
  if (candidateSlot.coreId != bestSlot.coreId)
    return candidateSlot.coreId < bestSlot.coreId;
  if (candidateSlot.localArrayId != bestSlot.localArrayId)
    return candidateSlot.localArrayId < bestSlot.localArrayId;
  return candidateSlot.physicalArrayId < bestSlot.physicalArrayId;
}

static llvm::SmallVector<int64_t, 8>
buildSlotOccupancy(const BoundaryAwarePlacementState &state) {
  llvm::SmallVector<int64_t, 8> groupBySlot(state.slots.size(), kUnassigned);
  for (auto indexedPhysicalArray :
       llvm::enumerate(state.physicalArrayByGroup)) {
    if (indexedPhysicalArray.value() == kUnassigned)
      continue;

    unsigned slotIndex =
        getSlotIndexForPhysicalArray(state, indexedPhysicalArray.value());
    if (slotIndex != std::numeric_limits<unsigned>::max())
      groupBySlot[slotIndex] =
          static_cast<int64_t>(indexedPhysicalArray.index());
  }

  return groupBySlot;
}

static void
refinePlacementWithPairSwaps(BoundaryAwarePlacementState &state,
                             llvm::ArrayRef<WeightedGroupEdge> edges,
                             const task_schedulers::HardwareBudget &budget,
                             const EndpointBoundaryConstraint &constraint) {
  constexpr unsigned kPairSwapPasses = 4;
  if (state.physicalArrayByGroup.size() < 2 || edges.empty())
    return;

  for (unsigned pass = 0; pass < kPairSwapPasses; ++pass) {
    int64_t currentScore = computeExactPlacementScore(state, edges);
    int64_t bestScore = currentScore;
    unsigned bestGroupA = std::numeric_limits<unsigned>::max();
    unsigned bestGroupB = std::numeric_limits<unsigned>::max();

    for (unsigned groupA = 0; groupA < state.physicalArrayByGroup.size();
         ++groupA) {
      if (state.physicalArrayByGroup[groupA] == kUnassigned)
        continue;

      for (unsigned groupB = groupA + 1;
           groupB < state.physicalArrayByGroup.size(); ++groupB) {
        if (state.physicalArrayByGroup[groupB] == kUnassigned)
          continue;

        std::swap(state.physicalArrayByGroup[groupA],
                  state.physicalArrayByGroup[groupB]);
        if (!endpointBoundarySatisfied(state, budget, constraint)) {
          std::swap(state.physicalArrayByGroup[groupA],
                    state.physicalArrayByGroup[groupB]);
          continue;
        }
        int64_t candidateScore = computeExactPlacementScore(state, edges);
        std::swap(state.physicalArrayByGroup[groupA],
                  state.physicalArrayByGroup[groupB]);

        if (candidateScore < bestScore) {
          bestScore = candidateScore;
          bestGroupA = groupA;
          bestGroupB = groupB;
        }
      }
    }

    if (bestGroupA == std::numeric_limits<unsigned>::max())
      return;

    std::swap(state.physicalArrayByGroup[bestGroupA],
              state.physicalArrayByGroup[bestGroupB]);
  }
}

static int64_t computeEdgeContribution(const BoundaryAwarePlacementState &state,
                                       const WeightedGroupEdge &edge) {
  unsigned slotA = getSlotIndexForGroup(state, edge.groupA);
  unsigned slotB = getSlotIndexForGroup(state, edge.groupB);
  if (slotA == std::numeric_limits<unsigned>::max() ||
      slotB == std::numeric_limits<unsigned>::max())
    return kInfiniteScore;

  return edge.totalBytes *
         getMeshDistance(state.slots[slotA], state.slots[slotB]);
}

static int64_t
computeChainDistanceScore(const BoundaryAwarePlacementState &state,
                          llvm::ArrayRef<unsigned> groups) {
  if (groups.size() < 2)
    return 0;

  int64_t score = 0;
  for (auto groupPair : llvm::zip(groups.drop_back(), groups.drop_front())) {
    unsigned previousGroup = std::get<0>(groupPair);
    unsigned nextGroup = std::get<1>(groupPair);
    unsigned previousSlot = getSlotIndexForGroup(state, previousGroup);
    unsigned nextSlot = getSlotIndexForGroup(state, nextGroup);
    if (previousSlot == std::numeric_limits<unsigned>::max() ||
        nextSlot == std::numeric_limits<unsigned>::max())
      return kInfiniteScore;

    int64_t distance =
        getMeshDistance(state.slots[previousSlot], state.slots[nextSlot]);
    score += distance * distance;
  }

  return score;
}

static int64_t
computeCompactFootprintScore(const BoundaryAwarePlacementState &state,
                             const task_schedulers::HardwareBudget &budget,
                             llvm::ArrayRef<unsigned> orderedGroups) {
  if (orderedGroups.empty())
    return 0;

  llvm::SmallBitVector occupiedCores(budget.numCores, false);
  int64_t minRow = std::numeric_limits<int64_t>::max();
  int64_t maxRow = std::numeric_limits<int64_t>::min();
  int64_t minCol = std::numeric_limits<int64_t>::max();
  int64_t maxCol = std::numeric_limits<int64_t>::min();

  for (unsigned group : orderedGroups) {
    unsigned slotIndex = getSlotIndexForGroup(state, group);
    if (slotIndex == std::numeric_limits<unsigned>::max())
      return kInfiniteScore;

    const PhysicalArraySlot &slot = state.slots[slotIndex];
    occupiedCores.set(static_cast<unsigned>(slot.coreId));
    minRow = std::min(minRow, slot.row);
    maxRow = std::max(maxRow, slot.row);
    minCol = std::min(minCol, slot.col);
    maxCol = std::max(maxCol, slot.col);
  }

  int64_t occupiedCoreCount = occupiedCores.count();
  int64_t boundingArea = (maxRow - minRow + 1) * (maxCol - minCol + 1);
  return boundingArea * 1000 + occupiedCoreCount * 100 +
         computeChainDistanceScore(state, orderedGroups);
}

static int64_t
getPhysicalArrayForCoreOrdinal(llvm::ArrayRef<PhysicalArraySlot> slots,
                               int64_t coreId, int64_t localOrdinal) {
  int64_t ordinal = 0;
  for (const PhysicalArraySlot &slot : slots) {
    if (slot.coreId != coreId)
      continue;
    if (ordinal == localOrdinal)
      return slot.physicalArrayId;
    ++ordinal;
  }
  return kUnassigned;
}

static int64_t getCoreId(int64_t row, int64_t col,
                         const task_schedulers::HardwareBudget &budget) {
  return row * budget.meshCols + col;
}

static bool areAdjacentCores(int64_t lhs, int64_t rhs,
                             const task_schedulers::HardwareBudget &budget) {
  int64_t lhsRow = lhs / budget.meshCols;
  int64_t lhsCol = lhs % budget.meshCols;
  int64_t rhsRow = rhs / budget.meshCols;
  int64_t rhsCol = rhs % budget.meshCols;
  return getAbsDiff(lhsRow, rhsRow) + getAbsDiff(lhsCol, rhsCol) == 1;
}

static bool isUniquePath(llvm::ArrayRef<llvm::SmallVector<int64_t, 8>> paths,
                         llvm::ArrayRef<int64_t> candidate) {
  for (llvm::ArrayRef<int64_t> path : paths) {
    if (path == candidate)
      return false;
  }
  return true;
}

static void
appendPathSlice(llvm::ArrayRef<int64_t> sequence, int64_t startCore,
                int64_t endCore, const task_schedulers::HardwareBudget &budget,
                unsigned minPathCapacity,
                llvm::SmallVectorImpl<llvm::SmallVector<int64_t, 8>> &paths) {
  int64_t startIndex = -1;
  int64_t endIndex = -1;
  for (auto indexedCore : llvm::enumerate(sequence)) {
    if (indexedCore.value() == startCore && startIndex < 0)
      startIndex = static_cast<int64_t>(indexedCore.index());
    if (indexedCore.value() == endCore && startIndex >= 0) {
      endIndex = static_cast<int64_t>(indexedCore.index());
      break;
    }
  }

  if (startIndex < 0 || endIndex < startIndex)
    return;

  llvm::SmallVector<int64_t, 8> path;
  for (int64_t index = startIndex; index <= endIndex; ++index) {
    if (!path.empty() &&
        !areAdjacentCores(path.back(), sequence[index], budget))
      return;
    path.push_back(sequence[index]);
  }

  if (path.empty() ||
      path.size() * static_cast<unsigned>(budget.arraysPerCore) <
          minPathCapacity)
    return;

  if (isUniquePath(paths, path))
    paths.push_back(std::move(path));
}

static void appendSnakePathCandidatesForRect(
    const MeshRegion &rect, int64_t startCore, int64_t endCore,
    const task_schedulers::HardwareBudget &budget, unsigned minPathCapacity,
    llvm::SmallVectorImpl<llvm::SmallVector<int64_t, 8>> &paths) {
  auto appendColumnSnake = [&](bool leftToRight, bool startAtTop) {
    llvm::SmallVector<int64_t, 16> sequence;
    for (int64_t colOffset = 0; colOffset < rect.cols(); ++colOffset) {
      int64_t col =
          leftToRight ? rect.colBegin + colOffset : rect.colEnd - 1 - colOffset;
      bool topToBottom = startAtTop == (colOffset % 2 == 0);
      if (topToBottom) {
        for (int64_t row = rect.rowBegin; row < rect.rowEnd; ++row)
          sequence.push_back(getCoreId(row, col, budget));
      } else {
        for (int64_t row = rect.rowEnd - 1; row >= rect.rowBegin; --row)
          sequence.push_back(getCoreId(row, col, budget));
      }
    }
    appendPathSlice(sequence, startCore, endCore, budget, minPathCapacity,
                    paths);
    std::reverse(sequence.begin(), sequence.end());
    appendPathSlice(sequence, startCore, endCore, budget, minPathCapacity,
                    paths);
  };

  auto appendRowSnake = [&](bool topToBottom, bool startAtLeft) {
    llvm::SmallVector<int64_t, 16> sequence;
    for (int64_t rowOffset = 0; rowOffset < rect.rows(); ++rowOffset) {
      int64_t row =
          topToBottom ? rect.rowBegin + rowOffset : rect.rowEnd - 1 - rowOffset;
      bool leftToRight = startAtLeft == (rowOffset % 2 == 0);
      if (leftToRight) {
        for (int64_t col = rect.colBegin; col < rect.colEnd; ++col)
          sequence.push_back(getCoreId(row, col, budget));
      } else {
        for (int64_t col = rect.colEnd - 1; col >= rect.colBegin; --col)
          sequence.push_back(getCoreId(row, col, budget));
      }
    }
    appendPathSlice(sequence, startCore, endCore, budget, minPathCapacity,
                    paths);
    std::reverse(sequence.begin(), sequence.end());
    appendPathSlice(sequence, startCore, endCore, budget, minPathCapacity,
                    paths);
  };

  appendColumnSnake(/*leftToRight=*/true, /*startAtTop=*/true);
  appendColumnSnake(/*leftToRight=*/true, /*startAtTop=*/false);
  appendColumnSnake(/*leftToRight=*/false, /*startAtTop=*/true);
  appendColumnSnake(/*leftToRight=*/false, /*startAtTop=*/false);
  appendRowSnake(/*topToBottom=*/true, /*startAtLeft=*/true);
  appendRowSnake(/*topToBottom=*/true, /*startAtLeft=*/false);
  appendRowSnake(/*topToBottom=*/false, /*startAtLeft=*/true);
  appendRowSnake(/*topToBottom=*/false, /*startAtLeft=*/false);
}

static llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 16>
buildCompactEndpointPaths(const BoundaryAwarePlacementState &state,
                          llvm::ArrayRef<unsigned> orderedGroups,
                          const task_schedulers::HardwareBudget &budget,
                          const EndpointBoundaryConstraint &constraint) {
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 16> paths;
  if (!constraint.isActive() || orderedGroups.empty())
    return paths;

  unsigned firstSlot = getSlotIndexForGroup(
      state, static_cast<unsigned>(constraint.firstEndpointGroup));
  if (firstSlot == std::numeric_limits<unsigned>::max())
    return paths;

  int64_t startCore = state.slots[firstSlot].coreId;
  int64_t startRow = state.slots[firstSlot].row;
  int64_t startCol = state.slots[firstSlot].col;
  unsigned maxPathCores = static_cast<unsigned>(
      std::min<int64_t>(budget.numCores, orderedGroups.size()));

  for (int64_t endCore = 0; endCore < budget.numCores; ++endCore) {
    int64_t endRow = endCore / budget.meshCols;
    int64_t endCol = endCore % budget.meshCols;
    PhysicalArraySlot endSlot;
    endSlot.coreId = endCore;
    endSlot.row = endRow;
    endSlot.col = endCol;
    if (!isSlotOnCorridor(endSlot, budget, constraint.corridor))
      continue;

    int64_t minRow = std::min(startRow, endRow);
    int64_t maxRow = std::max(startRow, endRow);
    int64_t minCol = std::min(startCol, endCol);
    int64_t maxCol = std::max(startCol, endCol);

    for (int64_t rowBegin = 0; rowBegin <= minRow; ++rowBegin) {
      for (int64_t rowEnd = maxRow + 1; rowEnd <= budget.meshRows; ++rowEnd) {
        for (int64_t colBegin = 0; colBegin <= minCol; ++colBegin) {
          for (int64_t colEnd = maxCol + 1; colEnd <= budget.meshCols;
               ++colEnd) {
            MeshRegion rect{rowBegin, rowEnd, colBegin, colEnd};
            if (static_cast<unsigned>(rect.coreCount()) > maxPathCores)
              continue;

            appendSnakePathCandidatesForRect(
                rect, startCore, endCore, budget,
                static_cast<unsigned>(orderedGroups.size()), paths);
          }
        }
      }
    }
  }

  return paths;
}

static mlir::FailureOr<BoundaryAwarePlacementState>
buildBestCompactPathAssignment(const BoundaryAwarePlacementState &state,
                               llvm::ArrayRef<WeightedGroupEdge> edges,
                               llvm::ArrayRef<unsigned> orderedGroups,
                               llvm::ArrayRef<int64_t> path,
                               const task_schedulers::HardwareBudget &budget,
                               const EndpointBoundaryConstraint &constraint) {
  constexpr unsigned kMaxCompactPathGroups = 16;
  if (orderedGroups.empty() || orderedGroups.size() > kMaxCompactPathGroups)
    return mlir::failure();

  llvm::SmallVector<unsigned, 16> positions(orderedGroups.size(), 0);
  llvm::SmallVector<unsigned, 8> pathCounts(path.size(), 0);
  BoundaryAwarePlacementState bestState;
  int64_t bestScore = kInfiniteScore;
  int64_t bestCompactScore = kInfiniteScore;
  bool hasBest = false;

  auto evaluateAssignment = [&]() {
    BoundaryAwarePlacementState candidate = state;
    llvm::SmallVector<int64_t, 8> localUseByCore(budget.numCores, 0);

    for (auto indexedGroup : llvm::enumerate(orderedGroups)) {
      unsigned group = indexedGroup.value();
      int64_t coreId = path[positions[indexedGroup.index()]];
      int64_t localOrdinal = localUseByCore[coreId]++;
      int64_t physicalArrayId =
          getPhysicalArrayForCoreOrdinal(state.slots, coreId, localOrdinal);
      if (physicalArrayId == kUnassigned)
        return;
      candidate.physicalArrayByGroup[group] = physicalArrayId;
    }

    if (!endpointBoundarySatisfied(candidate, budget, constraint))
      return;

    int64_t candidateScore = computeExactPlacementScore(candidate, edges);
    int64_t candidateCompactScore =
        computeCompactFootprintScore(candidate, budget, orderedGroups);
    if (candidateScore < bestScore ||
        (candidateScore == bestScore &&
         candidateCompactScore < bestCompactScore)) {
      bestState = std::move(candidate);
      bestScore = candidateScore;
      bestCompactScore = candidateCompactScore;
      hasBest = true;
    }
  };

  auto recurse = [&](auto &&self, unsigned groupIndex,
                     unsigned minimumPathIndex) -> void {
    if (groupIndex == orderedGroups.size()) {
      evaluateAssignment();
      return;
    }

    unsigned group = orderedGroups[groupIndex];
    unsigned firstPathIndex = minimumPathIndex;
    unsigned lastPathIndex = static_cast<unsigned>(path.size() - 1);
    if (static_cast<int64_t>(group) == constraint.firstEndpointGroup) {
      firstPathIndex = 0;
      lastPathIndex = 0;
    } else if (static_cast<int64_t>(group) == constraint.lastEndpointGroup) {
      firstPathIndex = static_cast<unsigned>(path.size() - 1);
      lastPathIndex = static_cast<unsigned>(path.size() - 1);
    }

    for (unsigned pathIndex = firstPathIndex; pathIndex <= lastPathIndex;
         ++pathIndex) {
      if (pathIndex < minimumPathIndex)
        continue;
      if (pathCounts[pathIndex] >= static_cast<unsigned>(budget.arraysPerCore))
        continue;

      positions[groupIndex] = pathIndex;
      ++pathCounts[pathIndex];
      self(self, groupIndex + 1, pathIndex);
      --pathCounts[pathIndex];
    }
  };

  recurse(recurse, /*groupIndex=*/0, /*minimumPathIndex=*/0);

  if (!hasBest)
    return mlir::failure();
  return bestState;
}

static void refinePlacementWithCompactChainPaths(
    BoundaryAwarePlacementState &state, llvm::ArrayRef<WeightedGroupEdge> edges,
    llvm::ArrayRef<unsigned> orderedGroups,
    const task_schedulers::HardwareBudget &budget,
    const EndpointBoundaryConstraint &constraint) {
  if (!constraint.isActive() || orderedGroups.size() < 2)
    return;

  int64_t currentScore = computeExactPlacementScore(state, edges);
  int64_t currentCompactScore =
      computeCompactFootprintScore(state, budget, orderedGroups);

  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 16> paths =
      buildCompactEndpointPaths(state, orderedGroups, budget, constraint);
  for (llvm::ArrayRef<int64_t> path : paths) {
    auto candidate = buildBestCompactPathAssignment(state, edges, orderedGroups,
                                                    path, budget, constraint);
    if (mlir::failed(candidate))
      continue;

    int64_t candidateScore = computeExactPlacementScore(*candidate, edges);
    if (candidateScore > currentScore)
      continue;

    int64_t candidateCompactScore =
        computeCompactFootprintScore(*candidate, budget, orderedGroups);
    if (candidateScore < currentScore ||
        candidateCompactScore < currentCompactScore) {
      state = std::move(*candidate);
      currentScore = candidateScore;
      currentCompactScore = candidateCompactScore;
    }
  }
}

static void
refinePlacementWithHotEdgeMoves(BoundaryAwarePlacementState &state,
                                llvm::ArrayRef<WeightedGroupEdge> edges,
                                const task_schedulers::HardwareBudget &budget,
                                const EndpointBoundaryConstraint &constraint) {
  constexpr unsigned kHotEdgeMovePasses = 4;
  if (edges.empty())
    return;

  for (unsigned pass = 0; pass < kHotEdgeMovePasses; ++pass) {
    llvm::SmallVector<int64_t, 8> groupBySlot = buildSlotOccupancy(state);
    if (!llvm::is_contained(groupBySlot, kUnassigned))
      return;

    llvm::SmallVector<WeightedGroupEdge, 16> hotEdges(edges.begin(),
                                                      edges.end());
    llvm::sort(hotEdges,
               [&](const WeightedGroupEdge &lhs, const WeightedGroupEdge &rhs) {
                 int64_t lhsContribution = computeEdgeContribution(state, lhs);
                 int64_t rhsContribution = computeEdgeContribution(state, rhs);
                 if (lhsContribution != rhsContribution)
                   return lhsContribution > rhsContribution;
                 if (lhs.totalBytes != rhs.totalBytes)
                   return lhs.totalBytes > rhs.totalBytes;
                 if (lhs.groupA != rhs.groupA)
                   return lhs.groupA < rhs.groupA;
                 return lhs.groupB < rhs.groupB;
               });

    int64_t currentScore = computeExactPlacementScore(state, edges);
    int64_t bestScore = currentScore;
    unsigned bestGroup = std::numeric_limits<unsigned>::max();
    unsigned bestSlot = std::numeric_limits<unsigned>::max();

    for (const WeightedGroupEdge &edge : hotEdges) {
      int64_t edgeContribution = computeEdgeContribution(state, edge);
      if (edgeContribution <= 0 || edgeContribution >= kInfiniteScore)
        continue;

      unsigned endpoints[2] = {edge.groupA, edge.groupB};
      for (unsigned endpointIndex = 0; endpointIndex < 2; ++endpointIndex) {
        unsigned moveGroup = endpoints[endpointIndex];
        unsigned anchorGroup = endpoints[1 - endpointIndex];
        unsigned anchorSlotIndex = getSlotIndexForGroup(state, anchorGroup);
        unsigned moveSlotIndex = getSlotIndexForGroup(state, moveGroup);
        if (anchorSlotIndex == std::numeric_limits<unsigned>::max() ||
            moveSlotIndex == std::numeric_limits<unsigned>::max())
          continue;

        int64_t currentDistance = getMeshDistance(state.slots[moveSlotIndex],
                                                  state.slots[anchorSlotIndex]);
        int64_t originalPhysicalArray = state.physicalArrayByGroup[moveGroup];

        for (auto indexedSlot : llvm::enumerate(state.slots)) {
          unsigned candidateSlotIndex =
              static_cast<unsigned>(indexedSlot.index());
          if (groupBySlot[candidateSlotIndex] != kUnassigned)
            continue;
          if (constraint.isEndpointGroup(moveGroup) &&
              !isSlotOnCorridor(indexedSlot.value(), budget,
                                constraint.corridor))
            continue;

          int64_t candidateDistance = getMeshDistance(
              indexedSlot.value(), state.slots[anchorSlotIndex]);
          if (candidateDistance > currentDistance)
            continue;

          state.physicalArrayByGroup[moveGroup] =
              indexedSlot.value().physicalArrayId;
          if (!endpointBoundarySatisfied(state, budget, constraint)) {
            state.physicalArrayByGroup[moveGroup] = originalPhysicalArray;
            continue;
          }
          int64_t candidateScore = computeExactPlacementScore(state, edges);
          state.physicalArrayByGroup[moveGroup] = originalPhysicalArray;

          if (candidateScore < bestScore) {
            bestScore = candidateScore;
            bestGroup = moveGroup;
            bestSlot = candidateSlotIndex;
          }
        }
      }
    }

    if (bestGroup == std::numeric_limits<unsigned>::max())
      return;

    state.physicalArrayByGroup[bestGroup] =
        state.slots[bestSlot].physicalArrayId;
  }
}

static void appendUniqueCut(llvm::SmallVectorImpl<int64_t> &cuts, int64_t cut,
                            int64_t begin, int64_t end) {
  if (cut <= begin || cut >= end)
    return;
  if (llvm::is_contained(cuts, cut))
    return;
  cuts.push_back(cut);
}

static llvm::SmallVector<RegionSplit, 4>
buildCandidateSplits(const MeshRegion &region) {
  llvm::SmallVector<RegionSplit, 4> splits;

  auto appendVerticalSplits = [&]() {
    llvm::SmallVector<int64_t, 4> cuts;
    int64_t midpoint = region.colBegin + region.cols() / 2;
    appendUniqueCut(cuts, midpoint, region.colBegin, region.colEnd);
    appendUniqueCut(cuts, midpoint - 1, region.colBegin, region.colEnd);
    appendUniqueCut(cuts, midpoint + 1, region.colBegin, region.colEnd);

    for (int64_t cut : cuts) {
      RegionSplit split;
      split.vertical = true;
      split.cutLine = cut;
      split.first =
          MeshRegion{region.rowBegin, region.rowEnd, region.colBegin, cut};
      split.second =
          MeshRegion{region.rowBegin, region.rowEnd, cut, region.colEnd};
      splits.push_back(split);
    }
  };

  auto appendHorizontalSplits = [&]() {
    llvm::SmallVector<int64_t, 4> cuts;
    int64_t midpoint = region.rowBegin + region.rows() / 2;
    appendUniqueCut(cuts, midpoint, region.rowBegin, region.rowEnd);
    appendUniqueCut(cuts, midpoint - 1, region.rowBegin, region.rowEnd);
    appendUniqueCut(cuts, midpoint + 1, region.rowBegin, region.rowEnd);

    for (int64_t cut : cuts) {
      RegionSplit split;
      split.vertical = false;
      split.cutLine = cut;
      split.first =
          MeshRegion{region.rowBegin, cut, region.colBegin, region.colEnd};
      split.second =
          MeshRegion{cut, region.rowEnd, region.colBegin, region.colEnd};
      splits.push_back(split);
    }
  };

  if (region.cols() >= region.rows()) {
    if (region.cols() > 1)
      appendVerticalSplits();
    if (region.rows() > 1)
      appendHorizontalSplits();
  } else {
    if (region.rows() > 1)
      appendHorizontalSplits();
    if (region.cols() > 1)
      appendVerticalSplits();
  }

  return splits;
}

static int64_t clampTarget(int64_t ideal, int64_t lower, int64_t upper) {
  return std::max(lower, std::min(upper, ideal));
}

static int64_t computeConnectionToSet(
    unsigned group,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
    const llvm::SmallBitVector &set) {
  int64_t connection = 0;
  for (const GroupAdjacency &edge : adjacency[group]) {
    if (set.test(edge.neighborGroup))
      connection += edge.totalBytes;
  }
  return connection;
}

static unsigned selectSeedGroupForRegion(
    llvm::ArrayRef<unsigned> groups, const llvm::SmallBitVector &assigned,
    const llvm::SmallBitVector &targetSet, const MeshRegion &region,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<int64_t> incidentWeights,
    const EndpointBoundaryConstraint &boundaryConstraint) {
  unsigned selected = std::numeric_limits<unsigned>::max();
  int64_t selectedWeight = std::numeric_limits<int64_t>::min();

  for (unsigned group : groups) {
    if (assigned.test(group))
      continue;
    if (!canAddGroupToBoundaryRegion(group, targetSet, region, budget,
                                     boundaryConstraint))
      continue;

    if (selected == std::numeric_limits<unsigned>::max() ||
        incidentWeights[group] > selectedWeight ||
        (incidentWeights[group] == selectedWeight && group < selected)) {
      selected = group;
      selectedWeight = incidentWeights[group];
    }
  }

  return selected;
}

static unsigned selectOppositeSeedForRegion(
    unsigned seed, llvm::ArrayRef<unsigned> groups,
    const llvm::SmallBitVector &assigned, const llvm::SmallBitVector &targetSet,
    const MeshRegion &region, const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
    llvm::ArrayRef<int64_t> incidentWeights,
    const EndpointBoundaryConstraint &boundaryConstraint) {
  unsigned selected = std::numeric_limits<unsigned>::max();
  int64_t selectedConnection = std::numeric_limits<int64_t>::max();
  int64_t selectedIncident = std::numeric_limits<int64_t>::min();

  for (unsigned group : groups) {
    if (group == seed || assigned.test(group))
      continue;
    if (!canAddGroupToBoundaryRegion(group, targetSet, region, budget,
                                     boundaryConstraint))
      continue;

    int64_t connection = 0;
    for (const GroupAdjacency &edge : adjacency[seed]) {
      if (edge.neighborGroup == group)
        connection += edge.totalBytes;
    }

    if (connection < selectedConnection ||
        (connection == selectedConnection &&
         incidentWeights[group] > selectedIncident) ||
        (connection == selectedConnection &&
         incidentWeights[group] == selectedIncident && group < selected)) {
      selected = group;
      selectedConnection = connection;
      selectedIncident = incidentWeights[group];
    }
  }

  return selected;
}

static int64_t computeCutWeight(llvm::ArrayRef<WeightedGroupEdge> edges,
                                const llvm::SmallBitVector &inFirst,
                                const llvm::SmallBitVector &inRegion) {
  int64_t cutWeight = 0;
  for (const WeightedGroupEdge &edge : edges) {
    if (!inRegion.test(edge.groupA) || !inRegion.test(edge.groupB))
      continue;
    if (inFirst.test(edge.groupA) != inFirst.test(edge.groupB))
      cutWeight += edge.totalBytes;
  }
  return cutWeight;
}

static PartitionResult partitionGroupsForSplit(
    llvm::ArrayRef<unsigned> groups, const RegionSplit &split,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<WeightedGroupEdge> edges,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
    llvm::ArrayRef<int64_t> incidentWeights,
    const EndpointBoundaryConstraint &boundaryConstraint) {
  PartitionResult result;
  if (groups.empty()) {
    result.score = 0;
    return result;
  }

  int64_t capacityFirst = getRegionCapacity(split.first, budget);
  int64_t capacitySecond = getRegionCapacity(split.second, budget);
  int64_t numGroups = static_cast<int64_t>(groups.size());
  if (numGroups > capacityFirst + capacitySecond)
    return result;

  int64_t lowerFirst = std::max<int64_t>(0, numGroups - capacitySecond);
  int64_t upperFirst = std::min(numGroups, capacityFirst);
  if (lowerFirst > upperFirst)
    return result;

  int64_t totalCapacity = capacityFirst + capacitySecond;
  int64_t idealFirst =
      (numGroups * capacityFirst + totalCapacity / 2) / totalCapacity;
  int64_t targetFirst = clampTarget(idealFirst, lowerFirst, upperFirst);
  int64_t targetSecond = numGroups - targetFirst;

  llvm::SmallBitVector inRegion(adjacency.size(), false);
  llvm::SmallBitVector assigned(adjacency.size(), false);
  llvm::SmallBitVector inFirst(adjacency.size(), false);
  llvm::SmallBitVector inSecond(adjacency.size(), false);
  for (unsigned group : groups)
    inRegion.set(group);

  if (targetFirst > 0) {
    unsigned seedFirst =
        selectSeedGroupForRegion(groups, assigned, inFirst, split.first, budget,
                                 incidentWeights, boundaryConstraint);
    if (seedFirst == std::numeric_limits<unsigned>::max())
      return PartitionResult();
    result.firstGroups.push_back(seedFirst);
    assigned.set(seedFirst);
    inFirst.set(seedFirst);
  }

  if (targetSecond > 0) {
    unsigned seedSecond =
        targetFirst > 0
            ? selectOppositeSeedForRegion(result.firstGroups.front(), groups,
                                          assigned, inSecond, split.second,
                                          budget, adjacency, incidentWeights,
                                          boundaryConstraint)
            : selectSeedGroupForRegion(groups, assigned, inSecond, split.second,
                                       budget, incidentWeights,
                                       boundaryConstraint);
    if (seedSecond == std::numeric_limits<unsigned>::max())
      return PartitionResult();
    result.secondGroups.push_back(seedSecond);
    assigned.set(seedSecond);
    inSecond.set(seedSecond);
  }

  while (static_cast<int64_t>(result.firstGroups.size()) < targetFirst ||
         static_cast<int64_t>(result.secondGroups.size()) < targetSecond) {
    unsigned selectedGroup = std::numeric_limits<unsigned>::max();
    bool assignFirst = false;
    int64_t selectedScore = std::numeric_limits<int64_t>::min();
    int64_t selectedIncident = std::numeric_limits<int64_t>::min();

    for (unsigned group : groups) {
      if (assigned.test(group))
        continue;

      int64_t connectionFirst =
          computeConnectionToSet(group, adjacency, inFirst);
      int64_t connectionSecond =
          computeConnectionToSet(group, adjacency, inSecond);

      bool canAssignFirst =
          static_cast<int64_t>(result.firstGroups.size()) < targetFirst &&
          canAddGroupToBoundaryRegion(group, inFirst, split.first, budget,
                                      boundaryConstraint);
      bool canAssignSecond =
          static_cast<int64_t>(result.secondGroups.size()) < targetSecond &&
          canAddGroupToBoundaryRegion(group, inSecond, split.second, budget,
                                      boundaryConstraint);

      auto consider = [&](bool toFirst, int64_t score) {
        if (selectedGroup == std::numeric_limits<unsigned>::max() ||
            score > selectedScore ||
            (score == selectedScore &&
             incidentWeights[group] > selectedIncident) ||
            (score == selectedScore &&
             incidentWeights[group] == selectedIncident &&
             group < selectedGroup)) {
          selectedGroup = group;
          assignFirst = toFirst;
          selectedScore = score;
          selectedIncident = incidentWeights[group];
        }
      };

      if (canAssignFirst)
        consider(/*toFirst=*/true, connectionFirst - connectionSecond);
      if (canAssignSecond)
        consider(/*toFirst=*/false, connectionSecond - connectionFirst);
    }

    if (selectedGroup == std::numeric_limits<unsigned>::max())
      break;

    assigned.set(selectedGroup);
    if (assignFirst) {
      result.firstGroups.push_back(selectedGroup);
      inFirst.set(selectedGroup);
    } else {
      result.secondGroups.push_back(selectedGroup);
      inSecond.set(selectedGroup);
    }
  }

  if (static_cast<int64_t>(result.firstGroups.size()) != targetFirst ||
      static_cast<int64_t>(result.secondGroups.size()) != targetSecond)
    return PartitionResult();

  if (!hasEndpointBoundaryCapacity(result.firstGroups, split.first, budget,
                                   boundaryConstraint) ||
      !hasEndpointBoundaryCapacity(result.secondGroups, split.second, budget,
                                   boundaryConstraint))
    return PartitionResult();

  result.cutWeight = computeCutWeight(edges, inFirst, inRegion);
  int64_t balanceWaste =
      (capacityFirst - static_cast<int64_t>(result.firstGroups.size())) +
      (capacitySecond - static_cast<int64_t>(result.secondGroups.size()));
  result.score = result.cutWeight + balanceWaste;
  llvm::sort(result.firstGroups);
  llvm::sort(result.secondGroups);
  return result;
}

static bool isBetterCandidateSplit(const CandidateSplit &candidate,
                                   const CandidateSplit &best, bool hasBest) {
  if (!hasBest)
    return true;
  if (candidate.partition.score != best.partition.score)
    return candidate.partition.score < best.partition.score;
  if (candidate.partition.cutWeight != best.partition.cutWeight)
    return candidate.partition.cutWeight < best.partition.cutWeight;
  if (candidate.split.vertical != best.split.vertical)
    return candidate.split.vertical;
  return candidate.split.cutLine < best.split.cutLine;
}

static mlir::FailureOr<CandidateSplit>
chooseBestSplit(mlir::Operation *diagnosticOp, llvm::ArrayRef<unsigned> groups,
                const MeshRegion &region,
                const task_schedulers::HardwareBudget &budget,
                llvm::ArrayRef<WeightedGroupEdge> edges,
                llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
                llvm::ArrayRef<int64_t> incidentWeights,
                const EndpointBoundaryConstraint &boundaryConstraint) {
  (void)diagnosticOp;

  llvm::SmallVector<RegionSplit, 4> splits = buildCandidateSplits(region);
  CandidateSplit best;
  bool hasBest = false;

  for (const RegionSplit &split : splits) {
    PartitionResult partition =
        partitionGroupsForSplit(groups, split, budget, edges, adjacency,
                                incidentWeights, boundaryConstraint);
    if (partition.score >= kInfiniteScore)
      continue;

    CandidateSplit candidate{split, std::move(partition)};
    if (isBetterCandidateSplit(candidate, best, hasBest)) {
      best = std::move(candidate);
      hasBest = true;
    }
  }

  if (!hasBest)
    return mlir::failure();

  return best;
}

static mlir::LogicalResult
assignLeafGroups(mlir::Operation *diagnosticOp, llvm::ArrayRef<unsigned> groups,
                 const MeshRegion &region,
                 const task_schedulers::HardwareBudget &budget,
                 llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
                 llvm::ArrayRef<int64_t> incidentWeights,
                 const EndpointBoundaryConstraint &boundaryConstraint,
                 BoundaryAwarePlacementState &state) {
  llvm::SmallVector<unsigned, 8> slotIndices =
      collectRegionSlotIndices(state.slots, region);
  if (groups.size() > slotIndices.size()) {
    diagnosticOp->emitError("expected leaf region capacity to fit assigned "
                            "matrix setup groups");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 8> sortedGroups(groups.begin(), groups.end());
  llvm::sort(sortedGroups, [&](unsigned lhs, unsigned rhs) {
    bool lhsEndpoint = boundaryConstraint.isEndpointGroup(lhs);
    bool rhsEndpoint = boundaryConstraint.isEndpointGroup(rhs);
    if (lhsEndpoint != rhsEndpoint)
      return lhsEndpoint;
    if (incidentWeights[lhs] != incidentWeights[rhs])
      return incidentWeights[lhs] > incidentWeights[rhs];
    return lhs < rhs;
  });

  llvm::SmallBitVector usedSlots = collectUsedSlots(state);
  for (unsigned group : sortedGroups) {
    unsigned bestSlotIndex = std::numeric_limits<unsigned>::max();
    int64_t bestCost = kInfiniteScore;
    for (unsigned slotIndex : slotIndices) {
      if (usedSlots.test(slotIndex))
        continue;
      if (boundaryConstraint.isEndpointGroup(group) &&
          !isSlotOnCorridor(state.slots[slotIndex], budget,
                            boundaryConstraint.corridor))
        continue;

      int64_t candidateCost =
          computeIncrementalPlacementCost(group, slotIndex, adjacency, state);
      if (isBetterSlotCandidate(state, slotIndex, candidateCost, bestSlotIndex,
                                bestCost)) {
        bestSlotIndex = slotIndex;
        bestCost = candidateCost;
      }
    }

    if (bestSlotIndex == std::numeric_limits<unsigned>::max()) {
      diagnosticOp->emitError("failed to find an unused analog array slot for "
                              "boundary-aware leaf placement");
      return mlir::failure();
    }

    usedSlots.set(bestSlotIndex);
    state.physicalArrayByGroup[group] =
        state.slots[bestSlotIndex].physicalArrayId;
  }

  return mlir::success();
}

static mlir::LogicalResult
placeRegion(mlir::Operation *diagnosticOp, llvm::ArrayRef<unsigned> groups,
            const MeshRegion &region,
            const task_schedulers::HardwareBudget &budget,
            llvm::ArrayRef<WeightedGroupEdge> edges,
            llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
            llvm::ArrayRef<int64_t> incidentWeights,
            const EndpointBoundaryConstraint &boundaryConstraint,
            BoundaryAwarePlacementState &state) {
  if (groups.empty())
    return mlir::success();

  int64_t regionCapacity = getRegionCapacity(region, budget);
  if (static_cast<int64_t>(groups.size()) > regionCapacity) {
    diagnosticOp->emitError("expected boundary-aware region capacity to fit "
                            "assigned matrix setup groups");
    return mlir::failure();
  }

  if (region.isSingleCore() ||
      static_cast<int64_t>(groups.size()) <= budget.arraysPerCore)
    return assignLeafGroups(diagnosticOp, groups, region, budget, adjacency,
                            incidentWeights, boundaryConstraint, state);

  auto split = chooseBestSplit(diagnosticOp, groups, region, budget, edges,
                               adjacency, incidentWeights, boundaryConstraint);
  if (mlir::failed(split))
    return mlir::failure();

  if (mlir::failed(placeRegion(diagnosticOp, split->partition.firstGroups,
                               split->split.first, budget, edges, adjacency,
                               incidentWeights, boundaryConstraint, state)))
    return mlir::failure();

  return placeRegion(diagnosticOp, split->partition.secondGroups,
                     split->split.second, budget, edges, adjacency,
                     incidentWeights, boundaryConstraint, state);
}

static mlir::FailureOr<BoundaryAwarePlacementState>
buildPlacement(mlir::Operation *diagnosticOp, unsigned numGroups,
               const task_schedulers::HardwareBudget &budget,
               llvm::ArrayRef<WeightedGroupEdge> edges,
               llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
               llvm::ArrayRef<int64_t> incidentWeights,
               const EndpointBoundaryConstraint &boundaryConstraint) {
  auto slots = buildPhysicalArraySlots(diagnosticOp, budget);
  if (mlir::failed(slots))
    return mlir::failure();

  if (static_cast<int64_t>(numGroups) > budget.numAnalogArrays) {
    diagnosticOp->emitError(
        "expected boundary-aware-cut-optimized scheduler to have "
        "enough physical analog arrays for matrix setup "
        "groups");
    return mlir::failure();
  }

  BoundaryAwarePlacementState state;
  state.physicalArrayByGroup.assign(numGroups, kUnassigned);
  state.slots = std::move(*slots);
  for (auto indexedSlot : llvm::enumerate(state.slots)) {
    state.slotIndexByPhysicalArrayId.try_emplace(
        indexedSlot.value().physicalArrayId,
        static_cast<unsigned>(indexedSlot.index()));
  }

  llvm::SmallVector<unsigned, 8> allGroups;
  allGroups.reserve(numGroups);
  for (unsigned group = 0; group < numGroups; ++group)
    allGroups.push_back(group);

  MeshRegion root{0, budget.meshRows, 0, budget.meshCols};
  if (mlir::failed(placeRegion(diagnosticOp, allGroups, root, budget, edges,
                               adjacency, incidentWeights, boundaryConstraint,
                               state)))
    return mlir::failure();

  if (!endpointBoundarySatisfied(state, budget, boundaryConstraint))
    return mlir::failure();

  refinePlacementWithPairSwaps(state, edges, budget, boundaryConstraint);
  refinePlacementWithHotEdgeMoves(state, edges, budget, boundaryConstraint);
  refinePlacementWithCompactChainPaths(state, edges, allGroups, budget,
                                       boundaryConstraint);

  return state;
}

static mlir::FailureOr<BoundaryAwarePlacementState> buildBestPlacement(
    mlir::Operation *diagnosticOp, unsigned numGroups,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<WeightedGroupEdge> edges,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
    llvm::ArrayRef<int64_t> incidentWeights, int64_t firstEndpointGroup,
    int64_t lastEndpointGroup) {
  if (firstEndpointGroup == kUnassigned || lastEndpointGroup == kUnassigned) {
    EndpointBoundaryConstraint unconstrained;
    return buildPlacement(diagnosticOp, numGroups, budget, edges, adjacency,
                          incidentWeights, unconstrained);
  }

  BoundaryAwarePlacementState bestState;
  int64_t bestScore = kInfiniteScore;
  bool hasBest = false;
  BoundaryCorridor corridors[] = {
      BoundaryCorridor::Top, BoundaryCorridor::Bottom, BoundaryCorridor::Left,
      BoundaryCorridor::Right};
  for (BoundaryCorridor corridor : corridors) {
    EndpointBoundaryConstraint constraint;
    constraint.corridor = corridor;
    constraint.firstEndpointGroup = firstEndpointGroup;
    constraint.lastEndpointGroup = lastEndpointGroup;

    auto candidate = buildPlacement(diagnosticOp, numGroups, budget, edges,
                                    adjacency, incidentWeights, constraint);
    if (mlir::failed(candidate))
      continue;

    int64_t candidateScore = computeExactPlacementScore(*candidate, edges);
    if (!hasBest || candidateScore < bestScore) {
      bestState = std::move(*candidate);
      bestScore = candidateScore;
      hasBest = true;
    }
  }

  if (!hasBest) {
    diagnosticOp->emitError("failed to find a feasible boundary corridor "
                            "placement for boundary-aware-cut-optimized");
    return mlir::failure();
  }

  return bestState;
}

class BoundaryAwareCutOptimizedTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final {
    return "boundary-aware-cut-optimized";
  }

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

    llvm::SmallVector<llvm::SmallVector<GroupAdjacency, 4>, 8> adjacency =
        buildGroupAdjacency(static_cast<unsigned>(groups.size()), *edges);
    llvm::SmallVector<int64_t, 8> incidentWeights =
        computeIncidentWeights(adjacency);

    int64_t firstEndpointGroup = kUnassigned;
    int64_t lastEndpointGroup = kUnassigned;
    if (!dag.nodes.empty()) {
      firstEndpointGroup = (*taskGroupOwnership)[dag.nodes.front().index];
      lastEndpointGroup = (*taskGroupOwnership)[dag.nodes.back().index];
    }

    auto placement = buildBestPlacement(
        taskGraphFunc.getOperation(), static_cast<unsigned>(groups.size()),
        budget, *edges, adjacency, incidentWeights, firstEndpointGroup,
        lastEndpointGroup);
    if (mlir::failed(placement))
      return mlir::failure();

    llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>
        groupPlacements;
    groupPlacements.reserve(groups.size());
    for (const PlacementGroup &group : groups) {
      int64_t physicalArrayId = placement->physicalArrayByGroup[group.index];
      if (physicalArrayId == kUnassigned) {
        taskGraphFunc.emitError(
            "expected boundary-aware-cut-optimized scheduler to "
            "place every matrix setup group");
        return mlir::failure();
      }

      groupPlacements.push_back(task_schedulers::MatrixSetupGroupPlacement{
          group.matrixSetupTaskIndex, physicalArrayId});
    }

    return task_schedulers::placeMatrixSetupGroupsAndSurroundingTasks(
        module, taskGraphFunc, budget, dag, groupPlacements);
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerBoundaryAwareCutOptimizedTaskScheduler(
    TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(
      registry, std::make_unique<BoundaryAwareCutOptimizedTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
