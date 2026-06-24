#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
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
constexpr int64_t kInfinity = std::numeric_limits<int64_t>::max() / 4;

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

struct CutCoordinates {
  llvm::SmallVector<int64_t, 8> phiX;
  llvm::SmallVector<int64_t, 8> phiY;
};

struct CandidatePlacement {
  llvm::SmallVector<int64_t, 8> physicalArrayByGroup;
  int64_t estimatedScore = kInfinity;
};

enum class ProjectionKind { XFirst, YFirst };

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

  return slots;
}

static int64_t getMaxEdgeWeight(
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency) {
  int64_t maxWeight = 1;
  for (const llvm::SmallVector<GroupAdjacency, 4> &edges : adjacency) {
    for (const GroupAdjacency &edge : edges)
      maxWeight = std::max(maxWeight, edge.totalBytes);
  }
  return maxWeight;
}

static llvm::SmallVector<int64_t, 8> computeWeightedDistances(
    unsigned source,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency) {
  llvm::SmallVector<int64_t, 8> distances(adjacency.size(), kInfinity);
  llvm::SmallVector<bool, 8> visited(adjacency.size(), false);
  if (source >= adjacency.size())
    return distances;

  distances[source] = 0;
  int64_t maxWeight = getMaxEdgeWeight(adjacency);

  for (unsigned iteration = 0; iteration < adjacency.size(); ++iteration) {
    unsigned selected = std::numeric_limits<unsigned>::max();
    int64_t selectedDistance = kInfinity;
    for (unsigned node = 0; node < adjacency.size(); ++node) {
      if (visited[node] || distances[node] >= selectedDistance)
        continue;
      selected = node;
      selectedDistance = distances[node];
    }

    if (selected == std::numeric_limits<unsigned>::max())
      break;

    visited[selected] = true;
    for (const GroupAdjacency &edge : adjacency[selected]) {
      if (visited[edge.neighborGroup])
        continue;

      int64_t edgeLength = std::max<int64_t>(1, maxWeight / edge.totalBytes);
      if (distances[selected] + edgeLength < distances[edge.neighborGroup])
        distances[edge.neighborGroup] = distances[selected] + edgeLength;
    }
  }

  return distances;
}

static unsigned findFarthestFiniteNode(llvm::ArrayRef<int64_t> distances) {
  unsigned selected = 0;
  int64_t selectedDistance = std::numeric_limits<int64_t>::min();

  for (auto indexedDistance : llvm::enumerate(distances)) {
    int64_t distance = indexedDistance.value();
    if (distance >= kInfinity)
      continue;

    if (distance > selectedDistance) {
      selected = static_cast<unsigned>(indexedDistance.index());
      selectedDistance = distance;
    }
  }

  return selected;
}

static unsigned findHighestIncidentGroup(llvm::ArrayRef<int64_t> weights) {
  unsigned selected = 0;
  int64_t selectedWeight = std::numeric_limits<int64_t>::min();
  for (auto indexedWeight : llvm::enumerate(weights)) {
    if (indexedWeight.value() > selectedWeight) {
      selected = static_cast<unsigned>(indexedWeight.index());
      selectedWeight = indexedWeight.value();
    }
  }
  return selected;
}

static int64_t fallbackDistance(int64_t maxFiniteDistance, unsigned groupIndex,
                                unsigned numGroups) {
  return maxFiniteDistance + static_cast<int64_t>(groupIndex) +
         static_cast<int64_t>(numGroups) + 1;
}

static llvm::SmallVector<int64_t, 8>
buildCoordinateFromAnchors(llvm::ArrayRef<int64_t> distancesA,
                           llvm::ArrayRef<int64_t> distancesB) {
  llvm::SmallVector<int64_t, 8> coordinates;
  coordinates.reserve(distancesA.size());

  int64_t maxFiniteDistance = 0;
  for (int64_t distance : distancesA) {
    if (distance < kInfinity)
      maxFiniteDistance = std::max(maxFiniteDistance, distance);
  }
  for (int64_t distance : distancesB) {
    if (distance < kInfinity)
      maxFiniteDistance = std::max(maxFiniteDistance, distance);
  }

  unsigned numGroups = static_cast<unsigned>(distancesA.size());
  for (unsigned groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
    int64_t distanceA =
        distancesA[groupIndex] < kInfinity
            ? distancesA[groupIndex]
            : fallbackDistance(maxFiniteDistance, groupIndex, numGroups);
    int64_t distanceB =
        distancesB[groupIndex] < kInfinity
            ? distancesB[groupIndex]
            : fallbackDistance(maxFiniteDistance, numGroups - groupIndex,
                               numGroups);
    coordinates.push_back(distanceA - distanceB);
  }

  return coordinates;
}

static CutCoordinates computeCutCoordinates(
    unsigned numGroups,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
    llvm::ArrayRef<int64_t> incidentWeights) {
  CutCoordinates coordinates;
  if (numGroups == 0)
    return coordinates;
  if (numGroups == 1) {
    coordinates.phiX.push_back(0);
    coordinates.phiY.push_back(0);
    return coordinates;
  }

  unsigned xAnchorA = 0;
  llvm::SmallVector<int64_t, 8> xAnchorADistances =
      computeWeightedDistances(xAnchorA, adjacency);
  unsigned xAnchorB = findFarthestFiniteNode(xAnchorADistances);
  llvm::SmallVector<int64_t, 8> xAnchorBDistances =
      computeWeightedDistances(xAnchorB, adjacency);
  coordinates.phiX =
      buildCoordinateFromAnchors(xAnchorADistances, xAnchorBDistances);

  unsigned yAnchorA = findHighestIncidentGroup(incidentWeights);
  llvm::SmallVector<int64_t, 8> yAnchorADistances =
      computeWeightedDistances(yAnchorA, adjacency);
  unsigned yAnchorB = findFarthestFiniteNode(yAnchorADistances);
  if (yAnchorB == yAnchorA && numGroups > 1)
    yAnchorB = numGroups - 1 - yAnchorA;

  llvm::SmallVector<int64_t, 8> yAnchorBDistances =
      computeWeightedDistances(yAnchorB, adjacency);
  coordinates.phiY =
      buildCoordinateFromAnchors(yAnchorADistances, yAnchorBDistances);

  return coordinates;
}

static bool compareCoordinate(unsigned lhs, unsigned rhs,
                              llvm::ArrayRef<int64_t> primary,
                              llvm::ArrayRef<int64_t> secondary,
                              bool reversePrimary, bool reverseSecondary) {
  if (primary[lhs] != primary[rhs])
    return reversePrimary ? primary[lhs] > primary[rhs]
                          : primary[lhs] < primary[rhs];
  if (secondary[lhs] != secondary[rhs])
    return reverseSecondary ? secondary[lhs] > secondary[rhs]
                            : secondary[lhs] < secondary[rhs];
  return lhs < rhs;
}

static llvm::SmallVector<unsigned, 8>
buildSortedGroups(unsigned numGroups, llvm::ArrayRef<int64_t> primary,
                  llvm::ArrayRef<int64_t> secondary, bool reversePrimary,
                  bool reverseSecondary) {
  llvm::SmallVector<unsigned, 8> groups;
  groups.reserve(numGroups);
  for (unsigned groupIndex = 0; groupIndex < numGroups; ++groupIndex)
    groups.push_back(groupIndex);

  llvm::sort(groups, [&](unsigned lhs, unsigned rhs) {
    return compareCoordinate(lhs, rhs, primary, secondary, reversePrimary,
                             reverseSecondary);
  });
  return groups;
}

static llvm::SmallVector<unsigned, 8>
sortSlotsForProjection(llvm::ArrayRef<PhysicalArraySlot> slots,
                       ProjectionKind projectionKind) {
  llvm::SmallVector<unsigned, 8> slotIndices;
  slotIndices.reserve(slots.size());
  for (unsigned slotIndex = 0; slotIndex < slots.size(); ++slotIndex)
    slotIndices.push_back(slotIndex);

  llvm::sort(slotIndices, [&](unsigned lhs, unsigned rhs) {
    const PhysicalArraySlot &lhsSlot = slots[lhs];
    const PhysicalArraySlot &rhsSlot = slots[rhs];
    if (projectionKind == ProjectionKind::XFirst) {
      if (lhsSlot.col != rhsSlot.col)
        return lhsSlot.col < rhsSlot.col;
      if (lhsSlot.row != rhsSlot.row)
        return lhsSlot.row < rhsSlot.row;
    } else {
      if (lhsSlot.row != rhsSlot.row)
        return lhsSlot.row < rhsSlot.row;
      if (lhsSlot.col != rhsSlot.col)
        return lhsSlot.col < rhsSlot.col;
    }
    if (lhsSlot.localArrayId != rhsSlot.localArrayId)
      return lhsSlot.localArrayId < rhsSlot.localArrayId;
    return lhsSlot.physicalArrayId < rhsSlot.physicalArrayId;
  });

  return slotIndices;
}

static llvm::SmallVector<int64_t, 8>
computeSliceCapacities(unsigned numGroups, int64_t numSlices,
                       int64_t maxSliceCapacity) {
  llvm::SmallVector<int64_t, 8> capacities;
  capacities.reserve(static_cast<size_t>(numSlices));

  int64_t remainingGroups = numGroups;
  for (int64_t slice = 0; slice < numSlices; ++slice) {
    int64_t remainingSlices = numSlices - slice;
    int64_t capacity = 0;
    if (remainingGroups > 0)
      capacity =
          std::min(maxSliceCapacity,
                   (remainingGroups + remainingSlices - 1) / remainingSlices);
    capacities.push_back(capacity);
    remainingGroups -= capacity;
  }

  return capacities;
}

static llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 8>
groupSlotsBySlice(llvm::ArrayRef<PhysicalArraySlot> slots,
                  ProjectionKind projectionKind) {
  int64_t numSlices = 0;
  for (const PhysicalArraySlot &slot : slots) {
    int64_t slice =
        projectionKind == ProjectionKind::XFirst ? slot.col : slot.row;
    numSlices = std::max(numSlices, slice + 1);
  }

  llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 8> slotsBySlice;
  slotsBySlice.resize(static_cast<size_t>(numSlices));
  for (unsigned slotIndex = 0; slotIndex < slots.size(); ++slotIndex) {
    const PhysicalArraySlot &slot = slots[slotIndex];
    int64_t slice =
        projectionKind == ProjectionKind::XFirst ? slot.col : slot.row;
    slotsBySlice[static_cast<size_t>(slice)].push_back(slotIndex);
  }

  for (llvm::SmallVector<unsigned, 8> &sliceSlots : slotsBySlice) {
    llvm::sort(sliceSlots, [&](unsigned lhs, unsigned rhs) {
      const PhysicalArraySlot &lhsSlot = slots[lhs];
      const PhysicalArraySlot &rhsSlot = slots[rhs];
      if (projectionKind == ProjectionKind::XFirst) {
        if (lhsSlot.row != rhsSlot.row)
          return lhsSlot.row < rhsSlot.row;
      } else {
        if (lhsSlot.col != rhsSlot.col)
          return lhsSlot.col < rhsSlot.col;
      }
      if (lhsSlot.localArrayId != rhsSlot.localArrayId)
        return lhsSlot.localArrayId < rhsSlot.localArrayId;
      return lhsSlot.physicalArrayId < rhsSlot.physicalArrayId;
    });
  }

  return slotsBySlice;
}

static int64_t getMeshDistance(const PhysicalArraySlot &lhs,
                               const PhysicalArraySlot &rhs) {
  int64_t rowDistance =
      lhs.row > rhs.row ? lhs.row - rhs.row : rhs.row - lhs.row;
  int64_t colDistance =
      lhs.col > rhs.col ? lhs.col - rhs.col : rhs.col - lhs.col;
  return rowDistance + colDistance;
}

static unsigned
getMeshBoundaryMask(int64_t coreId,
                    const task_schedulers::HardwareBudget &budget) {
  constexpr unsigned kTop = 1u << 0;
  constexpr unsigned kBottom = 1u << 1;
  constexpr unsigned kLeft = 1u << 2;
  constexpr unsigned kRight = 1u << 3;

  int64_t row = coreId / budget.meshCols;
  int64_t col = coreId % budget.meshCols;
  unsigned mask = 0;
  if (row == 0)
    mask |= kTop;
  if (row == budget.meshRows - 1)
    mask |= kBottom;
  if (col == 0)
    mask |= kLeft;
  if (col == budget.meshCols - 1)
    mask |= kRight;
  return mask;
}

static int64_t scorePlacement(
    const CandidatePlacement &placement,
    llvm::ArrayRef<WeightedGroupEdge> edges,
    llvm::ArrayRef<PhysicalArraySlot> slots,
    const llvm::DenseMap<int64_t, unsigned> &slotIndexByPhysicalArrayId,
    const task_schedulers::HardwareBudget &budget) {
  int64_t transferCost = 0;
  for (const WeightedGroupEdge &edge : edges) {
    int64_t lhsPhysicalArray = placement.physicalArrayByGroup[edge.groupA];
    int64_t rhsPhysicalArray = placement.physicalArrayByGroup[edge.groupB];
    auto lhsSlotIt = slotIndexByPhysicalArrayId.find(lhsPhysicalArray);
    auto rhsSlotIt = slotIndexByPhysicalArrayId.find(rhsPhysicalArray);
    if (lhsSlotIt == slotIndexByPhysicalArrayId.end() ||
        rhsSlotIt == slotIndexByPhysicalArrayId.end())
      continue;

    transferCost += edge.totalBytes * getMeshDistance(slots[lhsSlotIt->second],
                                                      slots[rhsSlotIt->second]);
  }

  if (placement.physicalArrayByGroup.size() < 2)
    return transferCost;

  auto firstSlotIt =
      slotIndexByPhysicalArrayId.find(placement.physicalArrayByGroup.front());
  auto lastSlotIt =
      slotIndexByPhysicalArrayId.find(placement.physicalArrayByGroup.back());
  if (firstSlotIt == slotIndexByPhysicalArrayId.end() ||
      lastSlotIt == slotIndexByPhysicalArrayId.end())
    return transferCost;

  unsigned firstMask =
      getMeshBoundaryMask(slots[firstSlotIt->second].coreId, budget);
  unsigned lastMask =
      getMeshBoundaryMask(slots[lastSlotIt->second].coreId, budget);
  if ((firstMask & lastMask) != 0)
    return transferCost;

  return transferCost + ((transferCost + 4) / 5);
}

static CandidatePlacement projectPlacement(
    unsigned numGroups, const CutCoordinates &coordinates,
    llvm::ArrayRef<PhysicalArraySlot> slots,
    llvm::ArrayRef<WeightedGroupEdge> edges,
    const llvm::DenseMap<int64_t, unsigned> &slotIndexByPhysicalArrayId,
    const task_schedulers::HardwareBudget &budget,
    ProjectionKind projectionKind, bool reversePrimary, bool reverseSecondary) {
  CandidatePlacement placement;
  placement.physicalArrayByGroup.assign(numGroups, kUnassigned);
  if (numGroups == 0 || slots.empty()) {
    placement.estimatedScore = 0;
    return placement;
  }

  llvm::ArrayRef<int64_t> primary = projectionKind == ProjectionKind::XFirst
                                        ? coordinates.phiX
                                        : coordinates.phiY;
  llvm::ArrayRef<int64_t> secondary = projectionKind == ProjectionKind::XFirst
                                          ? coordinates.phiY
                                          : coordinates.phiX;
  llvm::SmallVector<unsigned, 8> sortedGroups = buildSortedGroups(
      numGroups, primary, secondary, reversePrimary, reverseSecondary);
  llvm::SmallVector<unsigned, 8> globalSlotOrder =
      sortSlotsForProjection(slots, projectionKind);
  llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 8> slotsBySlice =
      groupSlotsBySlice(slots, projectionKind);

  int64_t numSlices = projectionKind == ProjectionKind::XFirst
                          ? budget.meshCols
                          : budget.meshRows;
  int64_t maxSliceCapacity = projectionKind == ProjectionKind::XFirst
                                 ? budget.meshRows * budget.arraysPerCore
                                 : budget.meshCols * budget.arraysPerCore;
  llvm::SmallVector<int64_t, 8> capacities =
      computeSliceCapacities(numGroups, numSlices, maxSliceCapacity);

  unsigned groupOffset = 0;
  for (int64_t slice = 0; slice < numSlices && groupOffset < numGroups;
       ++slice) {
    int64_t capacity = capacities[static_cast<size_t>(slice)];
    if (capacity <= 0)
      continue;

    int64_t sliceIndex = reversePrimary ? numSlices - 1 - slice : slice;
    llvm::ArrayRef<unsigned> sliceSlots;
    if (sliceIndex >= 0 &&
        static_cast<size_t>(sliceIndex) < slotsBySlice.size())
      sliceSlots = slotsBySlice[static_cast<size_t>(sliceIndex)];
    if (sliceSlots.empty())
      continue;

    unsigned chunkEnd = std::min<unsigned>(
        numGroups, groupOffset + static_cast<unsigned>(capacity));
    llvm::SmallVector<unsigned, 8> chunkGroups;
    for (unsigned index = groupOffset; index < chunkEnd; ++index)
      chunkGroups.push_back(sortedGroups[index]);

    llvm::sort(chunkGroups, [&](unsigned lhs, unsigned rhs) {
      return compareCoordinate(lhs, rhs, secondary, primary, reverseSecondary,
                               reversePrimary);
    });

    for (auto indexedGroup : llvm::enumerate(chunkGroups)) {
      unsigned slotOffset = static_cast<unsigned>(indexedGroup.index());
      if (reverseSecondary)
        slotOffset = static_cast<unsigned>(sliceSlots.size() - 1) -
                     (slotOffset % static_cast<unsigned>(sliceSlots.size()));

      unsigned slotIndex =
          sliceSlots[slotOffset % static_cast<unsigned>(sliceSlots.size())];
      placement.physicalArrayByGroup[indexedGroup.value()] =
          slots[slotIndex].physicalArrayId;
    }

    groupOffset = chunkEnd;
  }

  for (unsigned index = groupOffset; index < numGroups; ++index) {
    unsigned groupIndex = sortedGroups[index];
    unsigned slotIndex =
        globalSlotOrder[index % static_cast<unsigned>(globalSlotOrder.size())];
    placement.physicalArrayByGroup[groupIndex] =
        slots[slotIndex].physicalArrayId;
  }

  placement.estimatedScore = scorePlacement(placement, edges, slots,
                                            slotIndexByPhysicalArrayId, budget);
  return placement;
}

static bool isBetterCandidate(const CandidatePlacement &candidate,
                              const CandidatePlacement &best) {
  if (candidate.estimatedScore != best.estimatedScore)
    return candidate.estimatedScore < best.estimatedScore;
  return candidate.physicalArrayByGroup < best.physicalArrayByGroup;
}

static mlir::FailureOr<CandidatePlacement> buildBestPlacement(
    mlir::Operation *diagnosticOp, unsigned numGroups,
    const task_schedulers::HardwareBudget &budget,
    llvm::ArrayRef<WeightedGroupEdge> edges,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
    llvm::ArrayRef<int64_t> incidentWeights) {
  auto slots = buildPhysicalArraySlots(diagnosticOp, budget);
  if (mlir::failed(slots))
    return mlir::failure();

  if (slots->empty()) {
    diagnosticOp->emitError("expected manhattan-cut scheduler to have at "
                            "least one physical analog array");
    return mlir::failure();
  }

  llvm::DenseMap<int64_t, unsigned> slotIndexByPhysicalArrayId;
  for (auto indexedSlot : llvm::enumerate(*slots))
    slotIndexByPhysicalArrayId.try_emplace(
        indexedSlot.value().physicalArrayId,
        static_cast<unsigned>(indexedSlot.index()));

  CutCoordinates coordinates =
      computeCutCoordinates(numGroups, adjacency, incidentWeights);

  CandidatePlacement best;
  bool foundCandidate = false;
  for (ProjectionKind projectionKind :
       {ProjectionKind::XFirst, ProjectionKind::YFirst}) {
    for (bool reversePrimary : {false, true}) {
      for (bool reverseSecondary : {false, true}) {
        CandidatePlacement candidate = projectPlacement(
            numGroups, coordinates, *slots, edges, slotIndexByPhysicalArrayId,
            budget, projectionKind, reversePrimary, reverseSecondary);
        if (!foundCandidate || isBetterCandidate(candidate, best)) {
          foundCandidate = true;
          best = std::move(candidate);
        }
      }
    }
  }

  if (!foundCandidate) {
    diagnosticOp->emitError("failed to build manhattan-cut placement");
    return mlir::failure();
  }

  return best;
}

class ManhattanCutTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "manhattan-cut"; }

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

    auto placement = buildBestPlacement(
        taskGraphFunc.getOperation(), static_cast<unsigned>(groups.size()),
        budget, *edges, adjacency, incidentWeights);
    if (mlir::failed(placement))
      return mlir::failure();

    llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>
        groupPlacements;
    groupPlacements.reserve(groups.size());
    for (const PlacementGroup &group : groups) {
      int64_t physicalArrayId = placement->physicalArrayByGroup[group.index];
      if (physicalArrayId == kUnassigned) {
        taskGraphFunc.emitError("expected manhattan-cut scheduler to place "
                                "every matrix setup group");
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

void registerManhattanCutTaskScheduler(TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(
      registry, std::make_unique<ManhattanCutTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
