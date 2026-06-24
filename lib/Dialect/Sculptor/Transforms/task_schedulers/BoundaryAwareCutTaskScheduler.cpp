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

static unsigned selectSeedGroup(llvm::ArrayRef<unsigned> groups,
                                llvm::ArrayRef<int64_t> incidentWeights) {
  unsigned selected = groups.front();
  int64_t selectedWeight = std::numeric_limits<int64_t>::min();
  for (unsigned group : groups) {
    if (incidentWeights[group] > selectedWeight ||
        (incidentWeights[group] == selectedWeight && group < selected)) {
      selected = group;
      selectedWeight = incidentWeights[group];
    }
  }
  return selected;
}

static unsigned selectOppositeSeed(
    unsigned seed, llvm::ArrayRef<unsigned> groups,
    llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
    llvm::ArrayRef<int64_t> incidentWeights) {
  unsigned selected = seed;
  int64_t selectedConnection = std::numeric_limits<int64_t>::max();
  int64_t selectedIncident = std::numeric_limits<int64_t>::min();

  for (unsigned group : groups) {
    if (group == seed)
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
    llvm::ArrayRef<int64_t> incidentWeights) {
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
    unsigned seedFirst = selectSeedGroup(groups, incidentWeights);
    result.firstGroups.push_back(seedFirst);
    assigned.set(seedFirst);
    inFirst.set(seedFirst);
  }

  if (targetSecond > 0) {
    unsigned seedSecond =
        targetFirst > 0 ? selectOppositeSeed(result.firstGroups.front(), groups,
                                             adjacency, incidentWeights)
                        : selectSeedGroup(groups, incidentWeights);
    if (!assigned.test(seedSecond)) {
      result.secondGroups.push_back(seedSecond);
      assigned.set(seedSecond);
      inSecond.set(seedSecond);
    }
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
          static_cast<int64_t>(result.firstGroups.size()) < targetFirst;
      bool canAssignSecond =
          static_cast<int64_t>(result.secondGroups.size()) < targetSecond;

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
                llvm::ArrayRef<int64_t> incidentWeights) {
  llvm::SmallVector<RegionSplit, 4> splits = buildCandidateSplits(region);
  CandidateSplit best;
  bool hasBest = false;

  for (const RegionSplit &split : splits) {
    PartitionResult partition = partitionGroupsForSplit(
        groups, split, budget, edges, adjacency, incidentWeights);
    if (partition.score >= kInfiniteScore)
      continue;

    CandidateSplit candidate{split, std::move(partition)};
    if (isBetterCandidateSplit(candidate, best, hasBest)) {
      best = std::move(candidate);
      hasBest = true;
    }
  }

  if (!hasBest) {
    diagnosticOp->emitError("failed to find a feasible boundary-aware region "
                            "split");
    return mlir::failure();
  }

  return best;
}

static mlir::LogicalResult
assignLeafGroups(mlir::Operation *diagnosticOp, llvm::ArrayRef<unsigned> groups,
                 const MeshRegion &region, BoundaryAwarePlacementState &state) {
  llvm::SmallVector<unsigned, 8> slotIndices =
      collectRegionSlotIndices(state.slots, region);
  if (groups.size() > slotIndices.size()) {
    diagnosticOp->emitError("expected leaf region capacity to fit assigned "
                            "matrix setup groups");
    return mlir::failure();
  }

  llvm::SmallVector<unsigned, 8> sortedGroups(groups.begin(), groups.end());
  llvm::sort(sortedGroups);
  for (auto indexedGroup : llvm::enumerate(sortedGroups)) {
    unsigned group = indexedGroup.value();
    unsigned slotIndex = slotIndices[indexedGroup.index()];
    state.physicalArrayByGroup[group] = state.slots[slotIndex].physicalArrayId;
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
    return assignLeafGroups(diagnosticOp, groups, region, state);

  auto split = chooseBestSplit(diagnosticOp, groups, region, budget, edges,
                               adjacency, incidentWeights);
  if (mlir::failed(split))
    return mlir::failure();

  if (mlir::failed(placeRegion(diagnosticOp, split->partition.firstGroups,
                               split->split.first, budget, edges, adjacency,
                               incidentWeights, state)))
    return mlir::failure();

  return placeRegion(diagnosticOp, split->partition.secondGroups,
                     split->split.second, budget, edges, adjacency,
                     incidentWeights, state);
}

static mlir::FailureOr<BoundaryAwarePlacementState>
buildPlacement(mlir::Operation *diagnosticOp, unsigned numGroups,
               const task_schedulers::HardwareBudget &budget,
               llvm::ArrayRef<WeightedGroupEdge> edges,
               llvm::ArrayRef<llvm::SmallVector<GroupAdjacency, 4>> adjacency,
               llvm::ArrayRef<int64_t> incidentWeights) {
  auto slots = buildPhysicalArraySlots(diagnosticOp, budget);
  if (mlir::failed(slots))
    return mlir::failure();

  if (static_cast<int64_t>(numGroups) > budget.numAnalogArrays) {
    diagnosticOp->emitError("expected boundary-aware-cut scheduler to have "
                            "enough physical analog arrays for matrix setup "
                            "groups");
    return mlir::failure();
  }

  BoundaryAwarePlacementState state;
  state.physicalArrayByGroup.assign(numGroups, kUnassigned);
  state.slots = std::move(*slots);

  llvm::SmallVector<unsigned, 8> allGroups;
  allGroups.reserve(numGroups);
  for (unsigned group = 0; group < numGroups; ++group)
    allGroups.push_back(group);

  MeshRegion root{0, budget.meshRows, 0, budget.meshCols};
  if (mlir::failed(placeRegion(diagnosticOp, allGroups, root, budget, edges,
                               adjacency, incidentWeights, state)))
    return mlir::failure();

  return state;
}

class BoundaryAwareCutTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "boundary-aware-cut"; }

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

    auto placement = buildPlacement(taskGraphFunc.getOperation(),
                                    static_cast<unsigned>(groups.size()),
                                    budget, *edges, adjacency, incidentWeights);
    if (mlir::failed(placement))
      return mlir::failure();

    llvm::SmallVector<task_schedulers::MatrixSetupGroupPlacement, 8>
        groupPlacements;
    groupPlacements.reserve(groups.size());
    for (const PlacementGroup &group : groups) {
      int64_t physicalArrayId = placement->physicalArrayByGroup[group.index];
      if (physicalArrayId == kUnassigned) {
        taskGraphFunc.emitError("expected boundary-aware-cut scheduler to "
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

void registerBoundaryAwareCutTaskScheduler(
    TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(
      registry, std::make_unique<BoundaryAwareCutTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
