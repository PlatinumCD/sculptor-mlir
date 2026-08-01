#include "TaskGraphIslandInternals.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphWorkloadAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace workload_attrs = mlir::sculptor::workload_attrs;

constexpr long double kBalanceWeight = 0.25L;

struct WeightedNeighbor {
  unsigned nodeIndex = 0;
  int64_t byteSize = 0;
};

struct AssignmentProposal {
  unsigned localNode = 0;
  unsigned island = 0;
  long double knownAffinity = 0.0L;
  long double selectedAffinity = 0.0L;
  long double cost = 0.0L;
};

struct RefinementMove {
  unsigned localNode = 0;
  unsigned destinationIsland = 0;
  long double costDelta = 0.0L;
};

static bool isEligibleDigitalTask(const TaskGraphNode &node) {
  sculptor::TaskCreateOp taskOp = node.op;
  return !taskOp->hasAttr(runtime_attrs::kTaskCoreIdAttrName) &&
         isDigitalTask(taskOp) && !taskOp.getSourceLayer().empty();
}

static long double getDigitalWork(const TaskGraphNode &node) {
  for (llvm::StringRef attrName : {workload_attrs::kDigitalOpsAttrName,
                                   runtime_attrs::kTaskDigitalOpsAttrName}) {
    if (auto attr = node.op->getAttrOfType<IntegerAttr>(attrName)) {
      if (attr.getInt() > 0)
        return static_cast<long double>(attr.getInt());
    }
  }
  return 1.0L;
}

static void appendUniqueIsland(llvm::SmallVectorImpl<unsigned> &islands,
                               unsigned island) {
  if (!llvm::is_contained(islands, island))
    islands.push_back(island);
}

static bool isBetterCandidate(long double cost, long double affinity,
                              long double currentLoad, unsigned island,
                              const AssignmentProposal &best,
                              long double bestCurrentLoad, bool hasBest) {
  if (!hasBest)
    return true;

  constexpr long double epsilon = 1.0e-12L;
  if (cost + epsilon < best.cost)
    return true;
  if (std::abs(cost - best.cost) > epsilon)
    return false;
  if (affinity > best.selectedAffinity + epsilon)
    return true;
  if (std::abs(affinity - best.selectedAffinity) > epsilon)
    return false;
  if (currentLoad + epsilon < bestCurrentLoad)
    return true;
  if (std::abs(currentLoad - bestCurrentLoad) > epsilon)
    return false;
  return island < best.island;
}

static bool isBetterProposal(const AssignmentProposal &candidate,
                             const AssignmentProposal &best, bool hasBest) {
  if (!hasBest)
    return true;

  constexpr long double epsilon = 1.0e-12L;
  if (candidate.knownAffinity > best.knownAffinity + epsilon)
    return true;
  if (std::abs(candidate.knownAffinity - best.knownAffinity) > epsilon)
    return false;
  if (candidate.cost + epsilon < best.cost)
    return true;
  if (std::abs(candidate.cost - best.cost) > epsilon)
    return false;
  if (candidate.localNode != best.localNode)
    return candidate.localNode < best.localNode;
  return candidate.island < best.island;
}

static long double balanceContribution(long double load, long double target,
                                       long double total) {
  long double normalized = (load - target) / std::max(1.0L, total);
  return normalized * normalized;
}

static void refineComponentAssignments(
    llvm::ArrayRef<unsigned> component,
    llvm::ArrayRef<llvm::DenseMap<unsigned, long double>> boundaryAffinity,
    llvm::ArrayRef<llvm::SmallVector<WeightedNeighbor, 4>> internalNeighbors,
    const llvm::DenseMap<unsigned, unsigned> &localIndexByTask,
    llvm::ArrayRef<long double> digitalWork,
    llvm::ArrayRef<long double> storageWeight, long double totalWork,
    long double totalStorage, long double totalGraphBytes,
    long double targetWork, long double targetStorage,
    llvm::SmallVectorImpl<std::optional<unsigned>> &assignment,
    llvm::DenseMap<unsigned, long double> &workByIsland,
    llvm::DenseMap<unsigned, long double> &storageByIsland,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  unsigned moveLimit =
      std::max<unsigned>(1, static_cast<unsigned>(component.size()) * 4);
  for (unsigned moveCount = 0; moveCount < moveLimit; ++moveCount) {
    RefinementMove bestMove;
    bool hasBestMove = false;

    for (unsigned localNode = 0; localNode < component.size(); ++localNode) {
      if (!assignment[localNode])
        continue;
      unsigned currentIsland = *assignment[localNode];

      unsigned remainingCurrentTasks = 0;
      bool hasOtherBoundaryAnchor = false;
      for (unsigned otherNode = 0; otherNode < component.size(); ++otherNode) {
        if (otherNode == localNode || assignment[otherNode] != currentIsland)
          continue;
        ++remainingCurrentTasks;
        if (boundaryAffinity[otherNode].contains(currentIsland))
          hasOtherBoundaryAnchor = true;
      }
      if (remainingCurrentTasks > 0) {
        unsigned sameIslandNeighbors = 0;
        for (const WeightedNeighbor &neighbor : internalNeighbors[localNode]) {
          unsigned neighborLocal = localIndexByTask.lookup(neighbor.nodeIndex);
          if (assignment[neighborLocal] == currentIsland)
            ++sameIslandNeighbors;
        }
        if (sameIslandNeighbors > 1 || !hasOtherBoundaryAnchor)
          continue;
      }

      llvm::SmallVector<unsigned, 8> candidateIslands;
      for (const auto &entry : boundaryAffinity[localNode])
        appendUniqueIsland(candidateIslands, entry.first);
      for (const WeightedNeighbor &neighbor : internalNeighbors[localNode]) {
        unsigned neighborLocal = localIndexByTask.lookup(neighbor.nodeIndex);
        if (assignment[neighborLocal])
          appendUniqueIsland(candidateIslands, *assignment[neighborLocal]);
      }
      llvm::sort(candidateIslands);

      for (unsigned destinationIsland : candidateIslands) {
        if (destinationIsland == currentIsland)
          continue;

        long double communicationDelta = 0.0L;
        for (const auto &entry : boundaryAffinity[localNode]) {
          bool cutBefore = currentIsland != entry.first;
          bool cutAfter = destinationIsland != entry.first;
          communicationDelta +=
              static_cast<long double>(cutAfter) * entry.second -
              static_cast<long double>(cutBefore) * entry.second;
        }
        for (const WeightedNeighbor &neighbor : internalNeighbors[localNode]) {
          unsigned neighborLocal = localIndexByTask.lookup(neighbor.nodeIndex);
          if (!assignment[neighborLocal])
            continue;
          unsigned neighborIsland = *assignment[neighborLocal];
          bool cutBefore = currentIsland != neighborIsland;
          bool cutAfter = destinationIsland != neighborIsland;
          communicationDelta +=
              static_cast<long double>(cutAfter) * neighbor.byteSize -
              static_cast<long double>(cutBefore) * neighbor.byteSize;
        }

        long double currentWork = workByIsland.lookup(currentIsland);
        long double destinationWork = workByIsland.lookup(destinationIsland);
        long double currentStorage = storageByIsland.lookup(currentIsland);
        long double destinationStorage =
            storageByIsland.lookup(destinationIsland);

        long double balanceBefore =
            balanceContribution(currentWork, targetWork, totalWork) +
            balanceContribution(destinationWork, targetWork, totalWork) +
            balanceContribution(currentStorage, targetStorage, totalStorage) +
            balanceContribution(destinationStorage, targetStorage,
                                totalStorage);
        long double balanceAfter =
            balanceContribution(currentWork - digitalWork[localNode],
                                targetWork, totalWork) +
            balanceContribution(destinationWork + digitalWork[localNode],
                                targetWork, totalWork) +
            balanceContribution(currentStorage - storageWeight[localNode],
                                targetStorage, totalStorage) +
            balanceContribution(destinationStorage + storageWeight[localNode],
                                targetStorage, totalStorage);
        long double costDelta =
            communicationDelta +
            kBalanceWeight * totalGraphBytes * (balanceAfter - balanceBefore);

        constexpr long double epsilon = 1.0e-12L;
        if (costDelta >= -epsilon)
          continue;
        if (hasBestMove &&
            (costDelta > bestMove.costDelta - epsilon ||
             (std::abs(costDelta - bestMove.costDelta) <= epsilon &&
              (localNode > bestMove.localNode ||
               (localNode == bestMove.localNode &&
                destinationIsland >= bestMove.destinationIsland)))))
          continue;

        hasBestMove = true;
        bestMove = {localNode, destinationIsland, costDelta};
      }
    }

    if (!hasBestMove)
      break;

    unsigned sourceIsland = *assignment[bestMove.localNode];
    workByIsland[sourceIsland] -= digitalWork[bestMove.localNode];
    storageByIsland[sourceIsland] -= storageWeight[bestMove.localNode];
    workByIsland[bestMove.destinationIsland] += digitalWork[bestMove.localNode];
    storageByIsland[bestMove.destinationIsland] +=
        storageWeight[bestMove.localNode];
    assignment[bestMove.localNode] = bestMove.destinationIsland;
    islandByTaskIndex[component[bestMove.localNode]] =
        bestMove.destinationIsland;
  }
}

static LogicalResult
assignComponent(const TaskGraphDAG &dag,
                llvm::ArrayRef<ResourceEdge> resourceEdges,
                llvm::ArrayRef<unsigned> component,
                llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::DenseMap<unsigned, unsigned> localIndexByTask;
  for (auto indexedTask : llvm::enumerate(component))
    localIndexByTask.try_emplace(indexedTask.value(), indexedTask.index());

  llvm::SmallVector<unsigned, 8> terminalIslands;
  llvm::SmallVector<llvm::DenseMap<unsigned, long double>, 16> boundaryAffinity(
      component.size());
  llvm::SmallVector<llvm::SmallVector<WeightedNeighbor, 4>, 16>
      internalNeighbors(component.size());
  llvm::SmallVector<long double, 16> storageWeight(component.size(), 0.0L);
  long double totalGraphBytes = 0.0L;

  for (const ResourceEdge &edge : resourceEdges) {
    auto producerIt = localIndexByTask.find(edge.producerIndex);
    auto consumerIt = localIndexByTask.find(edge.consumerIndex);
    bool producerInComponent = producerIt != localIndexByTask.end();
    bool consumerInComponent = consumerIt != localIndexByTask.end();
    if (!producerInComponent && !consumerInComponent)
      continue;

    const TaskGraphNode &producer = dag.nodes[edge.producerIndex];
    const TaskGraphNode &consumer = dag.nodes[edge.consumerIndex];
    if (!sameNonEmptySourceLayer(producer.op, consumer.op))
      continue;

    long double bytes =
        static_cast<long double>(std::max<int64_t>(0, edge.byteSize));
    if (producerInComponent)
      storageWeight[producerIt->second] += bytes;
    if (consumerInComponent)
      storageWeight[consumerIt->second] += bytes;

    if (producerInComponent && consumerInComponent) {
      int64_t propagationWeight = std::max<int64_t>(1, edge.byteSize);
      internalNeighbors[producerIt->second].push_back(
          {edge.consumerIndex, propagationWeight});
      internalNeighbors[consumerIt->second].push_back(
          {edge.producerIndex, propagationWeight});
      totalGraphBytes += bytes;
      continue;
    }

    unsigned localIndex =
        producerInComponent ? producerIt->second : consumerIt->second;
    unsigned boundaryTask =
        producerInComponent ? edge.consumerIndex : edge.producerIndex;
    if (isReductionTask(dag.nodes[boundaryTask].op))
      continue;

    auto islandIt = islandByTaskIndex.find(boundaryTask);
    if (islandIt == islandByTaskIndex.end())
      continue;

    appendUniqueIsland(terminalIslands, islandIt->second);
    boundaryAffinity[localIndex][islandIt->second] +=
        static_cast<long double>(std::max<int64_t>(1, edge.byteSize));
    totalGraphBytes += bytes;
  }

  if (terminalIslands.empty())
    return success();

  llvm::sort(terminalIslands);
  if (terminalIslands.size() == 1) {
    for (unsigned taskIndex : component)
      islandByTaskIndex[taskIndex] = terminalIslands.front();
    return success();
  }

  llvm::SmallVector<long double, 16> digitalWork;
  digitalWork.reserve(component.size());
  long double totalWork = 0.0L;
  long double totalStorage = 0.0L;
  for (auto indexedTask : llvm::enumerate(component)) {
    long double work = getDigitalWork(dag.nodes[indexedTask.value()]);
    digitalWork.push_back(work);
    totalWork += work;
    if (storageWeight[indexedTask.index()] <= 0.0L)
      storageWeight[indexedTask.index()] = 1.0L;
    totalStorage += storageWeight[indexedTask.index()];
  }
  totalGraphBytes = std::max(1.0L, totalGraphBytes);

  llvm::DenseMap<unsigned, long double> workByIsland;
  llvm::DenseMap<unsigned, long double> storageByIsland;
  for (unsigned island : terminalIslands) {
    workByIsland[island] = 0.0L;
    storageByIsland[island] = 0.0L;
  }

  llvm::SmallVector<std::optional<unsigned>, 16> assignment(component.size());
  unsigned assignedCount = 0;
  const long double targetWork = totalWork / terminalIslands.size();
  const long double targetStorage = totalStorage / terminalIslands.size();

  while (assignedCount < component.size()) {
    AssignmentProposal selected;
    bool hasSelected = false;

    for (auto indexedTask : llvm::enumerate(component)) {
      unsigned localNode = indexedTask.index();
      if (assignment[localNode])
        continue;

      llvm::DenseMap<unsigned, long double> affinityByIsland =
          boundaryAffinity[localNode];
      for (const WeightedNeighbor &neighbor : internalNeighbors[localNode]) {
        unsigned neighborLocal = localIndexByTask.lookup(neighbor.nodeIndex);
        if (!assignment[neighborLocal])
          continue;
        affinityByIsland[*assignment[neighborLocal]] += neighbor.byteSize;
      }
      if (affinityByIsland.empty())
        continue;

      long double knownAffinity = 0.0L;
      for (const auto &entry : affinityByIsland)
        knownAffinity += entry.second;

      AssignmentProposal bestForNode;
      bestForNode.localNode = localNode;
      bestForNode.knownAffinity = knownAffinity;
      bool hasBestForNode = false;
      long double bestCurrentLoad = 0.0L;

      for (const auto &entry : affinityByIsland) {
        unsigned island = entry.first;
        long double projectedWork =
            workByIsland.lookup(island) + digitalWork[localNode];
        long double projectedStorage =
            storageByIsland.lookup(island) + storageWeight[localNode];
        long double workOverload = std::max(0.0L, projectedWork - targetWork) /
                                   std::max(1.0L, totalWork);
        long double storageOverload =
            std::max(0.0L, projectedStorage - targetStorage) /
            std::max(1.0L, totalStorage);
        long double cost = knownAffinity - entry.second;
        cost +=
            kBalanceWeight * totalGraphBytes * (workOverload + storageOverload);

        long double currentLoad = workByIsland.lookup(island);
        if (!isBetterCandidate(cost, entry.second, currentLoad, island,
                               bestForNode, bestCurrentLoad, hasBestForNode))
          continue;

        hasBestForNode = true;
        bestCurrentLoad = currentLoad;
        bestForNode.island = island;
        bestForNode.selectedAffinity = entry.second;
        bestForNode.cost = cost;
      }

      if (hasBestForNode &&
          isBetterProposal(bestForNode, selected, hasSelected)) {
        selected = bestForNode;
        hasSelected = true;
      }
    }

    // A zero-terminal subregion remains available to the established fallback.
    if (!hasSelected)
      break;

    assignment[selected.localNode] = selected.island;
    islandByTaskIndex[component[selected.localNode]] = selected.island;
    workByIsland[selected.island] += digitalWork[selected.localNode];
    storageByIsland[selected.island] += storageWeight[selected.localNode];
    ++assignedCount;
  }

  if (assignedCount == component.size()) {
    refineComponentAssignments(
        component, boundaryAffinity, internalNeighbors, localIndexByTask,
        digitalWork, storageWeight, totalWork, totalStorage, totalGraphBytes,
        targetWork, targetStorage, assignment, workByIsland, storageByIsland,
        islandByTaskIndex);
  }

  return success();
}

} // namespace

LogicalResult assignMultiTerminalBalancedDigitalIslands(
    const TaskGraphDAG &dag, llvm::ArrayRef<ResourceEdge> resourceEdges,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::SmallVector<bool, 16> eligible(dag.nodes.size(), false);
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> adjacency(
      dag.nodes.size());

  for (const TaskGraphNode &node : dag.nodes) {
    eligible[node.index] =
        !islandByTaskIndex.contains(node.index) && isEligibleDigitalTask(node);
  }

  for (const ResourceEdge &edge : resourceEdges) {
    if (!eligible[edge.producerIndex] || !eligible[edge.consumerIndex])
      continue;
    if (!sameNonEmptySourceLayer(dag.nodes[edge.producerIndex].op,
                                 dag.nodes[edge.consumerIndex].op))
      continue;
    adjacency[edge.producerIndex].push_back(edge.consumerIndex);
    adjacency[edge.consumerIndex].push_back(edge.producerIndex);
  }

  llvm::SmallVector<bool, 16> visited(dag.nodes.size(), false);
  for (const TaskGraphNode &node : dag.nodes) {
    if (!eligible[node.index] || visited[node.index])
      continue;

    llvm::SmallVector<unsigned, 16> component;
    std::queue<unsigned> worklist;
    visited[node.index] = true;
    worklist.push(node.index);

    while (!worklist.empty()) {
      unsigned current = worklist.front();
      worklist.pop();
      component.push_back(current);
      for (unsigned neighbor : adjacency[current]) {
        if (visited[neighbor])
          continue;
        visited[neighbor] = true;
        worklist.push(neighbor);
      }
    }

    if (failed(
            assignComponent(dag, resourceEdges, component, islandByTaskIndex)))
      return failure();
  }

  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
