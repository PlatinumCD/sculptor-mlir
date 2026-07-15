#include "TaskGraphIslandInternals.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

static void addIslandAffinityEdge(
    llvm::SmallVectorImpl<IslandAffinityEdge> &edges,
    unsigned producerIsland, unsigned consumerIsland, int64_t byteSize) {
  if (producerIsland == consumerIsland || byteSize <= 0)
    return;

  edges.push_back(
      IslandAffinityEdge{producerIsland, consumerIsland, byteSize});
}

static uint64_t getIslandPairKey(unsigned lhs, unsigned rhs) {
  unsigned first = std::min(lhs, rhs);
  unsigned second = std::max(lhs, rhs);
  return (static_cast<uint64_t>(first) << 32) | static_cast<uint64_t>(second);
}

static llvm::SmallVector<IslandAffinityEdge, 16>
compactIslandAffinityEdges(
    llvm::ArrayRef<IslandAffinityEdge> islandEdges) {
  llvm::DenseMap<uint64_t, int64_t> bytesByPair;
  for (const IslandAffinityEdge &edge : islandEdges) {
    if (edge.firstIsland == edge.secondIsland || edge.byteSize <= 0)
      continue;
    bytesByPair[getIslandPairKey(edge.firstIsland, edge.secondIsland)] +=
        edge.byteSize;
  }

  llvm::SmallVector<IslandAffinityEdge, 16> compactedEdges;
  compactedEdges.reserve(bytesByPair.size());
  for (const auto &entry : bytesByPair) {
    unsigned first = static_cast<unsigned>(entry.first >> 32);
    unsigned second = static_cast<unsigned>(entry.first & 0xffffffffu);
    compactedEdges.push_back(
        IslandAffinityEdge{first, second, entry.second});
  }

  llvm::sort(compactedEdges, [](const IslandAffinityEdge &lhs,
                                const IslandAffinityEdge &rhs) {
    if (lhs.firstIsland != rhs.firstIsland)
      return lhs.firstIsland < rhs.firstIsland;
    return lhs.secondIsland < rhs.secondIsland;
  });
  return compactedEdges;
}

} // namespace

llvm::SmallVector<IslandAffinityEdge, 16> buildIslandAffinityEdges(
    const TaskGraphDAG &dag, llvm::ArrayRef<ResourceEdge> resourceEdges,
    const llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::SmallVector<IslandAffinityEdge, 16> islandEdges;

  for (const ResourceEdge &edge : resourceEdges) {
    auto producerIslandIt = islandByTaskIndex.find(edge.producerIndex);
    auto consumerIslandIt = islandByTaskIndex.find(edge.consumerIndex);
    if (producerIslandIt == islandByTaskIndex.end() ||
        consumerIslandIt == islandByTaskIndex.end())
      continue;

    addIslandAffinityEdge(islandEdges, producerIslandIt->second,
                          consumerIslandIt->second, edge.byteSize);
  }

  llvm::SmallVector<bool, 16> eligible(dag.nodes.size(), false);
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> adjacency(
      dag.nodes.size());

  for (const TaskGraphNode &node : dag.nodes) {
    eligible[node.index] =
        !islandByTaskIndex.contains(node.index) && isDigitalTask(node.op);
  }

  for (const ResourceEdge &edge : resourceEdges) {
    if (!eligible[edge.producerIndex] || !eligible[edge.consumerIndex])
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

    llvm::DenseSet<unsigned> componentNodes;
    for (unsigned nodeIndex : component)
      componentNodes.insert(nodeIndex);

    llvm::DenseMap<unsigned, int64_t> terminalBytesByIsland;
    for (const ResourceEdge &edge : resourceEdges) {
      bool producerInComponent = componentNodes.contains(edge.producerIndex);
      bool consumerInComponent = componentNodes.contains(edge.consumerIndex);
      if (producerInComponent == consumerInComponent)
        continue;

      unsigned boundaryIndex =
          producerInComponent ? edge.consumerIndex : edge.producerIndex;
      auto islandIt = islandByTaskIndex.find(boundaryIndex);
      if (islandIt == islandByTaskIndex.end())
        continue;

      terminalBytesByIsland[islandIt->second] += edge.byteSize;
    }

    if (terminalBytesByIsland.size() < 2)
      continue;

    llvm::SmallVector<std::pair<unsigned, int64_t>, 8> terminals;
    terminals.reserve(terminalBytesByIsland.size());
    for (const auto &entry : terminalBytesByIsland)
      terminals.push_back({entry.first, entry.second});
    llvm::sort(terminals, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });

    int64_t denominator = static_cast<int64_t>(terminals.size()) - 1;
    for (auto lhsIt = terminals.begin(); lhsIt != terminals.end(); ++lhsIt) {
      for (auto rhsIt = std::next(lhsIt); rhsIt != terminals.end(); ++rhsIt) {
        int64_t pairBytes = std::min(lhsIt->second, rhsIt->second);
        pairBytes = (pairBytes + denominator - 1) / denominator;
        addIslandAffinityEdge(islandEdges, lhsIt->first, rhsIt->first,
                              pairBytes);
      }
    }
  }

  return compactIslandAffinityEdges(islandEdges);
}

IslandExecutionGraph buildIslandExecutionGraph(
    const TaskExecutionGraph &executionGraph,
    const llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  IslandExecutionGraph islandGraph;
  llvm::DenseMap<uint64_t, unsigned> edgeByPair;

  for (const TaskExecutionEdge &edge : executionGraph.edges) {
    auto producer = islandByTaskIndex.find(edge.producerTask);
    auto consumer = islandByTaskIndex.find(edge.consumerTask);
    if (producer == islandByTaskIndex.end() ||
        consumer == islandByTaskIndex.end() ||
        producer->second == consumer->second)
      continue;

    uint64_t key = (static_cast<uint64_t>(producer->second) << 32) |
                   static_cast<uint64_t>(consumer->second);
    auto existing = edgeByPair.find(key);
    if (existing != edgeByPair.end()) {
      IslandExecutionEdge &islandEdge = islandGraph.edges[existing->second];
      islandEdge.controlDependency |= edge.controlDependency;
      islandEdge.dataDependency |= edge.dataDependency;
      islandEdge.transferredBytes += edge.transferredBytes;
      continue;
    }

    unsigned edgeIndex = islandGraph.edges.size();
    edgeByPair.try_emplace(key, edgeIndex);
    islandGraph.edges.push_back(IslandExecutionEdge{
        producer->second, consumer->second, edge.controlDependency,
        edge.dataDependency, edge.transferredBytes});
  }

  llvm::sort(islandGraph.edges,
             [](const IslandExecutionEdge &lhs,
                const IslandExecutionEdge &rhs) {
               if (lhs.producerIsland != rhs.producerIsland)
                 return lhs.producerIsland < rhs.producerIsland;
               return lhs.consumerIsland < rhs.consumerIsland;
             });
  for (const IslandExecutionEdge &edge : islandGraph.edges) {
    islandGraph.predecessors[edge.consumerIsland].push_back(
        edge.producerIsland);
    islandGraph.successors[edge.producerIsland].push_back(edge.consumerIsland);
  }
  return islandGraph;
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
