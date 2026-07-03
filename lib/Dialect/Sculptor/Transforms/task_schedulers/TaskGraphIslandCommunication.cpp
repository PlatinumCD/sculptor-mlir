#include "TaskGraphIslandInternals.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTaskKinds.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace {

using IslandCommunicationEdge = LogicalIslandCommunicationEdge;

static void addIslandCommunicationEdge(
    llvm::SmallVectorImpl<IslandCommunicationEdge> &edges,
    unsigned producerIsland, unsigned consumerIsland, int64_t byteSize) {
  if (producerIsland == consumerIsland || byteSize <= 0)
    return;

  edges.push_back(
      IslandCommunicationEdge{producerIsland, consumerIsland, byteSize});
}

static uint64_t getIslandPairKey(unsigned lhs, unsigned rhs) {
  unsigned first = std::min(lhs, rhs);
  unsigned second = std::max(lhs, rhs);
  return (static_cast<uint64_t>(first) << 32) | static_cast<uint64_t>(second);
}

static llvm::SmallVector<IslandCommunicationEdge, 16>
compactIslandCommunicationEdges(
    llvm::ArrayRef<IslandCommunicationEdge> islandEdges) {
  llvm::DenseMap<uint64_t, int64_t> bytesByPair;
  for (const IslandCommunicationEdge &edge : islandEdges) {
    if (edge.producerIsland == edge.consumerIsland || edge.byteSize <= 0)
      continue;
    bytesByPair[getIslandPairKey(edge.producerIsland, edge.consumerIsland)] +=
        edge.byteSize;
  }

  llvm::SmallVector<IslandCommunicationEdge, 16> compactedEdges;
  compactedEdges.reserve(bytesByPair.size());
  for (const auto &entry : bytesByPair) {
    unsigned first = static_cast<unsigned>(entry.first >> 32);
    unsigned second = static_cast<unsigned>(entry.first & 0xffffffffu);
    compactedEdges.push_back(
        IslandCommunicationEdge{first, second, entry.second});
  }

  llvm::sort(compactedEdges, [](const IslandCommunicationEdge &lhs,
                                const IslandCommunicationEdge &rhs) {
    if (lhs.producerIsland != rhs.producerIsland)
      return lhs.producerIsland < rhs.producerIsland;
    return lhs.consumerIsland < rhs.consumerIsland;
  });
  return compactedEdges;
}

} // namespace

llvm::SmallVector<LogicalIslandCommunicationEdge, 16>
buildIslandCommunicationEdges(
    const TaskGraphDAG &dag, llvm::ArrayRef<ResourceEdge> resourceEdges,
    const llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::SmallVector<IslandCommunicationEdge, 16> islandEdges;

  for (const ResourceEdge &edge : resourceEdges) {
    auto producerIslandIt = islandByTaskIndex.find(edge.producerIndex);
    auto consumerIslandIt = islandByTaskIndex.find(edge.consumerIndex);
    if (producerIslandIt == islandByTaskIndex.end() ||
        consumerIslandIt == islandByTaskIndex.end())
      continue;

    addIslandCommunicationEdge(islandEdges, producerIslandIt->second,
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
        addIslandCommunicationEdge(islandEdges, lhsIt->first, rhsIt->first,
                                   pairBytes);
      }
    }
  }

  return compactIslandCommunicationEdges(islandEdges);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
