#include "TaskGraphIslandInternals.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTaskKinds.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;

struct FlowEdge {
  unsigned to = 0;
  unsigned reverseIndex = 0;
  int64_t capacity = 0;
};

class MinCutGraph {
public:
  explicit MinCutGraph(unsigned nodeCount) : adjacency(nodeCount) {}

  void addEdge(unsigned from, unsigned to, int64_t capacity) {
    if (capacity <= 0)
      return;

    FlowEdge forward{to, static_cast<unsigned>(adjacency[to].size()),
                     capacity};
    FlowEdge reverse{from, static_cast<unsigned>(adjacency[from].size()), 0};
    adjacency[from].push_back(forward);
    adjacency[to].push_back(reverse);
  }

  int64_t computeMaxFlow(unsigned source, unsigned sink) {
    int64_t flow = 0;
    while (buildLevels(source, sink)) {
      nextEdge.assign(adjacency.size(), 0);
      while (true) {
        int64_t pushed =
            pushFlow(source, sink, std::numeric_limits<int64_t>::max());
        if (pushed == 0)
          break;
        flow += pushed;
      }
    }
    return flow;
  }

  llvm::SmallVector<bool, 16> collectSourceReachable(unsigned source) const {
    llvm::SmallVector<bool, 16> reachable(adjacency.size(), false);
    std::queue<unsigned> worklist;
    reachable[source] = true;
    worklist.push(source);

    while (!worklist.empty()) {
      unsigned node = worklist.front();
      worklist.pop();
      for (const FlowEdge &edge : adjacency[node]) {
        if (edge.capacity <= 0 || reachable[edge.to])
          continue;
        reachable[edge.to] = true;
        worklist.push(edge.to);
      }
    }

    return reachable;
  }

private:
  bool buildLevels(unsigned source, unsigned sink) {
    levels.assign(adjacency.size(), -1);
    std::queue<unsigned> worklist;
    levels[source] = 0;
    worklist.push(source);

    while (!worklist.empty()) {
      unsigned node = worklist.front();
      worklist.pop();
      for (const FlowEdge &edge : adjacency[node]) {
        if (edge.capacity <= 0 || levels[edge.to] >= 0)
          continue;
        levels[edge.to] = levels[node] + 1;
        worklist.push(edge.to);
      }
    }

    return levels[sink] >= 0;
  }

  int64_t pushFlow(unsigned node, unsigned sink, int64_t flow) {
    if (node == sink)
      return flow;

    for (unsigned &edgeIndex = nextEdge[node];
         edgeIndex < adjacency[node].size(); ++edgeIndex) {
      FlowEdge &edge = adjacency[node][edgeIndex];
      if (edge.capacity <= 0 || levels[edge.to] != levels[node] + 1)
        continue;

      int64_t pushed = pushFlow(edge.to, sink, std::min(flow, edge.capacity));
      if (pushed == 0)
        continue;

      edge.capacity -= pushed;
      adjacency[edge.to][edge.reverseIndex].capacity += pushed;
      return pushed;
    }

    return 0;
  }

  llvm::SmallVector<llvm::SmallVector<FlowEdge, 4>, 16> adjacency;
  llvm::SmallVector<int64_t, 16> levels;
  llvm::SmallVector<unsigned, 16> nextEdge;
};

static mlir::IntegerAttr getCoreIdAttr(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp->getAttrOfType<mlir::IntegerAttr>(
      runtime_attrs::kTaskCoreIdAttrName);
}

static bool isEligiblePrePlacementIslandDigitalTask(const TaskGraphNode &node) {
  mlir::sculptor::TaskCreateOp taskOp = node.op;
  return !getCoreIdAttr(taskOp) && isDigitalTask(taskOp) &&
         !taskOp.getSourceLayer().empty();
}

static void appendUniqueIsland(llvm::SmallVectorImpl<unsigned> &islands,
                               unsigned island) {
  for (unsigned existingIsland : islands) {
    if (existingIsland == island)
      return;
  }
  islands.push_back(island);
}

static mlir::LogicalResult assignPrePlacementMinCutComponentIsland(
    const TaskGraphDAG &dag, llvm::ArrayRef<ResourceEdge> resourceEdges,
    llvm::ArrayRef<unsigned> component,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::DenseMap<unsigned, unsigned> localIndexByNodeIndex;
  for (auto indexedNode : llvm::enumerate(component)) {
    localIndexByNodeIndex.try_emplace(
        indexedNode.value(), static_cast<unsigned>(indexedNode.index()));
  }

  llvm::SmallVector<unsigned, 4> terminalIslands;
  for (const ResourceEdge &edge : resourceEdges) {
    auto producerLocalIt = localIndexByNodeIndex.find(edge.producerIndex);
    auto consumerLocalIt = localIndexByNodeIndex.find(edge.consumerIndex);
    bool producerInComponent = producerLocalIt != localIndexByNodeIndex.end();
    bool consumerInComponent = consumerLocalIt != localIndexByNodeIndex.end();
    if (producerInComponent == consumerInComponent)
      continue;

    const TaskGraphNode &producer = dag.nodes[edge.producerIndex];
    const TaskGraphNode &consumer = dag.nodes[edge.consumerIndex];
    if (!sameNonEmptySourceLayer(producer.op, consumer.op))
      continue;

    unsigned boundaryIndex =
        producerInComponent ? edge.consumerIndex : edge.producerIndex;
    auto islandIt = islandByTaskIndex.find(boundaryIndex);
    if (islandIt == islandByTaskIndex.end())
      continue;

    appendUniqueIsland(terminalIslands, islandIt->second);
  }

  if (terminalIslands.empty())
    return mlir::success();

  if (terminalIslands.size() == 1) {
    for (unsigned nodeIndex : component)
      islandByTaskIndex[nodeIndex] = terminalIslands.front();
    return mlir::success();
  }

  if (terminalIslands.size() != 2)
    return mlir::success();

  unsigned sourceIsland = terminalIslands[0];
  unsigned sinkIsland = terminalIslands[1];
  unsigned source = static_cast<unsigned>(component.size());
  unsigned sink = source + 1;
  MinCutGraph cutGraph(/*nodeCount=*/sink + 1);

  auto addUndirectedEdge = [&cutGraph](unsigned lhs, unsigned rhs,
                                       int64_t byteSize) {
    cutGraph.addEdge(lhs, rhs, byteSize);
    cutGraph.addEdge(rhs, lhs, byteSize);
  };

  for (const ResourceEdge &edge : resourceEdges) {
    const TaskGraphNode &producer = dag.nodes[edge.producerIndex];
    const TaskGraphNode &consumer = dag.nodes[edge.consumerIndex];
    if (!sameNonEmptySourceLayer(producer.op, consumer.op))
      continue;

    auto producerLocalIt = localIndexByNodeIndex.find(edge.producerIndex);
    auto consumerLocalIt = localIndexByNodeIndex.find(edge.consumerIndex);
    bool producerInComponent = producerLocalIt != localIndexByNodeIndex.end();
    bool consumerInComponent = consumerLocalIt != localIndexByNodeIndex.end();

    if (producerInComponent && consumerInComponent) {
      addUndirectedEdge(producerLocalIt->second, consumerLocalIt->second,
                        edge.byteSize);
      continue;
    }

    if (producerInComponent == consumerInComponent)
      continue;

    unsigned localIndex =
        producerInComponent ? producerLocalIt->second : consumerLocalIt->second;
    unsigned boundaryIndex =
        producerInComponent ? edge.consumerIndex : edge.producerIndex;
    auto islandIt = islandByTaskIndex.find(boundaryIndex);
    if (islandIt == islandByTaskIndex.end())
      continue;

    if (islandIt->second == sourceIsland) {
      addUndirectedEdge(source, localIndex, edge.byteSize);
      continue;
    }

    if (islandIt->second == sinkIsland)
      addUndirectedEdge(sink, localIndex, edge.byteSize);
  }

  cutGraph.computeMaxFlow(source, sink);
  llvm::SmallVector<bool, 16> sourceReachable =
      cutGraph.collectSourceReachable(source);

  for (auto indexedNode : llvm::enumerate(component)) {
    islandByTaskIndex[indexedNode.value()] =
        sourceReachable[static_cast<unsigned>(indexedNode.index())]
            ? sourceIsland
            : sinkIsland;
  }

  return mlir::success();
}

} // namespace

LogicalResult assignPrePlacementMinCutDigitalIslands(
    const TaskGraphDAG &dag,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  auto resourceEdges = collectResourceEdges(dag);
  if (mlir::failed(resourceEdges))
    return mlir::failure();

  llvm::SmallVector<bool, 16> eligible(dag.nodes.size(), false);
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> adjacency(
      dag.nodes.size());

  for (const TaskGraphNode &node : dag.nodes) {
    eligible[node.index] = !islandByTaskIndex.contains(node.index) &&
                           isEligiblePrePlacementIslandDigitalTask(node);
  }

  for (const ResourceEdge &edge : *resourceEdges) {
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

    if (mlir::failed(assignPrePlacementMinCutComponentIsland(
            dag, *resourceEdges, component, islandByTaskIndex)))
      return mlir::failure();
  }

  return mlir::success();
}

LogicalResult assignRemainingDigitalIslandsByLocalAffinity(
    const TaskGraphDAG &dag, llvm::ArrayRef<ResourceEdge> resourceEdges,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex) {
  llvm::SmallVector<llvm::SmallVector<const ResourceEdge *, 8>, 16>
      incidentEdges(dag.nodes.size());
  for (const ResourceEdge &edge : resourceEdges) {
    if (edge.byteSize <= 0)
      continue;
    if (edge.producerIndex < dag.nodes.size())
      incidentEdges[edge.producerIndex].push_back(&edge);
    if (edge.consumerIndex < dag.nodes.size())
      incidentEdges[edge.consumerIndex].push_back(&edge);
  }

  bool changed = false;
  do {
    changed = false;

    for (const TaskGraphNode &node : dag.nodes) {
      if (islandByTaskIndex.contains(node.index) ||
          !isEligiblePrePlacementIslandDigitalTask(node))
        continue;

      llvm::DenseMap<unsigned, int64_t> bytesByIsland;
      for (const ResourceEdge *edge : incidentEdges[node.index]) {
        unsigned otherIndex = edge->producerIndex == node.index
                                  ? edge->consumerIndex
                                  : edge->producerIndex;
        if (otherIndex >= dag.nodes.size())
          continue;

        auto islandIt = islandByTaskIndex.find(otherIndex);
        if (islandIt == islandByTaskIndex.end())
          continue;

        if (!sameNonEmptySourceLayer(node.op, dag.nodes[otherIndex].op))
          continue;

        bytesByIsland[islandIt->second] += edge->byteSize;
      }

      if (bytesByIsland.empty())
        continue;

      unsigned bestIsland = 0;
      int64_t bestBytes = std::numeric_limits<int64_t>::min();
      bool hasBest = false;
      for (const auto &entry : bytesByIsland) {
        if (!hasBest || entry.second > bestBytes ||
            (entry.second == bestBytes && entry.first < bestIsland)) {
          hasBest = true;
          bestIsland = entry.first;
          bestBytes = entry.second;
        }
      }

      if (!hasBest)
        continue;

      islandByTaskIndex[node.index] = bestIsland;
      changed = true;
    }
  } while (changed);

  return mlir::success();
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
