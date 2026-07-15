#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>
#include <queue>
#include <vector>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

uint64_t getDirectedEdgeKey(unsigned producer, unsigned consumer) {
  return (static_cast<uint64_t>(producer) << 32) |
         static_cast<uint64_t>(consumer);
}

LogicalResult addByteCount(Operation *anchor, int64_t bytes, int64_t &total) {
  if (bytes < 0 || total > std::numeric_limits<int64_t>::max() - bytes) {
    anchor->emitError("task graph byte count overflow while building execution "
                      "graph");
    return failure();
  }
  total += bytes;
  return success();
}

LogicalResult addOrUpdateEdge(TaskExecutionGraph &graph,
                              llvm::DenseMap<uint64_t, unsigned> &edgeByPair,
                              unsigned producer, unsigned consumer,
                              bool controlDependency, bool dataDependency,
                              int64_t transferredBytes, Operation *anchor) {
  uint64_t key = getDirectedEdgeKey(producer, consumer);
  auto existing = edgeByPair.find(key);
  if (existing != edgeByPair.end()) {
    TaskExecutionEdge &edge = graph.edges[existing->second];
    edge.controlDependency |= controlDependency;
    edge.dataDependency |= dataDependency;
    return addByteCount(anchor, transferredBytes, edge.transferredBytes);
  }

  unsigned edgeIndex = graph.edges.size();
  edgeByPair.try_emplace(key, edgeIndex);
  graph.edges.push_back(TaskExecutionEdge{producer, consumer, controlDependency,
                                          dataDependency, transferredBytes});
  return success();
}

LogicalResult buildTopologicalOrder(func::FuncOp taskGraphFunc,
                                    TaskExecutionGraph &graph) {
  llvm::SmallVector<unsigned, 16> indegree(graph.incomingEdges.size(), 0);
  std::priority_queue<unsigned, std::vector<unsigned>, std::greater<unsigned>>
      ready;

  for (unsigned taskIndex = 0; taskIndex < indegree.size(); ++taskIndex) {
    indegree[taskIndex] = graph.incomingEdges[taskIndex].size();
    if (indegree[taskIndex] == 0)
      ready.push(taskIndex);
  }

  while (!ready.empty()) {
    unsigned taskIndex = ready.top();
    ready.pop();
    graph.topologicalOrder.push_back(taskIndex);

    for (unsigned edgeIndex : graph.outgoingEdges[taskIndex]) {
      unsigned consumer = graph.edges[edgeIndex].consumerTask;
      if (--indegree[consumer] == 0)
        ready.push(consumer);
    }
  }

  if (graph.topologicalOrder.size() != graph.incomingEdges.size()) {
    taskGraphFunc.emitError("combined task dependency and resource dataflow "
                            "graph contains a cycle");
    return failure();
  }
  return success();
}

} // namespace

FailureOr<TaskExecutionGraph>
buildTaskExecutionGraph(func::FuncOp taskGraphFunc, const TaskGraphDAG &dag) {
  TaskExecutionGraph graph;
  llvm::DenseMap<uint64_t, unsigned> edgeByPair;

  for (const TaskGraphNode &consumer : dag.nodes) {
    for (unsigned producer : consumer.predecessors) {
      if (failed(addOrUpdateEdge(graph, edgeByPair, producer, consumer.index,
                                 true, false, 0, consumer.op)))
        return failure();
    }
  }

  auto resourceEdges = collectResourceEdges(dag);
  if (failed(resourceEdges))
    return failure();
  for (const ResourceEdge &edge : *resourceEdges) {
    if (failed(addOrUpdateEdge(graph, edgeByPair, edge.producerIndex,
                               edge.consumerIndex, false, true, edge.byteSize,
                               dag.nodes[edge.consumerIndex].op)))
      return failure();
  }

  llvm::sort(graph.edges,
             [](const TaskExecutionEdge &lhs, const TaskExecutionEdge &rhs) {
               if (lhs.producerTask != rhs.producerTask)
                 return lhs.producerTask < rhs.producerTask;
               return lhs.consumerTask < rhs.consumerTask;
             });

  graph.incomingEdges.resize(dag.nodes.size());
  graph.outgoingEdges.resize(dag.nodes.size());
  for (auto indexedEdge : llvm::enumerate(graph.edges)) {
    const TaskExecutionEdge &edge = indexedEdge.value();
    graph.outgoingEdges[edge.producerTask].push_back(indexedEdge.index());
    graph.incomingEdges[edge.consumerTask].push_back(indexedEdge.index());
    graph.controlEdgeCount += edge.controlDependency;
    graph.dataEdgeCount += edge.dataDependency;
    if (failed(addByteCount(dag.nodes[edge.consumerTask].op,
                            edge.transferredBytes, graph.totalDataBytes)))
      return failure();
  }

  if (failed(buildTopologicalOrder(taskGraphFunc, graph)))
    return failure();
  return graph;
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
