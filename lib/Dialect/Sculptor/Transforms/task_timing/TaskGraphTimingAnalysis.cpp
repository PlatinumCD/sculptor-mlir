#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDigitalOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskLatencyModel.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphNetworkTiming.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfileBuilder.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {
namespace {

using task_graph::LogicalPlacementIslandGraph;
using task_graph::TaskGraphDAG;

static FailureOr<TaskTiming>
analyzeTask(ModuleOp module, const TaskGraphDAG &dag,
            const TimingAnalysis &analysis, unsigned taskIndex,
            unsigned topologicalIndex, const TimingModel &model) {
  TaskTiming timing;
  timing.topologicalIndex = topologicalIndex;

  for (unsigned edgeIndex : analysis.incomingEdges[taskIndex]) {
    const ExecutionEdge &edge = analysis.executionEdges[edgeIndex];
    timing.controlPredecessorCount += edge.controlDependency;
    timing.dataPredecessorCount += edge.dataDependency;
    timing.incomingDataBytes += edge.transferredBytes;
    const TaskTiming &predecessor = analysis.tasks[edge.producerTask];
    timing.dependencyDepth =
        std::max(timing.dependencyDepth, predecessor.dependencyDepth + 1);
    timing.earliestStartNs =
        std::max(timing.earliestStartNs, predecessor.earliestFinishNs);
  }
  for (unsigned edgeIndex : analysis.outgoingEdges[taskIndex])
    timing.outgoingDataBytes +=
        analysis.executionEdges[edgeIndex].transferredBytes;

  FailureOr<int64_t> digitalOps =
      task_graph::estimateTaskDigitalOps(module, dag.nodes[taskIndex].op);
  if (failed(digitalOps))
    return failure();
  timing.digitalOps = *digitalOps;

  FailureOr<TaskLatencyEstimate> latency =
      estimateTaskLatency(dag.nodes[taskIndex].op, *digitalOps, model);
  if (failed(latency))
    return failure();
  timing.analogLoadLatencyNs = latency->analogLoadLatencyNs;
  timing.analogExecuteLatencyNs = latency->analogExecuteLatencyNs;
  timing.analogStoreLatencyNs = latency->analogStoreLatencyNs;
  timing.intrinsicLatencyNs = latency->intrinsicLatencyNs;
  timing.earliestFinishNs = timing.earliestStartNs + timing.intrinsicLatencyNs;
  return timing;
}

} // namespace

FailureOr<TimingAnalysis> analyzeTaskGraphTiming(
    ModuleOp module, func::FuncOp taskGraphFunc, const TaskGraphDAG &dag,
    const task_graph::TaskExecutionGraph &executionGraph,
    const LogicalPlacementIslandGraph &islandGraph, const TimingModel &model) {
  TimingAnalysis analysis;
  analysis.tasks.resize(dag.nodes.size());
  analysis.incomingEdges = executionGraph.incomingEdges;
  analysis.outgoingEdges = executionGraph.outgoingEdges;
  analysis.topologicalOrder = executionGraph.topologicalOrder;
  analysis.controlEdgeCount = executionGraph.controlEdgeCount;
  analysis.dataEdgeCount = executionGraph.dataEdgeCount;
  analysis.totalDataBytes = executionGraph.totalDataBytes;
  analysis.executionEdges.reserve(executionGraph.edges.size());
  for (const task_graph::TaskExecutionEdge &edge : executionGraph.edges) {
    analysis.executionEdges.push_back(ExecutionEdge{
        edge.producerTask, edge.consumerTask, edge.controlDependency,
        edge.dataDependency, edge.transferredBytes});
  }

  for (auto indexedTask : llvm::enumerate(analysis.topologicalOrder)) {
    unsigned taskIndex = indexedTask.value();
    FailureOr<TaskTiming> timing = analyzeTask(module, dag, analysis, taskIndex,
                                               indexedTask.index(), model);
    if (failed(timing))
      return failure();
    analysis.tasks[taskIndex] = *timing;
    analysis.executionDepth =
        std::max(analysis.executionDepth, timing->dependencyDepth + 1);
  }

  if (failed(applyPlacementAwareNetworkTimingIfAvailable(
          taskGraphFunc, dag, model, analysis)))
    return failure();

  buildSchedulingTimingProfile(islandGraph, model, analysis);
  return analysis;
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
