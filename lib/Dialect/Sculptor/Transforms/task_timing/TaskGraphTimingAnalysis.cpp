#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDigitalOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphNetworkTiming.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfileBuilder.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskLatencyModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {
namespace {

using task_graph::LogicalPlacementIslandGraph;
using task_graph::TaskGraphDAG;

static FailureOr<int64_t> estimateReplacementOps(
    ModuleOp module, sculptor::TaskCreateOp task,
    const task_graph::ResourceProducerMap &producerByResource,
    MVMCostMode mvmCostMode) {
  if (task_graph::isAnalogComputeTask(task)) {
    return task_graph::estimateDigitalReplacementOps(module, task,
                                                     producerByResource);
  }

  if (task.getTaskKind() == task_graph_names::kStreamingConvolutionTaskKind) {
    auto replacement = task->getAttrOfType<IntegerAttr>(
        runtime_attrs::kTaskDigitalReplacementOpsAttrName);
    if (replacement) {
      if (replacement.getInt() < 0) {
        return task.emitOpError(
            "expected non-negative digital replacement operation count");
      }
      return replacement.getInt();
    }
    if (mvmCostMode == MVMCostMode::Digital) {
      return task.emitOpError(
          "digital MVM cost mode requires an explicit "
          "'sculptor.runtime.digital_replacement_ops' count for a mixed "
          "streaming-convolution task");
    }
  }
  return int64_t{0};
}

static FailureOr<TaskTiming>
analyzeTask(ModuleOp module, const TaskGraphDAG &dag,
            const TimingAnalysis &analysis, unsigned taskIndex,
            unsigned topologicalIndex, const TimingModel &model,
            MVMCostMode mvmCostMode,
            const task_graph::ResourceProducerMap &producerByResource) {
  TaskTiming timing;
  timing.topologicalIndex = topologicalIndex;

  for (unsigned edgeIndex : analysis.incomingEdges[taskIndex]) {
    const ExecutionEdge &edge = analysis.executionEdges[edgeIndex];
    timing.workload.controlPredecessorCount += edge.controlDependency;
    timing.workload.dataPredecessorCount += edge.dataDependency;
    timing.workload.incomingDataBytes += edge.transferredBytes;
    const TaskTiming &predecessor = analysis.tasks[edge.producerTask];
    timing.dependencyDepth =
        std::max(timing.dependencyDepth, predecessor.dependencyDepth + 1);
    timing.earliestStartNs =
        std::max(timing.earliestStartNs, predecessor.earliestFinishNs);
  }
  for (unsigned edgeIndex : analysis.outgoingEdges[taskIndex])
    timing.workload.outgoingDataBytes +=
        analysis.executionEdges[edgeIndex].transferredBytes;

  FailureOr<int64_t> digitalOps =
      task_graph::estimateTaskDigitalOps(module, dag.nodes[taskIndex].op);
  if (failed(digitalOps))
    return failure();
  timing.workload.digitalOps = *digitalOps;

  FailureOr<int64_t> digitalReplacementOps = estimateReplacementOps(
      module, dag.nodes[taskIndex].op, producerByResource, mvmCostMode);
  if (failed(digitalReplacementOps))
    return failure();
  timing.workload.digitalReplacementOps = *digitalReplacementOps;

  int64_t effectiveDigitalOps = timing.workload.digitalOps;
  if (mvmCostMode == MVMCostMode::Digital) {
    std::optional<int64_t> sum = llvm::checkedAdd(
        effectiveDigitalOps, timing.workload.digitalReplacementOps);
    if (!sum) {
      dag.nodes[taskIndex].op->emitError(
          "effective digital operation count overflow");
      return failure();
    }
    effectiveDigitalOps = *sum;
  }

  FailureOr<TaskLatencyEstimate> latency = estimateTaskLatency(
      module, dag.nodes[taskIndex].op, timing.workload, effectiveDigitalOps,
      mvmCostMode == MVMCostMode::Digital
          ? timing.workload.digitalReplacementOps
          : 0,
      model, mvmCostMode);
  if (failed(latency))
    return failure();
  timing.cost = latency->cost;
  timing.analogLoadLatencyNs = latency->analogLoadLatencyNs;
  timing.analogExecuteLatencyNs = latency->analogExecuteLatencyNs;
  timing.analogStoreLatencyNs = latency->analogStoreLatencyNs;
  timing.analogPipelineLatencyNs = latency->analogPipelineLatencyNs;
  timing.intrinsicLatencyNs = latency->intrinsicLatencyNs;
  timing.earliestFinishNs = timing.earliestStartNs + timing.intrinsicLatencyNs;
  return timing;
}

} // namespace

FailureOr<TimingAnalysis>
analyzeTaskGraphTiming(ModuleOp module, func::FuncOp taskGraphFunc,
                       const TaskGraphDAG &dag,
                       const task_graph::TaskExecutionGraph &executionGraph,
                       const LogicalPlacementIslandGraph &islandGraph,
                       const TimingModel &model, MVMCostMode mvmCostMode) {
  TimingAnalysis analysis;
  analysis.mvmCostMode = mvmCostMode;
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

  task_graph::ResourceProducerMap producerByResource;
  if (failed(task_graph::collectResourceProducers(dag, producerByResource)))
    return failure();

  for (auto indexedTask : llvm::enumerate(analysis.topologicalOrder)) {
    unsigned taskIndex = indexedTask.value();
    FailureOr<TaskTiming> timing =
        analyzeTask(module, dag, analysis, taskIndex, indexedTask.index(),
                    model, mvmCostMode, producerByResource);
    if (failed(timing))
      return failure();
    analysis.tasks[taskIndex] = *timing;
    analysis.sumTaskWorkNs += timing->intrinsicLatencyNs;
    std::optional<int64_t> totalReplacement =
        llvm::checkedAdd(analysis.totalDigitalReplacementOps,
                         timing->workload.digitalReplacementOps);
    if (!totalReplacement) {
      dag.nodes[taskIndex].op->emitError(
          "total digital replacement operation count overflow");
      return failure();
    }
    analysis.totalDigitalReplacementOps = *totalReplacement;
    analysis.executionDepth =
        std::max(analysis.executionDepth, timing->dependencyDepth + 1);
  }

  if (failed(applyPlacementAwareNetworkTimingIfAvailable(taskGraphFunc, dag,
                                                         model, analysis)))
    return failure();

  buildSchedulingTimingProfile(islandGraph, model, analysis);
  if (!analysis.placementAware) {
    analysis.noContentionMakespanNs = analysis.criticalPathNs;
    analysis.zeroNetworkMakespanNs = analysis.criticalPathNs;
  }
  return analysis;
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
