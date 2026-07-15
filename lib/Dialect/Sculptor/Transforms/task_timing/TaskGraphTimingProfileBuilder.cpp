#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfileBuilder.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlir {
namespace sculptor {
namespace task_timing {
namespace {

static uint64_t getDirectedEdgeKey(unsigned producer, unsigned consumer) {
  return (static_cast<uint64_t>(producer) << 32) |
         static_cast<uint64_t>(consumer);
}

static void computeReverseCriticalPaths(TimingAnalysis &analysis) {
  for (unsigned taskIndex : llvm::reverse(analysis.topologicalOrder)) {
    double successorPathNs = 0.0;
    for (unsigned edgeIndex : analysis.outgoingEdges[taskIndex]) {
      const ExecutionEdge &edge = analysis.executionEdges[edgeIndex];
      unsigned successor = edge.consumerTask;
      successorPathNs =
          std::max(successorPathNs,
                   edge.networkLatencyNs +
                       analysis.tasks[successor].criticalPathRemainingNs);
    }
    analysis.tasks[taskIndex].criticalPathRemainingNs =
        analysis.tasks[taskIndex].intrinsicLatencyNs + successorPathNs;
  }

  for (const TaskTiming &timing : analysis.tasks)
    analysis.criticalPathNs =
        std::max(analysis.criticalPathNs, timing.earliestFinishNs);

  double tolerance = std::max(1.0, analysis.criticalPathNs) * 1.0e-9;
  for (TaskTiming &timing : analysis.tasks) {
    double throughTask =
        timing.earliestStartNs + timing.criticalPathRemainingNs;
    timing.slackNs = std::max(0.0, analysis.criticalPathNs - throughTask);
    timing.isCritical = timing.slackNs <= tolerance;
  }
}

static void
buildIslandTiming(const task_graph::LogicalPlacementIslandGraph &islandGraph,
                  TimingAnalysis &analysis) {
  analysis.islands.reserve(islandGraph.islands.size());
  for (const task_graph::LogicalPlacementIsland &island : islandGraph.islands) {
    IslandTimingProfile islandTiming;
    islandTiming.islandId = island.islandIndex;
    islandTiming.taskCount = island.taskIndices.size();
    islandTiming.earliestStartNs = std::numeric_limits<double>::infinity();
    islandTiming.slackNs = std::numeric_limits<double>::infinity();
    for (unsigned taskIndex : island.taskIndices) {
      const TaskTiming &taskTiming = analysis.tasks[taskIndex];
      islandTiming.totalWorkNs += taskTiming.intrinsicLatencyNs;
      islandTiming.earliestStartNs =
          std::min(islandTiming.earliestStartNs, taskTiming.earliestStartNs);
      islandTiming.earliestFinishNs =
          std::max(islandTiming.earliestFinishNs, taskTiming.earliestFinishNs);
      islandTiming.criticalPathRemainingNs =
          std::max(islandTiming.criticalPathRemainingNs,
                   taskTiming.criticalPathRemainingNs);
      islandTiming.slackNs = std::min(islandTiming.slackNs, taskTiming.slackNs);
      islandTiming.isCritical |= taskTiming.isCritical;
      double analogWorkNs = taskTiming.analogLoadLatencyNs +
                            taskTiming.analogExecuteLatencyNs +
                            taskTiming.analogStoreLatencyNs;
      islandTiming.analogWorkNs += analogWorkNs;
      islandTiming.digitalWorkNs +=
          std::max(0.0, taskTiming.intrinsicLatencyNs - analogWorkNs);
    }
    if (island.taskIndices.empty()) {
      islandTiming.earliestStartNs = 0.0;
      islandTiming.slackNs = 0.0;
    }
    analysis.islands.push_back(islandTiming);
  }
}

struct TimedEdgeAccumulator {
  double producerReadyTimeNs = 0.0;
  double minimumConsumerSlackNs = std::numeric_limits<double>::infinity();
  double maximumConsumerCriticalPathRemainingNs = 0.0;
  bool foundTaskEdge = false;
};

static void buildTimedIslandEdges(
    const task_graph::LogicalPlacementIslandGraph &islandGraph,
    const TimingModel &model, TimingAnalysis &analysis) {
  llvm::DenseMap<uint64_t, TimedEdgeAccumulator> accumulatorByPair;

  for (const ExecutionEdge &edge : analysis.executionEdges) {
    if (!edge.dataDependency || edge.transferredBytes <= 0)
      continue;
    auto producerIsland = islandGraph.islandByTaskIndex.find(edge.producerTask);
    auto consumerIsland = islandGraph.islandByTaskIndex.find(edge.consumerTask);
    if (producerIsland == islandGraph.islandByTaskIndex.end() ||
        consumerIsland == islandGraph.islandByTaskIndex.end() ||
        producerIsland->second == consumerIsland->second)
      continue;

    TimedEdgeAccumulator &accumulator = accumulatorByPair[getDirectedEdgeKey(
        producerIsland->second, consumerIsland->second)];
    const TaskTiming &producer = analysis.tasks[edge.producerTask];
    const TaskTiming &consumer = analysis.tasks[edge.consumerTask];
    accumulator.foundTaskEdge = true;
    accumulator.producerReadyTimeNs =
        std::max(accumulator.producerReadyTimeNs, producer.earliestFinishNs);
    accumulator.minimumConsumerSlackNs =
        std::min(accumulator.minimumConsumerSlackNs, consumer.slackNs);
    accumulator.maximumConsumerCriticalPathRemainingNs =
        std::max(accumulator.maximumConsumerCriticalPathRemainingNs,
                 consumer.criticalPathRemainingNs);
  }

  llvm::DenseMap<unsigned, const IslandTimingProfile *> timingByIsland;
  for (const IslandTimingProfile &island : analysis.islands)
    timingByIsland[island.islandId] = &island;

  analysis.timedIslandEdges.reserve(islandGraph.executionGraph.edges.size());
  for (const task_graph::IslandExecutionEdge &edge :
       islandGraph.executionGraph.edges) {
    TimedIslandEdge timedEdge;
    timedEdge.producerIsland = edge.producerIsland;
    timedEdge.consumerIsland = edge.consumerIsland;
    timedEdge.bytes = edge.transferredBytes;
    timedEdge.estimatedTransferNsPerHop =
        estimateNetworkTransferLatencyNs(edge.transferredBytes, 1, model);
    timedEdge.estimatedAdditionalHopNs =
        estimateNetworkTransferLatencyNs(edge.transferredBytes, 2, model) -
        timedEdge.estimatedTransferNsPerHop;

    TimedEdgeAccumulator &accumulator = accumulatorByPair[getDirectedEdgeKey(
        edge.producerIsland, edge.consumerIsland)];
    const IslandTimingProfile *producer =
        timingByIsland.lookup(edge.producerIsland);
    const IslandTimingProfile *consumer =
        timingByIsland.lookup(edge.consumerIsland);
    timedEdge.producerReadyTimeNs = accumulator.foundTaskEdge
                                        ? accumulator.producerReadyTimeNs
                                        : producer->earliestFinishNs;
    double consumerSlackNs = accumulator.foundTaskEdge
                                 ? accumulator.minimumConsumerSlackNs
                                 : consumer->slackNs;
    double consumerCriticalPathRemainingNs =
        accumulator.foundTaskEdge
            ? accumulator.maximumConsumerCriticalPathRemainingNs
            : consumer->criticalPathRemainingNs;
    if (analysis.criticalPathNs > 0.0) {
      timedEdge.consumerTimingPressure =
          1.0 - std::clamp(consumerSlackNs / analysis.criticalPathNs, 0.0, 1.0);
      double throughEdge = timedEdge.producerReadyTimeNs +
                           timedEdge.estimatedTransferNsPerHop +
                           consumerCriticalPathRemainingNs;
      double edgeSlack = std::max(0.0, analysis.criticalPathNs - throughEdge);
      timedEdge.criticality =
          1.0 - std::clamp(edgeSlack / analysis.criticalPathNs, 0.0, 1.0);
    }
    analysis.timedIslandEdges.push_back(timedEdge);
  }
}

} // namespace

void buildSchedulingTimingProfile(
    const task_graph::LogicalPlacementIslandGraph &islandGraph,
    const TimingModel &model, TimingAnalysis &analysis) {
  computeReverseCriticalPaths(analysis);
  buildIslandTiming(islandGraph, analysis);
  buildTimedIslandEdges(islandGraph, model, analysis);
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
