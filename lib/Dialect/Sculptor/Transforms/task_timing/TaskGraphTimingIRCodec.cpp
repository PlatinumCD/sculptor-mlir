#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTimingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {
using task_graph::LogicalPlacementIslandGraph;
using task_graph::TaskGraphDAG;

void attachTaskGraphTimingAnalysis(func::FuncOp taskGraphFunc,
                                   const TaskGraphDAG &dag,
                                   const TimingAnalysis &analysis,
                                   const TimingModel &model) {
  Builder builder(taskGraphFunc.getContext());
  auto i64Attr = [&](int64_t value) {
    return builder.getI64IntegerAttr(value);
  };
  auto f64Attr = [&](double value) { return builder.getF64FloatAttr(value); };

  for (const task_graph::TaskGraphNode &node : dag.nodes) {
    const TaskTiming &timing = analysis.tasks[node.index];
    node.op->setAttr(timing_attrs::kTopologicalIndexAttrName,
                     i64Attr(timing.topologicalIndex));
    node.op->setAttr(timing_attrs::kDependencyDepthAttrName,
                     i64Attr(timing.dependencyDepth));
    node.op->setAttr(timing_attrs::kControlPredecessorCountAttrName,
                     i64Attr(timing.controlPredecessorCount));
    node.op->setAttr(timing_attrs::kDataPredecessorCountAttrName,
                     i64Attr(timing.dataPredecessorCount));
    node.op->setAttr(timing_attrs::kIncomingDataBytesAttrName,
                     i64Attr(timing.incomingDataBytes));
    node.op->setAttr(timing_attrs::kOutgoingDataBytesAttrName,
                     i64Attr(timing.outgoingDataBytes));
    node.op->setAttr(timing_attrs::kDigitalOpsAttrName,
                     i64Attr(timing.digitalOps));
    node.op->setAttr(timing_attrs::kAnalogLoadLatencyNsAttrName,
                     f64Attr(timing.analogLoadLatencyNs));
    node.op->setAttr(timing_attrs::kAnalogExecuteLatencyNsAttrName,
                     f64Attr(timing.analogExecuteLatencyNs));
    node.op->setAttr(timing_attrs::kAnalogStoreLatencyNsAttrName,
                     f64Attr(timing.analogStoreLatencyNs));
    node.op->setAttr(timing_attrs::kIntrinsicLatencyNsAttrName,
                     f64Attr(timing.intrinsicLatencyNs));
    node.op->setAttr(timing_attrs::kEarliestStartNsAttrName,
                     f64Attr(timing.earliestStartNs));
    node.op->setAttr(timing_attrs::kEarliestFinishNsAttrName,
                     f64Attr(timing.earliestFinishNs));
    node.op->setAttr(timing_attrs::kCriticalPathRemainingNsAttrName,
                     f64Attr(timing.criticalPathRemainingNs));
    node.op->setAttr(timing_attrs::kSlackNsAttrName, f64Attr(timing.slackNs));
    node.op->setAttr(timing_attrs::kIncomingNetworkDelayNsAttrName,
                     f64Attr(timing.incomingNetworkDelayNs));
    node.op->setAttr(timing_attrs::kIsCriticalAttrName,
                     builder.getBoolAttr(timing.isCritical));
  }

  taskGraphFunc->setAttr(timing_attrs::kTaskCountAttrName,
                         i64Attr(dag.nodes.size()));
  taskGraphFunc->setAttr(timing_attrs::kExecutionEdgeCountAttrName,
                         i64Attr(analysis.executionEdges.size()));
  taskGraphFunc->setAttr(timing_attrs::kControlEdgeCountAttrName,
                         i64Attr(analysis.controlEdgeCount));
  taskGraphFunc->setAttr(timing_attrs::kDataEdgeCountAttrName,
                         i64Attr(analysis.dataEdgeCount));
  taskGraphFunc->setAttr(timing_attrs::kExecutionDepthAttrName,
                         i64Attr(analysis.executionDepth));
  taskGraphFunc->setAttr(timing_attrs::kCriticalPathNsAttrName,
                         f64Attr(analysis.criticalPathNs));
  taskGraphFunc->setAttr(timing_attrs::kTotalDataBytesAttrName,
                         i64Attr(analysis.totalDataBytes));
  taskGraphFunc->setAttr(timing_attrs::kPlacementAwareAttrName,
                         builder.getBoolAttr(analysis.placementAware));
  taskGraphFunc->setAttr(timing_attrs::kTotalNetworkLatencyNsAttrName,
                         f64Attr(analysis.totalNetworkLatencyNs));
  taskGraphFunc->setAttr(timing_attrs::kTotalNetworkContentionDelayNsAttrName,
                         f64Attr(analysis.totalNetworkContentionDelayNs));

  llvm::SmallVector<Attribute, 16> networkEdges;
  networkEdges.reserve(analysis.executionEdges.size());
  for (const ExecutionEdge &edge : analysis.executionEdges) {
    networkEdges.push_back(NetworkEdgeTimingAttr::get(
        taskGraphFunc.getContext(), i64Attr(edge.producerTask),
        i64Attr(edge.consumerTask), i64Attr(edge.sourceCore),
        i64Attr(edge.destinationCore), i64Attr(edge.meshHops),
        f64Attr(edge.transferStartNs), f64Attr(edge.transferFinishNs),
        f64Attr(edge.networkLatencyNs), f64Attr(edge.contentionDelayNs)));
  }
  taskGraphFunc->setAttr(timing_attrs::kNetworkEdgesAttrName,
                         builder.getArrayAttr(networkEdges));

  llvm::SmallVector<Attribute, 16> islandProfiles;
  islandProfiles.reserve(analysis.islands.size());
  for (const IslandTimingProfile &island : analysis.islands) {
    islandProfiles.push_back(IslandTimingAttr::get(
        taskGraphFunc.getContext(), i64Attr(island.islandId),
        i64Attr(island.taskCount), f64Attr(island.totalWorkNs),
        f64Attr(island.analogWorkNs), f64Attr(island.digitalWorkNs),
        f64Attr(island.earliestStartNs), f64Attr(island.earliestFinishNs),
        f64Attr(island.criticalPathRemainingNs), f64Attr(island.slackNs),
        builder.getBoolAttr(island.isCritical)));
  }
  taskGraphFunc->setAttr(timing_attrs::kIslandProfilesAttrName,
                         builder.getArrayAttr(islandProfiles));

  llvm::SmallVector<Attribute, 16> timedIslandEdges;
  timedIslandEdges.reserve(analysis.timedIslandEdges.size());
  for (const TimedIslandEdge &edge : analysis.timedIslandEdges) {
    timedIslandEdges.push_back(TimedIslandEdgeAttr::get(
        taskGraphFunc.getContext(), i64Attr(edge.producerIsland),
        i64Attr(edge.consumerIsland), i64Attr(edge.bytes),
        f64Attr(edge.estimatedTransferNsPerHop),
        f64Attr(edge.estimatedAdditionalHopNs), f64Attr(edge.criticality),
        f64Attr(edge.producerReadyTimeNs),
        f64Attr(edge.consumerTimingPressure)));
  }
  taskGraphFunc->setAttr(timing_attrs::kTimedIslandEdgesAttrName,
                         builder.getArrayAttr(timedIslandEdges));

  taskGraphFunc->setAttr(
      timing_attrs::kTimingModelAttrName,
      TimingModelAttr::get(
          taskGraphFunc.getContext(), i64Attr(model.analogMVMLatencyNs),
          i64Attr(model.analogIOBitsPerCycle),
          builder.getBoolAttr(model.analogIOShared),
          f64Attr(model.digitalClockGHz), i64Attr(model.digitalIssueWidth),
          i64Attr(model.digitalVectorBitsPerCycle),
          i64Attr(model.networkLinkBitsPerCycle),
          i64Attr(model.networkHopLatencyCycles),
          builder.getBoolAttr(model.networkPipelined)));
}

namespace {

template <typename AttrT>
static FailureOr<AttrT> getRequiredAttr(func::FuncOp taskGraphFunc,
                                        StringRef name) {
  AttrT attr = taskGraphFunc->getAttrOfType<AttrT>(name);
  if (!attr) {
    taskGraphFunc.emitError("expected pre-placement timing attribute '")
        << name << "'";
    return failure();
  }
  return attr;
}

static FailureOr<int64_t> getRequiredI64(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr) {
    op->emitError("expected pre-placement timing attribute '") << name << "'";
    return failure();
  }
  return attr.getInt();
}

static FailureOr<double> getRequiredF64(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<FloatAttr>(name);
  if (!attr) {
    op->emitError("expected pre-placement timing attribute '") << name << "'";
    return failure();
  }
  double value = attr.getValueAsDouble();
  if (!std::isfinite(value) || value < 0.0) {
    op->emitError("expected finite non-negative timing attribute '")
        << name << "'";
    return failure();
  }
  return value;
}

static FailureOr<bool> getRequiredBool(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<BoolAttr>(name);
  if (!attr) {
    op->emitError("expected pre-placement timing attribute '") << name << "'";
    return failure();
  }
  return attr.getValue();
}

static LogicalResult validateRecordCount(func::FuncOp taskGraphFunc,
                                         StringRef name, size_t actual,
                                         size_t expected) {
  if (actual == expected)
    return success();
  taskGraphFunc.emitError("timing profile attribute '")
      << name << "' has " << actual << " entries; expected " << expected;
  return failure();
}

static FailureOr<double> getFiniteNonNegative(func::FuncOp taskGraphFunc,
                                              FloatAttr attr,
                                              StringRef recordName,
                                              StringRef fieldName) {
  double value = attr.getValueAsDouble();
  if (std::isfinite(value) && value >= 0.0)
    return value;
  taskGraphFunc.emitError("expected '")
      << recordName << "." << fieldName
      << "' to be a finite non-negative value";
  return failure();
}

} // namespace

FailureOr<TimingModel> loadTimingModel(func::FuncOp taskGraphFunc) {
  auto timingModelAttr = getRequiredAttr<TimingModelAttr>(
      taskGraphFunc, timing_attrs::kTimingModelAttrName);
  if (failed(timingModelAttr))
    return failure();

  TimingModel model;
  model.analogMVMLatencyNs = timingModelAttr->getAnalogMVMLatencyNs().getInt();
  model.analogIOBitsPerCycle =
      timingModelAttr->getAnalogIOBitsPerCycle().getInt();
  model.analogIOShared = timingModelAttr->getAnalogIOShared().getValue();
  model.digitalClockGHz =
      timingModelAttr->getDigitalClockGHz().getValueAsDouble();
  model.digitalIssueWidth = timingModelAttr->getDigitalIssueWidth().getInt();
  model.digitalVectorBitsPerCycle =
      timingModelAttr->getDigitalVectorBitsPerCycle().getInt();
  model.networkLinkBitsPerCycle =
      timingModelAttr->getNetworkLinkBitsPerCycle().getInt();
  model.networkHopLatencyCycles =
      timingModelAttr->getNetworkHopLatencyCycles().getInt();
  model.networkPipelined = timingModelAttr->getNetworkPipelined().getValue();
  if (failed(validateTimingModel(taskGraphFunc, model)))
    return failure();
  return model;
}

FailureOr<SchedulingTimingProfile>
loadSchedulingTimingProfile(func::FuncOp taskGraphFunc, const TaskGraphDAG &dag,
                            const LogicalPlacementIslandGraph &islandGraph) {
  auto placementAware =
      getRequiredBool(taskGraphFunc, timing_attrs::kPlacementAwareAttrName);
  if (failed(placementAware))
    return failure();
  if (*placementAware) {
    taskGraphFunc.emitError(
        "expected a pre-placement timing profile before scheduling");
    return failure();
  }

  auto taskCount =
      getRequiredI64(taskGraphFunc, timing_attrs::kTaskCountAttrName);
  auto criticalPathNs =
      getRequiredF64(taskGraphFunc, timing_attrs::kCriticalPathNsAttrName);
  if (failed(taskCount) || failed(criticalPathNs))
    return failure();
  if (*taskCount < 0 || static_cast<size_t>(*taskCount) != dag.nodes.size()) {
    taskGraphFunc.emitError(
        "timing profile task count does not match task DAG");
    return failure();
  }

  SchedulingTimingProfile profile;
  profile.criticalPathNs = *criticalPathNs;
  profile.tasks.resize(dag.nodes.size());
  llvm::SmallVector<bool, 16> seenTopologicalIndex(dag.nodes.size(), false);
  for (const task_graph::TaskGraphNode &node : dag.nodes) {
    TaskTiming timing;
    auto topologicalIndex =
        getRequiredI64(node.op, timing_attrs::kTopologicalIndexAttrName);
    auto dependencyDepth =
        getRequiredI64(node.op, timing_attrs::kDependencyDepthAttrName);
    auto controlPredecessors =
        getRequiredI64(node.op, timing_attrs::kControlPredecessorCountAttrName);
    auto dataPredecessors =
        getRequiredI64(node.op, timing_attrs::kDataPredecessorCountAttrName);
    auto incomingBytes =
        getRequiredI64(node.op, timing_attrs::kIncomingDataBytesAttrName);
    auto outgoingBytes =
        getRequiredI64(node.op, timing_attrs::kOutgoingDataBytesAttrName);
    auto digitalOps =
        getRequiredI64(node.op, timing_attrs::kDigitalOpsAttrName);
    auto analogLoad =
        getRequiredF64(node.op, timing_attrs::kAnalogLoadLatencyNsAttrName);
    auto analogExecute =
        getRequiredF64(node.op, timing_attrs::kAnalogExecuteLatencyNsAttrName);
    auto analogStore =
        getRequiredF64(node.op, timing_attrs::kAnalogStoreLatencyNsAttrName);
    auto intrinsic =
        getRequiredF64(node.op, timing_attrs::kIntrinsicLatencyNsAttrName);
    auto earliestStart =
        getRequiredF64(node.op, timing_attrs::kEarliestStartNsAttrName);
    auto earliestFinish =
        getRequiredF64(node.op, timing_attrs::kEarliestFinishNsAttrName);
    auto criticalRemaining =
        getRequiredF64(node.op, timing_attrs::kCriticalPathRemainingNsAttrName);
    auto slack = getRequiredF64(node.op, timing_attrs::kSlackNsAttrName);
    auto incomingNetworkDelay =
        getRequiredF64(node.op, timing_attrs::kIncomingNetworkDelayNsAttrName);
    auto isCritical =
        getRequiredBool(node.op, timing_attrs::kIsCriticalAttrName);
    if (failed(topologicalIndex) || failed(dependencyDepth) ||
        failed(controlPredecessors) || failed(dataPredecessors) ||
        failed(incomingBytes) || failed(outgoingBytes) || failed(digitalOps) ||
        failed(analogLoad) || failed(analogExecute) || failed(analogStore) ||
        failed(intrinsic) || failed(earliestStart) || failed(earliestFinish) ||
        failed(criticalRemaining) || failed(slack) ||
        failed(incomingNetworkDelay) || failed(isCritical))
      return failure();
    if (*topologicalIndex < 0 || *dependencyDepth < 0 ||
        *controlPredecessors < 0 || *dataPredecessors < 0 ||
        *incomingBytes < 0 || *outgoingBytes < 0 || *digitalOps < 0) {
      node.op->emitError("expected non-negative task timing counters");
      return failure();
    }
    if (node.index >= profile.tasks.size() ||
        static_cast<size_t>(*topologicalIndex) >= profile.tasks.size() ||
        seenTopologicalIndex[*topologicalIndex]) {
      node.op->emitError(
          "task timing profile has invalid or duplicate topological index");
      return failure();
    }
    seenTopologicalIndex[*topologicalIndex] = true;
    timing.topologicalIndex = *topologicalIndex;
    timing.dependencyDepth = *dependencyDepth;
    timing.controlPredecessorCount = *controlPredecessors;
    timing.dataPredecessorCount = *dataPredecessors;
    timing.incomingDataBytes = *incomingBytes;
    timing.outgoingDataBytes = *outgoingBytes;
    timing.digitalOps = *digitalOps;
    timing.analogLoadLatencyNs = *analogLoad;
    timing.analogExecuteLatencyNs = *analogExecute;
    timing.analogStoreLatencyNs = *analogStore;
    timing.intrinsicLatencyNs = *intrinsic;
    timing.earliestStartNs = *earliestStart;
    timing.earliestFinishNs = *earliestFinish;
    timing.criticalPathRemainingNs = *criticalRemaining;
    timing.slackNs = *slack;
    timing.incomingNetworkDelayNs = *incomingNetworkDelay;
    timing.isCritical = *isCritical;
    profile.tasks[node.index] = timing;
  }

  auto islandAttrs = getRequiredAttr<ArrayAttr>(
      taskGraphFunc, timing_attrs::kIslandProfilesAttrName);
  if (failed(islandAttrs))
    return failure();

  size_t islandCount = islandGraph.islands.size();
  if (failed(validateRecordCount(taskGraphFunc,
                                 timing_attrs::kIslandProfilesAttrName,
                                 islandAttrs->size(), islandCount)))
    return failure();

  profile.islands.reserve(islandCount);
  for (auto indexedIsland : llvm::enumerate(islandGraph.islands)) {
    size_t index = indexedIsland.index();
    const auto &logicalIsland = indexedIsland.value();
    auto attr = dyn_cast<IslandTimingAttr>((*islandAttrs)[index]);
    if (!attr) {
      taskGraphFunc.emitError("expected '")
          << timing_attrs::kIslandProfilesAttrName
          << "' to contain #sculptor.island_timing records";
      return failure();
    }
    int64_t islandId = attr.getIslandId().getInt();
    int64_t taskCount = attr.getTaskCount().getInt();
    if (islandId < 0 ||
        static_cast<unsigned>(islandId) != logicalIsland.islandIndex ||
        taskCount < 0 ||
        static_cast<size_t>(taskCount) != logicalIsland.taskIndices.size()) {
      taskGraphFunc.emitError(
          "timing profile island metadata does not match logical islands");
      return failure();
    }
    auto totalWork = getFiniteNonNegative(taskGraphFunc, attr.getTotalWorkNs(),
                                          "island_timing", "totalWorkNs");
    auto analogWork = getFiniteNonNegative(
        taskGraphFunc, attr.getAnalogWorkNs(), "island_timing", "analogWorkNs");
    auto digitalWork =
        getFiniteNonNegative(taskGraphFunc, attr.getDigitalWorkNs(),
                             "island_timing", "digitalWorkNs");
    auto earliestStart =
        getFiniteNonNegative(taskGraphFunc, attr.getEarliestStartNs(),
                             "island_timing", "earliestStartNs");
    auto earliestFinish =
        getFiniteNonNegative(taskGraphFunc, attr.getEarliestFinishNs(),
                             "island_timing", "earliestFinishNs");
    auto criticalRemaining =
        getFiniteNonNegative(taskGraphFunc, attr.getCriticalPathRemainingNs(),
                             "island_timing", "criticalPathRemainingNs");
    auto slack = getFiniteNonNegative(taskGraphFunc, attr.getSlackNs(),
                                      "island_timing", "slackNs");
    if (failed(totalWork) || failed(analogWork) || failed(digitalWork) ||
        failed(earliestStart) || failed(earliestFinish) ||
        failed(criticalRemaining) || failed(slack))
      return failure();
    profile.islands.push_back(IslandTimingProfile{
        logicalIsland.islandIndex, taskCount, *totalWork, *analogWork,
        *digitalWork, *earliestStart, *earliestFinish, *criticalRemaining,
        *slack, attr.getIsCritical().getValue()});
  }

  auto edgeAttrs = getRequiredAttr<ArrayAttr>(
      taskGraphFunc, timing_attrs::kTimedIslandEdgesAttrName);
  if (failed(edgeAttrs))
    return failure();

  size_t edgeCount = islandGraph.executionGraph.edges.size();
  if (failed(validateRecordCount(taskGraphFunc,
                                 timing_attrs::kTimedIslandEdgesAttrName,
                                 edgeAttrs->size(), edgeCount)))
    return failure();

  profile.islandEdges.reserve(edgeCount);
  for (auto indexedEdge : llvm::enumerate(islandGraph.executionGraph.edges)) {
    size_t index = indexedEdge.index();
    const auto &logicalEdge = indexedEdge.value();
    auto attr = dyn_cast<TimedIslandEdgeAttr>((*edgeAttrs)[index]);
    if (!attr) {
      taskGraphFunc.emitError("expected '")
          << timing_attrs::kTimedIslandEdgesAttrName
          << "' to contain #sculptor.timed_island_edge records";
      return failure();
    }
    int64_t producer = attr.getProducerIsland().getInt();
    int64_t consumer = attr.getConsumerIsland().getInt();
    int64_t bytes = attr.getBytes().getInt();
    auto transfer =
        getFiniteNonNegative(taskGraphFunc, attr.getEstimatedTransferNsPerHop(),
                             "timed_island_edge", "estimatedTransferNsPerHop");
    auto additionalHop =
        getFiniteNonNegative(taskGraphFunc, attr.getEstimatedAdditionalHopNs(),
                             "timed_island_edge", "estimatedAdditionalHopNs");
    auto criticality =
        getFiniteNonNegative(taskGraphFunc, attr.getCriticality(),
                             "timed_island_edge", "criticality");
    auto producerReady =
        getFiniteNonNegative(taskGraphFunc, attr.getProducerReadyTimeNs(),
                             "timed_island_edge", "producerReadyTimeNs");
    auto consumerPressure =
        getFiniteNonNegative(taskGraphFunc, attr.getConsumerTimingPressure(),
                             "timed_island_edge", "consumerTimingPressure");
    if (failed(transfer) || failed(additionalHop) || failed(criticality) ||
        failed(producerReady) || failed(consumerPressure))
      return failure();
    if (producer < 0 || consumer < 0 || bytes < 0 ||
        static_cast<unsigned>(producer) != logicalEdge.producerIsland ||
        static_cast<unsigned>(consumer) != logicalEdge.consumerIsland ||
        bytes != logicalEdge.transferredBytes || *criticality > 1.0 ||
        *consumerPressure > 1.0) {
      taskGraphFunc.emitError(
          "timed island edge does not match logical island graph");
      return failure();
    }
    profile.islandEdges.push_back(
        TimedIslandEdge{logicalEdge.producerIsland, logicalEdge.consumerIsland,
                        logicalEdge.transferredBytes, *transfer, *additionalHop,
                        *criticality, *producerReady, *consumerPressure});
  }
  return profile;
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
