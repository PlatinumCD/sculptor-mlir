#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTimingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphWorkloadAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {
using task_graph::LogicalPlacementIslandGraph;
using task_graph::TaskGraphDAG;

static void removeAttrsWithPrefix(Operation *op, StringRef prefix) {
  llvm::SmallVector<StringAttr, 16> toRemove;
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().getValue().starts_with(prefix))
      toRemove.push_back(attr.getName());
  }
  for (StringAttr name : toRemove)
    op->removeAttr(name);
}

static void removeTaskGraphAttrsWithPrefix(func::FuncOp taskGraphFunc,
                                           StringRef prefix) {
  removeAttrsWithPrefix(taskGraphFunc, prefix);
  taskGraphFunc.walk([&](sculptor::TaskCreateOp task) {
    removeAttrsWithPrefix(task, prefix);
  });
}

void invalidateTaskGraphTiming(func::FuncOp taskGraphFunc,
                               bool advanceGeneration) {
  int64_t generation = 0;
  if (auto current = taskGraphFunc->getAttrOfType<IntegerAttr>(
          timing_attrs::kGraphGenerationAttrName))
    generation = current.getInt();
  if (advanceGeneration)
    ++generation;

  removeTaskGraphAttrsWithPrefix(taskGraphFunc, "sculptor.timing.");
  taskGraphFunc->setAttr(
      timing_attrs::kGraphGenerationAttrName,
      IntegerAttr::get(IntegerType::get(taskGraphFunc.getContext(), 64),
                       generation));
}

void invalidateTaskGraphWorkload(func::FuncOp taskGraphFunc) {
  removeTaskGraphAttrsWithPrefix(taskGraphFunc, "sculptor.workload.");
}

void invalidateTaskGraphStructure(func::FuncOp taskGraphFunc) {
  invalidateTaskGraphWorkload(taskGraphFunc);
  invalidateTaskGraphTiming(taskGraphFunc, /*advanceGeneration=*/true);
}

void attachTaskGraphTimingAnalysis(func::FuncOp taskGraphFunc,
                                   const TaskGraphDAG &dag,
                                   const TimingAnalysis &analysis,
                                   const TimingModel &model,
                                   MVMCostMode mvmCostMode) {
  Builder builder(taskGraphFunc.getContext());
  auto i64Attr = [&](int64_t value) {
    return builder.getI64IntegerAttr(value);
  };
  auto f64Attr = [&](double value) { return builder.getF64FloatAttr(value); };
  int64_t generation = 0;
  if (auto attr = taskGraphFunc->getAttrOfType<IntegerAttr>(
          timing_attrs::kGraphGenerationAttrName))
    generation = attr.getInt();
  taskGraphFunc->setAttr(timing_attrs::kGraphGenerationAttrName,
                         i64Attr(generation));
  taskGraphFunc->setAttr(timing_attrs::kAnalysisGenerationAttrName,
                         i64Attr(generation));

  for (const task_graph::TaskGraphNode &node : dag.nodes) {
    const TaskTiming &timing = analysis.tasks[node.index];
    node.op->setAttr(timing_attrs::kTopologicalIndexAttrName,
                     i64Attr(timing.topologicalIndex));
    node.op->setAttr(timing_attrs::kLocalRuntimeIndexAttrName,
                     i64Attr(timing.localRuntimeIndex));
    node.op->setAttr(timing_attrs::kDependencyDepthAttrName,
                     i64Attr(timing.dependencyDepth));
    node.op->setAttr(timing_attrs::kControlPredecessorCountAttrName,
                     i64Attr(timing.workload.controlPredecessorCount));
    node.op->setAttr(timing_attrs::kDataPredecessorCountAttrName,
                     i64Attr(timing.workload.dataPredecessorCount));
    node.op->setAttr(workload_attrs::kIncomingDataBytesAttrName,
                     i64Attr(timing.workload.incomingDataBytes));
    node.op->setAttr(workload_attrs::kOutgoingDataBytesAttrName,
                     i64Attr(timing.workload.outgoingDataBytes));
    node.op->setAttr(workload_attrs::kDigitalOpsAttrName,
                     i64Attr(timing.workload.digitalOps));
    node.op->setAttr(workload_attrs::kDigitalReplacementOpsAttrName,
                     i64Attr(timing.workload.digitalReplacementOps));
    node.op->setAttr(workload_attrs::kAnalogLoadBytesAttrName,
                     i64Attr(timing.workload.analogLoadBytes));
    node.op->setAttr(workload_attrs::kAnalogExecutionCountAttrName,
                     i64Attr(timing.workload.analogExecutionCount));
    node.op->setAttr(workload_attrs::kAnalogStoreBytesAttrName,
                     i64Attr(timing.workload.analogStoreBytes));
    node.op->setAttr(workload_attrs::kStaticElementsAttrName,
                     i64Attr(timing.workload.staticElements));
    node.op->setAttr(workload_attrs::kLocalBytesReadAttrName,
                     i64Attr(timing.workload.localBytesRead));
    node.op->setAttr(workload_attrs::kLocalBytesWrittenAttrName,
                     i64Attr(timing.workload.localBytesWritten));
    node.op->setAttr(workload_attrs::kLoopIterationsAttrName,
                     i64Attr(timing.workload.loopIterations));
    node.op->setAttr(timing_attrs::kScalarInstructionEstimateAttrName,
                     i64Attr(timing.cost.scalarInstructions));
    node.op->setAttr(timing_attrs::kVectorInstructionEstimateAttrName,
                     i64Attr(timing.cost.vectorInstructions));
    node.op->setAttr(timing_attrs::kLoadInstructionEstimateAttrName,
                     i64Attr(timing.cost.loadInstructions));
    node.op->setAttr(timing_attrs::kStoreInstructionEstimateAttrName,
                     i64Attr(timing.cost.storeInstructions));
    node.op->setAttr(timing_attrs::kControlInstructionEstimateAttrName,
                     i64Attr(timing.cost.controlInstructions));
    node.op->setAttr(timing_attrs::kRuntimeDispatchCyclesAttrName,
                     i64Attr(timing.cost.runtimeDispatchCycles));
    node.op->setAttr(timing_attrs::kTaskEntryCyclesAttrName,
                     i64Attr(timing.cost.taskEntryCycles));
    node.op->setAttr(timing_attrs::kTaskExitCyclesAttrName,
                     i64Attr(timing.cost.taskExitCycles));
    node.op->setAttr(timing_attrs::kPredictedCpuCyclesAttrName,
                     f64Attr(timing.cost.predictedCpuCycles));
    node.op->setAttr(
        timing_attrs::kCostSourceAttrName,
        builder.getStringAttr(stringifyTaskCostSource(timing.cost.source)));
    node.op->setAttr(timing_attrs::kCostConfidenceAttrName,
                     builder.getStringAttr(
                         stringifyTaskCostConfidence(timing.cost.confidence)));
    node.op->setAttr(timing_attrs::kAnalogLoadLatencyNsAttrName,
                     f64Attr(timing.analogLoadLatencyNs));
    node.op->setAttr(timing_attrs::kAnalogExecuteLatencyNsAttrName,
                     f64Attr(timing.analogExecuteLatencyNs));
    node.op->setAttr(timing_attrs::kAnalogStoreLatencyNsAttrName,
                     f64Attr(timing.analogStoreLatencyNs));
    node.op->setAttr(timing_attrs::kAnalogPipelineLatencyNsAttrName,
                     f64Attr(timing.analogPipelineLatencyNs));
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
    node.op->setAttr(timing_attrs::kCoreQueueDelayNsAttrName,
                     f64Attr(timing.coreQueueDelayNs));
    node.op->setAttr(timing_attrs::kCausalInputEdgeAttrName,
                     i64Attr(timing.causalInputEdge));
    node.op->setAttr(timing_attrs::kCausalPreviousTaskAttrName,
                     i64Attr(timing.causalPreviousTask));
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
  taskGraphFunc->setAttr(timing_attrs::kTotalDigitalReplacementOpsAttrName,
                         i64Attr(analysis.totalDigitalReplacementOps));
  taskGraphFunc->setAttr(
      timing_attrs::kMVMCostModeAttrName,
      builder.getStringAttr(stringifyMVMCostMode(mvmCostMode)));
  taskGraphFunc->setAttr(timing_attrs::kPlacementAwareAttrName,
                         builder.getBoolAttr(analysis.placementAware));
  taskGraphFunc->setAttr(timing_attrs::kSumEdgeNetworkServiceNsAttrName,
                         f64Attr(analysis.sumEdgeNetworkServiceNs));
  taskGraphFunc->setAttr(timing_attrs::kSumEdgeNetworkQueueDelayNsAttrName,
                         f64Attr(analysis.sumEdgeNetworkQueueDelayNs));
  taskGraphFunc->setAttr(timing_attrs::kSumTaskWorkNsAttrName,
                         f64Attr(analysis.sumTaskWorkNs));
  taskGraphFunc->setAttr(timing_attrs::kSumCoreQueueDelayNsAttrName,
                         f64Attr(analysis.sumCoreQueueDelayNs));
  taskGraphFunc->setAttr(timing_attrs::kSumNicQueueDelayNsAttrName,
                         f64Attr(analysis.sumNicQueueDelayNs));
  taskGraphFunc->setAttr(timing_attrs::kSumLinkQueueDelayNsAttrName,
                         f64Attr(analysis.sumLinkQueueDelayNs));
  taskGraphFunc->setAttr(timing_attrs::kSumReceiveQueueDelayNsAttrName,
                         f64Attr(analysis.sumReceiveQueueDelayNs));
  taskGraphFunc->setAttr(timing_attrs::kNoContentionMakespanNsAttrName,
                         f64Attr(analysis.noContentionMakespanNs));
  taskGraphFunc->setAttr(timing_attrs::kZeroNetworkMakespanNsAttrName,
                         f64Attr(analysis.zeroNetworkMakespanNs));
  taskGraphFunc->setAttr(timing_attrs::kExposedTransportNsAttrName,
                         f64Attr(analysis.exposedTransportNs));
  taskGraphFunc->setAttr(timing_attrs::kExposedContentionNsAttrName,
                         f64Attr(analysis.exposedContentionNs));
  taskGraphFunc->setAttr(timing_attrs::kTotalPayloadWordsAttrName,
                         i64Attr(analysis.totalPayloadWords));
  taskGraphFunc->setAttr(timing_attrs::kTotalProtocolWordsAttrName,
                         i64Attr(analysis.totalProtocolWords));
  taskGraphFunc->setAttr(timing_attrs::kTotalWordHopsAttrName,
                         i64Attr(analysis.totalWordHops));

  llvm::SmallVector<Attribute, 16> networkEdges;
  networkEdges.reserve(analysis.executionEdges.size());
  for (const ExecutionEdge &edge : analysis.executionEdges) {
    networkEdges.push_back(NetworkEdgeTimingAttr::get(
        taskGraphFunc.getContext(), i64Attr(edge.producerTask),
        i64Attr(edge.consumerTask), i64Attr(edge.sourceCore),
        i64Attr(edge.destinationCore), i64Attr(edge.meshHops),
        i64Attr(edge.payloadWords), i64Attr(edge.protocolWords),
        f64Attr(edge.transferStartNs), f64Attr(edge.injectionStartNs),
        f64Attr(edge.injectionFinishNs), f64Attr(edge.routeArrivalNs),
        f64Attr(edge.receiveStartNs), f64Attr(edge.receiveCompleteNs),
        f64Attr(edge.transferFinishNs), f64Attr(edge.networkLatencyNs),
        f64Attr(edge.contentionDelayNs), f64Attr(edge.nicQueueDelayNs),
        f64Attr(edge.linkQueueDelayNs), f64Attr(edge.receiveQueueDelayNs),
        i64Attr(edge.causalParentTask), i64Attr(edge.causalParentEdge),
        builder.getStringAttr(edge.causalResource)));
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

  llvm::SmallVector<Attribute, 16> causalEvents;
  causalEvents.reserve(analysis.causalCriticalChain.size());
  for (const CausalTimingEvent &event : analysis.causalCriticalChain) {
    causalEvents.push_back(CausalTimingEventAttr::get(
        taskGraphFunc.getContext(), i64Attr(event.id),
        builder.getStringAttr(event.kind), i64Attr(event.taskIndex),
        i64Attr(event.edgeIndex), i64Attr(event.coreId), f64Attr(event.startNs),
        f64Attr(event.finishNs), i64Attr(event.parentEvent),
        builder.getStringAttr(event.resource)));
  }
  taskGraphFunc->setAttr(timing_attrs::kCausalCriticalChainAttrName,
                         builder.getArrayAttr(causalEvents));

  taskGraphFunc->setAttr(
      timing_attrs::kTimingModelAttrName,
      TimingModelAttr::get(
          taskGraphFunc.getContext(), builder.getStringAttr(model.costModel),
          i64Attr(model.costModelRevision),
          builder.getStringAttr(model.compilerRevision),
          builder.getStringAttr(model.timingBoundary),
          builder.getStringAttr(model.runtimeTaskPolicy),
          builder.getStringAttr(model.runtimeTransmitPolicy),
          builder.getStringAttr(model.memoryBackend),
          i64Attr(model.analogMVMLatencyNs),
          i64Attr(model.analogIOBitsPerCycle),
          builder.getBoolAttr(model.analogIOShared),
          f64Attr(model.digitalClockGHz), i64Attr(model.digitalIssueWidth),
          i64Attr(model.digitalVectorBitsPerCycle),
          i64Attr(model.fixedRuntimeDispatchCycles),
          i64Attr(model.fixedTaskEntryCycles),
          i64Attr(model.fixedTaskExitCycles),
          i64Attr(model.networkLinkBitsPerCycle),
          i64Attr(model.networkHopLatencyCycles),
          builder.getBoolAttr(model.networkPipelined),
          i64Attr(model.networkLinkWordBits),
          i64Attr(model.protocolWordsPerRoute),
          i64Attr(model.nicInjectionWordsPerCycle),
          i64Attr(model.rxDmaWordsPerCycle),
          builder.getStringAttr(model.routingPolicy)));
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
  model.costModel = timingModelAttr->getCostModel().getValue().str();
  model.costModelRevision = timingModelAttr->getCostModelRevision().getInt();
  model.compilerRevision =
      timingModelAttr->getCompilerRevision().getValue().str();
  model.timingBoundary = timingModelAttr->getTimingBoundary().getValue().str();
  model.runtimeTaskPolicy =
      timingModelAttr->getRuntimeTaskPolicy().getValue().str();
  model.runtimeTransmitPolicy =
      timingModelAttr->getRuntimeTransmitPolicy().getValue().str();
  model.memoryBackend = timingModelAttr->getMemoryBackend().getValue().str();
  model.analogMVMLatencyNs = timingModelAttr->getAnalogMVMLatencyNs().getInt();
  model.analogIOBitsPerCycle =
      timingModelAttr->getAnalogIOBitsPerCycle().getInt();
  model.analogIOShared = timingModelAttr->getAnalogIOShared().getValue();
  model.digitalClockGHz =
      timingModelAttr->getDigitalClockGHz().getValueAsDouble();
  model.digitalIssueWidth = timingModelAttr->getDigitalIssueWidth().getInt();
  model.digitalVectorBitsPerCycle =
      timingModelAttr->getDigitalVectorBitsPerCycle().getInt();
  model.fixedRuntimeDispatchCycles =
      timingModelAttr->getFixedRuntimeDispatchCycles().getInt();
  model.fixedTaskEntryCycles =
      timingModelAttr->getFixedTaskEntryCycles().getInt();
  model.fixedTaskExitCycles =
      timingModelAttr->getFixedTaskExitCycles().getInt();
  model.networkLinkBitsPerCycle =
      timingModelAttr->getNetworkLinkBitsPerCycle().getInt();
  model.networkHopLatencyCycles =
      timingModelAttr->getNetworkHopLatencyCycles().getInt();
  model.networkPipelined = timingModelAttr->getNetworkPipelined().getValue();
  model.networkLinkWordBits =
      timingModelAttr->getNetworkLinkWordBits().getInt();
  model.protocolWordsPerRoute =
      timingModelAttr->getProtocolWordsPerRoute().getInt();
  model.nicInjectionWordsPerCycle =
      timingModelAttr->getNicInjectionWordsPerCycle().getInt();
  model.rxDmaWordsPerCycle = timingModelAttr->getRxDmaWordsPerCycle().getInt();
  model.routingPolicy = timingModelAttr->getRoutingPolicy().getValue().str();
  if (failed(validateTimingModel(taskGraphFunc, model)))
    return failure();
  return model;
}

FailureOr<SchedulingTimingProfile>
loadSchedulingTimingProfile(func::FuncOp taskGraphFunc, const TaskGraphDAG &dag,
                            const LogicalPlacementIslandGraph &islandGraph) {
  int64_t graphGeneration = 0;
  if (auto attr = taskGraphFunc->getAttrOfType<IntegerAttr>(
          timing_attrs::kGraphGenerationAttrName))
    graphGeneration = attr.getInt();
  auto analysisGeneration =
      getRequiredI64(taskGraphFunc, timing_attrs::kAnalysisGenerationAttrName);
  if (failed(analysisGeneration))
    return failure();
  if (*analysisGeneration != graphGeneration) {
    taskGraphFunc.emitError(
        "stale timing metadata: timing generation does not match the latest "
        "task-graph structural generation; rerun "
        "--sculptor-analyze-task-graph-timing");
    return failure();
  }

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
  auto totalDigitalReplacementOps = getRequiredI64(
      taskGraphFunc, timing_attrs::kTotalDigitalReplacementOpsAttrName);
  if (failed(taskCount) || failed(criticalPathNs) ||
      failed(totalDigitalReplacementOps))
    return failure();
  if (*taskCount < 0 || static_cast<size_t>(*taskCount) != dag.nodes.size()) {
    taskGraphFunc.emitError(
        "timing profile task count does not match task DAG");
    return failure();
  }

  SchedulingTimingProfile profile;
  auto mvmCostModeAttr = getRequiredAttr<StringAttr>(
      taskGraphFunc, timing_attrs::kMVMCostModeAttrName);
  if (failed(mvmCostModeAttr))
    return failure();
  std::optional<MVMCostMode> mvmCostMode =
      symbolizeMVMCostMode(mvmCostModeAttr->getValue());
  if (!mvmCostMode) {
    taskGraphFunc.emitError("unknown timing MVM cost mode '")
        << mvmCostModeAttr->getValue() << "'";
    return failure();
  }
  profile.mvmCostMode = *mvmCostMode;
  profile.criticalPathNs = *criticalPathNs;
  profile.tasks.resize(dag.nodes.size());
  int64_t observedDigitalReplacementOps = 0;
  llvm::SmallVector<bool, 16> seenTopologicalIndex(dag.nodes.size(), false);
  for (const task_graph::TaskGraphNode &node : dag.nodes) {
    TaskTiming timing;
    auto topologicalIndex =
        getRequiredI64(node.op, timing_attrs::kTopologicalIndexAttrName);
    auto dependencyDepth =
        getRequiredI64(node.op, timing_attrs::kDependencyDepthAttrName);
    auto localRuntimeIndex =
        getRequiredI64(node.op, timing_attrs::kLocalRuntimeIndexAttrName);
    auto controlPredecessors =
        getRequiredI64(node.op, timing_attrs::kControlPredecessorCountAttrName);
    auto dataPredecessors =
        getRequiredI64(node.op, timing_attrs::kDataPredecessorCountAttrName);
    auto incomingBytes =
        getRequiredI64(node.op, workload_attrs::kIncomingDataBytesAttrName);
    auto outgoingBytes =
        getRequiredI64(node.op, workload_attrs::kOutgoingDataBytesAttrName);
    auto digitalOps =
        getRequiredI64(node.op, workload_attrs::kDigitalOpsAttrName);
    auto digitalReplacementOps =
        getRequiredI64(node.op, workload_attrs::kDigitalReplacementOpsAttrName);
    auto analogLoadBytes =
        getRequiredI64(node.op, workload_attrs::kAnalogLoadBytesAttrName);
    auto analogExecutionCount =
        getRequiredI64(node.op, workload_attrs::kAnalogExecutionCountAttrName);
    auto analogStoreBytes =
        getRequiredI64(node.op, workload_attrs::kAnalogStoreBytesAttrName);
    auto staticElements =
        getRequiredI64(node.op, workload_attrs::kStaticElementsAttrName);
    auto localBytesRead =
        getRequiredI64(node.op, workload_attrs::kLocalBytesReadAttrName);
    auto localBytesWritten =
        getRequiredI64(node.op, workload_attrs::kLocalBytesWrittenAttrName);
    auto loopIterations =
        getRequiredI64(node.op, workload_attrs::kLoopIterationsAttrName);
    auto scalarInstructions = getRequiredI64(
        node.op, timing_attrs::kScalarInstructionEstimateAttrName);
    auto vectorInstructions = getRequiredI64(
        node.op, timing_attrs::kVectorInstructionEstimateAttrName);
    auto loadInstructions =
        getRequiredI64(node.op, timing_attrs::kLoadInstructionEstimateAttrName);
    auto storeInstructions = getRequiredI64(
        node.op, timing_attrs::kStoreInstructionEstimateAttrName);
    auto controlInstructions = getRequiredI64(
        node.op, timing_attrs::kControlInstructionEstimateAttrName);
    auto runtimeDispatchCycles =
        getRequiredI64(node.op, timing_attrs::kRuntimeDispatchCyclesAttrName);
    auto taskEntryCycles =
        getRequiredI64(node.op, timing_attrs::kTaskEntryCyclesAttrName);
    auto taskExitCycles =
        getRequiredI64(node.op, timing_attrs::kTaskExitCyclesAttrName);
    auto predictedCpuCycles =
        getRequiredF64(node.op, timing_attrs::kPredictedCpuCyclesAttrName);
    auto costSource =
        node.op->getAttrOfType<StringAttr>(timing_attrs::kCostSourceAttrName);
    auto costConfidence = node.op->getAttrOfType<StringAttr>(
        timing_attrs::kCostConfidenceAttrName);
    auto analogLoad =
        getRequiredF64(node.op, timing_attrs::kAnalogLoadLatencyNsAttrName);
    auto analogExecute =
        getRequiredF64(node.op, timing_attrs::kAnalogExecuteLatencyNsAttrName);
    auto analogStore =
        getRequiredF64(node.op, timing_attrs::kAnalogStoreLatencyNsAttrName);
    auto analogPipeline =
        getRequiredF64(node.op, timing_attrs::kAnalogPipelineLatencyNsAttrName);
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
    if (failed(topologicalIndex) || failed(localRuntimeIndex) ||
        failed(dependencyDepth) || failed(controlPredecessors) ||
        failed(dataPredecessors) || failed(incomingBytes) ||
        failed(outgoingBytes) || failed(digitalOps) ||
        failed(digitalReplacementOps) || failed(analogLoadBytes) ||
        failed(analogExecutionCount) || failed(analogStoreBytes) ||
        failed(staticElements) || failed(localBytesRead) ||
        failed(localBytesWritten) || failed(loopIterations) ||
        failed(scalarInstructions) || failed(vectorInstructions) ||
        failed(loadInstructions) || failed(storeInstructions) ||
        failed(controlInstructions) || failed(runtimeDispatchCycles) ||
        failed(taskEntryCycles) || failed(taskExitCycles) ||
        failed(predictedCpuCycles) || !costSource || !costConfidence ||
        failed(analogLoad) || failed(analogExecute) || failed(analogStore) ||
        failed(intrinsic) || failed(analogPipeline) || failed(earliestStart) ||
        failed(earliestFinish) || failed(criticalRemaining) || failed(slack) ||
        failed(incomingNetworkDelay) || failed(isCritical))
      return failure();
    if (*topologicalIndex < 0 || *localRuntimeIndex < 0 ||
        *dependencyDepth < 0 || *controlPredecessors < 0 ||
        *dataPredecessors < 0 || *incomingBytes < 0 || *outgoingBytes < 0 ||
        *digitalOps < 0 || *digitalReplacementOps < 0 || *analogLoadBytes < 0 ||
        *analogExecutionCount < 0 || *analogStoreBytes < 0 ||
        *staticElements < 0 || *localBytesRead < 0 || *localBytesWritten < 0 ||
        *loopIterations < 0 || *scalarInstructions < 0 ||
        *vectorInstructions < 0 || *loadInstructions < 0 ||
        *storeInstructions < 0 || *controlInstructions < 0 ||
        *runtimeDispatchCycles < 0 || *taskEntryCycles < 0 ||
        *taskExitCycles < 0) {
      node.op->emitError("expected non-negative task timing counters");
      return failure();
    }
    auto source = symbolizeTaskCostSource(costSource.getValue());
    auto confidence = symbolizeTaskCostConfidence(costConfidence.getValue());
    if (!source || !confidence) {
      node.op->emitError("expected known task cost source and confidence");
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
    timing.localRuntimeIndex = *localRuntimeIndex;
    timing.dependencyDepth = *dependencyDepth;
    timing.workload.controlPredecessorCount = *controlPredecessors;
    timing.workload.dataPredecessorCount = *dataPredecessors;
    timing.workload.incomingDataBytes = *incomingBytes;
    timing.workload.outgoingDataBytes = *outgoingBytes;
    timing.workload.digitalOps = *digitalOps;
    timing.workload.digitalReplacementOps = *digitalReplacementOps;
    timing.workload.analogLoadBytes = *analogLoadBytes;
    timing.workload.analogExecutionCount = *analogExecutionCount;
    timing.workload.analogStoreBytes = *analogStoreBytes;
    timing.workload.staticElements = *staticElements;
    timing.workload.localBytesRead = *localBytesRead;
    timing.workload.localBytesWritten = *localBytesWritten;
    timing.workload.loopIterations = *loopIterations;
    timing.cost.scalarInstructions = *scalarInstructions;
    timing.cost.vectorInstructions = *vectorInstructions;
    timing.cost.loadInstructions = *loadInstructions;
    timing.cost.storeInstructions = *storeInstructions;
    timing.cost.controlInstructions = *controlInstructions;
    timing.cost.runtimeDispatchCycles = *runtimeDispatchCycles;
    timing.cost.taskEntryCycles = *taskEntryCycles;
    timing.cost.taskExitCycles = *taskExitCycles;
    timing.cost.predictedCpuCycles = *predictedCpuCycles;
    timing.cost.source = *source;
    timing.cost.confidence = *confidence;
    std::optional<int64_t> replacementTotal = llvm::checkedAdd(
        observedDigitalReplacementOps, timing.workload.digitalReplacementOps);
    if (!replacementTotal) {
      node.op->emitError("digital replacement operation count overflow");
      return failure();
    }
    observedDigitalReplacementOps = *replacementTotal;
    timing.analogLoadLatencyNs = *analogLoad;
    timing.analogExecuteLatencyNs = *analogExecute;
    timing.analogStoreLatencyNs = *analogStore;
    timing.analogPipelineLatencyNs = *analogPipeline;
    timing.intrinsicLatencyNs = *intrinsic;
    timing.earliestStartNs = *earliestStart;
    timing.earliestFinishNs = *earliestFinish;
    timing.criticalPathRemainingNs = *criticalRemaining;
    timing.slackNs = *slack;
    timing.incomingNetworkDelayNs = *incomingNetworkDelay;
    timing.isCritical = *isCritical;
    profile.tasks[node.index] = timing;
  }
  if (*totalDigitalReplacementOps < 0 ||
      observedDigitalReplacementOps != *totalDigitalReplacementOps) {
    taskGraphFunc.emitError(
        "timing profile total digital replacement operation count does not "
        "match task records");
    return failure();
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
