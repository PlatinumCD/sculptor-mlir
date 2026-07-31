#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphSimModel.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTimingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphWorkloadAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace task_graph_attrs = mlir::sculptor::task_graph_attrs;
namespace timing_attrs = mlir::sculptor::timing_attrs;
namespace workload_attrs = mlir::sculptor::workload_attrs;

bool isAnalogArrayOp(mlir::Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();
  return opName.starts_with("sculptor.array.") || opName.starts_with("analog.");
}

struct HardwareModel {
  int64_t numCores = 0;
  int64_t arraysPerCore = 0;
  std::string topology;
  int64_t meshRows = 0;
  int64_t meshCols = 0;
  int64_t numAnalogArrays = 0;
  llvm::SmallVector<int64_t> analogArrays;
};

struct SummaryModel {
  int64_t taskCount = 0;
  int64_t dependencyCount = 0;
  int64_t graphScore = 0;
  int64_t boundaryPenalty = 0;
  int64_t interCoreTransferBytes = 0;
  int64_t totalTransferCost = 0;
  double transferCostPerInterCoreByte = 0.0;
  int64_t totalDigitalOps = 0;
  int64_t numLogicalArrays = 0;
  std::optional<std::string> placementCostMode;
  std::optional<double> searchCompletionTimeProxy;
  std::optional<double> searchCommunicationProxy;
  std::optional<double> searchResourceLoadProxy;
  std::optional<double> predictedMakespanNs;
  std::optional<double> predictedExposedContentionNs;
  std::optional<double> predictedExposedTransportNs;
  std::optional<int64_t> predictedTotalWordHops;
  std::optional<int64_t> timingRerankCandidateCount;
  std::optional<int64_t> timingRerankSelectedProxyRank;
  llvm::SmallVector<int64_t> coreTransferBytes;
  llvm::SmallVector<int64_t> coreTransferCost;
  llvm::SmallVector<int64_t> logicalArrayToAnalogArray;
};

struct ResourceModel {
  mlir::Value value;
  mlir::Operation *op = nullptr;
  int64_t id = 0;
  std::string kind;
  std::string valueType;
  std::optional<int64_t> slot;
  std::optional<int64_t> byteSize;
  std::optional<int64_t> tempIndex;
  std::optional<int64_t> tempOffset;
  std::optional<int64_t> logicalArrayIndex;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> coreId;
  std::optional<int64_t> localArrayId;
};

struct AnalogOpModel {
  int64_t index = 0;
  std::string name;
};

struct TaskTimingModel {
  std::optional<int64_t> topologicalIndex;
  std::optional<int64_t> localRuntimeIndex;
  std::optional<int64_t> dependencyDepth;
  std::optional<int64_t> incomingDataBytes;
  std::optional<int64_t> outgoingDataBytes;
  std::optional<int64_t> digitalReplacementOps;
  std::optional<int64_t> analogLoadBytes;
  std::optional<int64_t> analogExecutionCount;
  std::optional<int64_t> analogStoreBytes;
  std::optional<int64_t> staticElements;
  std::optional<int64_t> localBytesRead;
  std::optional<int64_t> localBytesWritten;
  std::optional<int64_t> loopIterations;
  std::optional<int64_t> scalarInstructionEstimate;
  std::optional<int64_t> vectorInstructionEstimate;
  std::optional<int64_t> loadInstructionEstimate;
  std::optional<int64_t> storeInstructionEstimate;
  std::optional<int64_t> controlInstructionEstimate;
  std::optional<int64_t> runtimeDispatchCycles;
  std::optional<int64_t> taskEntryCycles;
  std::optional<int64_t> taskExitCycles;
  std::optional<double> predictedCpuCycles;
  std::optional<std::string> costSource;
  std::optional<std::string> costConfidence;
  std::optional<double> analogLoadLatencyNs;
  std::optional<double> analogExecuteLatencyNs;
  std::optional<double> analogStoreLatencyNs;
  std::optional<double> analogPipelineLatencyNs;
  std::optional<double> intrinsicLatencyNs;
  std::optional<double> earliestStartNs;
  std::optional<double> earliestFinishNs;
  std::optional<double> incomingNetworkDelayNs;
  std::optional<double> coreQueueDelayNs;
  std::optional<int64_t> causalInputEdge;
  std::optional<int64_t> causalPreviousTask;
  std::optional<double> criticalPathRemainingNs;
  std::optional<bool> isCritical;
};

struct NetworkEdgeTimingModel {
  int64_t sourceCore = -1;
  int64_t destinationCore = -1;
  int64_t hops = 0;
  int64_t payloadWords = 0;
  int64_t protocolWords = 0;
  double transferStartNs = 0.0;
  double injectionStartNs = 0.0;
  double injectionFinishNs = 0.0;
  double routeArrivalNs = 0.0;
  double receiveStartNs = 0.0;
  double receiveCompleteNs = 0.0;
  double transferFinishNs = 0.0;
  double latencyNs = 0.0;
  double contentionDelayNs = 0.0;
  double nicQueueDelayNs = 0.0;
  double linkQueueDelayNs = 0.0;
  double receiveQueueDelayNs = 0.0;
  int64_t causalParentTask = -1;
  int64_t causalParentEdge = -1;
  std::string causalResource;
};

struct CausalTimingEventModel {
  int64_t id = -1;
  std::string kind;
  int64_t taskIndex = -1;
  int64_t edgeIndex = -1;
  int64_t coreId = -1;
  double startNs = 0.0;
  double finishNs = 0.0;
  int64_t parentEvent = -1;
  std::string resource;
};

struct TaskModel {
  mlir::sculptor::TaskCreateOp op;
  int64_t index = 0;
  std::string callee;
  std::string domain;
  std::string kind;
  std::string name;
  std::string sourceLayer;
  uint64_t sourceTaskOrdinal = 0;
  int64_t coreId = 0;
  int64_t digitalOps = 0;
  std::optional<int64_t> islandId;
  std::optional<int64_t> reductionTreeId;
  std::optional<int64_t> reductionLevel;
  std::optional<int64_t> reductionLane;
  std::optional<int64_t> reductionWidth;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> localArrayId;
  TaskTimingModel timing;
  llvm::SmallVector<int64_t> inputResourceIds;
  llvm::SmallVector<int64_t> outputResourceIds;
  llvm::SmallVector<int64_t> dependencyTaskIndices;
  llvm::SmallVector<AnalogOpModel, 4> analogOps;
};

struct GraphTimingModel {
  std::optional<int64_t> taskCount;
  std::optional<int64_t> executionEdgeCount;
  std::optional<int64_t> executionDepth;
  std::optional<int64_t> totalDataBytes;
  std::optional<int64_t> totalDigitalReplacementOps;
  std::optional<std::string> mvmCostMode;
  std::optional<double> criticalPathNs;
  std::optional<bool> placementAware;
  std::optional<double> sumTaskWorkNs;
  std::optional<double> sumCoreQueueDelayNs;
  std::optional<double> sumEdgeNetworkServiceNs;
  std::optional<double> sumEdgeNetworkQueueDelayNs;
  std::optional<double> sumNicQueueDelayNs;
  std::optional<double> sumLinkQueueDelayNs;
  std::optional<double> sumReceiveQueueDelayNs;
  std::optional<double> noContentionMakespanNs;
  std::optional<double> zeroNetworkMakespanNs;
  std::optional<double> exposedTransportNs;
  std::optional<double> exposedContentionNs;
  std::optional<int64_t> totalPayloadWords;
  std::optional<int64_t> totalProtocolWords;
  std::optional<int64_t> totalWordHops;
  std::optional<std::string> costModel;
  std::optional<int64_t> costModelRevision;
  std::optional<std::string> compilerRevision;
  std::optional<std::string> timingBoundary;
  std::optional<std::string> runtimeTaskPolicy;
  std::optional<std::string> runtimeTransmitPolicy;
  std::optional<std::string> memoryBackend;
  std::optional<int64_t> analogMVMLatencyNs;
  std::optional<int64_t> analogIOBitsPerCycle;
  std::optional<bool> analogIOShared;
  std::optional<double> digitalClockGHz;
  std::optional<int64_t> digitalIssueWidth;
  std::optional<int64_t> digitalVectorBitsPerCycle;
  std::optional<int64_t> fixedRuntimeDispatchCycles;
  std::optional<int64_t> fixedTaskEntryCycles;
  std::optional<int64_t> fixedTaskExitCycles;
  std::optional<int64_t> networkLinkBitsPerCycle;
  std::optional<int64_t> networkHopLatencyCycles;
  std::optional<bool> networkPipelined;
  std::optional<int64_t> networkLinkWordBits;
  std::optional<int64_t> protocolWordsPerRoute;
  std::optional<int64_t> nicInjectionWordsPerCycle;
  std::optional<int64_t> rxDmaWordsPerCycle;
  std::optional<std::string> routingPolicy;
};

struct IoBoundaryModel {
  int64_t entryTaskId = 0;
  int64_t entryCore = 0;
  llvm::SmallVector<std::string, 4> entryEdges;
  int64_t exitTaskId = 0;
  int64_t exitCore = 0;
  llvm::SmallVector<std::string, 4> exitEdges;
  bool sharesEdge = false;
};

struct GraphModel {
  mlir::func::FuncOp func;
  std::string name;
  HardwareModel hardware;
  SummaryModel summary;
  GraphTimingModel timing;
  std::optional<IoBoundaryModel> ioBoundary;
  llvm::SmallVector<ResourceModel, 0> resources;
  llvm::SmallVector<TaskModel, 0> tasks;
  llvm::DenseMap<mlir::Value, int64_t> resourceIdByValue;
  llvm::DenseMap<mlir::Value, int64_t> taskIndexByResult;
  llvm::DenseMap<mlir::Value, int64_t> producerTaskIndexByResource;
  llvm::DenseMap<uint64_t, NetworkEdgeTimingModel> networkTimingByTaskPair;
  llvm::SmallVector<CausalTimingEventModel> causalCriticalChain;
};

uint64_t getTaskPairKey(int64_t producer, int64_t consumer) {
  return (static_cast<uint64_t>(producer) << 32) |
         static_cast<uint32_t>(consumer);
}

bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

std::string stringifyType(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  type.print(os);
  return result;
}

mlir::FailureOr<int64_t> getRequiredI64Attr(mlir::Operation *op,
                                            llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr) {
    op->emitError("expected required attr '") << attrName << "'";
    return mlir::failure();
  }

  return attr.getInt();
}

std::optional<int64_t> getOptionalI64Attr(mlir::Operation *op,
                                          llvm::StringRef attrName) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

std::optional<double> getOptionalF64Attr(mlir::Operation *op,
                                         llvm::StringRef attrName) {
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>(attrName))
    return attr.getValueAsDouble();
  return std::nullopt;
}

std::optional<bool> getOptionalBoolAttr(mlir::Operation *op,
                                        llvm::StringRef attrName) {
  if (auto attr = op->getAttrOfType<mlir::BoolAttr>(attrName))
    return attr.getValue();
  return std::nullopt;
}

std::optional<std::string> getOptionalStringAttr(mlir::Operation *op,
                                                 llvm::StringRef attrName) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
    return attr.getValue().str();
  return std::nullopt;
}

mlir::FailureOr<std::string> getRequiredStringAttr(mlir::Operation *op,
                                                   llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr) {
    op->emitError("expected required attr '") << attrName << "'";
    return mlir::failure();
  }

  return attr.getValue().str();
}

mlir::FailureOr<llvm::SmallVector<int64_t>>
getRequiredI64ArrayAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::ArrayAttr>(attrName);
  if (!attr) {
    op->emitError("expected required attr '") << attrName << "'";
    return mlir::failure();
  }

  llvm::SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (mlir::Attribute element : attr) {
    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(element);
    if (!intAttr) {
      op->emitError("expected attr '")
          << attrName << "' to contain only integer attrs";
      return mlir::failure();
    }
    values.push_back(intAttr.getInt());
  }

  return values;
}

mlir::FailureOr<HardwareModel> buildHardwareModel(mlir::func::FuncOp func) {
  HardwareModel hardware;
  auto numCores = getRequiredI64Attr(func, schedule_attrs::kNumCoresAttrName);
  auto arraysPerCore =
      getRequiredI64Attr(func, schedule_attrs::kArraysPerCoreAttrName);
  auto topology =
      getRequiredStringAttr(func, schedule_attrs::kTopologyAttrName);
  auto meshRows = getRequiredI64Attr(func, schedule_attrs::kMeshRowsAttrName);
  auto meshCols = getRequiredI64Attr(func, schedule_attrs::kMeshColsAttrName);
  auto numAnalogArrays =
      getRequiredI64Attr(func, schedule_attrs::kNumAnalogArraysAttrName);
  auto analogArrays =
      getRequiredI64ArrayAttr(func, schedule_attrs::kAnalogArraysAttrName);

  if (mlir::failed(numCores) || mlir::failed(arraysPerCore) ||
      mlir::failed(topology) || mlir::failed(meshRows) ||
      mlir::failed(meshCols) || mlir::failed(numAnalogArrays) ||
      mlir::failed(analogArrays))
    return mlir::failure();

  hardware.numCores = *numCores;
  hardware.arraysPerCore = *arraysPerCore;
  hardware.topology = std::move(*topology);
  hardware.meshRows = *meshRows;
  hardware.meshCols = *meshCols;
  hardware.numAnalogArrays = *numAnalogArrays;
  hardware.analogArrays = std::move(*analogArrays);

  if (hardware.numCores <= 0 || hardware.arraysPerCore <= 0) {
    func.emitError("expected positive scheduled core and array budgets");
    return mlir::failure();
  }
  if (hardware.topology != "mesh") {
    func.emitError("expected mesh topology for simulation model export");
    return mlir::failure();
  }
  if (hardware.meshRows <= 0 || hardware.meshCols <= 0 ||
      hardware.meshRows * hardware.meshCols != hardware.numCores) {
    func.emitError("expected mesh dimensions to match scheduled core count");
    return mlir::failure();
  }

  return hardware;
}

mlir::FailureOr<SummaryModel> buildSummaryModel(mlir::func::FuncOp func) {
  SummaryModel summary;
  auto taskCount = getRequiredI64Attr(func, schedule_attrs::kTaskCountAttrName);
  auto dependencyCount =
      getRequiredI64Attr(func, schedule_attrs::kDependencyCountAttrName);
  auto graphScore =
      getRequiredI64Attr(func, schedule_attrs::kGraphScoreAttrName);
  auto boundaryPenalty =
      getRequiredI64Attr(func, schedule_attrs::kBoundaryPenaltyAttrName);
  auto coreTransferBytes =
      getRequiredI64ArrayAttr(func, schedule_attrs::kCoreTransferBytesAttrName);
  auto interCoreTransferBytes =
      getRequiredI64Attr(func, schedule_attrs::kInterCoreTransferBytesAttrName);
  auto coreTransferCost =
      getRequiredI64ArrayAttr(func, schedule_attrs::kCoreTransferCostAttrName);
  auto totalTransferCost =
      getRequiredI64Attr(func, schedule_attrs::kTotalTransferCostAttrName);
  auto totalDigitalOps =
      getRequiredI64Attr(func, schedule_attrs::kTotalDigitalOpsAttrName);
  auto numLogicalArrays =
      getRequiredI64Attr(func, schedule_attrs::kNumLogicalArraysAttrName);
  auto logicalArrayToAnalogArray = getRequiredI64ArrayAttr(
      func, schedule_attrs::kLogicalArrayToAnalogArrayAttrName);

  if (mlir::failed(taskCount) || mlir::failed(dependencyCount) ||
      mlir::failed(graphScore) || mlir::failed(boundaryPenalty) ||
      mlir::failed(coreTransferBytes) || mlir::failed(interCoreTransferBytes) ||
      mlir::failed(coreTransferCost) || mlir::failed(totalTransferCost) ||
      mlir::failed(totalDigitalOps) || mlir::failed(numLogicalArrays) ||
      mlir::failed(logicalArrayToAnalogArray))
    return mlir::failure();

  summary.taskCount = *taskCount;
  summary.dependencyCount = *dependencyCount;
  summary.graphScore = *graphScore;
  summary.boundaryPenalty = *boundaryPenalty;
  summary.coreTransferBytes = std::move(*coreTransferBytes);
  summary.interCoreTransferBytes = *interCoreTransferBytes;
  summary.coreTransferCost = std::move(*coreTransferCost);
  summary.totalTransferCost = *totalTransferCost;
  if (summary.interCoreTransferBytes > 0) {
    summary.transferCostPerInterCoreByte =
        static_cast<double>(summary.totalTransferCost) /
        static_cast<double>(summary.interCoreTransferBytes);
  }
  summary.totalDigitalOps = *totalDigitalOps;
  summary.numLogicalArrays = *numLogicalArrays;
  summary.placementCostMode =
      getOptionalStringAttr(func, schedule_attrs::kPlacementCostModeAttrName);
  summary.searchCompletionTimeProxy = getOptionalF64Attr(
      func, schedule_attrs::kSearchCompletionTimeProxyAttrName);
  summary.searchCommunicationProxy = getOptionalF64Attr(
      func, schedule_attrs::kSearchCommunicationProxyAttrName);
  summary.searchResourceLoadProxy = getOptionalF64Attr(
      func, schedule_attrs::kSearchResourceLoadProxyAttrName);
  summary.predictedMakespanNs =
      getOptionalF64Attr(func, schedule_attrs::kPredictedMakespanNsAttrName);
  summary.predictedExposedContentionNs = getOptionalF64Attr(
      func, schedule_attrs::kPredictedExposedContentionNsAttrName);
  summary.predictedExposedTransportNs = getOptionalF64Attr(
      func, schedule_attrs::kPredictedExposedTransportNsAttrName);
  summary.predictedTotalWordHops =
      getOptionalI64Attr(func, schedule_attrs::kPredictedTotalWordHopsAttrName);
  summary.timingRerankCandidateCount = getOptionalI64Attr(
      func, schedule_attrs::kTimingRerankCandidateCountAttrName);
  summary.timingRerankSelectedProxyRank = getOptionalI64Attr(
      func, schedule_attrs::kTimingRerankSelectedProxyRankAttrName);
  summary.logicalArrayToAnalogArray = std::move(*logicalArrayToAnalogArray);
  return summary;
}

GraphTimingModel buildGraphTimingModel(mlir::func::FuncOp func) {
  GraphTimingModel timing;
  timing.taskCount = getOptionalI64Attr(func, timing_attrs::kTaskCountAttrName);
  timing.executionEdgeCount =
      getOptionalI64Attr(func, timing_attrs::kExecutionEdgeCountAttrName);
  timing.executionDepth =
      getOptionalI64Attr(func, timing_attrs::kExecutionDepthAttrName);
  timing.totalDataBytes =
      getOptionalI64Attr(func, timing_attrs::kTotalDataBytesAttrName);
  timing.totalDigitalReplacementOps = getOptionalI64Attr(
      func, timing_attrs::kTotalDigitalReplacementOpsAttrName);
  timing.mvmCostMode =
      getOptionalStringAttr(func, timing_attrs::kMVMCostModeAttrName);
  timing.criticalPathNs =
      getOptionalF64Attr(func, timing_attrs::kCriticalPathNsAttrName);
  timing.placementAware =
      getOptionalBoolAttr(func, timing_attrs::kPlacementAwareAttrName);
  timing.sumTaskWorkNs =
      getOptionalF64Attr(func, timing_attrs::kSumTaskWorkNsAttrName);
  timing.sumCoreQueueDelayNs =
      getOptionalF64Attr(func, timing_attrs::kSumCoreQueueDelayNsAttrName);
  timing.sumEdgeNetworkServiceNs =
      getOptionalF64Attr(func, timing_attrs::kSumEdgeNetworkServiceNsAttrName);
  timing.sumEdgeNetworkQueueDelayNs = getOptionalF64Attr(
      func, timing_attrs::kSumEdgeNetworkQueueDelayNsAttrName);
  timing.sumNicQueueDelayNs =
      getOptionalF64Attr(func, timing_attrs::kSumNicQueueDelayNsAttrName);
  timing.sumLinkQueueDelayNs =
      getOptionalF64Attr(func, timing_attrs::kSumLinkQueueDelayNsAttrName);
  timing.sumReceiveQueueDelayNs =
      getOptionalF64Attr(func, timing_attrs::kSumReceiveQueueDelayNsAttrName);
  timing.noContentionMakespanNs =
      getOptionalF64Attr(func, timing_attrs::kNoContentionMakespanNsAttrName);
  timing.zeroNetworkMakespanNs =
      getOptionalF64Attr(func, timing_attrs::kZeroNetworkMakespanNsAttrName);
  timing.exposedTransportNs =
      getOptionalF64Attr(func, timing_attrs::kExposedTransportNsAttrName);
  timing.exposedContentionNs =
      getOptionalF64Attr(func, timing_attrs::kExposedContentionNsAttrName);
  timing.totalPayloadWords =
      getOptionalI64Attr(func, timing_attrs::kTotalPayloadWordsAttrName);
  timing.totalProtocolWords =
      getOptionalI64Attr(func, timing_attrs::kTotalProtocolWordsAttrName);
  timing.totalWordHops =
      getOptionalI64Attr(func, timing_attrs::kTotalWordHopsAttrName);
  if (auto model = func->getAttrOfType<mlir::sculptor::TimingModelAttr>(
          timing_attrs::kTimingModelAttrName)) {
    timing.costModel = model.getCostModel().getValue().str();
    timing.costModelRevision = model.getCostModelRevision().getInt();
    timing.compilerRevision = model.getCompilerRevision().getValue().str();
    timing.timingBoundary = model.getTimingBoundary().getValue().str();
    timing.runtimeTaskPolicy = model.getRuntimeTaskPolicy().getValue().str();
    timing.runtimeTransmitPolicy =
        model.getRuntimeTransmitPolicy().getValue().str();
    timing.memoryBackend = model.getMemoryBackend().getValue().str();
    timing.analogMVMLatencyNs = model.getAnalogMVMLatencyNs().getInt();
    timing.analogIOBitsPerCycle = model.getAnalogIOBitsPerCycle().getInt();
    timing.analogIOShared = model.getAnalogIOShared().getValue();
    timing.digitalClockGHz = model.getDigitalClockGHz().getValueAsDouble();
    timing.digitalIssueWidth = model.getDigitalIssueWidth().getInt();
    timing.digitalVectorBitsPerCycle =
        model.getDigitalVectorBitsPerCycle().getInt();
    timing.fixedRuntimeDispatchCycles =
        model.getFixedRuntimeDispatchCycles().getInt();
    timing.fixedTaskEntryCycles = model.getFixedTaskEntryCycles().getInt();
    timing.fixedTaskExitCycles = model.getFixedTaskExitCycles().getInt();
    timing.networkLinkBitsPerCycle =
        model.getNetworkLinkBitsPerCycle().getInt();
    timing.networkHopLatencyCycles =
        model.getNetworkHopLatencyCycles().getInt();
    timing.networkPipelined = model.getNetworkPipelined().getValue();
    timing.networkLinkWordBits = model.getNetworkLinkWordBits().getInt();
    timing.protocolWordsPerRoute = model.getProtocolWordsPerRoute().getInt();
    timing.nicInjectionWordsPerCycle =
        model.getNicInjectionWordsPerCycle().getInt();
    timing.rxDmaWordsPerCycle = model.getRxDmaWordsPerCycle().getInt();
    timing.routingPolicy = model.getRoutingPolicy().getValue().str();
  }
  return timing;
}

mlir::LogicalResult collectNetworkEdgeTiming(
    mlir::func::FuncOp func,
    llvm::DenseMap<uint64_t, NetworkEdgeTimingModel> &timingByTaskPair) {
  auto edgeAttrs =
      func->getAttrOfType<mlir::ArrayAttr>(timing_attrs::kNetworkEdgesAttrName);
  if (!edgeAttrs)
    return mlir::success();

  for (mlir::Attribute element : edgeAttrs) {
    auto edge = llvm::dyn_cast<mlir::sculptor::NetworkEdgeTimingAttr>(element);
    if (!edge) {
      func.emitError("expected '")
          << timing_attrs::kNetworkEdgesAttrName
          << "' to contain #sculptor.network_edge_timing records";
      return mlir::failure();
    }
    int64_t producer = edge.getProducerTask().getInt();
    int64_t consumer = edge.getConsumerTask().getInt();
    uint64_t key = getTaskPairKey(producer, consumer);
    NetworkEdgeTimingModel timing;
    timing.sourceCore = edge.getSourceCore().getInt();
    timing.destinationCore = edge.getDestinationCore().getInt();
    timing.hops = edge.getMeshHops().getInt();
    timing.payloadWords = edge.getPayloadWords().getInt();
    timing.protocolWords = edge.getProtocolWords().getInt();
    timing.transferStartNs = edge.getTransferStartNs().getValueAsDouble();
    timing.injectionStartNs = edge.getInjectionStartNs().getValueAsDouble();
    timing.injectionFinishNs = edge.getInjectionFinishNs().getValueAsDouble();
    timing.routeArrivalNs = edge.getRouteArrivalNs().getValueAsDouble();
    timing.receiveStartNs = edge.getReceiveStartNs().getValueAsDouble();
    timing.receiveCompleteNs = edge.getReceiveCompleteNs().getValueAsDouble();
    timing.transferFinishNs = edge.getTransferFinishNs().getValueAsDouble();
    timing.latencyNs = edge.getNetworkLatencyNs().getValueAsDouble();
    timing.contentionDelayNs = edge.getContentionDelayNs().getValueAsDouble();
    timing.nicQueueDelayNs = edge.getNicQueueDelayNs().getValueAsDouble();
    timing.linkQueueDelayNs = edge.getLinkQueueDelayNs().getValueAsDouble();
    timing.receiveQueueDelayNs =
        edge.getReceiveQueueDelayNs().getValueAsDouble();
    timing.causalParentTask = edge.getCausalParentTask().getInt();
    timing.causalParentEdge = edge.getCausalParentEdge().getInt();
    timing.causalResource = edge.getCausalResource().getValue().str();
    if (!timingByTaskPair.try_emplace(key, std::move(timing)).second) {
      func.emitError("expected unique network timing task pairs");
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult collectCausalCriticalChain(
    mlir::func::FuncOp func,
    llvm::SmallVectorImpl<CausalTimingEventModel> &events) {
  auto eventAttrs = func->getAttrOfType<mlir::ArrayAttr>(
      timing_attrs::kCausalCriticalChainAttrName);
  if (!eventAttrs)
    return mlir::success();

  for (mlir::Attribute element : eventAttrs) {
    auto event = llvm::dyn_cast<mlir::sculptor::CausalTimingEventAttr>(element);
    if (!event) {
      func.emitError("expected '")
          << timing_attrs::kCausalCriticalChainAttrName
          << "' to contain #sculptor.causal_timing_event records";
      return mlir::failure();
    }
    events.push_back(CausalTimingEventModel{
        event.getId().getInt(),
        event.getKind().getValue().str(),
        event.getTaskIndex().getInt(),
        event.getEdgeIndex().getInt(),
        event.getCoreId().getInt(),
        event.getStartNs().getValueAsDouble(),
        event.getFinishNs().getValueAsDouble(),
        event.getParentEvent().getInt(),
        event.getResource().getValue().str(),
    });
  }
  return mlir::success();
}

std::optional<std::pair<mlir::Value, llvm::StringRef>>
getTaskGraphResource(mlir::Operation &op) {
  if (auto input = llvm::dyn_cast<mlir::sculptor::TaskGraphInputOp>(&op))
    return std::make_pair(input.getResult(), llvm::StringRef("input"));
  if (auto output = llvm::dyn_cast<mlir::sculptor::TaskGraphOutputOp>(&op))
    return std::make_pair(output.getResult(), llvm::StringRef("output"));
  if (auto intermediate =
          llvm::dyn_cast<mlir::sculptor::TaskGraphIntermediateOp>(&op))
    return std::make_pair(intermediate.getResult(),
                          llvm::StringRef("intermediate"));
  if (auto persistent =
          llvm::dyn_cast<mlir::sculptor::TaskGraphPersistentOp>(&op))
    return std::make_pair(persistent.getResult(),
                          llvm::StringRef("persistent"));
  return std::nullopt;
}

bool isLogicalArrayResource(mlir::Value resource) {
  auto resourceType =
      llvm::dyn_cast<mlir::sculptor::TaskResourceType>(resource.getType());
  return resourceType && llvm::isa<mlir::sculptor::LogicalArrayType>(
                             resourceType.getValueType());
}

std::string getResourceValueTypeString(mlir::Value resource) {
  auto resourceType =
      llvm::dyn_cast<mlir::sculptor::TaskResourceType>(resource.getType());
  if (!resourceType)
    return stringifyType(resource.getType());
  return stringifyType(resourceType.getValueType());
}

mlir::FailureOr<llvm::SmallVector<ResourceModel, 0>>
collectResources(mlir::func::FuncOp func, const HardwareModel &hardware,
                 const SummaryModel &summary,
                 llvm::DenseMap<mlir::Value, int64_t> &resourceIdByValue) {
  llvm::SmallVector<ResourceModel, 0> resources;

  for (mlir::Operation &op : func.getBody().front()) {
    std::optional<std::pair<mlir::Value, llvm::StringRef>> resourceInfo =
        getTaskGraphResource(op);
    if (!resourceInfo)
      continue;

    mlir::Value resource = resourceInfo->first;
    int64_t resourceId = static_cast<int64_t>(resources.size());
    if (!resourceIdByValue.try_emplace(resource, resourceId).second) {
      op.emitError("expected task graph resource value to be unique");
      return mlir::failure();
    }

    ResourceModel model;
    model.value = resource;
    model.op = &op;
    model.id = resourceId;
    model.kind = resourceInfo->second.str();
    model.valueType = getResourceValueTypeString(resource);
    model.slot = getOptionalI64Attr(&op, runtime_attrs::kResourceSlotAttrName);
    model.byteSize =
        getOptionalI64Attr(&op, runtime_attrs::kResourceByteSizeAttrName);
    model.tempIndex =
        getOptionalI64Attr(&op, runtime_attrs::kResourceTempIndexAttrName);
    model.tempOffset =
        getOptionalI64Attr(&op, runtime_attrs::kResourceTempOffsetAttrName);
    model.logicalArrayIndex =
        getOptionalI64Attr(&op, schedule_attrs::kLogicalArrayIndexAttrName);

    if (isLogicalArrayResource(resource) && model.logicalArrayIndex) {
      int64_t logicalIndex = *model.logicalArrayIndex;
      if (logicalIndex < 0 ||
          logicalIndex >=
              static_cast<int64_t>(summary.logicalArrayToAnalogArray.size())) {
        op.emitError("expected logical array index to reference scheduled "
                     "logical array placement");
        return mlir::failure();
      }
      int64_t physicalArrayId =
          summary.logicalArrayToAnalogArray[static_cast<size_t>(logicalIndex)];
      model.physicalArrayId = physicalArrayId;
      model.coreId = physicalArrayId / hardware.arraysPerCore;
      model.localArrayId = physicalArrayId % hardware.arraysPerCore;
    }

    resources.push_back(std::move(model));
  }

  return resources;
}

mlir::FailureOr<llvm::SmallVector<TaskModel, 0>> collectTasks(
    mlir::ModuleOp module, mlir::func::FuncOp func,
    const HardwareModel &hardware,
    const llvm::DenseMap<mlir::Value, int64_t> &resourceIdByValue,
    llvm::DenseMap<mlir::Value, int64_t> &taskIndexByResult,
    llvm::DenseMap<mlir::Value, int64_t> &producerTaskIndexByResource) {
  llvm::SmallVector<TaskModel, 0> tasks;
  llvm::DenseMap<int64_t, llvm::DenseSet<int64_t>> seenTaskIndicesByCore;

  for (mlir::Operation &op : func.getBody().front()) {
    auto taskOp = llvm::dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    auto taskIndex =
        getRequiredI64Attr(taskOp, runtime_attrs::kTaskIndexAttrName);
    auto coreId =
        getRequiredI64Attr(taskOp, runtime_attrs::kTaskCoreIdAttrName);
    auto digitalOps =
        getRequiredI64Attr(taskOp, runtime_attrs::kTaskDigitalOpsAttrName);
    if (mlir::failed(taskIndex) || mlir::failed(coreId) ||
        mlir::failed(digitalOps))
      return mlir::failure();

    if (*coreId < 0 || *coreId >= hardware.numCores) {
      taskOp.emitError("expected task core id to be inside scheduled core "
                       "budget");
      return mlir::failure();
    }
    if (*taskIndex < 0 ||
        !seenTaskIndicesByCore[*coreId].insert(*taskIndex).second) {
      taskOp.emitError(
          "expected unique non-negative task index within each core");
      return mlir::failure();
    }

    TaskModel task;
    task.op = taskOp;
    task.index = static_cast<int64_t>(tasks.size());
    task.callee = taskOp.getCallee().str();
    task.domain = taskOp.getDomain().str();
    task.kind = taskOp.getTaskKind().str();
    task.name = taskOp.getTaskName().str();
    task.sourceLayer = taskOp.getSourceLayer().str();
    task.sourceTaskOrdinal = taskOp.getSourceTaskOrdinal();
    task.coreId = *coreId;
    task.digitalOps = *digitalOps;
    task.islandId =
        getOptionalI64Attr(taskOp, schedule_attrs::kIslandIdAttrName);
    task.reductionTreeId = getOptionalI64Attr(
        taskOp, task_graph_attrs::kTaskReductionTreeIdAttrName);
    task.reductionLevel = getOptionalI64Attr(
        taskOp, task_graph_attrs::kTaskReductionLevelAttrName);
    task.reductionLane = getOptionalI64Attr(
        taskOp, task_graph_attrs::kTaskReductionLaneAttrName);
    task.reductionWidth = getOptionalI64Attr(
        taskOp, task_graph_attrs::kTaskReductionWidthAttrName);
    task.timing.topologicalIndex =
        getOptionalI64Attr(taskOp, timing_attrs::kTopologicalIndexAttrName);
    task.timing.localRuntimeIndex =
        getOptionalI64Attr(taskOp, timing_attrs::kLocalRuntimeIndexAttrName);
    if (!task.timing.localRuntimeIndex)
      task.timing.localRuntimeIndex = *taskIndex;
    task.timing.dependencyDepth =
        getOptionalI64Attr(taskOp, timing_attrs::kDependencyDepthAttrName);
    task.timing.incomingDataBytes =
        getOptionalI64Attr(taskOp, workload_attrs::kIncomingDataBytesAttrName);
    task.timing.outgoingDataBytes =
        getOptionalI64Attr(taskOp, workload_attrs::kOutgoingDataBytesAttrName);
    task.timing.digitalReplacementOps = getOptionalI64Attr(
        taskOp, workload_attrs::kDigitalReplacementOpsAttrName);
    task.timing.analogLoadBytes =
        getOptionalI64Attr(taskOp, workload_attrs::kAnalogLoadBytesAttrName);
    task.timing.analogExecutionCount = getOptionalI64Attr(
        taskOp, workload_attrs::kAnalogExecutionCountAttrName);
    task.timing.analogStoreBytes =
        getOptionalI64Attr(taskOp, workload_attrs::kAnalogStoreBytesAttrName);
    task.timing.staticElements =
        getOptionalI64Attr(taskOp, workload_attrs::kStaticElementsAttrName);
    task.timing.localBytesRead =
        getOptionalI64Attr(taskOp, workload_attrs::kLocalBytesReadAttrName);
    task.timing.localBytesWritten =
        getOptionalI64Attr(taskOp, workload_attrs::kLocalBytesWrittenAttrName);
    task.timing.loopIterations =
        getOptionalI64Attr(taskOp, workload_attrs::kLoopIterationsAttrName);
    task.timing.scalarInstructionEstimate = getOptionalI64Attr(
        taskOp, timing_attrs::kScalarInstructionEstimateAttrName);
    task.timing.vectorInstructionEstimate = getOptionalI64Attr(
        taskOp, timing_attrs::kVectorInstructionEstimateAttrName);
    task.timing.loadInstructionEstimate = getOptionalI64Attr(
        taskOp, timing_attrs::kLoadInstructionEstimateAttrName);
    task.timing.storeInstructionEstimate = getOptionalI64Attr(
        taskOp, timing_attrs::kStoreInstructionEstimateAttrName);
    task.timing.controlInstructionEstimate = getOptionalI64Attr(
        taskOp, timing_attrs::kControlInstructionEstimateAttrName);
    task.timing.runtimeDispatchCycles = getOptionalI64Attr(
        taskOp, timing_attrs::kRuntimeDispatchCyclesAttrName);
    task.timing.taskEntryCycles =
        getOptionalI64Attr(taskOp, timing_attrs::kTaskEntryCyclesAttrName);
    task.timing.taskExitCycles =
        getOptionalI64Attr(taskOp, timing_attrs::kTaskExitCyclesAttrName);
    task.timing.predictedCpuCycles =
        getOptionalF64Attr(taskOp, timing_attrs::kPredictedCpuCyclesAttrName);
    task.timing.costSource =
        getOptionalStringAttr(taskOp, timing_attrs::kCostSourceAttrName);
    task.timing.costConfidence =
        getOptionalStringAttr(taskOp, timing_attrs::kCostConfidenceAttrName);
    task.timing.analogLoadLatencyNs =
        getOptionalF64Attr(taskOp, timing_attrs::kAnalogLoadLatencyNsAttrName);
    task.timing.analogExecuteLatencyNs = getOptionalF64Attr(
        taskOp, timing_attrs::kAnalogExecuteLatencyNsAttrName);
    task.timing.analogStoreLatencyNs =
        getOptionalF64Attr(taskOp, timing_attrs::kAnalogStoreLatencyNsAttrName);
    task.timing.analogPipelineLatencyNs = getOptionalF64Attr(
        taskOp, timing_attrs::kAnalogPipelineLatencyNsAttrName);
    task.timing.intrinsicLatencyNs =
        getOptionalF64Attr(taskOp, timing_attrs::kIntrinsicLatencyNsAttrName);
    task.timing.earliestStartNs =
        getOptionalF64Attr(taskOp, timing_attrs::kEarliestStartNsAttrName);
    task.timing.earliestFinishNs =
        getOptionalF64Attr(taskOp, timing_attrs::kEarliestFinishNsAttrName);
    task.timing.incomingNetworkDelayNs = getOptionalF64Attr(
        taskOp, timing_attrs::kIncomingNetworkDelayNsAttrName);
    task.timing.coreQueueDelayNs =
        getOptionalF64Attr(taskOp, timing_attrs::kCoreQueueDelayNsAttrName);
    task.timing.causalInputEdge =
        getOptionalI64Attr(taskOp, timing_attrs::kCausalInputEdgeAttrName);
    task.timing.causalPreviousTask =
        getOptionalI64Attr(taskOp, timing_attrs::kCausalPreviousTaskAttrName);
    task.timing.criticalPathRemainingNs = getOptionalF64Attr(
        taskOp, timing_attrs::kCriticalPathRemainingNsAttrName);
    task.timing.isCritical =
        getOptionalBoolAttr(taskOp, timing_attrs::kIsCriticalAttrName);
    auto callee = module.lookupSymbol<mlir::func::FuncOp>(
        taskOp.getCalleeAttr().getValue());
    if (!callee) {
      taskOp.emitError("expected task callee '")
          << taskOp.getCalleeAttr().getValue()
          << "' to resolve to a func.func for simulation model export";
      return mlir::failure();
    }
    if (!callee.isDeclaration()) {
      int64_t analogOpIndex = 0;
      callee.walk([&](mlir::Operation *nestedOp) {
        if (!isAnalogArrayOp(nestedOp))
          return;
        task.analogOps.push_back(AnalogOpModel{
            analogOpIndex++, nestedOp->getName().getStringRef().str()});
      });
    }
    task.physicalArrayId =
        getOptionalI64Attr(taskOp, runtime_attrs::kTaskPhysicalArrayIdAttrName);
    if (task.physicalArrayId)
      task.localArrayId = *task.physicalArrayId % hardware.arraysPerCore;

    for (mlir::Value input : taskOp.getInputs()) {
      auto resourceIt = resourceIdByValue.find(input);
      if (resourceIt == resourceIdByValue.end()) {
        taskOp.emitError("expected every task input to reference a task graph "
                         "resource");
        return mlir::failure();
      }
      task.inputResourceIds.push_back(resourceIt->second);
    }

    for (mlir::Value output : taskOp.getOutputs()) {
      auto resourceIt = resourceIdByValue.find(output);
      if (resourceIt == resourceIdByValue.end()) {
        taskOp.emitError("expected every task output to reference a task graph "
                         "resource");
        return mlir::failure();
      }
      task.outputResourceIds.push_back(resourceIt->second);
      if (!producerTaskIndexByResource.try_emplace(output, task.index).second) {
        taskOp.emitError("expected task graph resource to have one producer");
        return mlir::failure();
      }
    }

    taskIndexByResult.try_emplace(taskOp.getResult(), task.index);
    tasks.push_back(std::move(task));
  }

  for (TaskModel &task : tasks) {
    for (mlir::Value dependency : task.op.getDependencies()) {
      auto dependencyIt = taskIndexByResult.find(dependency);
      if (dependencyIt == taskIndexByResult.end()) {
        task.op.emitError("expected task dependency to reference a task in "
                          "the same graph");
        return mlir::failure();
      }
      task.dependencyTaskIndices.push_back(dependencyIt->second);
    }
  }

  llvm::sort(tasks, [](const TaskModel &lhs, const TaskModel &rhs) {
    return lhs.index < rhs.index;
  });
  return tasks;
}

const ResourceModel *lookupResourceById(llvm::ArrayRef<ResourceModel> resources,
                                        int64_t resourceId);

bool taskTouchesResourceKind(const TaskModel &task,
                             llvm::ArrayRef<ResourceModel> resources,
                             llvm::ArrayRef<int64_t> resourceIds,
                             llvm::StringRef kind) {
  for (int64_t resourceId : resourceIds) {
    const ResourceModel *resource = lookupResourceById(resources, resourceId);
    if (resource && kind == resource->kind)
      return true;
  }
  return false;
}

llvm::SmallVector<std::string, 4>
getMeshEdgeMembership(int64_t coreId, const HardwareModel &hardware) {
  llvm::SmallVector<std::string, 4> edges;
  int64_t row = coreId / hardware.meshCols;
  int64_t col = coreId % hardware.meshCols;

  if (row == 0)
    edges.push_back("top");
  if (col == hardware.meshCols - 1)
    edges.push_back("right");
  if (row == hardware.meshRows - 1)
    edges.push_back("bottom");
  if (col == 0)
    edges.push_back("left");

  return edges;
}

bool shareMeshEdge(llvm::ArrayRef<std::string> lhs,
                   llvm::ArrayRef<std::string> rhs) {
  for (const std::string &lhsEdge : lhs) {
    for (const std::string &rhsEdge : rhs) {
      if (lhsEdge == rhsEdge)
        return true;
    }
  }
  return false;
}

mlir::FailureOr<IoBoundaryModel> buildIoBoundaryModel(mlir::func::FuncOp func,
                                                      const GraphModel &graph) {
  const TaskModel *entryTask = nullptr;
  const TaskModel *exitTask = nullptr;

  for (const TaskModel &task : graph.tasks) {
    if (!entryTask &&
        llvm::StringRef(task.kind) != task_graph_names::kMatrixSetupTaskKind &&
        taskTouchesResourceKind(task, graph.resources, task.inputResourceIds,
                                "input"))
      entryTask = &task;

    if (taskTouchesResourceKind(task, graph.resources, task.outputResourceIds,
                                "output"))
      exitTask = &task;
  }

  if (!entryTask || !exitTask) {
    func.emitError("expected task graph to expose entry and exit tasks for "
                   "I/O boundary metadata");
    return mlir::failure();
  }

  IoBoundaryModel boundary;
  boundary.entryTaskId = entryTask->index;
  boundary.entryCore = entryTask->coreId;
  boundary.entryEdges =
      getMeshEdgeMembership(entryTask->coreId, graph.hardware);
  boundary.exitTaskId = exitTask->index;
  boundary.exitCore = exitTask->coreId;
  boundary.exitEdges = getMeshEdgeMembership(exitTask->coreId, graph.hardware);
  boundary.sharesEdge = shareMeshEdge(boundary.entryEdges, boundary.exitEdges);
  return boundary;
}

mlir::FailureOr<GraphModel> buildGraphModel(mlir::ModuleOp module,
                                            mlir::func::FuncOp func) {
  if (!func.getBody().hasOneBlock()) {
    func.emitError("expected task graph function to have one block");
    return mlir::failure();
  }

  GraphModel graph;
  graph.func = func;
  graph.name = func.getName().str();
  auto hardware = buildHardwareModel(func);
  auto summary = buildSummaryModel(func);
  if (mlir::failed(hardware) || mlir::failed(summary))
    return mlir::failure();
  graph.hardware = std::move(*hardware);
  graph.summary = std::move(*summary);
  graph.timing = buildGraphTimingModel(func);

  auto resources = collectResources(func, graph.hardware, graph.summary,
                                    graph.resourceIdByValue);
  if (mlir::failed(resources))
    return mlir::failure();
  graph.resources = std::move(*resources);

  auto tasks =
      collectTasks(module, func, graph.hardware, graph.resourceIdByValue,
                   graph.taskIndexByResult, graph.producerTaskIndexByResource);
  if (mlir::failed(tasks))
    return mlir::failure();
  graph.tasks = std::move(*tasks);

  if (mlir::failed(
          collectNetworkEdgeTiming(func, graph.networkTimingByTaskPair)))
    return mlir::failure();
  if (mlir::failed(collectCausalCriticalChain(func, graph.causalCriticalChain)))
    return mlir::failure();

  auto ioBoundary = buildIoBoundaryModel(func, graph);
  if (mlir::failed(ioBoundary))
    return mlir::failure();
  graph.ioBoundary = std::move(*ioBoundary);

  return graph;
}

void emitOptionalI64Attr(llvm::json::OStream &json, llvm::StringRef key,
                         std::optional<int64_t> value) {
  if (value) {
    json.attribute(key, *value);
    return;
  }

  json.attribute(key, nullptr);
}

void emitOptionalF64Attr(llvm::json::OStream &json, llvm::StringRef key,
                         std::optional<double> value) {
  if (value) {
    json.attribute(key, *value);
    return;
  }
  json.attribute(key, nullptr);
}

void emitOptionalBoolAttr(llvm::json::OStream &json, llvm::StringRef key,
                          std::optional<bool> value) {
  if (value) {
    json.attribute(key, *value);
    return;
  }
  json.attribute(key, nullptr);
}

void emitOptionalStringAttr(llvm::json::OStream &json, llvm::StringRef key,
                            const std::optional<std::string> &value) {
  if (value) {
    json.attribute(key, *value);
    return;
  }
  json.attribute(key, nullptr);
}

void emitI64ArrayAttr(llvm::json::OStream &json, llvm::StringRef key,
                      llvm::ArrayRef<int64_t> values) {
  json.attributeArray(key, [&] {
    for (int64_t value : values)
      json.value(value);
  });
}

void emitStringArrayAttr(llvm::json::OStream &json, llvm::StringRef key,
                         llvm::ArrayRef<std::string> values) {
  json.attributeArray(key, [&] {
    for (const std::string &value : values)
      json.value(value);
  });
}

void emitHardware(llvm::json::OStream &json, const HardwareModel &hardware) {
  json.attributeObject("hardware", [&] {
    json.attribute("topology", hardware.topology);
    json.attribute("num_cores", hardware.numCores);
    json.attribute("arrays_per_core", hardware.arraysPerCore);
    json.attribute("mesh_rows", hardware.meshRows);
    json.attribute("mesh_cols", hardware.meshCols);
    json.attribute("num_analog_arrays", hardware.numAnalogArrays);
    emitI64ArrayAttr(json, "analog_arrays", hardware.analogArrays);
  });
}

void emitResources(llvm::json::OStream &json,
                   llvm::ArrayRef<ResourceModel> resources) {
  json.attributeArray("resources", [&] {
    for (const ResourceModel &resource : resources) {
      json.object([&] {
        json.attribute("id", resource.id);
        json.attribute("kind", resource.kind);
        json.attribute("value_type", resource.valueType);
        emitOptionalI64Attr(json, "slot", resource.slot);
        emitOptionalI64Attr(json, "byte_size", resource.byteSize);
        emitOptionalI64Attr(json, "temp_index", resource.tempIndex);
        emitOptionalI64Attr(json, "temp_offset", resource.tempOffset);
        emitOptionalI64Attr(json, "logical_array_index",
                            resource.logicalArrayIndex);
        emitOptionalI64Attr(json, "physical_array_id",
                            resource.physicalArrayId);
        emitOptionalI64Attr(json, "core_id", resource.coreId);
        emitOptionalI64Attr(json, "local_array_id", resource.localArrayId);
      });
    }
  });
}

void emitTaskResourceIds(llvm::json::OStream &json, llvm::StringRef key,
                         llvm::ArrayRef<int64_t> resourceIds) {
  json.attributeArray(key, [&] {
    for (int64_t resourceId : resourceIds)
      json.value(resourceId);
  });
}

void emitSculptorOps(llvm::json::OStream &json,
                     llvm::ArrayRef<AnalogOpModel> analogOps) {
  json.attributeArray("analog_ops", [&] {
    for (const AnalogOpModel &op : analogOps) {
      json.object([&] {
        json.attribute("index", op.index);
        json.attribute("name", op.name);
      });
    }
  });

  llvm::SmallVector<std::pair<std::string, int64_t>, 4> counts;
  for (const AnalogOpModel &op : analogOps) {
    auto existing = llvm::find_if(
        counts, [&](const auto &entry) { return entry.first == op.name; });
    if (existing != counts.end()) {
      ++existing->second;
      continue;
    }
    counts.push_back({op.name, 1});
  }

  json.attributeArray("analog_op_counts", [&] {
    for (const auto &entry : counts) {
      json.object([&] {
        json.attribute("name", entry.first);
        json.attribute("count", entry.second);
      });
    }
  });
}

void emitTaskTiming(llvm::json::OStream &json, const TaskTimingModel &timing) {
  json.attributeObject("timing", [&] {
    emitOptionalI64Attr(json, "topological_index", timing.topologicalIndex);
    emitOptionalI64Attr(json, "local_runtime_index", timing.localRuntimeIndex);
    emitOptionalI64Attr(json, "dependency_depth", timing.dependencyDepth);
    emitOptionalI64Attr(json, "incoming_data_bytes", timing.incomingDataBytes);
    emitOptionalI64Attr(json, "outgoing_data_bytes", timing.outgoingDataBytes);
    emitOptionalI64Attr(json, "digital_replacement_ops",
                        timing.digitalReplacementOps);
    emitOptionalI64Attr(json, "analog_load_bytes", timing.analogLoadBytes);
    emitOptionalI64Attr(json, "analog_execution_count",
                        timing.analogExecutionCount);
    emitOptionalI64Attr(json, "analog_store_bytes", timing.analogStoreBytes);
    emitOptionalI64Attr(json, "static_elements", timing.staticElements);
    emitOptionalI64Attr(json, "local_bytes_read", timing.localBytesRead);
    emitOptionalI64Attr(json, "local_bytes_written", timing.localBytesWritten);
    emitOptionalI64Attr(json, "loop_iterations", timing.loopIterations);
    emitOptionalI64Attr(json, "scalar_instruction_estimate",
                        timing.scalarInstructionEstimate);
    emitOptionalI64Attr(json, "vector_instruction_estimate",
                        timing.vectorInstructionEstimate);
    emitOptionalI64Attr(json, "load_instruction_estimate",
                        timing.loadInstructionEstimate);
    emitOptionalI64Attr(json, "store_instruction_estimate",
                        timing.storeInstructionEstimate);
    emitOptionalI64Attr(json, "control_instruction_estimate",
                        timing.controlInstructionEstimate);
    emitOptionalI64Attr(json, "runtime_dispatch_cycles",
                        timing.runtimeDispatchCycles);
    emitOptionalI64Attr(json, "task_entry_cycles", timing.taskEntryCycles);
    emitOptionalI64Attr(json, "task_exit_cycles", timing.taskExitCycles);
    emitOptionalF64Attr(json, "predicted_cpu_cycles",
                        timing.predictedCpuCycles);
    if (timing.costSource)
      json.attribute("cost_source", *timing.costSource);
    if (timing.costConfidence)
      json.attribute("cost_confidence", *timing.costConfidence);
    emitOptionalF64Attr(json, "analog_load_latency_ns",
                        timing.analogLoadLatencyNs);
    emitOptionalF64Attr(json, "analog_execute_latency_ns",
                        timing.analogExecuteLatencyNs);
    emitOptionalF64Attr(json, "analog_store_latency_ns",
                        timing.analogStoreLatencyNs);
    emitOptionalF64Attr(json, "analog_pipeline_latency_ns",
                        timing.analogPipelineLatencyNs);
    emitOptionalF64Attr(json, "intrinsic_latency_ns",
                        timing.intrinsicLatencyNs);
    emitOptionalF64Attr(json, "earliest_start_ns", timing.earliestStartNs);
    emitOptionalF64Attr(json, "earliest_finish_ns", timing.earliestFinishNs);
    emitOptionalF64Attr(json, "incoming_network_delay_ns",
                        timing.incomingNetworkDelayNs);
    emitOptionalF64Attr(json, "core_queue_delay_ns", timing.coreQueueDelayNs);
    emitOptionalI64Attr(json, "causal_input_edge", timing.causalInputEdge);
    emitOptionalI64Attr(json, "causal_previous_task",
                        timing.causalPreviousTask);
    emitOptionalF64Attr(json, "critical_path_remaining_ns",
                        timing.criticalPathRemainingNs);
    emitOptionalBoolAttr(json, "is_critical", timing.isCritical);
  });
}

void emitTasks(llvm::json::OStream &json, llvm::ArrayRef<TaskModel> tasks) {
  json.attributeArray("tasks", [&] {
    for (const TaskModel &task : tasks) {
      json.object([&] {
        json.attribute("index", task.index);
        json.attribute("callee", task.callee);
        json.attribute("domain", task.domain);
        json.attribute("kind", task.kind);
        json.attribute("name", task.name);
        json.attribute("source_layer", task.sourceLayer);
        json.attribute("source_task_ordinal",
                       static_cast<int64_t>(task.sourceTaskOrdinal));
        json.attribute("core_id", task.coreId);
        emitOptionalI64Attr(json, "island_id", task.islandId);
        emitOptionalI64Attr(json, "reduction_tree_id", task.reductionTreeId);
        emitOptionalI64Attr(json, "reduction_level", task.reductionLevel);
        emitOptionalI64Attr(json, "reduction_lane", task.reductionLane);
        emitOptionalI64Attr(json, "reduction_width", task.reductionWidth);
        emitOptionalI64Attr(json, "physical_array_id", task.physicalArrayId);
        emitOptionalI64Attr(json, "local_array_id", task.localArrayId);
        json.attribute("digital_ops", task.digitalOps);
        emitTaskTiming(json, task.timing);
        emitSculptorOps(json, task.analogOps);
        emitTaskResourceIds(json, "inputs", task.inputResourceIds);
        emitTaskResourceIds(json, "outputs", task.outputResourceIds);
        emitTaskResourceIds(json, "dependencies", task.dependencyTaskIndices);
      });
    }
  });
}

void emitIoBoundary(llvm::json::OStream &json,
                    const std::optional<IoBoundaryModel> &boundary) {
  if (!boundary)
    return;

  json.attributeObject("io_boundary", [&] {
    json.attribute("entry_task_id", boundary->entryTaskId);
    json.attribute("entry_core", boundary->entryCore);
    emitStringArrayAttr(json, "entry_edges", boundary->entryEdges);
    json.attribute("exit_task_id", boundary->exitTaskId);
    json.attribute("exit_core", boundary->exitCore);
    emitStringArrayAttr(json, "exit_edges", boundary->exitEdges);
    json.attribute("shares_edge", boundary->sharesEdge);
  });
}

int64_t getMeshDistance(int64_t sourceCore, int64_t destinationCore,
                        const HardwareModel &hardware) {
  int64_t sourceRow = sourceCore / hardware.meshCols;
  int64_t sourceCol = sourceCore % hardware.meshCols;
  int64_t destinationRow = destinationCore / hardware.meshCols;
  int64_t destinationCol = destinationCore % hardware.meshCols;
  return std::llabs(sourceRow - destinationRow) +
         std::llabs(sourceCol - destinationCol);
}

const ResourceModel *lookupResourceById(llvm::ArrayRef<ResourceModel> resources,
                                        int64_t resourceId) {
  if (resourceId < 0 || resourceId >= static_cast<int64_t>(resources.size()))
    return nullptr;
  return &resources[static_cast<size_t>(resourceId)];
}

void emitControlEdges(llvm::json::OStream &json,
                      llvm::ArrayRef<TaskModel> tasks) {
  json.attributeArray("control_edges", [&] {
    int64_t edgeId = 0;
    for (const TaskModel &consumer : tasks) {
      for (int64_t producerIndex : consumer.dependencyTaskIndices) {
        json.object([&] {
          json.attribute("id", edgeId++);
          json.attribute("producer_task", producerIndex);
          json.attribute("consumer_task", consumer.index);
        });
      }
    }
  });
}

void emitDataEdges(llvm::json::OStream &json, const GraphModel &graph) {
  json.attributeArray("data_edges", [&] {
    int64_t edgeId = 0;
    for (const TaskModel &consumer : graph.tasks) {
      for (int64_t resourceId : consumer.inputResourceIds) {
        const ResourceModel *resource =
            lookupResourceById(graph.resources, resourceId);
        if (!resource)
          continue;

        auto producerIt =
            graph.producerTaskIndexByResource.find(resource->value);
        if (producerIt == graph.producerTaskIndexByResource.end())
          continue;

        int64_t producerIndex = producerIt->second;
        auto producerItByIndex =
            llvm::find_if(graph.tasks, [&](const TaskModel &task) {
              return task.index == producerIndex;
            });
        if (producerItByIndex == graph.tasks.end())
          continue;

        int64_t sourceCore = producerItByIndex->coreId;
        int64_t destinationCore = consumer.coreId;
        int64_t byteSize = resource->byteSize.value_or(0);
        int64_t meshDistance =
            getMeshDistance(sourceCore, destinationCore, graph.hardware);
        int64_t transferCost = byteSize * meshDistance;
        auto networkTiming = graph.networkTimingByTaskPair.find(
            getTaskPairKey(producerIndex, consumer.index));

        json.object([&] {
          json.attribute("id", edgeId++);
          json.attribute("producer_task", producerIndex);
          json.attribute("consumer_task", consumer.index);
          json.attribute("resource", resourceId);
          json.attribute("byte_size", byteSize);
          json.attribute("source_core", sourceCore);
          json.attribute("destination_core", destinationCore);
          json.attribute("mesh_distance", meshDistance);
          json.attribute("transfer_cost", transferCost);
          json.attribute("inter_core", sourceCore != destinationCore);
          if (networkTiming != graph.networkTimingByTaskPair.end()) {
            const NetworkEdgeTimingModel &timing = networkTiming->second;
            json.attribute("network_hops", timing.hops);
            json.attribute("network_payload_words", timing.payloadWords);
            json.attribute("network_protocol_words", timing.protocolWords);
            json.attribute("network_transfer_start_ns", timing.transferStartNs);
            json.attribute("network_injection_start_ns",
                           timing.injectionStartNs);
            json.attribute("network_injection_finish_ns",
                           timing.injectionFinishNs);
            json.attribute("network_route_arrival_ns", timing.routeArrivalNs);
            json.attribute("network_receive_start_ns", timing.receiveStartNs);
            json.attribute("network_receive_complete_ns",
                           timing.receiveCompleteNs);
            json.attribute("network_transfer_finish_ns",
                           timing.transferFinishNs);
            json.attribute("network_latency_ns", timing.latencyNs);
            json.attribute("network_contention_delay_ns",
                           timing.contentionDelayNs);
            json.attribute("network_nic_queue_delay_ns",
                           timing.nicQueueDelayNs);
            json.attribute("network_link_queue_delay_ns",
                           timing.linkQueueDelayNs);
            json.attribute("network_receive_queue_delay_ns",
                           timing.receiveQueueDelayNs);
            json.attribute("network_causal_parent_task",
                           timing.causalParentTask);
            json.attribute("network_causal_parent_edge",
                           timing.causalParentEdge);
            json.attribute("network_causal_resource", timing.causalResource);
          }
        });
      }
    }
  });
}

void emitSummary(llvm::json::OStream &json, const SummaryModel &summary) {
  json.attributeObject("summary", [&] {
    json.attribute("task_count", summary.taskCount);
    json.attribute("dependency_count", summary.dependencyCount);
    json.attribute("graph_score", summary.graphScore);
    json.attribute("boundary_penalty", summary.boundaryPenalty);
    json.attribute("inter_core_transfer_bytes", summary.interCoreTransferBytes);
    json.attribute("total_transfer_cost", summary.totalTransferCost);
    json.attribute("transfer_cost_per_inter_core_byte",
                   summary.transferCostPerInterCoreByte);
    json.attribute("total_digital_ops", summary.totalDigitalOps);
    emitOptionalStringAttr(json, "placement_cost_mode",
                           summary.placementCostMode);
    emitOptionalF64Attr(json, "search_completion_time_proxy",
                        summary.searchCompletionTimeProxy);
    emitOptionalF64Attr(json, "search_communication_proxy",
                        summary.searchCommunicationProxy);
    emitOptionalF64Attr(json, "search_resource_load_proxy",
                        summary.searchResourceLoadProxy);
    emitOptionalF64Attr(json, "predicted_makespan_ns",
                        summary.predictedMakespanNs);
    emitOptionalF64Attr(json, "predicted_exposed_contention_ns",
                        summary.predictedExposedContentionNs);
    emitOptionalF64Attr(json, "predicted_exposed_transport_ns",
                        summary.predictedExposedTransportNs);
    emitOptionalI64Attr(json, "predicted_total_word_hops",
                        summary.predictedTotalWordHops);
    emitOptionalI64Attr(json, "timing_rerank_candidate_count",
                        summary.timingRerankCandidateCount);
    emitOptionalI64Attr(json, "timing_rerank_selected_proxy_rank",
                        summary.timingRerankSelectedProxyRank);
    json.attribute("num_logical_arrays", summary.numLogicalArrays);
    emitI64ArrayAttr(json, "core_transfer_bytes", summary.coreTransferBytes);
    emitI64ArrayAttr(json, "core_transfer_cost", summary.coreTransferCost);
    emitI64ArrayAttr(json, "logical_array_to_analog_array",
                     summary.logicalArrayToAnalogArray);
  });
}

void emitGraphTiming(llvm::json::OStream &json,
                     const GraphTimingModel &timing) {
  json.attributeObject("timing", [&] {
    emitOptionalI64Attr(json, "task_count", timing.taskCount);
    emitOptionalI64Attr(json, "execution_edge_count",
                        timing.executionEdgeCount);
    emitOptionalI64Attr(json, "execution_depth", timing.executionDepth);
    emitOptionalI64Attr(json, "total_data_bytes", timing.totalDataBytes);
    emitOptionalI64Attr(json, "total_digital_replacement_ops",
                        timing.totalDigitalReplacementOps);
    emitOptionalStringAttr(json, "mvm_cost_mode", timing.mvmCostMode);
    emitOptionalF64Attr(json, "critical_path_ns", timing.criticalPathNs);
    emitOptionalBoolAttr(json, "placement_aware", timing.placementAware);
    emitOptionalF64Attr(json, "sum_task_work_ns", timing.sumTaskWorkNs);
    emitOptionalF64Attr(json, "sum_core_queue_delay_ns",
                        timing.sumCoreQueueDelayNs);
    emitOptionalF64Attr(json, "sum_edge_network_service_ns",
                        timing.sumEdgeNetworkServiceNs);
    emitOptionalF64Attr(json, "sum_edge_network_queue_delay_ns",
                        timing.sumEdgeNetworkQueueDelayNs);
    emitOptionalF64Attr(json, "sum_nic_queue_delay_ns",
                        timing.sumNicQueueDelayNs);
    emitOptionalF64Attr(json, "sum_link_queue_delay_ns",
                        timing.sumLinkQueueDelayNs);
    emitOptionalF64Attr(json, "sum_receive_queue_delay_ns",
                        timing.sumReceiveQueueDelayNs);
    emitOptionalF64Attr(json, "no_contention_makespan_ns",
                        timing.noContentionMakespanNs);
    emitOptionalF64Attr(json, "zero_network_makespan_ns",
                        timing.zeroNetworkMakespanNs);
    emitOptionalF64Attr(json, "exposed_transport_ns",
                        timing.exposedTransportNs);
    emitOptionalF64Attr(json, "exposed_contention_ns",
                        timing.exposedContentionNs);
    emitOptionalI64Attr(json, "total_payload_words", timing.totalPayloadWords);
    emitOptionalI64Attr(json, "total_protocol_words",
                        timing.totalProtocolWords);
    emitOptionalI64Attr(json, "total_word_hops", timing.totalWordHops);
    json.attributeObject("model", [&] {
      emitOptionalStringAttr(json, "cost_model", timing.costModel);
      emitOptionalI64Attr(json, "cost_model_revision",
                          timing.costModelRevision);
      emitOptionalStringAttr(json, "compiler_revision",
                             timing.compilerRevision);
      emitOptionalStringAttr(json, "timing_boundary", timing.timingBoundary);
      emitOptionalStringAttr(json, "runtime_task_policy",
                             timing.runtimeTaskPolicy);
      emitOptionalStringAttr(json, "runtime_transmit_policy",
                             timing.runtimeTransmitPolicy);
      emitOptionalStringAttr(json, "memory_backend", timing.memoryBackend);
      emitOptionalI64Attr(json, "analog_mvm_latency_ns",
                          timing.analogMVMLatencyNs);
      emitOptionalI64Attr(json, "analog_io_bits_per_cycle",
                          timing.analogIOBitsPerCycle);
      emitOptionalBoolAttr(json, "analog_io_shared", timing.analogIOShared);
      emitOptionalF64Attr(json, "digital_clock_ghz", timing.digitalClockGHz);
      emitOptionalI64Attr(json, "digital_issue_width",
                          timing.digitalIssueWidth);
      emitOptionalI64Attr(json, "digital_vector_bits_per_cycle",
                          timing.digitalVectorBitsPerCycle);
      emitOptionalI64Attr(json, "fixed_runtime_dispatch_cycles",
                          timing.fixedRuntimeDispatchCycles);
      emitOptionalI64Attr(json, "fixed_task_entry_cycles",
                          timing.fixedTaskEntryCycles);
      emitOptionalI64Attr(json, "fixed_task_exit_cycles",
                          timing.fixedTaskExitCycles);
      emitOptionalI64Attr(json, "network_link_bits_per_cycle",
                          timing.networkLinkBitsPerCycle);
      emitOptionalI64Attr(json, "network_hop_latency_cycles",
                          timing.networkHopLatencyCycles);
      emitOptionalBoolAttr(json, "network_pipelined", timing.networkPipelined);
      emitOptionalI64Attr(json, "network_link_word_bits",
                          timing.networkLinkWordBits);
      emitOptionalI64Attr(json, "protocol_words_per_route",
                          timing.protocolWordsPerRoute);
      emitOptionalI64Attr(json, "nic_injection_words_per_cycle",
                          timing.nicInjectionWordsPerCycle);
      emitOptionalI64Attr(json, "rx_dma_words_per_cycle",
                          timing.rxDmaWordsPerCycle);
      emitOptionalStringAttr(json, "routing_policy", timing.routingPolicy);
    });
  });
}

void emitCausalCriticalChain(
    llvm::json::OStream &json,
    llvm::ArrayRef<CausalTimingEventModel> causalCriticalChain) {
  json.attributeArray("causal_critical_chain", [&] {
    for (const CausalTimingEventModel &event : causalCriticalChain) {
      json.object([&] {
        json.attribute("id", event.id);
        json.attribute("kind", event.kind);
        json.attribute("task_index", event.taskIndex);
        json.attribute("edge_index", event.edgeIndex);
        json.attribute("core_id", event.coreId);
        json.attribute("start_ns", event.startNs);
        json.attribute("finish_ns", event.finishNs);
        json.attribute("parent_event", event.parentEvent);
        json.attribute("resource", event.resource);
      });
    }
  });
}

void emitGraph(llvm::json::OStream &json, const GraphModel &graph) {
  json.object([&] {
    json.attribute("name", graph.name);
    emitHardware(json, graph.hardware);
    emitResources(json, graph.resources);
    emitTasks(json, graph.tasks);
    emitIoBoundary(json, graph.ioBoundary);
    emitControlEdges(json, graph.tasks);
    emitDataEdges(json, graph);
    emitSummary(json, graph.summary);
    emitGraphTiming(json, graph.timing);
    emitCausalCriticalChain(json, graph.causalCriticalChain);
  });
}

} // namespace

namespace mlir {
namespace sculptor {

void ExportTaskGraphSimModelPass::runOnOperation() {
  if (output.empty()) {
    getOperation().emitError("expected non-empty output path for "
                             "sculptor-export-task-graph-sim-model");
    signalPassFailure();
    return;
  }

  ModuleOp module = getOperation();
  llvm::SmallVector<func::FuncOp> graphFuncs;
  for (func::FuncOp func : module.getOps<func::FuncOp>())
    if (returnsTaskGraph(func))
      graphFuncs.push_back(func);

  if (graphFuncs.empty()) {
    module.emitError("expected at least one function returning "
                     "!sculptor.task_graph");
    signalPassFailure();
    return;
  }

  llvm::SmallVector<GraphModel, 1> graphs;
  graphs.reserve(graphFuncs.size());
  for (func::FuncOp func : graphFuncs) {
    auto graph = buildGraphModel(module, func);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    graphs.push_back(std::move(*graph));
  }

  std::error_code error;
  llvm::raw_fd_ostream os(output, error, llvm::sys::fs::OF_Text);
  if (error) {
    module.emitError("failed to open task graph simulation model output file '")
        << output << "': " << error.message();
    signalPassFailure();
    return;
  }

  llvm::json::OStream json(os, /*IndentSize=*/2);
  json.object([&] {
    json.attribute("schema_version", 1);
    json.attribute("format", "sculptor.task_graph.sim_model");
    json.attributeArray("graphs", [&] {
      for (const GraphModel &graph : graphs)
        emitGraph(json, graph);
    });
  });
  os << "\n";
}

void registerExportTaskGraphSimModelPass() {
  PassRegistration<ExportTaskGraphSimModelPass>();
}

} // namespace sculptor
} // namespace mlir
