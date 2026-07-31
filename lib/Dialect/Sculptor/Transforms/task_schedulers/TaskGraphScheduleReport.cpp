#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleReport.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace {

static FailureOr<int64_t> getIntegerSummaryAttr(func::FuncOp func,
                                                StringRef attrName) {
  auto attr = func->getAttrOfType<IntegerAttr>(attrName);
  if (!attr) {
    func.emitError("expected schedule summary attribute '") << attrName << "'";
    return failure();
  }
  return attr.getInt();
}

static FailureOr<double> getFloatSummaryAttr(func::FuncOp func,
                                             StringRef attrName) {
  auto attr = func->getAttrOfType<FloatAttr>(attrName);
  if (!attr) {
    func.emitError("expected schedule summary attribute '") << attrName << "'";
    return failure();
  }
  return attr.getValueAsDouble();
}

static void writeCsvString(llvm::raw_ostream &os, StringRef value) {
  if (!value.contains(',') && !value.contains('"') && !value.contains('\n')) {
    os << value;
    return;
  }

  os << '"';
  for (char c : value) {
    if (c == '"')
      os << "\"\"";
    else
      os << c;
  }
  os << '"';
}

static const GreedyScheduleConfig &
getReportingGreedyConfig(const TaskGraphSchedulerOptions &options) {
  if (const auto *greedy = std::get_if<GreedySchedulerOptions>(&options))
    return greedy->greedy;
  if (const auto *annealing =
          std::get_if<AnnealingSchedulerOptions>(&options)) {
    if (annealing->annealing.initialSchedule ==
        AnnealingInitialSchedule::Greedy)
      return annealing->greedyInitialPlacement;
  }
  static const GreedyScheduleConfig defaultGreedy;
  return defaultGreedy;
}

} // namespace

LogicalResult appendScheduleSummary(func::FuncOp taskGraphFunc,
                                    const HardwareBudget &budget,
                                    const TaskGraphSchedulerOptions &options,
                                    StringRef scheduleName,
                                    StringRef outputPath) {
  if (outputPath.empty())
    return success();

  auto taskCount =
      getIntegerSummaryAttr(taskGraphFunc, schedule_attrs::kTaskCountAttrName);
  auto dependencyCount = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kDependencyCountAttrName);
  auto numLogicalArrays = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kNumLogicalArraysAttrName);
  auto totalDigitalOps = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kTotalDigitalOpsAttrName);
  auto interCoreTransferBytes = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kInterCoreTransferBytesAttrName);
  auto totalTransferCost = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kTotalTransferCostAttrName);
  auto boundaryPenalty = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kBoundaryPenaltyAttrName);
  auto graphScore =
      getIntegerSummaryAttr(taskGraphFunc, schedule_attrs::kGraphScoreAttrName);
  auto transferCostPerByte = getFloatSummaryAttr(
      taskGraphFunc, schedule_attrs::kTransferCostPerInterCoreByteAttrName);

  if (failed(taskCount) || failed(dependencyCount) ||
      failed(numLogicalArrays) || failed(totalDigitalOps) ||
      failed(interCoreTransferBytes) || failed(totalTransferCost) ||
      failed(boundaryPenalty) || failed(graphScore) ||
      failed(transferCostPerByte))
    return failure();

  int64_t annealingEpochs = 0;
  int64_t annealingEvaluations = 0;
  int64_t annealingInitialScore = 0;
  int64_t annealingBestScore = 0;
  double annealingUphillAcceptanceRate = 0.0;
  double annealingSearchSeconds = 0.0;
  StringAttr annealingStopReason;
  if (scheduleName == "annealing") {
    auto epochs = getIntegerSummaryAttr(
        taskGraphFunc, schedule_attrs::kAnnealingEpochsAttrName);
    auto evaluations = getIntegerSummaryAttr(
        taskGraphFunc, schedule_attrs::kAnnealingEvaluationsAttrName);
    auto initialScore = getIntegerSummaryAttr(
        taskGraphFunc, schedule_attrs::kAnnealingInitialScoreAttrName);
    auto bestScore = getIntegerSummaryAttr(
        taskGraphFunc, schedule_attrs::kAnnealingBestScoreAttrName);
    auto uphillAcceptanceRate = getFloatSummaryAttr(
        taskGraphFunc, schedule_attrs::kAnnealingUphillAcceptanceRateAttrName);
    auto searchSeconds = getFloatSummaryAttr(
        taskGraphFunc, schedule_attrs::kAnnealingSearchSecondsAttrName);
    annealingStopReason = taskGraphFunc->getAttrOfType<StringAttr>(
        schedule_attrs::kAnnealingStopReasonAttrName);
    if (failed(epochs) || failed(evaluations) || failed(initialScore) ||
        failed(bestScore) || failed(uphillAcceptanceRate) ||
        failed(searchSeconds) || !annealingStopReason) {
      if (!annealingStopReason)
        taskGraphFunc.emitError("expected annealing stop-reason attribute");
      return failure();
    }
    annealingEpochs = *epochs;
    annealingEvaluations = *evaluations;
    annealingInitialScore = *initialScore;
    annealingBestScore = *bestScore;
    annealingUphillAcceptanceRate = *uphillAcceptanceRate;
    annealingSearchSeconds = *searchSeconds;
  }

  StringRef placementCostMode = "n/a";
  double searchCompletionTimeProxy = 0.0;
  double searchCommunicationProxy = 0.0;
  double searchResourceLoadProxy = 0.0;
  double predictedMakespanNs = 0.0;
  double predictedExposedContentionNs = 0.0;
  double predictedExposedTransportNs = 0.0;
  int64_t predictedTotalWordHops = 0;
  int64_t timingRerankCandidateCount = 0;
  int64_t timingRerankSelectedProxyRank = 0;
  if (auto mode = taskGraphFunc->getAttrOfType<StringAttr>(
          schedule_attrs::kPlacementCostModeAttrName)) {
    auto completionTimeProxy = getFloatSummaryAttr(
        taskGraphFunc, schedule_attrs::kSearchCompletionTimeProxyAttrName);
    auto communicationProxy = getFloatSummaryAttr(
        taskGraphFunc, schedule_attrs::kSearchCommunicationProxyAttrName);
    auto resourceLoadProxy = getFloatSummaryAttr(
        taskGraphFunc, schedule_attrs::kSearchResourceLoadProxyAttrName);
    auto makespan = getFloatSummaryAttr(
        taskGraphFunc, schedule_attrs::kPredictedMakespanNsAttrName);
    auto exposedContention = getFloatSummaryAttr(
        taskGraphFunc, schedule_attrs::kPredictedExposedContentionNsAttrName);
    auto exposedTransport = getFloatSummaryAttr(
        taskGraphFunc, schedule_attrs::kPredictedExposedTransportNsAttrName);
    auto wordHops = getIntegerSummaryAttr(
        taskGraphFunc, schedule_attrs::kPredictedTotalWordHopsAttrName);
    auto rerankCandidateCount = getIntegerSummaryAttr(
        taskGraphFunc, schedule_attrs::kTimingRerankCandidateCountAttrName);
    auto rerankSelectedProxyRank = getIntegerSummaryAttr(
        taskGraphFunc, schedule_attrs::kTimingRerankSelectedProxyRankAttrName);
    if (failed(completionTimeProxy) || failed(communicationProxy) ||
        failed(resourceLoadProxy) || failed(makespan) ||
        failed(exposedContention) || failed(exposedTransport) ||
        failed(wordHops) || failed(rerankCandidateCount) ||
        failed(rerankSelectedProxyRank))
      return failure();
    placementCostMode = mode.getValue();
    searchCompletionTimeProxy = *completionTimeProxy;
    searchCommunicationProxy = *communicationProxy;
    searchResourceLoadProxy = *resourceLoadProxy;
    predictedMakespanNs = *makespan;
    predictedExposedContentionNs = *exposedContention;
    predictedExposedTransportNs = *exposedTransport;
    predictedTotalWordHops = *wordHops;
    timingRerankCandidateCount = *rerankCandidateCount;
    timingRerankSelectedProxyRank = *rerankSelectedProxyRank;
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec, llvm::sys::fs::OF_Append);
  if (ec) {
    taskGraphFunc.emitError("failed to open schedule summary output '")
        << outputPath << "': " << ec.message();
    return failure();
  }

  const GreedyScheduleConfig &greedy = getReportingGreedyConfig(options);
  writeCsvString(os, taskGraphFunc.getName());
  os << ',';
  writeCsvString(os, scheduleName);
  os << ',' << budget.numCores << ',' << budget.arraysPerCore << ','
     << budget.meshRows << ',' << budget.meshCols << ',' << greedy.lookahead
     << ',' << greedy.beamWidth << ',';
  writeCsvString(os, stringifyGreedyCandidateScope(greedy.candidateScope));
  os << ',';
  writeCsvString(os, greedy.specification);
  os << ',' << *taskCount << ',' << *dependencyCount << ',' << *numLogicalArrays
     << ',' << *totalDigitalOps << ',' << *interCoreTransferBytes << ','
     << *totalTransferCost << ',' << *transferCostPerByte << ','
     << *boundaryPenalty << ',' << *graphScore;

  if (scheduleName == "annealing") {
    os << ',';
    writeCsvString(os, annealingStopReason.getValue());
    os << ',' << annealingEpochs << ',' << annealingEvaluations << ','
       << annealingInitialScore << ',' << annealingBestScore << ','
       << annealingUphillAcceptanceRate << ',' << annealingSearchSeconds;
  } else {
    os << ",,,,,,,";
  }
  os << ',';
  writeCsvString(os, placementCostMode);
  os << ',' << searchCompletionTimeProxy << ',' << searchCommunicationProxy
     << ',' << searchResourceLoadProxy << ',' << predictedMakespanNs << ','
     << predictedExposedContentionNs << ',' << predictedExposedTransportNs
     << ',' << predictedTotalWordHops << ',' << timingRerankCandidateCount
     << ',' << timingRerankSelectedProxyRank;
  os << '\n';
  return success();
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
