#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskLatencyModel.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResourceUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskCostAnalysis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mlir {
namespace sculptor {
namespace task_timing {
namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;

LogicalResult addByteCount(Operation *anchor, int64_t bytes, int64_t &total) {
  if (bytes < 0 || total > std::numeric_limits<int64_t>::max() - bytes) {
    anchor->emitError("task graph byte count overflow during timing analysis");
    return failure();
  }
  total += bytes;
  return success();
}

LogicalResult collectTaskIOBytes(sculptor::TaskCreateOp taskOp,
                                 int64_t &inputBytes, int64_t &outputBytes) {
  for (Value input : taskOp.getInputs()) {
    FailureOr<int64_t> bytes = getTaskResourceByteSize(input);
    if (failed(bytes)) {
      taskOp.emitError(
          "expected statically sized task inputs for timing analysis");
      return failure();
    }
    if (failed(addByteCount(taskOp, *bytes, inputBytes)))
      return failure();
  }

  for (Value output : taskOp.getOutputs()) {
    FailureOr<int64_t> bytes = getTaskResourceByteSize(output);
    if (failed(bytes)) {
      taskOp.emitError(
          "expected statically sized task outputs for timing analysis");
      return failure();
    }
    if (failed(addByteCount(taskOp, *bytes, outputBytes)))
      return failure();
  }
  return success();
}

FailureOr<TaskLatencyEstimate>
estimateStreamingConvolutionLatency(sculptor::TaskCreateOp taskOp,
                                    const TaskCost &cost,
                                    const TimingModel &model) {
  auto counts = taskOp->getAttrOfType<ArrayAttr>(
      runtime_attrs::kTaskAnalogExecutionCountsAttrName);
  auto loadBytes = taskOp->getAttrOfType<IntegerAttr>(
      runtime_attrs::kTaskAnalogLoadBytesAttrName);
  auto storeBytes = taskOp->getAttrOfType<IntegerAttr>(
      runtime_attrs::kTaskAnalogStoreBytesAttrName);
  auto loadBytesPerArray = taskOp->getAttrOfType<ArrayAttr>(
      runtime_attrs::kTaskAnalogLoadBytesPerArrayAttrName);
  auto storeBytesPerArray = taskOp->getAttrOfType<ArrayAttr>(
      runtime_attrs::kTaskAnalogStoreBytesPerArrayAttrName);
  if (!counts || counts.empty() || !loadBytes || !storeBytes ||
      !loadBytesPerArray || !storeBytesPerArray ||
      loadBytesPerArray.size() != counts.size() ||
      storeBytesPerArray.size() != counts.size() || loadBytes.getInt() < 0 ||
      storeBytes.getInt() < 0) {
    taskOp.emitError(
        "expected streaming convolution per-array timing counts and bytes");
    return failure();
  }

  llvm::SmallVector<int64_t, 4> executionCounts;
  llvm::SmallVector<int64_t, 4> perArrayLoadBytes;
  llvm::SmallVector<int64_t, 4> perArrayStoreBytes;
  int64_t observedLoadBytes = 0;
  int64_t observedStoreBytes = 0;
  for (auto [countAttr, loadAttr, storeAttr] :
       llvm::zip_equal(counts, loadBytesPerArray, storeBytesPerArray)) {
    auto count = dyn_cast<IntegerAttr>(countAttr);
    auto arrayLoad = dyn_cast<IntegerAttr>(loadAttr);
    auto arrayStore = dyn_cast<IntegerAttr>(storeAttr);
    if (!count || !arrayLoad || !arrayStore || count.getInt() <= 0 ||
        arrayLoad.getInt() < 0 || arrayStore.getInt() < 0 ||
        observedLoadBytes >
            std::numeric_limits<int64_t>::max() - arrayLoad.getInt() ||
        observedStoreBytes >
            std::numeric_limits<int64_t>::max() - arrayStore.getInt()) {
      taskOp.emitError(
          "expected valid per-array analog execution and byte counts");
      return failure();
    }
    executionCounts.push_back(count.getInt());
    perArrayLoadBytes.push_back(arrayLoad.getInt());
    perArrayStoreBytes.push_back(arrayStore.getInt());
    observedLoadBytes += arrayLoad.getInt();
    observedStoreBytes += arrayStore.getInt();
  }
  if (observedLoadBytes != loadBytes.getInt() ||
      observedStoreBytes != storeBytes.getInt()) {
    taskOp.emitError(
        "per-array analog byte counts do not match aggregate totals");
    return failure();
  }

  llvm::SmallVector<double, 4> arrayAvailableCycles(counts.size(), 0.0);
  double sharedIOAvailableCycles = 0.0;
  double loadAvailableCycles = 0.0;
  double storeAvailableCycles = 0.0;
  int64_t maximumExecutionCount = *llvm::max_element(executionCounts);
  double executeCyclesPerOperation =
      static_cast<double>(model.analogMVMLatencyNs) * model.digitalClockGHz;
  for (int64_t execution = 0; execution < maximumExecutionCount; ++execution) {
    for (size_t array = 0; array < executionCounts.size(); ++array) {
      if (execution >= executionCounts[array])
        continue;
      double loadCycles =
          std::ceil(static_cast<double>(perArrayLoadBytes[array]) * 8.0 /
                    static_cast<double>(executionCounts[array]) /
                    static_cast<double>(model.analogIOBitsPerCycle));
      double &ioAvailable =
          model.analogIOShared ? sharedIOAvailableCycles : loadAvailableCycles;
      double loadStart = std::max(arrayAvailableCycles[array], ioAvailable);
      double loadFinish = loadStart + loadCycles;
      ioAvailable = loadFinish;
      arrayAvailableCycles[array] = loadFinish + executeCyclesPerOperation;
    }
    for (size_t array = 0; array < executionCounts.size(); ++array) {
      if (execution >= executionCounts[array])
        continue;
      double storeCycles =
          std::ceil(static_cast<double>(perArrayStoreBytes[array]) * 8.0 /
                    static_cast<double>(executionCounts[array]) /
                    static_cast<double>(model.analogIOBitsPerCycle));
      double &ioAvailable =
          model.analogIOShared ? sharedIOAvailableCycles : storeAvailableCycles;
      double storeStart = std::max(arrayAvailableCycles[array], ioAvailable);
      double storeFinish = storeStart + storeCycles;
      ioAvailable = storeFinish;
      arrayAvailableCycles[array] = storeFinish;
    }
  }

  int64_t totalExecutions = 0;
  for (int64_t count : executionCounts) {
    if (totalExecutions > std::numeric_limits<int64_t>::max() - count) {
      taskOp.emitOpError("analog execution count overflow");
      return failure();
    }
    totalExecutions += count;
  }
  double loadCycles =
      std::ceil(static_cast<double>(observedLoadBytes) * 8.0 /
                static_cast<double>(model.analogIOBitsPerCycle));
  double storeCycles =
      std::ceil(static_cast<double>(observedStoreBytes) * 8.0 /
                static_cast<double>(model.analogIOBitsPerCycle));
  double pipelineCycles = *llvm::max_element(arrayAvailableCycles);

  TaskLatencyEstimate estimate;
  estimate.cost = cost;
  estimate.analogLoadLatencyNs = cyclesToNanoseconds(loadCycles, model);
  estimate.analogExecuteLatencyNs =
      model.analogMVMLatencyNs * static_cast<double>(totalExecutions);
  estimate.analogStoreLatencyNs = cyclesToNanoseconds(storeCycles, model);
  estimate.analogPipelineLatencyNs = cyclesToNanoseconds(pipelineCycles, model);
  estimate.intrinsicLatencyNs =
      estimate.analogPipelineLatencyNs +
      cyclesToNanoseconds(cost.predictedCpuCycles, model);
  return estimate;
}

} // namespace

FailureOr<TaskLatencyEstimate>
estimateTaskLatency(ModuleOp module, sculptor::TaskCreateOp taskOp,
                    TaskWorkloadFeatures &workload, int64_t effectiveDigitalOps,
                    int64_t digitalReplacementOps, const TimingModel &model,
                    MVMCostMode mvmCostMode) {
  FailureOr<TaskCost> taskCost =
      estimateTaskCost(module, taskOp, workload, effectiveDigitalOps,
                       digitalReplacementOps, model);
  if (failed(taskCost))
    return failure();

  bool hasBodyAnalogWork = workload.analogExecutionCount > 0 ||
                           workload.analogLoadBytes > 0 ||
                           workload.analogStoreBytes > 0;
  if (mvmCostMode == MVMCostMode::Digital &&
      (task_graph::isAnalogComputeTask(taskOp) ||
       taskOp.getTaskKind() ==
           task_graph_names::kStreamingConvolutionTaskKind ||
       hasBodyAnalogWork)) {
    if (hasBodyAnalogWork && digitalReplacementOps == 0) {
      taskOp.emitError(
          "digital MVM cost mode requires explicit replacement work for a "
          "task whose final body contains analog array operations");
      return failure();
    }
    TaskLatencyEstimate estimate;
    estimate.cost = *taskCost;
    estimate.intrinsicLatencyNs =
        cyclesToNanoseconds(taskCost->predictedCpuCycles, model);
    return estimate;
  }

  if (taskOp.getTaskKind() == task_graph_names::kStreamingConvolutionTaskKind)
    return estimateStreamingConvolutionLatency(taskOp, *taskCost, model);

  int64_t inputBytes = 0;
  int64_t outputBytes = 0;
  if (failed(collectTaskIOBytes(taskOp, inputBytes, outputBytes)))
    return failure();

  TaskLatencyEstimate estimate;
  estimate.cost = *taskCost;
  double digitalLatencyNs =
      cyclesToNanoseconds(taskCost->predictedCpuCycles, model);

  if (!task_graph::isAnalogComputeTask(taskOp) && !hasBodyAnalogWork) {
    estimate.intrinsicLatencyNs =
        model.timingBoundary == "warm" && task_graph::isMatrixSetupTask(taskOp)
            ? 0.0
            : digitalLatencyNs;
    return estimate;
  }

  if (hasBodyAnalogWork) {
    if (workload.analogExecutionCount == 0 || workload.analogLoadBytes == 0 ||
        workload.analogStoreBytes == 0) {
      taskOp.emitError(
          "expected analog task body to contain load, execute, and store work");
      return failure();
    }
    inputBytes = static_cast<int64_t>(workload.analogLoadBytes);
    outputBytes = static_cast<int64_t>(workload.analogStoreBytes);
  }

  double inputBits = static_cast<double>(inputBytes) * 8.0;
  double outputBits = static_cast<double>(outputBytes) * 8.0;
  double loadCycles =
      std::ceil(inputBits / static_cast<double>(model.analogIOBitsPerCycle));
  double storeCycles =
      std::ceil(outputBits / static_cast<double>(model.analogIOBitsPerCycle));

  estimate.analogLoadLatencyNs = cyclesToNanoseconds(loadCycles, model);
  int64_t analogExecutionCount =
      hasBodyAnalogWork ? static_cast<int64_t>(workload.analogExecutionCount)
                        : 1;
  if (!hasBodyAnalogWork) {
    auto countAttr = taskOp->getAttrOfType<IntegerAttr>(
        runtime_attrs::kTaskAnalogExecutionCountAttrName);
    if (countAttr)
      analogExecutionCount = countAttr.getInt();
  }
  if (analogExecutionCount <= 0) {
    taskOp.emitError("expected positive analog execution count");
    return failure();
  }
  estimate.analogExecuteLatencyNs =
      model.analogMVMLatencyNs * static_cast<double>(analogExecutionCount);
  estimate.analogStoreLatencyNs = cyclesToNanoseconds(storeCycles, model);
  estimate.analogPipelineLatencyNs = estimate.analogLoadLatencyNs +
                                     estimate.analogExecuteLatencyNs +
                                     estimate.analogStoreLatencyNs;
  estimate.intrinsicLatencyNs =
      estimate.analogPipelineLatencyNs + digitalLatencyNs;
  return estimate;
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
