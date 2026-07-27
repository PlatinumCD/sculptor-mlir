#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskLatencyModel.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTimingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResourceUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include "mlir/IR/BuiltinAttributes.h"

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

double computeDigitalLatencyNs(int64_t digitalOps, const TimingModel &model) {
  if (digitalOps <= 0)
    return 0.0;
  double vectorOpsPerCycle =
      static_cast<double>(model.digitalVectorBitsPerCycle) / 32.0;
  double operationsPerCycle =
      std::max<double>(model.digitalIssueWidth, vectorOpsPerCycle);
  return cyclesToNanoseconds(std::ceil(digitalOps / operationsPerCycle), model);
}

FailureOr<TaskLatencyEstimate>
estimateStreamingConvolutionLatency(sculptor::TaskCreateOp taskOp,
                                    int64_t digitalOps,
                                    const TimingModel &model) {
  auto counts = taskOp->getAttrOfType<ArrayAttr>(
      runtime_attrs::kTaskAnalogExecutionCountsAttrName);
  auto loadBytes = taskOp->getAttrOfType<IntegerAttr>(
      runtime_attrs::kTaskAnalogLoadBytesAttrName);
  auto storeBytes = taskOp->getAttrOfType<IntegerAttr>(
      runtime_attrs::kTaskAnalogStoreBytesAttrName);
  if (!counts || counts.empty() || !loadBytes || !storeBytes ||
      loadBytes.getInt() < 0 || storeBytes.getInt() < 0) {
    taskOp.emitError(
        "expected streaming convolution timing counts and byte totals");
    return failure();
  }

  int64_t maximumExecutionCount = 0;
  for (Attribute attribute : counts) {
    auto count = dyn_cast<IntegerAttr>(attribute);
    if (!count || count.getInt() <= 0) {
      taskOp.emitError("expected positive per-array analog execution counts");
      return failure();
    }
    maximumExecutionCount = std::max(maximumExecutionCount, count.getInt());
  }

  double loadCycles =
      std::ceil(static_cast<double>(loadBytes.getInt()) * 8.0 /
                static_cast<double>(model.analogIOBitsPerCycle));
  double storeCycles =
      std::ceil(static_cast<double>(storeBytes.getInt()) * 8.0 /
                static_cast<double>(model.analogIOBitsPerCycle));

  TaskLatencyEstimate estimate;
  estimate.analogLoadLatencyNs = cyclesToNanoseconds(loadCycles, model);
  estimate.analogExecuteLatencyNs =
      model.analogMVMLatencyNs * static_cast<double>(maximumExecutionCount);
  estimate.analogStoreLatencyNs = cyclesToNanoseconds(storeCycles, model);
  estimate.intrinsicLatencyNs = estimate.analogLoadLatencyNs +
                                estimate.analogExecuteLatencyNs +
                                estimate.analogStoreLatencyNs +
                                computeDigitalLatencyNs(digitalOps, model);
  return estimate;
}

} // namespace

FailureOr<TaskLatencyEstimate>
estimateTaskLatency(sculptor::TaskCreateOp taskOp, int64_t digitalOps,
                    const TimingModel &model) {
  if (taskOp.getTaskKind() == "mixed.fused") {
    auto load = taskOp->getAttrOfType<FloatAttr>(
        timing_attrs::kAnalogLoadLatencyNsAttrName);
    auto execute = taskOp->getAttrOfType<FloatAttr>(
        timing_attrs::kAnalogExecuteLatencyNsAttrName);
    auto store = taskOp->getAttrOfType<FloatAttr>(
        timing_attrs::kAnalogStoreLatencyNsAttrName);
    auto intrinsic = taskOp->getAttrOfType<FloatAttr>(
        timing_attrs::kIntrinsicLatencyNsAttrName);
    if (load && execute && store && intrinsic) {
      return TaskLatencyEstimate{
          load.getValueAsDouble(), execute.getValueAsDouble(),
          store.getValueAsDouble(), intrinsic.getValueAsDouble()};
    }
  }

  if (taskOp.getTaskKind() == task_graph_names::kStreamingConvolutionTaskKind)
    return estimateStreamingConvolutionLatency(taskOp, digitalOps, model);

  int64_t inputBytes = 0;
  int64_t outputBytes = 0;
  if (failed(collectTaskIOBytes(taskOp, inputBytes, outputBytes)))
    return failure();

  TaskLatencyEstimate estimate;
  double digitalLatencyNs = computeDigitalLatencyNs(digitalOps, model);

  if (!task_graph::isAnalogComputeTask(taskOp)) {
    estimate.intrinsicLatencyNs = digitalLatencyNs;
    return estimate;
  }

  double inputBits = static_cast<double>(inputBytes) * 8.0;
  double outputBits = static_cast<double>(outputBytes) * 8.0;
  double loadCycles =
      std::ceil(inputBits / static_cast<double>(model.analogIOBitsPerCycle));
  double storeCycles =
      std::ceil(outputBits / static_cast<double>(model.analogIOBitsPerCycle));

  estimate.analogLoadLatencyNs = cyclesToNanoseconds(loadCycles, model);
  int64_t analogExecutionCount = 1;
  if (auto countAttr = taskOp->getAttrOfType<IntegerAttr>(
          runtime_attrs::kTaskAnalogExecutionCountAttrName)) {
    analogExecutionCount = countAttr.getInt();
    if (analogExecutionCount <= 0) {
      taskOp.emitError("expected positive analog execution count");
      return failure();
    }
  }
  estimate.analogExecuteLatencyNs =
      model.analogMVMLatencyNs * static_cast<double>(analogExecutionCount);
  estimate.analogStoreLatencyNs = cyclesToNanoseconds(storeCycles, model);
  estimate.intrinsicLatencyNs =
      estimate.analogLoadLatencyNs + estimate.analogExecuteLatencyNs +
      estimate.analogStoreLatencyNs + digitalLatencyNs;
  return estimate;
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
