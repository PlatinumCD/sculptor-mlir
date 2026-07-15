#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskLatencyModel.h"

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

  int64_t inputBytes = 0;
  int64_t outputBytes = 0;
  if (failed(collectTaskIOBytes(taskOp, inputBytes, outputBytes)))
    return failure();

  TaskLatencyEstimate estimate;
  double digitalLatencyNs = 0.0;
  if (digitalOps > 0) {
    double vectorOpsPerCycle =
        static_cast<double>(model.digitalVectorBitsPerCycle) / 32.0;
    double operationsPerCycle =
        std::max<double>(model.digitalIssueWidth, vectorOpsPerCycle);
    digitalLatencyNs =
        cyclesToNanoseconds(std::ceil(digitalOps / operationsPerCycle), model);
  }

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
  estimate.analogExecuteLatencyNs = model.analogMVMLatencyNs;
  estimate.analogStoreLatencyNs = cyclesToNanoseconds(storeCycles, model);
  estimate.intrinsicLatencyNs =
      estimate.analogLoadLatencyNs + estimate.analogExecuteLatencyNs +
      estimate.analogStoreLatencyNs + digitalLatencyNs;
  return estimate;
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
