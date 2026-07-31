#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include <algorithm>
#include <cmath>

namespace mlir {
namespace sculptor {
namespace task_timing {

llvm::StringRef stringifyTaskCostSource(TaskCostSource source) {
  switch (source) {
  case TaskCostSource::StaticAnalysis:
    return "static-analysis";
  case TaskCostSource::CalibratedFallback:
    return "calibrated-fallback";
  case TaskCostSource::ExplicitMetadata:
    return "explicit-metadata";
  }
  llvm_unreachable("unknown task cost source");
}

llvm::StringRef stringifyTaskCostConfidence(TaskCostConfidence confidence) {
  switch (confidence) {
  case TaskCostConfidence::High:
    return "high";
  case TaskCostConfidence::Medium:
    return "medium";
  case TaskCostConfidence::Low:
    return "low";
  }
  llvm_unreachable("unknown task cost confidence");
}

std::optional<TaskCostSource> symbolizeTaskCostSource(llvm::StringRef value) {
  if (value == "static-analysis")
    return TaskCostSource::StaticAnalysis;
  if (value == "calibrated-fallback")
    return TaskCostSource::CalibratedFallback;
  if (value == "explicit-metadata")
    return TaskCostSource::ExplicitMetadata;
  return std::nullopt;
}

std::optional<TaskCostConfidence>
symbolizeTaskCostConfidence(llvm::StringRef value) {
  if (value == "high")
    return TaskCostConfidence::High;
  if (value == "medium")
    return TaskCostConfidence::Medium;
  if (value == "low")
    return TaskCostConfidence::Low;
  return std::nullopt;
}

LogicalResult validateTimingModel(Operation *anchor, const TimingModel &model) {
  if (model.costModel != "golem-qemu-v1" || model.costModelRevision != 1)
    return anchor->emitError("unsupported timing cost model revision; expected "
                             "'golem-qemu-v1' revision 1 but received '")
           << model.costModel << "' revision " << model.costModelRevision;
  if (model.compilerRevision.empty())
    return anchor->emitError("expected compiler revision provenance");
  if (model.timingBoundary != "warm" && model.timingBoundary != "cold")
    return anchor->emitError("expected timing boundary to be 'warm' or 'cold'");
  if (model.runtimeTaskPolicy != "lowest-local-task-index")
    return anchor->emitError("unsupported runtime task policy; expected "
                             "'lowest-local-task-index'");
  if (model.runtimeTransmitPolicy != "overlap-ready-tasks")
    return anchor->emitError("unsupported runtime transmit policy; expected "
                             "'overlap-ready-tasks'");
  if (model.memoryBackend.empty())
    return anchor->emitError("expected local-memory backend provenance");
  if (model.analogMVMLatencyNs < 0)
    return anchor->emitError("expected analog MVM latency to be non-negative");
  if (model.analogIOBitsPerCycle <= 0)
    return anchor->emitError("expected analog I/O bandwidth to be positive");
  if (!std::isfinite(model.digitalClockGHz) || model.digitalClockGHz <= 0.0)
    return anchor->emitError("expected digital clock frequency to be positive");
  if (model.digitalIssueWidth <= 0)
    return anchor->emitError("expected digital issue width to be positive");
  if (model.digitalVectorBitsPerCycle <= 0)
    return anchor->emitError(
        "expected digital vector throughput to be positive");
  if (model.fixedRuntimeDispatchCycles < 0)
    return anchor->emitError(
        "expected fixed runtime dispatch cycles to be non-negative");
  if (model.fixedTaskEntryCycles < 0)
    return anchor->emitError(
        "expected fixed task entry cycles to be non-negative");
  if (model.fixedTaskExitCycles < 0)
    return anchor->emitError(
        "expected fixed task exit cycles to be non-negative");
  if (model.networkLinkBitsPerCycle <= 0)
    return anchor->emitError("expected network link bandwidth to be positive");
  if (model.networkHopLatencyCycles < 0)
    return anchor->emitError("expected network hop latency to be non-negative");
  if (model.networkLinkWordBits <= 0)
    return anchor->emitError("expected network word width to be positive");
  if (model.protocolWordsPerRoute < 0)
    return anchor->emitError(
        "expected protocol words per route to be non-negative");
  if (model.nicInjectionWordsPerCycle <= 0)
    return anchor->emitError(
        "expected NIC injection throughput to be positive");
  if (model.rxDmaWordsPerCycle <= 0)
    return anchor->emitError("expected receive DMA throughput to be positive");
  if (model.routingPolicy != "xy")
    return anchor->emitError("unsupported routing policy; expected 'xy'");
  return success();
}

double cyclesToNanoseconds(double cycles, const TimingModel &model) {
  return cycles / model.digitalClockGHz;
}

double estimateNetworkTransferLatencyNs(int64_t bytes, int64_t meshHops,
                                        const TimingModel &model) {
  if (bytes <= 0 || meshHops <= 0)
    return 0.0;

  double flits = std::ceil(static_cast<double>(bytes) * 8.0 /
                           static_cast<double>(model.networkLinkBitsPerCycle));
  double hopLatency = model.networkHopLatencyCycles;
  double cycles =
      model.networkPipelined
          ? flits + static_cast<double>(meshHops) * hopLatency - 1.0
          : static_cast<double>(meshHops) * (flits + hopLatency - 1.0);
  return cyclesToNanoseconds(std::max(0.0, cycles), model);
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
