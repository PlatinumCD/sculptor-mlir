#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include <algorithm>
#include <cmath>

namespace mlir {
namespace sculptor {
namespace task_timing {

LogicalResult validateTimingModel(Operation *anchor, const TimingModel &model) {
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
  if (model.networkLinkBitsPerCycle <= 0)
    return anchor->emitError("expected network link bandwidth to be positive");
  if (model.networkHopLatencyCycles < 0)
    return anchor->emitError("expected network hop latency to be non-negative");
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
  double cycles = model.networkPipelined
                      ? flits + static_cast<double>(meshHops) * hopLatency - 1.0
                      : static_cast<double>(meshHops) *
                            (flits + hopLatency - 1.0);
  return cyclesToNanoseconds(std::max(0.0, cycles), model);
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
