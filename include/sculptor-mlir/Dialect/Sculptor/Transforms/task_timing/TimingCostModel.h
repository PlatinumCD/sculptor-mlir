#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TIMINGCOSTMODEL_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TIMINGCOSTMODEL_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {

LogicalResult validateTimingModel(Operation *anchor, const TimingModel &model);

double cyclesToNanoseconds(double cycles, const TimingModel &model);

double estimateNetworkTransferLatencyNs(int64_t bytes, int64_t meshHops,
                                        const TimingModel &model);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TIMINGCOSTMODEL_H
