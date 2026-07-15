#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKLATENCYMODEL_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKLATENCYMODEL_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {

struct TaskLatencyEstimate {
  double analogLoadLatencyNs = 0.0;
  double analogExecuteLatencyNs = 0.0;
  double analogStoreLatencyNs = 0.0;
  double intrinsicLatencyNs = 0.0;
};

FailureOr<TaskLatencyEstimate>
estimateTaskLatency(sculptor::TaskCreateOp taskOp, int64_t digitalOps,
                    const TimingModel &model);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKLATENCYMODEL_H
