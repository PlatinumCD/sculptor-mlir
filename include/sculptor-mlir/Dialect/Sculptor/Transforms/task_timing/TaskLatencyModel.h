#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKLATENCYMODEL_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKLATENCYMODEL_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {

struct TaskLatencyEstimate {
  TaskCost cost;
  double analogLoadLatencyNs = 0.0;
  double analogExecuteLatencyNs = 0.0;
  double analogStoreLatencyNs = 0.0;
  double analogPipelineLatencyNs = 0.0;
  double intrinsicLatencyNs = 0.0;
};

FailureOr<TaskLatencyEstimate>
estimateTaskLatency(ModuleOp module, sculptor::TaskCreateOp taskOp,
                    TaskWorkloadFeatures &workload, int64_t effectiveDigitalOps,
                    int64_t digitalReplacementOps, const TimingModel &model,
                    MVMCostMode mvmCostMode);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKLATENCYMODEL_H
