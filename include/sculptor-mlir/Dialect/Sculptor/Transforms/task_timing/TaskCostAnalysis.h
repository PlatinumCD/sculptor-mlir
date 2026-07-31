#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKCOSTANALYSIS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKCOSTANALYSIS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {

FailureOr<TaskCost>
estimateTaskCost(ModuleOp module, sculptor::TaskCreateOp task,
                 TaskWorkloadFeatures &workload, int64_t effectiveDigitalOps,
                 int64_t digitalReplacementOps, const TimingModel &model);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKCOSTANALYSIS_H
