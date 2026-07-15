#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDYSEARCH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDYSEARCH_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<IslandPlacementPlan>
buildGreedyPlacementPlan(const TaskGraphPlacementProblem &problem,
                         llvm::ArrayRef<int64_t> physicalArrayOrder,
                         const GreedyScheduleConfig &config);

FailureOr<IslandPlacementPlan> buildGreedyTimingPlacementPlan(
    const TaskGraphPlacementProblem &problem,
    const task_timing::SchedulingTimingProfile &timingProfile,
    llvm::ArrayRef<int64_t> physicalArrayOrder,
    const GreedyScheduleConfig &config);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDYSEARCH_H
