#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_ANNEALING_ANNEALINGSEARCH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_ANNEALING_ANNEALINGSEARCH_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleConfig.h"

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace annealing_detail {

FailureOr<IslandPlacementPlan>
buildPlacementPlan(const TaskGraphPlacementProblem &problem,
                   const AnnealingScheduleConfig &config,
                   const GreedyScheduleConfig &greedyInitialPlacement,
                   int64_t randomSeed);

} // namespace annealing_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_ANNEALING_ANNEALINGSEARCH_H
