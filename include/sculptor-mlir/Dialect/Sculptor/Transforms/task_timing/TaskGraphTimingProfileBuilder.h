#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILEBUILDER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILEBUILDER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"

namespace mlir {
namespace sculptor {
namespace task_timing {

struct TimingAnalysis;

void buildSchedulingTimingProfile(
    const task_graph::LogicalPlacementIslandGraph &islandGraph,
    const TimingModel &model, TimingAnalysis &analysis);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILEBUILDER_H
