#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGIRCODEC_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGIRCODEC_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sculptor {
namespace task_timing {

struct TimingAnalysis;

void attachTaskGraphTimingAnalysis(func::FuncOp taskGraphFunc,
                                   const task_graph::TaskGraphDAG &dag,
                                   const TimingAnalysis &analysis,
                                   const TimingModel &model);

FailureOr<TimingModel> loadTimingModel(func::FuncOp taskGraphFunc);

FailureOr<SchedulingTimingProfile> loadSchedulingTimingProfile(
    func::FuncOp taskGraphFunc, const task_graph::TaskGraphDAG &dag,
    const task_graph::LogicalPlacementIslandGraph &islandGraph);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGIRCODEC_H
