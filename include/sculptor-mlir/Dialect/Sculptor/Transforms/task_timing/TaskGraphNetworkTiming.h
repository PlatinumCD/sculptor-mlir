#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHNETWORKTIMING_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHNETWORKTIMING_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sculptor {
namespace task_timing {

struct TimingAnalysis;

LogicalResult applyPlacementAwareNetworkTimingIfAvailable(
    func::FuncOp taskGraphFunc, const task_graph::TaskGraphDAG &dag,
    const TimingModel &model, TimingAnalysis &analysis);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHNETWORKTIMING_H
