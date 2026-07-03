#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDYPLACEMENT_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDYPLACEMENT_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"

namespace mlir {
namespace sculptor {
namespace task_schedulers {

LogicalResult
runGreedyIslandPlacement(ModuleOp module, func::FuncOp taskGraphFunc,
                         const HardwareBudget &budget, const TaskGraphDAG &dag,
                         llvm::ArrayRef<int64_t> physicalArrayOrder);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDYPLACEMENT_H
