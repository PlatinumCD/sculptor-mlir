#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

class TaskGraphScheduler {
public:
  virtual ~TaskGraphScheduler() = default;

  virtual StringRef getName() const = 0;

  virtual LogicalResult schedule(ModuleOp module, func::FuncOp taskGraphFunc,
                                 const HardwareBudget &budget,
                                 const TaskGraphDAG &dag) const = 0;
};

using TaskGraphSchedulerRegistry =
    llvm::StringMap<std::unique_ptr<TaskGraphScheduler>>;

LogicalResult
registerTaskGraphScheduler(TaskGraphSchedulerRegistry &registry,
                           std::unique_ptr<TaskGraphScheduler> scheduler);

const TaskGraphScheduler *
lookupTaskGraphScheduler(const TaskGraphSchedulerRegistry &registry,
                         StringRef name);

void registerRandomTaskScheduler(TaskGraphSchedulerRegistry &registry);
void registerSnakeTaskScheduler(TaskGraphSchedulerRegistry &registry);
void registerGreedyTaskScheduler(TaskGraphSchedulerRegistry &registry);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULER_H
