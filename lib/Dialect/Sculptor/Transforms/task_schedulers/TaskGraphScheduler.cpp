#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <utility>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

LogicalResult
registerTaskGraphScheduler(TaskGraphSchedulerRegistry &registry,
                           std::unique_ptr<TaskGraphScheduler> scheduler) {
  if (!scheduler)
    return failure();

  StringRef name = scheduler->getName();
  if (name.empty() || registry.contains(name))
    return failure();

  registry.try_emplace(name, std::move(scheduler));
  return success();
}

const TaskGraphScheduler *
lookupTaskGraphScheduler(const TaskGraphSchedulerRegistry &registry,
                         StringRef name) {
  auto it = registry.find(name);
  return it == registry.end() ? nullptr : it->second.get();
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
