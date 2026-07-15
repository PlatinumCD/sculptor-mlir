#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPhysicalArrayOrders.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <memory>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

class SnakeTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "snake"; }

  mlir::FailureOr<task_schedulers::IslandPlacementPlan> buildPlacementPlan(
      const task_schedulers::TaskGraphPlacementProblem &problem,
      const task_schedulers::TaskGraphSchedulerOptions &options) const final {
    if (!std::holds_alternative<task_schedulers::SnakeSchedulerOptions>(
            options)) {
      problem.diagnosticOp->emitError(
          "snake scheduler received incompatible scheduler options");
      return mlir::failure();
    }
    llvm::SmallVector<int64_t, 8> physicalArrayOrder =
        task_schedulers::buildSnakePhysicalArrayOrder(problem.budget);

    return task_schedulers::buildPlacementPlanFromPhysicalArrayOrder(
        problem, physicalArrayOrder);
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerSnakeTaskScheduler(TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(registry,
                                   std::make_unique<SnakeTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
