#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedySearch.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <memory>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

class GreedyTimingTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "greedy-timing"; }
  bool requiresTimingProfile() const final { return true; }

  mlir::FailureOr<task_schedulers::IslandPlacementPlan>
  buildTimingPlacementPlan(
      const task_schedulers::TaskGraphPlacementProblem &problem,
      const mlir::sculptor::task_timing::SchedulingTimingProfile &timingProfile,
      const task_schedulers::TaskGraphSchedulerOptions &options) const final {
    const auto *greedy =
        std::get_if<task_schedulers::GreedySchedulerOptions>(&options);
    if (!greedy) {
      problem.diagnosticOp->emitError(
          "greedy-timing scheduler received incompatible scheduler options");
      return mlir::failure();
    }
    if (problem.budget.analogArrays.empty()) {
      problem.diagnosticOp->emitError(
          "expected greedy-timing task scheduler to have at least one analog "
          "array");
      return mlir::failure();
    }

    return task_schedulers::buildGreedyTimingPlacementPlan(
        problem, timingProfile, problem.budget.analogArrays, greedy->greedy);
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerGreedyTimingTaskScheduler(TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(
      registry, std::make_unique<GreedyTimingTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
