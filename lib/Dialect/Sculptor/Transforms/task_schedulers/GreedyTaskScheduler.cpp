#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedySearch.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <memory>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

class GreedyTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "greedy"; }

  mlir::FailureOr<task_schedulers::IslandPlacementPlan> buildPlacementPlan(
      const task_schedulers::TaskGraphPlacementProblem &problem,
      const task_schedulers::TaskGraphSchedulerOptions &options) const final {
    const auto *greedy =
        std::get_if<task_schedulers::GreedySchedulerOptions>(&options);
    if (!greedy) {
      problem.diagnosticOp->emitError(
          "greedy scheduler received incompatible scheduler options");
      return mlir::failure();
    }
    if (problem.budget.analogArrays.empty()) {
      problem.diagnosticOp->emitError(
          "expected greedy task scheduler to have at least one analog array");
      return mlir::failure();
    }

    return task_schedulers::buildGreedyPlacementPlan(
        problem, problem.budget.analogArrays, greedy->greedy);
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerGreedyTaskScheduler(TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(registry,
                                   std::make_unique<GreedyTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
