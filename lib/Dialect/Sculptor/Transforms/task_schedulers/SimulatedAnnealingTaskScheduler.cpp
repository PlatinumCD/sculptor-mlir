#include "annealing/AnnealingSearch.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <memory>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;
namespace annealing = task_schedulers::annealing_detail;

class SimulatedAnnealingTaskScheduler final
    : public task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "annealing"; }

  mlir::FailureOr<task_schedulers::IslandPlacementPlan> buildPlacementPlan(
      const task_schedulers::TaskGraphPlacementProblem &problem,
      const task_schedulers::TaskGraphSchedulerOptions &options) const final {
    const auto *annealingOptions =
        std::get_if<task_schedulers::AnnealingSchedulerOptions>(&options);
    if (!annealingOptions) {
      problem.diagnosticOp->emitError(
          "annealing scheduler received incompatible scheduler options");
      return mlir::failure();
    }
    if (problem.budget.analogArrays.empty()) {
      problem.diagnosticOp->emitError(
          "expected simulated annealing task scheduler to have at least one "
          "analog array");
      return mlir::failure();
    }

    return annealing::buildPlacementPlan(
        problem, annealingOptions->annealing,
        annealingOptions->greedyInitialPlacement,
        annealingOptions->randomSeed);
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerSimulatedAnnealingTaskScheduler(
    TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(
      registry, std::make_unique<SimulatedAnnealingTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
