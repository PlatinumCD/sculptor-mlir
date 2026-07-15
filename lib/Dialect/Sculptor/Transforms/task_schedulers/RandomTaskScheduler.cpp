#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPhysicalArrayOrders.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <memory>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

class RandomTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "random"; }

  mlir::FailureOr<task_schedulers::IslandPlacementPlan> buildPlacementPlan(
      const task_schedulers::TaskGraphPlacementProblem &problem,
      const task_schedulers::TaskGraphSchedulerOptions &options) const final {
    const auto *random =
        std::get_if<task_schedulers::RandomSchedulerOptions>(&options);
    if (!random) {
      problem.diagnosticOp->emitError(
          "random scheduler received incompatible scheduler options");
      return mlir::failure();
    }
    if (problem.budget.analogArrays.empty()) {
      problem.diagnosticOp->emitError(
          "expected random task scheduler to have at least one analog array");
      return mlir::failure();
    }

    llvm::SmallVector<int64_t, 8> shuffledAnalogArrays =
        task_schedulers::buildRandomPhysicalArrayOrder(problem.budget,
                                                       random->randomSeed);

    return task_schedulers::buildPlacementPlanFromPhysicalArrayOrder(
        problem, shuffledAnalogArrays);
  }
};

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

void registerRandomTaskScheduler(TaskGraphSchedulerRegistry &registry) {
  (void)registerTaskGraphScheduler(registry,
                                   std::make_unique<RandomTaskScheduler>());
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
