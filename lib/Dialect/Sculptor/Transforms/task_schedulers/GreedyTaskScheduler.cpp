#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedyPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <memory>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

class GreedyTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "greedy"; }

  mlir::LogicalResult schedule(
      mlir::ModuleOp module, mlir::func::FuncOp taskGraphFunc,
      const mlir::sculptor::task_schedulers::HardwareBudget &budget,
      const mlir::sculptor::task_schedulers::TaskGraphDAG &dag) const final {
    if (budget.analogArrays.empty()) {
      taskGraphFunc.emitError("expected greedy task scheduler to have at "
                              "least one analog array");
      return mlir::failure();
    }

    // Build logical islands around matrix setups, then place them with greedy
    // same-core/neighbor growth over the mesh.
    return task_schedulers::runGreedyIslandPlacement(
        module, taskGraphFunc, budget, dag, budget.analogArrays);
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
