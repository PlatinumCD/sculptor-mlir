#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

class RandomTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "random"; }

  mlir::LogicalResult schedule(
      mlir::ModuleOp module, mlir::func::FuncOp taskGraphFunc,
      const mlir::sculptor::task_schedulers::HardwareBudget &budget,
      const mlir::sculptor::task_schedulers::TaskGraphDAG &dag) const final {
    if (budget.analogArrays.empty()) {
      taskGraphFunc.emitError("expected random task scheduler to have at "
                              "least one analog array");
      return mlir::failure();
    }

    llvm::SmallVector<int64_t, 8> shuffledAnalogArrays = budget.analogArrays;
    std::mt19937 randomEngine(static_cast<uint32_t>(budget.randomSeed));
    std::shuffle(shuffledAnalogArrays.begin(), shuffledAnalogArrays.end(),
                 randomEngine);

    return task_schedulers::placeLogicalPlacementIslands(
        module, taskGraphFunc, budget, dag, shuffledAnalogArrays);
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
