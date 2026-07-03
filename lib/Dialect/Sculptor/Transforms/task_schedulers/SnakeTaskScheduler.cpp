#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include <cstdint>
#include <memory>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

static void appendCorePhysicalArrays(
    llvm::SmallVectorImpl<int64_t> &physicalArrayOrder,
    const task_schedulers::HardwareBudget &budget, int64_t coreId) {
  for (int64_t localArrayId = 0; localArrayId < budget.arraysPerCore;
       ++localArrayId)
    physicalArrayOrder.push_back(coreId * budget.arraysPerCore + localArrayId);
}

static llvm::SmallVector<int64_t, 8>
buildSnakePhysicalArrayOrder(const task_schedulers::HardwareBudget &budget) {
  llvm::SmallVector<int64_t, 8> physicalArrayOrder;
  physicalArrayOrder.reserve(static_cast<size_t>(budget.numAnalogArrays));

  for (int64_t row = 0; row < budget.meshRows; ++row) {
    if (row % 2 == 0) {
      for (int64_t col = 0; col < budget.meshCols; ++col)
        appendCorePhysicalArrays(physicalArrayOrder, budget,
                                 row * budget.meshCols + col);
      continue;
    }

    for (int64_t col = budget.meshCols - 1; col >= 0; --col)
      appendCorePhysicalArrays(physicalArrayOrder, budget,
                               row * budget.meshCols + col);
  }

  return physicalArrayOrder;
}

class SnakeTaskScheduler final
    : public mlir::sculptor::task_schedulers::TaskGraphScheduler {
public:
  mlir::StringRef getName() const final { return "snake"; }

  mlir::LogicalResult schedule(
      mlir::ModuleOp module, mlir::func::FuncOp taskGraphFunc,
      const mlir::sculptor::task_schedulers::HardwareBudget &budget,
      const mlir::sculptor::task_schedulers::TaskGraphDAG &dag) const final {
    llvm::SmallVector<int64_t, 8> physicalArrayOrder =
        buildSnakePhysicalArrayOrder(budget);

    return task_schedulers::placeLogicalPlacementIslands(
        module, taskGraphFunc, budget, dag, physicalArrayOrder);
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
