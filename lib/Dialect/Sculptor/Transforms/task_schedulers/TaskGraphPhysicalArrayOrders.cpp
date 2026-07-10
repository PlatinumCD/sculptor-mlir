#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPhysicalArrayOrders.h"

#include <algorithm>
#include <random>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

static void
appendCorePhysicalArrays(llvm::SmallVectorImpl<int64_t> &physicalArrayOrder,
                         const task_schedulers::HardwareBudget &budget,
                         int64_t coreId) {
  for (int64_t localArrayId = 0; localArrayId < budget.arraysPerCore;
       ++localArrayId)
    physicalArrayOrder.push_back(coreId * budget.arraysPerCore + localArrayId);
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

llvm::SmallVector<int64_t, 8>
buildIdentityPhysicalArrayOrder(const HardwareBudget &budget) {
  return budget.analogArrays;
}

llvm::SmallVector<int64_t, 8>
buildRandomPhysicalArrayOrder(const HardwareBudget &budget) {
  llvm::SmallVector<int64_t, 8> physicalArrayOrder =
      buildIdentityPhysicalArrayOrder(budget);
  std::mt19937 randomEngine(static_cast<uint32_t>(budget.randomSeed));
  std::shuffle(physicalArrayOrder.begin(), physicalArrayOrder.end(),
               randomEngine);
  return physicalArrayOrder;
}

llvm::SmallVector<int64_t, 8>
buildSnakePhysicalArrayOrder(const HardwareBudget &budget) {
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

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
