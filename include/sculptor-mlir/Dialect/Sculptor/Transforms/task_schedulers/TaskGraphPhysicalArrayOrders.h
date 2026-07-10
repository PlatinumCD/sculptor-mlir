#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPHYSICALARRAYORDERS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPHYSICALARRAYORDERS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

llvm::SmallVector<int64_t, 8>
buildIdentityPhysicalArrayOrder(const HardwareBudget &budget);

llvm::SmallVector<int64_t, 8>
buildRandomPhysicalArrayOrder(const HardwareBudget &budget);

llvm::SmallVector<int64_t, 8>
buildSnakePhysicalArrayOrder(const HardwareBudget &budget);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPHYSICALARRAYORDERS_H
