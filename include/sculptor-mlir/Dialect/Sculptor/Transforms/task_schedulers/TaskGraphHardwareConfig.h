#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHHARDWARECONFIG_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHHARDWARECONFIG_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<HardwareBudget>
buildHardwareBudget(ModuleOp module, int64_t numCores,
                    int64_t arraysPerCore, llvm::StringRef topology,
                    int64_t meshRows, int64_t meshCols);

void attachHardwareBudgetAttrs(Operation *op, Builder &builder,
                               const HardwareBudget &budget);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHHARDWARECONFIG_H
