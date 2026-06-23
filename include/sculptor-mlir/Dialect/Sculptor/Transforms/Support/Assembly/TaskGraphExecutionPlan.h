#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_ASSEMBLY_TASKGRAPHEXECUTIONPLAN_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_ASSEMBLY_TASKGRAPHEXECUTIONPLAN_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sculptor {

LogicalResult rebuildTaskGraphExecutionPlan(func::FuncOp taskGraphFunc);

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_ASSEMBLY_TASKGRAPHEXECUTIONPLAN_H
