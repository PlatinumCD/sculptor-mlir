#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHCLEANUP_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHCLEANUP_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sculptor {
namespace task_graph {

LogicalResult
eraseUnusedTaskGraphIntermediateResources(func::FuncOp taskGraphFunc);

void eraseUnusedTaskCallees(ModuleOp module);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHCLEANUP_H
