#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_SEPARABLEREGIONOPTIMIZER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_SEPARABLEREGIONOPTIMIZER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sculptor {
namespace task_graph {

LogicalResult optimizeSeparableRegions(ModuleOp module,
                                       func::FuncOp taskGraphFunc,
                                       const TaskGraphDAG &dag, bool &changed);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_SEPARABLEREGIONOPTIMIZER_H
