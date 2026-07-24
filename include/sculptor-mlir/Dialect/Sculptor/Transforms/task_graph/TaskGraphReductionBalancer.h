#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHREDUCTIONBALANCER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHREDUCTIONBALANCER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_graph {

LogicalResult balanceTaskGraphReductions(ModuleOp module,
                                         func::FuncOp taskGraphFunc,
                                         int64_t reductionWidth);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHREDUCTIONBALANCER_H
