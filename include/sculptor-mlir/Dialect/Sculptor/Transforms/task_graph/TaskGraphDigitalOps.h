#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDIGITALOPS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDIGITALOPS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_graph {

FailureOr<int64_t> estimateTaskDigitalOps(ModuleOp module,
                                          sculptor::TaskCreateOp taskOp);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDIGITALOPS_H
