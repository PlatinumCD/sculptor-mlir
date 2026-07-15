#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHRESOURCEUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHRESOURCEUTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace sculptor {

/// Returns the static byte size of the payload wrapped by a task resource.
/// Runtime handles and logical arrays carry no transferred payload bytes.
FailureOr<int64_t> getTaskResourceByteSize(Value resource);

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHRESOURCEUTILS_H
