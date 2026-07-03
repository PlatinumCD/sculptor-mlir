#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHRESOURCES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHRESOURCES_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct ResourceEdge {
  unsigned producerIndex = 0;
  unsigned consumerIndex = 0;
  int64_t byteSize = 0;
};

int64_t getResourceByteSize(Value resource);

LogicalResult
collectResourceProducers(const TaskGraphDAG &dag,
                         llvm::DenseMap<Value, unsigned> &producerByResource);

LogicalResult collectResourceProducers(
    const TaskGraphDAG &dag,
    llvm::DenseMap<Value, const TaskGraphNode *> &producerByResource);

FailureOr<llvm::SmallVector<ResourceEdge, 16>>
collectResourceEdges(const TaskGraphDAG &dag);

llvm::SmallVector<const TaskGraphNode *, 8>
collectMatrixSetupTasks(const TaskGraphDAG &dag);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHRESOURCES_H
