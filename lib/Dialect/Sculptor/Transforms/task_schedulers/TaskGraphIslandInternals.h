#ifndef SCULPTOR_MLIR_LIB_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHISLANDINTERNALS_H
#define SCULPTOR_MLIR_LIB_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHISLANDINTERNALS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sculptor {
namespace task_schedulers {

LogicalResult assignPrePlacementMinCutDigitalIslands(
    const TaskGraphDAG &dag,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex);

LogicalResult assignRemainingDigitalIslandsByLocalAffinity(
    const TaskGraphDAG &dag, llvm::ArrayRef<ResourceEdge> resourceEdges,
    llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex);

llvm::SmallVector<LogicalIslandCommunicationEdge, 16>
buildIslandCommunicationEdges(
    const TaskGraphDAG &dag, llvm::ArrayRef<ResourceEdge> resourceEdges,
    const llvm::DenseMap<unsigned, unsigned> &islandByTaskIndex);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_LIB_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHISLANDINTERNALS_H
