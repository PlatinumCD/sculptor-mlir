#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHDIGITALSHARDPLACEMENT_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHDIGITALSHARDPLACEMENT_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct TaskGraphPlacementProblem;

FailureOr<llvm::DenseMap<unsigned, int64_t>> buildDigitalShardCorePlacements(
    const TaskGraphPlacementProblem &problem,
    const llvm::DenseMap<unsigned, int64_t> &initialCoreByIsland);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHDIGITALSHARDPLACEMENT_H
