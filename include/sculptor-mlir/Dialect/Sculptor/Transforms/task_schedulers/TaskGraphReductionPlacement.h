#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHREDUCTIONPLACEMENT_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHREDUCTIONPLACEMENT_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct TaskGraphPlacementProblem;

FailureOr<llvm::DenseMap<unsigned, int64_t>>
buildSpatialReductionCorePlacements(
    const TaskGraphPlacementProblem &problem,
    const llvm::DenseMap<unsigned, int64_t> &coreByAnalogIsland);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHREDUCTIONPLACEMENT_H
