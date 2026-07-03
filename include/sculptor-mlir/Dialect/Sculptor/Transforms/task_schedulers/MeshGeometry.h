#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_MESHGEOMETRY_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_MESHGEOMETRY_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

enum MeshBoundaryMask : unsigned {
  kMeshTopBoundary = 1u << 0,
  kMeshBottomBoundary = 1u << 1,
  kMeshLeftBoundary = 1u << 2,
  kMeshRightBoundary = 1u << 3,
};

int64_t getMeshRow(int64_t coreId, const HardwareBudget &budget);
int64_t getMeshCol(int64_t coreId, const HardwareBudget &budget);

int64_t getMeshDistance(int64_t sourceCore, int64_t destinationCore,
                        const HardwareBudget &budget);

unsigned getMeshBoundaryMask(int64_t coreId, const HardwareBudget &budget);

int64_t getBoundaryPenalty(int64_t totalTransferCost);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_MESHGEOMETRY_H
