#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"

namespace mlir {
namespace sculptor {
namespace task_schedulers {

int64_t getMeshRow(int64_t coreId, const HardwareBudget &budget) {
  return coreId / budget.meshCols;
}

int64_t getMeshCol(int64_t coreId, const HardwareBudget &budget) {
  return coreId % budget.meshCols;
}

int64_t getMeshDistance(int64_t sourceCore, int64_t destinationCore,
                        const HardwareBudget &budget) {
  int64_t sourceRow = getMeshRow(sourceCore, budget);
  int64_t sourceCol = getMeshCol(sourceCore, budget);
  int64_t destinationRow = getMeshRow(destinationCore, budget);
  int64_t destinationCol = getMeshCol(destinationCore, budget);
  int64_t rowDistance = sourceRow > destinationRow ? sourceRow - destinationRow
                                                   : destinationRow - sourceRow;
  int64_t colDistance = sourceCol > destinationCol ? sourceCol - destinationCol
                                                   : destinationCol - sourceCol;
  return rowDistance + colDistance;
}

unsigned getMeshBoundaryMask(int64_t coreId, const HardwareBudget &budget) {
  int64_t row = getMeshRow(coreId, budget);
  int64_t col = getMeshCol(coreId, budget);
  unsigned mask = 0;
  if (row == 0)
    mask |= kMeshTopBoundary;
  if (row == budget.meshRows - 1)
    mask |= kMeshBottomBoundary;
  if (col == 0)
    mask |= kMeshLeftBoundary;
  if (col == budget.meshCols - 1)
    mask |= kMeshRightBoundary;
  return mask;
}

int64_t getBoundaryPenalty(int64_t totalTransferCost) {
  if (totalTransferCost <= 0)
    return 0;
  return (totalTransferCost + 4) / 5;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
