#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_GOLEMMVMPLANNING_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_GOLEMMVMPLANNING_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace mapping {

struct GolemMVMTile {
  int64_t row = 0;
  int64_t column = 0;
  int64_t physicalRows = 0;
  int64_t physicalColumns = 0;
  int64_t validRows = 0;
  int64_t validColumns = 0;
};

// Describes the physical Golem resources required by one logical MVM without
// creating arrays, constants, IR operations, or placement decisions.
struct GolemMVMPlan {
  int64_t logicalRows = 0;
  int64_t logicalColumns = 0;
  int64_t arrayRows = 0;
  int64_t arrayColumns = 0;
  int64_t gridRows = 0;
  int64_t gridColumns = 0;
  int64_t arrayCount = 0;
  int64_t paddedMatrixBytes = 0;
  int64_t physicalLoadElements = 0;
  int64_t physicalStoreElements = 0;
  int64_t executionCount = 0;
  int64_t recombinationAddOps = 0;
  SmallVector<GolemMVMTile, 4> tiles;
};

FailureOr<GolemMVMPlan> planGolemMVM(Operation *anchor, int64_t logicalRows,
                                     int64_t logicalColumns, int64_t arrayRows,
                                     int64_t arrayColumns);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_GOLEMMVMPLANNING_H
