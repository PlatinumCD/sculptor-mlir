#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/GolemMVMPlanning.h"

#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <optional>

namespace {

using namespace mlir;
using namespace mlir::sculptor::mapping;

FailureOr<int64_t> checkedMultiply(Operation *anchor, int64_t lhs, int64_t rhs,
                                   StringRef description) {
  std::optional<int64_t> result = llvm::checkedMul(lhs, rhs);
  if (!result) {
    anchor->emitError(description) << " overflows a signed 64-bit integer";
    return failure();
  }
  return *result;
}

int64_t divideCeilPositive(int64_t value, int64_t divisor) {
  return value / divisor + (value % divisor != 0);
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<GolemMVMPlan> planGolemMVM(Operation *anchor, int64_t logicalRows,
                                     int64_t logicalColumns, int64_t arrayRows,
                                     int64_t arrayColumns) {
  if (!anchor)
    return failure();
  if (logicalRows <= 0 || logicalColumns <= 0) {
    anchor->emitError("expected positive logical MVM matrix dimensions");
    return failure();
  }
  if (arrayRows <= 0 || arrayColumns <= 0) {
    anchor->emitError("expected positive physical Golem array dimensions");
    return failure();
  }

  GolemMVMPlan plan;
  plan.logicalRows = logicalRows;
  plan.logicalColumns = logicalColumns;
  plan.arrayRows = arrayRows;
  plan.arrayColumns = arrayColumns;
  plan.gridRows = divideCeilPositive(logicalRows, arrayRows);
  plan.gridColumns = divideCeilPositive(logicalColumns, arrayColumns);

  FailureOr<int64_t> arrayCount = checkedMultiply(
      anchor, plan.gridRows, plan.gridColumns, "Golem MVM array count");
  if (failed(arrayCount))
    return failure();
  plan.arrayCount = *arrayCount;
  plan.executionCount = *arrayCount;

  FailureOr<int64_t> physicalElementsPerArray = checkedMultiply(
      anchor, arrayRows, arrayColumns, "Golem MVM array element count");
  if (failed(physicalElementsPerArray))
    return failure();
  FailureOr<int64_t> physicalElements =
      checkedMultiply(anchor, *arrayCount, *physicalElementsPerArray,
                      "Golem MVM padded matrix element count");
  if (failed(physicalElements))
    return failure();
  FailureOr<int64_t> paddedBytes = checkedMultiply(
      anchor, *physicalElements, int64_t{4}, "Golem MVM padded matrix bytes");
  if (failed(paddedBytes))
    return failure();
  plan.paddedMatrixBytes = *paddedBytes;

  FailureOr<int64_t> loadElements = checkedMultiply(
      anchor, *arrayCount, arrayColumns, "Golem MVM physical load elements");
  FailureOr<int64_t> storeElements = checkedMultiply(
      anchor, *arrayCount, arrayRows, "Golem MVM physical store elements");
  FailureOr<int64_t> recombinationAdds =
      checkedMultiply(anchor, logicalRows, plan.gridColumns - 1,
                      "Golem MVM recombination operation count");
  if (failed(loadElements) || failed(storeElements) ||
      failed(recombinationAdds))
    return failure();
  plan.physicalLoadElements = *loadElements;
  plan.physicalStoreElements = *storeElements;
  plan.recombinationAddOps = *recombinationAdds;

  plan.tiles.reserve(plan.arrayCount);
  for (int64_t row = 0; row < plan.gridRows; ++row) {
    for (int64_t column = 0; column < plan.gridColumns; ++column) {
      int64_t rowOffset = row * arrayRows;
      int64_t columnOffset = column * arrayColumns;
      plan.tiles.push_back(
          {row, column, arrayRows, arrayColumns,
           std::min(arrayRows, logicalRows - rowOffset),
           std::min(arrayColumns, logicalColumns - columnOffset)});
    }
  }
  return plan;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
