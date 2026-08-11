#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGCOSTMODEL_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGCOSTMODEL_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace mapping {

struct TaskCostFeatures {
  int64_t operationId = -1;
  int64_t workUnitId = -1;
  StringRef semanticTaskKind;
  int64_t workItems = 0;
  int64_t inputBytes = 0;
  int64_t outputBytes = 0;
};

struct TaskCostEstimate {
  double computeNs = 0.0;
  double memoryNs = 0.0;
  double runtimeNs = 0.0;
  double totalNs = 0.0;
};

FailureOr<TaskCostEstimate>
estimateDigitalTaskCost(const MappingCostProfile &profile,
                        const TaskCostFeatures &features, Operation *anchor);

FailureOr<TaskCostEstimate>
estimateAnalogTaskCost(const MappingCostProfile &profile, int64_t loadBytes,
                       int64_t storeBytes, int64_t executionCount,
                       Operation *anchor);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif
