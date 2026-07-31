#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHMATMULDISTRIBUTOR_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHMATMULDISTRIBUTOR_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace sculptor {
namespace task_graph {

struct DigitalMatmulDistributionOptions {
  int64_t maxShards = 8;
  int64_t minOpsPerShard = 65536;
  std::optional<MatmulDistributionStrategy> strategy;
  DistributionPlacementPolicy placement =
      DistributionPlacementPolicy::PreferDistinct;
};

FailureOr<unsigned>
distributeDigitalMatmuls(ModuleOp module, func::FuncOp taskGraphFunc,
                         const DigitalMatmulDistributionOptions &options);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHMATMULDISTRIBUTOR_H
