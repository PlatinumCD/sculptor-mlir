#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDIGITALOPS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDIGITALOPS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_graph {

using ResourceProducerMap = llvm::DenseMap<Value, const TaskGraphNode *>;

struct DigitalMatmulGeometry {
  sculptor::TaskCreateOp matrixSetupTask;
  func::FuncOp matrixSetupCallee;
  func::FuncOp mvmCallee;
  unsigned logicalArrayInputIndex = 0;
  unsigned tensorInputIndex = 0;
  RankedTensorType inputType;
  RankedTensorType resultType;
  RankedTensorType weightType;
  int64_t executionRows = 0;
  int64_t physicalRows = 0;
  int64_t physicalColumns = 0;
  bool needsVectorTileExtraction = false;
  int64_t vectorTile = 0;
  int64_t validColumns = 0;
};

FailureOr<int64_t> estimateTaskDigitalOps(ModuleOp module,
                                          sculptor::TaskCreateOp taskOp);

FailureOr<DigitalMatmulGeometry>
resolveDigitalMatmulGeometry(ModuleOp module, sculptor::TaskCreateOp taskOp,
                             const ResourceProducerMap &producerByResource);

FailureOr<int64_t> computeDigitalMatmulScalarOps(Operation *anchor,
                                                 int64_t executionRows,
                                                 int64_t physicalRows,
                                                 int64_t physicalColumns);

FailureOr<int64_t>
estimateDigitalReplacementOps(ModuleOp module, sculptor::TaskCreateOp taskOp,
                              const ResourceProducerMap &producerByResource);

FailureOr<int64_t> estimateDigitalReplacementOps(ModuleOp module,
                                                 sculptor::TaskCreateOp taskOp);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDIGITALOPS_H
