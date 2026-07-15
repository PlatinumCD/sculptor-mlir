#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHTASKKINDS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHTASKKINDS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace task_graph {

inline bool hasTaskKind(sculptor::TaskCreateOp taskOp,
                        llvm::StringRef taskKind) {
  return taskOp.getTaskKind() == taskKind;
}

inline bool isDigitalTask(sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() == task_graph_names::kDigitalDomain;
}

inline bool isMatrixSetupTask(sculptor::TaskCreateOp taskOp) {
  return hasTaskKind(taskOp, task_graph_names::kMatrixSetupTaskKind);
}

inline bool isAnalogComputeTask(sculptor::TaskCreateOp taskOp) {
  return hasTaskKind(taskOp, task_graph_names::kMVMTaskKind) ||
         hasTaskKind(taskOp, task_graph_names::kConvTileMVMTaskKind);
}

inline bool sameNonEmptySourceLayer(sculptor::TaskCreateOp lhs,
                                    sculptor::TaskCreateOp rhs) {
  llvm::StringRef lhsLayer = lhs.getSourceLayer();
  return !lhsLayer.empty() && lhsLayer == rhs.getSourceLayer();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHTASKKINDS_H
