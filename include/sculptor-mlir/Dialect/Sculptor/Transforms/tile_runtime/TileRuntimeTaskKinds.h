#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMETASKKINDS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMETASKKINDS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/SemanticOperationNames.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace tile_runtime {

inline bool hasTaskKind(sculptor::TaskCreateOp taskOp,
                        llvm::StringRef taskKind) {
  return taskOp.getTaskKind() == taskKind;
}

inline bool isDigitalTask(sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() == semantic_operation_names::kDigitalDomain;
}

inline bool isMatrixSetupTask(sculptor::TaskCreateOp taskOp) {
  return hasTaskKind(taskOp, semantic_operation_names::kMatrixSetupTaskKind);
}

inline bool isAnalogComputeTask(sculptor::TaskCreateOp taskOp) {
  return hasTaskKind(taskOp, semantic_operation_names::kMVMTaskKind) ||
         hasTaskKind(taskOp, semantic_operation_names::kConvTileMVMTaskKind);
}

inline bool isReductionTask(sculptor::TaskCreateOp taskOp) {
  return hasTaskKind(taskOp, semantic_operation_names::kReductionTaskKind) &&
         taskOp->hasAttr(task_graph_attrs::kTaskReductionTreeIdAttrName) &&
         taskOp->hasAttr(task_graph_attrs::kTaskReductionLevelAttrName) &&
         taskOp->hasAttr(task_graph_attrs::kTaskReductionWidthAttrName);
}

inline bool sameNonEmptySourceLayer(sculptor::TaskCreateOp lhs,
                                    sculptor::TaskCreateOp rhs) {
  llvm::StringRef lhsLayer = lhs.getSourceLayer();
  return !lhsLayer.empty() && lhsLayer == rhs.getSourceLayer();
}

} // namespace tile_runtime
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMETASKKINDS_H
