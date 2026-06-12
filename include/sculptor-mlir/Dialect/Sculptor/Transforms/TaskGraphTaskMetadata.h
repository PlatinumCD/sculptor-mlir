#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKMETADATA_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKMETADATA_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sculptor {
namespace task_metadata {

struct TaskFunctionMetadata {
  mlir::StringAttr domain;
  mlir::StringAttr taskKind;
  mlir::StringAttr taskName;
  mlir::StringAttr sourceLayer;
  mlir::IntegerAttr sourceTaskOrdinal;
};

void setTaskFunctionMetadata(mlir::func::FuncOp func,
                             const TaskFunctionMetadata &metadata);

bool hasTaskFunctionMetadata(mlir::func::FuncOp func);

mlir::FailureOr<TaskFunctionMetadata>
getTaskFunctionMetadata(mlir::func::CallOp call, mlir::func::FuncOp callee);

} // namespace task_metadata
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKMETADATA_H
