#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskMetadata.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace task_metadata {
namespace {

mlir::FailureOr<mlir::StringAttr>
getRequiredStringTaskAttr(mlir::func::CallOp call, mlir::func::FuncOp callee,
                          llvm::StringRef attrName) {
  auto attr = callee->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr || attr.getValue().empty()) {
    call.emitError("expected materialized task callee '")
        << callee.getSymName() << "' to carry non-empty string attr '"
        << attrName << "'";
    return mlir::failure();
  }

  return attr;
}

} // namespace

void setTaskFunctionMetadata(mlir::func::FuncOp func,
                             const TaskFunctionMetadata &metadata) {
  func->setAttr(task_attrs::kTaskDomainAttrName, metadata.domain);
  func->setAttr(task_attrs::kTaskKindAttrName, metadata.taskKind);
  func->setAttr(task_attrs::kTaskNameAttrName, metadata.taskName);
  func->setAttr(task_attrs::kSourceLayerAttrName, metadata.sourceLayer);
  func->setAttr(task_attrs::kSourceTaskOrdinalAttrName,
                metadata.sourceTaskOrdinal);
}

bool hasTaskFunctionMetadata(mlir::func::FuncOp func) {
  return func && static_cast<bool>(func->getAttrOfType<mlir::StringAttr>(
                     task_attrs::kTaskKindAttrName));
}

mlir::FailureOr<TaskFunctionMetadata>
getTaskFunctionMetadata(mlir::func::CallOp call, mlir::func::FuncOp callee) {
  auto domain =
      getRequiredStringTaskAttr(call, callee, task_attrs::kTaskDomainAttrName);
  auto taskKind =
      getRequiredStringTaskAttr(call, callee, task_attrs::kTaskKindAttrName);
  auto taskName =
      getRequiredStringTaskAttr(call, callee, task_attrs::kTaskNameAttrName);
  auto sourceLayer =
      getRequiredStringTaskAttr(call, callee, task_attrs::kSourceLayerAttrName);
  auto sourceTaskOrdinal = callee->getAttrOfType<mlir::IntegerAttr>(
      task_attrs::kSourceTaskOrdinalAttrName);
  if (!sourceTaskOrdinal) {
    call.emitError("expected materialized task callee '")
        << callee.getSymName() << "' to carry integer attr "
        << "'" << task_attrs::kSourceTaskOrdinalAttrName << "'";
    return mlir::failure();
  }

  if (mlir::failed(domain) || mlir::failed(taskKind) ||
      mlir::failed(taskName) || mlir::failed(sourceLayer))
    return mlir::failure();

  return TaskFunctionMetadata{*domain, *taskKind, *taskName, *sourceLayer,
                              sourceTaskOrdinal};
}

} // namespace task_metadata
} // namespace sculptor
} // namespace mlir
