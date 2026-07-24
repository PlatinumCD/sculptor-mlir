#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::sculptor;

namespace {

// Keeps the accepted task execution domains centralized for verifier checks.
bool isValidTaskDomain(StringRef domain) {
  return domain == "analog" || domain == "digital";
}

// Checks whether a resource handle comes from the expected op in this graph.
template <typename ResourceOpT>
bool belongsToSameGraph(TaskCreateOp taskOp, Value resourceValue) {
  auto resourceOp = resourceValue.getDefiningOp<ResourceOpT>();
  return resourceOp && resourceOp.getGraph() == taskOp.getGraph();
}

// Accepts only graph-owned resources that can be consumed or produced by tasks.
bool isValidTaskResource(TaskCreateOp taskOp, Value resourceValue) {
  return belongsToSameGraph<TaskGraphInputOp>(taskOp, resourceValue) ||
         belongsToSameGraph<TaskGraphOutputOp>(taskOp, resourceValue) ||
         belongsToSameGraph<TaskGraphIntermediateOp>(taskOp, resourceValue) ||
         belongsToSameGraph<TaskGraphPersistentOp>(taskOp, resourceValue);
}

LogicalResult verifyReductionMetadata(TaskCreateOp taskOp) {
  Attribute rawReduction =
      taskOp->getAttr(task_graph_attrs::kTaskReductionAttrName);
  if (!rawReduction)
    return success();

  auto reduction = dyn_cast<TaskReductionAttr>(rawReduction);
  if (!reduction) {
    return taskOp.emitOpError("expected '")
           << task_graph_attrs::kTaskReductionAttrName
           << "' to be a #sculptor.task_reduction attribute";
  }
  if (taskOp.getDomain() != "digital") {
    return taskOp.emitOpError(
        "expected a marked task reduction to use the digital domain");
  }
  if (!reduction.getReassociate().getValue()) {
    return taskOp.emitOpError(
        "expected task reduction metadata to permit reassociation");
  }
  if (taskOp.getInputs().size() < 2 || taskOp.getOutputs().size() != 1) {
    return taskOp.emitOpError(
        "expected a marked task reduction to have at least two inputs and "
        "exactly one output");
  }

  auto outputResource =
      dyn_cast<TaskResourceType>(taskOp.getOutputs().front().getType());
  auto outputType =
      outputResource ? dyn_cast<RankedTensorType>(outputResource.getValueType())
                     : RankedTensorType{};
  if (!outputType || !outputType.hasStaticShape() ||
      outputType.getRank() == 0 ||
      !isa<FloatType>(outputType.getElementType())) {
    return taskOp.emitOpError(
        "expected a marked task reduction to use a non-scalar, statically "
        "shaped floating-point tensor resource");
  }

  for (Value input : taskOp.getInputs()) {
    auto inputResource = dyn_cast<TaskResourceType>(input.getType());
    if (!inputResource || inputResource.getValueType() != outputType) {
      return taskOp.emitOpError(
          "expected every task reduction input to match the output tensor "
          "type");
    }
  }
  return success();
}

// Verifies one resource segment while sharing the diagnostic wording.
template <typename ValuesT>
LogicalResult verifyTaskResources(TaskCreateOp taskOp, ValuesT resources,
                                  StringRef name) {
  for (Value resource : resources) {
    if (!isValidTaskResource(taskOp, resource)) {
      return taskOp.emitOpError("expected ")
             << name << " to be produced by sculptor.task_graph.input, "
             << "sculptor.task_graph.output, or "
             << "sculptor.task_graph.intermediate, or "
             << "sculptor.task_graph.persistent in the same graph";
    }
  }

  return success();
}

} // namespace

// Enforces that task nodes reference only well-formed graph-local edges.
mlir::LogicalResult mlir::sculptor::TaskCreateOp::verify() {
  // Validate the task's execution placement before checking graph topology.
  if (!isValidTaskDomain(getDomain())) {
    return emitOpError(
        "expected domain to be either \"analog\" or \"digital\"");
  }

  // Dependencies are task results, and must not cross graph boundaries.
  for (Value dependency : getDependencies()) {
    auto dependencyTask = dependency.getDefiningOp<TaskCreateOp>();
    if (!dependencyTask) {
      return emitOpError(
          "expected dependencies to be produced by sculptor.task.create");
    }

    if (dependencyTask.getGraph() != getGraph()) {
      return emitOpError(
          "expected dependencies to belong to the same task graph");
    }
  }

  // Inputs and outputs must be resource handles owned by this task graph.
  if (failed(verifyTaskResources(*this, getInputs(), "inputs")))
    return failure();

  if (failed(verifyTaskResources(*this, getOutputs(), "outputs")))
    return failure();

  if (failed(verifyReductionMetadata(*this)))
    return failure();

  return mlir::success();
}
