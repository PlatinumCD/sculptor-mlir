#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

namespace {

static bool isLogicalArrayResource(mlir::Value value) {
  auto resourceType =
      mlir::dyn_cast<mlir::sculptor::TaskResourceType>(value.getType());
  return resourceType && mlir::isa<mlir::sculptor::LogicalArrayType>(
                             resourceType.getValueType());
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_graph {

FailureOr<TaskGraphDAG> parseTaskGraphDAG(func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected scheduled task graph function to have "
                            "one block");
    return failure();
  }

  TaskGraphDAG dag;
  Block &block = taskGraphFunc.getBody().front();
  for (Operation &op : block) {
    for (Value result : op.getResults()) {
      if (isLogicalArrayResource(result))
        dag.logicalArrayResources.push_back(result);
    }

    auto taskOp = dyn_cast<sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    TaskGraphNode node;
    node.op = taskOp;
    node.index = dag.nodes.size();
    dag.nodeIndexByTaskResult.try_emplace(taskOp.getResult(), node.index);
    dag.nodes.push_back(std::move(node));
  }

  for (TaskGraphNode &node : dag.nodes) {
    for (Value dependency : node.op.getDependencies()) {
      auto predecessorIt = dag.nodeIndexByTaskResult.find(dependency);
      if (predecessorIt == dag.nodeIndexByTaskResult.end()) {
        node.op.emitError("expected task dependency to reference an "
                          "sculptor.task.create result in the same task graph");
        return failure();
      }

      unsigned predecessorIndex = predecessorIt->second;
      if (predecessorIndex >= node.index) {
        node.op.emitError("expected task dependency to reference an earlier "
                          "task in the task graph");
        return failure();
      }

      node.predecessors.push_back(predecessorIndex);
      dag.nodes[predecessorIndex].successors.push_back(node.index);
      ++dag.dependencyCount;
    }
  }
  return dag;
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
