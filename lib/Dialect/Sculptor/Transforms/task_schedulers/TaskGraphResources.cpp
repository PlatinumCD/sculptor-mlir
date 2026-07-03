#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTaskKinds.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace sculptor {
namespace task_schedulers {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;

int64_t getResourceByteSize(Value resource) {
  Operation *resourceOp = resource.getDefiningOp();
  if (!resourceOp)
    return 0;

  auto byteSizeAttr = resourceOp->getAttrOfType<IntegerAttr>(
      runtime_attrs::kResourceByteSizeAttrName);
  if (!byteSizeAttr)
    return 0;
  return byteSizeAttr.getInt();
}

LogicalResult
collectResourceProducers(const TaskGraphDAG &dag,
                         llvm::DenseMap<Value, unsigned> &producerByResource) {
  for (const TaskGraphNode &node : dag.nodes) {
    sculptor::TaskCreateOp taskOp = node.op;
    for (Value output : taskOp.getOutputs()) {
      if (!producerByResource.try_emplace(output, node.index).second) {
        taskOp.emitError("expected task graph resource to have one producer");
        return failure();
      }
    }
  }

  return success();
}

LogicalResult collectResourceProducers(
    const TaskGraphDAG &dag,
    llvm::DenseMap<Value, const TaskGraphNode *> &producerByResource) {
  for (const TaskGraphNode &node : dag.nodes) {
    sculptor::TaskCreateOp taskOp = node.op;
    for (Value output : taskOp.getOutputs()) {
      if (!producerByResource.try_emplace(output, &node).second) {
        taskOp.emitError("expected task graph resource to have one producer");
        return failure();
      }
    }
  }

  return success();
}

FailureOr<llvm::SmallVector<ResourceEdge, 16>>
collectResourceEdges(const TaskGraphDAG &dag) {
  llvm::DenseMap<Value, unsigned> producerByResource;
  if (failed(collectResourceProducers(dag, producerByResource)))
    return failure();

  llvm::SmallVector<ResourceEdge, 16> edges;
  for (const TaskGraphNode &consumer : dag.nodes) {
    sculptor::TaskCreateOp consumerTask = consumer.op;
    for (Value input : consumerTask.getInputs()) {
      auto producerIt = producerByResource.find(input);
      if (producerIt == producerByResource.end())
        continue;

      if (producerIt->second == consumer.index)
        continue;

      ResourceEdge edge;
      edge.producerIndex = producerIt->second;
      edge.consumerIndex = consumer.index;
      edge.byteSize = getResourceByteSize(input);
      edges.push_back(edge);
    }
  }

  return edges;
}

llvm::SmallVector<const TaskGraphNode *, 8>
collectMatrixSetupTasks(const TaskGraphDAG &dag) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks;
  for (const TaskGraphNode &node : dag.nodes) {
    if (isMatrixSetupTask(node.op))
      matrixSetupTasks.push_back(&node);
  }
  return matrixSetupTasks;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
