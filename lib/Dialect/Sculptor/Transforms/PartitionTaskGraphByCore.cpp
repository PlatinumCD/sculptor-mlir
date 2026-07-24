#include "sculptor-mlir/Dialect/Sculptor/Transforms/PartitionTaskGraphByCore.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDeploymentPartitioner.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sculptor {

void PartitionTaskGraphByCorePass::runOnOperation() {
  if (failed(task_graph::partitionTaskGraphByCore(getOperation())))
    signalPassFailure();
}

void registerPartitionTaskGraphByCorePass() {
  PassRegistration<PartitionTaskGraphByCorePass>();
}

} // namespace sculptor
} // namespace mlir
