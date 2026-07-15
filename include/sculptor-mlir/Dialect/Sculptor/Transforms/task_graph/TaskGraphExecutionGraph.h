#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHEXECUTIONGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHEXECUTIONGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_graph {

struct TaskExecutionEdge {
  unsigned producerTask = 0;
  unsigned consumerTask = 0;
  bool controlDependency = false;
  bool dataDependency = false;
  int64_t transferredBytes = 0;
};

struct TaskExecutionGraph {
  llvm::SmallVector<TaskExecutionEdge, 16> edges;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> incomingEdges;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> outgoingEdges;
  llvm::SmallVector<unsigned, 16> topologicalOrder;
  unsigned controlEdgeCount = 0;
  unsigned dataEdgeCount = 0;
  int64_t totalDataBytes = 0;
};

FailureOr<TaskExecutionGraph>
buildTaskExecutionGraph(func::FuncOp taskGraphFunc, const TaskGraphDAG &dag);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHEXECUTIONGRAPH_H
