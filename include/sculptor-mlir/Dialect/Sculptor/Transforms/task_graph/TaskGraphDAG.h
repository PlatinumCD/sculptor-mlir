#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDAG_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDAG_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sculptor {
namespace task_graph {

struct TaskGraphNode {
  sculptor::TaskCreateOp op;
  unsigned index = 0;
  llvm::SmallVector<unsigned, 4> predecessors;
  llvm::SmallVector<unsigned, 4> successors;
};

struct TaskGraphDAG {
  llvm::SmallVector<TaskGraphNode, 16> nodes;
  llvm::SmallVector<Value, 8> logicalArrayResources;
  llvm::DenseMap<Value, unsigned> nodeIndexByTaskResult;
  unsigned dependencyCount = 0;
};

FailureOr<TaskGraphDAG> parseTaskGraphDAG(func::FuncOp taskGraphFunc);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHDAG_H
