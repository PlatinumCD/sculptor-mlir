#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHOPTIMIZER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHOPTIMIZER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace task_graph {

enum class TaskGraphOptimizationStage {
  PreSchedule,
  PostSchedule,
};

struct TaskGraphOptimizationPattern {
  llvm::StringRef name;
  TaskGraphOptimizationStage stage;
  LogicalResult (*apply)(ModuleOp module, func::FuncOp taskGraphFunc,
                         const TaskGraphDAG &dag, bool &changed);
};

llvm::ArrayRef<TaskGraphOptimizationPattern> getTaskGraphOptimizationPatterns();

LogicalResult optimizeTaskGraph(
    ModuleOp module, func::FuncOp taskGraphFunc, const TaskGraphDAG &dag,
    llvm::ArrayRef<llvm::StringRef> selectedPatterns, bool &changed);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHOPTIMIZER_H
