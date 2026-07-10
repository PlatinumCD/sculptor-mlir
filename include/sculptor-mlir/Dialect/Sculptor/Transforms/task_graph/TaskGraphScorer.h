#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHSCORER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHSCORER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct TaskGraphScore {
  int64_t score = 0;
  int64_t boundaryPenalty = 0;
  int64_t interCoreTransferBytes = 0;
  int64_t totalTransferCost = 0;
  double transferCostPerInterCoreByte = 0.0;
  llvm::SmallVector<int64_t, 16> coreTransferBytes;
  llvm::SmallVector<int64_t, 16> coreTransferCost;
};

class TaskGraphScorer {
public:
  virtual ~TaskGraphScorer() = default;

  virtual StringRef getName() const = 0;

  virtual FailureOr<TaskGraphScore> score(ModuleOp module,
                                          func::FuncOp taskGraphFunc,
                                          const HardwareBudget &budget,
                                          const TaskGraphDAG &dag) const = 0;
};

class MeshTaskGraphScorer final : public TaskGraphScorer {
public:
  StringRef getName() const final { return "mesh-transfer"; }

  FailureOr<TaskGraphScore> score(ModuleOp module, func::FuncOp taskGraphFunc,
                                  const HardwareBudget &budget,
                                  const TaskGraphDAG &dag) const final;
};

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHSCORER_H
