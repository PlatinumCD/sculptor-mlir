#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_ANNEALING_ANNEALINGMOVES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_ANNEALING_ANNEALINGMOVES_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphIslands.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <random>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace annealing_detail {

struct Placement {
  llvm::SmallVector<int64_t, 8> physicalArrayOrder;
};

using MoveKind = AnnealingMoveKind;

FailureOr<Placement>
perturbPlacement(func::FuncOp taskGraphFunc, const Placement &current,
                 const TaskGraphDAG &dag,
                 const LogicalPlacementIslandGraph &islandGraph,
                 llvm::ArrayRef<MoveKind> enabledMoveKinds, int64_t moveRadius,
                 std::mt19937 &randomEngine);

} // namespace annealing_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_ANNEALING_ANNEALINGMOVES_H
