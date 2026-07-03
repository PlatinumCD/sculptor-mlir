#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENT_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENT_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<PhysicalArrayPlacement>
resolvePhysicalArrayPlacement(Operation *diagnosticOp,
                              const HardwareBudget &budget,
                              int64_t physicalArrayId);

LogicalResult attachTaskCorePlacement(ModuleOp module,
                                      sculptor::TaskCreateOp taskOp,
                                      const HardwareBudget &budget,
                                      int64_t coreId);

LogicalResult attachTaskAnalogArrayPlacement(ModuleOp module,
                                             sculptor::TaskCreateOp taskOp,
                                             const HardwareBudget &budget,
                                             int64_t physicalArrayId);

LogicalResult placeLogicalPlacementIslands(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag, llvm::ArrayRef<int64_t> physicalArrayOrder);

LogicalResult placeLogicalPlacementIslands(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag, const LogicalPlacementIslandGraph &islandGraph,
    llvm::ArrayRef<MatrixSetupGroupPlacement> groupPlacements);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHPLACEMENT_H
