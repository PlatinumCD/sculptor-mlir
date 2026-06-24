#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULER_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct HardwareBudget {
  int64_t numCores = 0;
  int64_t arraysPerCore = 0;
  std::string topology;
  int64_t meshRows = 0;
  int64_t meshCols = 0;
  int64_t numAnalogArrays = 0;
  int64_t randomSeed = 0;
  llvm::SmallVector<int64_t, 8> analogArrays;
};

struct TaskGraphNode {
  sculptor::TaskCreateOp op;
  unsigned index = 0;
  llvm::SmallVector<unsigned, 4> predecessors;
  llvm::SmallVector<unsigned, 4> successors;
};

struct LogicalArrayPlacement {
  int64_t logicalArrayIndex = 0;
  int64_t analogArrayIndex = 0;
};

struct PhysicalArrayPlacement {
  int64_t physicalArrayId = 0;
  int64_t coreId = 0;
  int64_t localArrayId = 0;
};

struct MatrixSetupGroupPlacement {
  unsigned matrixSetupTaskIndex = 0;
  int64_t physicalArrayId = 0;
};

struct TaskGraphDAG {
  llvm::SmallVector<TaskGraphNode, 16> nodes;
  llvm::SmallVector<Value, 8> logicalArrayResources;
  llvm::DenseMap<Value, unsigned> nodeIndexByTaskResult;
  unsigned dependencyCount = 0;
};

FailureOr<TaskGraphDAG> parseTaskGraphDAG(func::FuncOp taskGraphFunc);

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

LogicalResult placeMatrixSetupGroupsAndSurroundingTasks(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag, llvm::ArrayRef<int64_t> physicalArrayOrder);

LogicalResult placeMatrixSetupGroupsAndSurroundingTasks(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag,
    llvm::ArrayRef<MatrixSetupGroupPlacement> groupPlacements);

FailureOr<sculptor::TaskCreateOp> fuseTasks(ModuleOp module,
                                            sculptor::TaskCreateOp parentTask,
                                            sculptor::TaskCreateOp childTask);

LogicalResult fuseTaskGraphRoutines(ModuleOp module, func::FuncOp taskGraphFunc,
                                    const HardwareBudget &budget,
                                    const TaskGraphDAG &dag);

LogicalResult finalizeTaskGraphScheduleMetadata(ModuleOp module,
                                                func::FuncOp taskGraphFunc,
                                                const HardwareBudget &budget,
                                                const TaskGraphDAG &dag);

class TaskGraphScheduler {
public:
  virtual ~TaskGraphScheduler() = default;

  virtual StringRef getName() const = 0;

  virtual LogicalResult schedule(ModuleOp module, func::FuncOp taskGraphFunc,
                                 const HardwareBudget &budget,
                                 const TaskGraphDAG &dag) const = 0;
};

using TaskGraphSchedulerRegistry =
    llvm::StringMap<std::unique_ptr<TaskGraphScheduler>>;

LogicalResult
registerTaskGraphScheduler(TaskGraphSchedulerRegistry &registry,
                           std::unique_ptr<TaskGraphScheduler> scheduler);

const TaskGraphScheduler *
lookupTaskGraphScheduler(const TaskGraphSchedulerRegistry &registry,
                         StringRef name);

void registerRandomTaskScheduler(TaskGraphSchedulerRegistry &registry);
void registerSnakeTaskScheduler(TaskGraphSchedulerRegistry &registry);
void registerGreedyHeavyEdgeTaskScheduler(TaskGraphSchedulerRegistry &registry);
void registerManhattanCutTaskScheduler(TaskGraphSchedulerRegistry &registry);
void registerBoundaryAwareCutTaskScheduler(
    TaskGraphSchedulerRegistry &registry);
void registerBoundaryAwareCutOptimizedTaskScheduler(
    TaskGraphSchedulerRegistry &registry);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULER_H
