#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULEHELPERS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULEHELPERS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct LogicalArrayRuntimePlacement {
  int64_t physicalArrayId = 0;
  int64_t coreId = 0;
};

struct TaskCoreAssignments {
  llvm::DenseMap<Operation *, int64_t> coreByTask;
  llvm::DenseMap<Operation *, int64_t> physicalArrayByTask;
};

struct CoreTransferSummary {
  llvm::SmallVector<int64_t> transferBytes;
  llvm::SmallVector<int64_t> transferCost;
  int64_t interCoreTransferBytes = 0;
  int64_t totalTransferCost = 0;
};

LogicalResult verifyLogicalArraysFitAnalogBudget(Operation *diagnosticOp,
                                                 int64_t numLogicalArrays,
                                                 int64_t numAnalogArrays);

bool isMatrixSetupTask(sculptor::TaskCreateOp taskOp);
bool isMVMTask(sculptor::TaskCreateOp taskOp);
bool isAnalogTask(sculptor::TaskCreateOp taskOp);
bool isDigitalTask(sculptor::TaskCreateOp taskOp);
bool isLogicalArrayResource(Value value);

ArrayAttr buildI64ArrayAttr(Builder &builder, llvm::ArrayRef<int64_t> values);

void attachLogicalArrayPlacementAttrs(
    const llvm::SmallVectorImpl<Value> &logicalArrayResources,
    llvm::ArrayRef<LogicalArrayPlacement> placements, Builder &builder);

ArrayAttr buildLogicalArrayToAnalogArrayAttr(
    llvm::ArrayRef<LogicalArrayPlacement> placements, Builder &builder);

FailureOr<llvm::DenseMap<Value, LogicalArrayRuntimePlacement>>
buildLogicalArrayRuntimePlacementMap(
    const llvm::SmallVectorImpl<Value> &logicalArrayResources,
    llvm::ArrayRef<LogicalArrayPlacement> placements,
    const HardwareBudget &budget);

FailureOr<LogicalArrayRuntimePlacement> getSingleLogicalArrayPlacement(
    sculptor::TaskCreateOp taskOp, OperandRange resources,
    const llvm::DenseMap<Value, LogicalArrayRuntimePlacement>
        &placementByLogicalArray,
    llvm::StringRef resourceRole);

FailureOr<llvm::SmallVector<sculptor::TaskCreateOp>>
buildMatrixSetupFirstTaskOrder(const TaskGraphDAG &dag);

void attachTaskIndices(llvm::ArrayRef<sculptor::TaskCreateOp> scheduledTasks,
                       Builder &builder);

void attachTaskCoreIds(const TaskCoreAssignments &assignments,
                       Builder &builder);

LogicalResult attachTaskFunctionPlacementAttrs(
    ModuleOp module, llvm::ArrayRef<sculptor::TaskCreateOp> tasks,
    const TaskCoreAssignments &assignments, const HardwareBudget &budget);

FailureOr<int64_t>
attachTaskDigitalOps(ModuleOp module,
                     llvm::ArrayRef<sculptor::TaskCreateOp> scheduledTasks,
                     Builder &builder);

CoreTransferSummary
computeCoreTransferSummary(llvm::ArrayRef<sculptor::TaskCreateOp> scheduledTasks,
                           const HardwareBudget &budget,
                           const TaskCoreAssignments &assignments);

void attachTaskGraphScheduleSummaryAttrs(
    func::FuncOp taskGraphFunc, const TaskGraphDAG &dag,
    llvm::ArrayRef<LogicalArrayPlacement> placements,
    const CoreTransferSummary &transferSummary, int64_t totalDigitalOps,
    Builder &builder);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULEHELPERS_H
