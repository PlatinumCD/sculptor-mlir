#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"

#include "llvm/ADT/STLExtras.h"

namespace {

static mlir::InFlightDiagnostic emitPlacementError(
    const mlir::sculptor::task_schedulers::TaskGraphPlacementProblem &problem) {
  return problem.diagnosticOp->emitError();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

LogicalResult validatePlacementPlan(const TaskGraphPlacementProblem &problem,
                                    const IslandPlacementPlan &plan) {
  if (!problem.taskGraphFunc || !problem.diagnosticOp)
    return failure();

  if (plan.physicalArrayByIsland.size() != problem.islandGraph.islands.size()) {
    emitPlacementError(problem)
        << "expected placement plan to contain one physical array per logical "
           "island";
    return failure();
  }

  for (int64_t physicalArrayId : plan.physicalArrayByIsland) {
    if (physicalArrayId < 0 ||
        physicalArrayId >= problem.budget.numAnalogArrays ||
        !llvm::is_contained(problem.budget.analogArrays, physicalArrayId)) {
      emitPlacementError(problem)
          << "expected placement plan physical array " << physicalArrayId
          << " to belong to the hardware budget";
      return failure();
    }
  }

  return success();
}

FailureOr<IslandPlacementPlan> buildPlacementPlanFromPhysicalArrayOrder(
    const TaskGraphPlacementProblem &problem,
    llvm::ArrayRef<int64_t> physicalArrayOrder) {
  if (!problem.islandGraph.islands.empty() && physicalArrayOrder.empty()) {
    emitPlacementError(problem)
        << "expected logical island placement to have at least one physical "
           "analog array";
    return failure();
  }

  IslandPlacementPlan plan;
  plan.physicalArrayByIsland.reserve(problem.islandGraph.islands.size());
  for (auto indexedIsland : llvm::enumerate(problem.islandGraph.islands)) {
    (void)indexedIsland.value();
    plan.physicalArrayByIsland.push_back(
        physicalArrayOrder[indexedIsland.index() % physicalArrayOrder.size()]);
  }

  if (failed(validatePlacementPlan(problem, plan)))
    return failure();
  return plan;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
