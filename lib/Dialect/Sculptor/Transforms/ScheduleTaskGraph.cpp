#include "sculptor-mlir/Dialect/Sculptor/Transforms/ScheduleTaskGraph.h"

// ScheduleTaskGraph makes the target hardware budget explicit in IR and runs
// the selected task scheduler to place prebuilt logical islands on cores and
// arrays.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphHardwareConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementConstraints.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleMetadata.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleReport.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <utility>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;

bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

} // namespace

namespace mlir {
namespace sculptor {

void ScheduleTaskGraphPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto budget = task_schedulers::buildHardwareBudget(
      module, cores, arraysPerCore, topology, meshRows, meshCols);
  if (failed(budget)) {
    signalPassFailure();
    return;
  }
  auto schedulerOptions = task_schedulers::buildTaskGraphSchedulerOptions(
      module.getOperation(), schedule, randomSeed, greedyHeuristic,
      annealingInitialSchedule, annealingMoveSet, annealingMoveRadius,
      annealingInitialTemperature, annealingFinalTemperature,
      annealingCoolingRate, annealingStepsPerTemperature);
  if (failed(schedulerOptions)) {
    signalPassFailure();
    return;
  }

  task_schedulers::TaskGraphSchedulerRegistry registry;
  task_schedulers::registerRandomTaskScheduler(registry);
  task_schedulers::registerSnakeTaskScheduler(registry);
  task_schedulers::registerGreedyTaskScheduler(registry);
  task_schedulers::registerGreedyTimingTaskScheduler(registry);
  task_schedulers::registerSimulatedAnnealingTaskScheduler(registry);
  const task_schedulers::TaskGraphScheduler *selectedScheduler =
      task_schedulers::lookupTaskGraphScheduler(registry, schedule);
  if (!selectedScheduler) {
    if (schedule.empty()) {
      module.emitError("expected task graph schedule name");
    } else {
      module.emitError("unknown task graph schedule '") << schedule << "'";
    }
    signalPassFailure();
    return;
  }

  mlir::Builder builder(module.getContext());
  task_schedulers::attachHardwareBudgetAttrs(module.getOperation(), builder,
                                             *budget);
  task_schedulers::attachTaskGraphSchedulerOptionAttrs(
      module.getOperation(), builder, *schedulerOptions);

  bool foundTaskGraph = false;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;

    task_schedulers::attachHardwareBudgetAttrs(func.getOperation(), builder,
                                               *budget);
    task_schedulers::attachTaskGraphSchedulerOptionAttrs(
        func.getOperation(), builder, *schedulerOptions);
    auto parsedDag = task_graph::parseTaskGraphDAG(func);
    if (failed(parsedDag)) {
      signalPassFailure();
      return;
    }

    task_schedulers::TaskGraphDAG dag = std::move(*parsedDag);
    auto executionGraph = task_graph::buildTaskExecutionGraph(func, dag);
    if (failed(executionGraph)) {
      signalPassFailure();
      return;
    }
    auto islandGraph =
        task_graph::loadLogicalPlacementIslandGraph(dag, *executionGraph);
    if (failed(islandGraph)) {
      func.emitError("failed to load logical placement islands");
      signalPassFailure();
      return;
    }
    auto constraints = task_schedulers::buildPlacementConstraints(
        func.getOperation(), *executionGraph, *islandGraph);
    if (failed(constraints)) {
      signalPassFailure();
      return;
    }

    task_schedulers::TaskGraphPlacementProblem placementProblem{
        func, *budget, dag, *executionGraph, *islandGraph, *constraints};
    auto buildPlacementPlan =
        [&]() -> FailureOr<task_schedulers::IslandPlacementPlan> {
      if (!selectedScheduler->requiresTimingProfile())
        return selectedScheduler->buildPlacementPlan(placementProblem,
                                                     *schedulerOptions);

      auto timingProfile =
          task_timing::loadSchedulingTimingProfile(func, dag, *islandGraph);
      if (failed(timingProfile)) {
        func.emitError(
            "failed to load pre-placement scheduling timing profile");
        return failure();
      }
      return selectedScheduler->buildTimingPlacementPlan(
          placementProblem, *timingProfile, *schedulerOptions);
    };
    auto placementPlan = buildPlacementPlan();
    if (failed(placementPlan)) {
      func.emitError("failed to build task graph placement plan for schedule '")
          << selectedScheduler->getName() << "'";
      signalPassFailure();
      return;
    }

    if (failed(task_schedulers::commitPlacementPlan(
            module, func, placementProblem, *placementPlan))) {
      func.emitError(
          "failed to commit task graph placement plan for schedule '")
          << selectedScheduler->getName() << "'";
      signalPassFailure();
      return;
    }

    auto scheduledDag = task_graph::parseTaskGraphDAG(func);
    if (failed(scheduledDag)) {
      signalPassFailure();
      return;
    }

    if (failed(task_schedulers::finalizeTaskGraphScheduleMetadata(
            module, func, *budget, *scheduledDag, *islandGraph,
            *constraints))) {
      func.emitError("failed to finalize task graph schedule metadata");
      signalPassFailure();
      return;
    }

    if (failed(task_schedulers::appendScheduleSummary(
            func, *budget, *schedulerOptions, selectedScheduler->getName(),
            summaryOutput))) {
      signalPassFailure();
      return;
    }

    foundTaskGraph = true;
  }

  if (!foundTaskGraph) {
    module.emitError("expected at least one task graph function returning "
                     "!sculptor.task_graph");
    signalPassFailure();
  }
}

void registerScheduleTaskGraphPass() {
  PassRegistration<ScheduleTaskGraphPass>();
}

} // namespace sculptor
} // namespace mlir
