#include "sculptor-mlir/Dialect/Sculptor/Transforms/ScheduleTaskGraph.h"

// ScheduleTaskGraph makes the target hardware budget explicit in IR and runs
// the selected task scheduler to assign task indices, cores, and arrays.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphExecutionPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphCleanup.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphRoutineFuser.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphScheduleMetadata.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>

namespace {

namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_schedulers = mlir::sculptor::task_schedulers;

bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

mlir::FailureOr<task_schedulers::HardwareBudget> buildHardwareBudget(
    mlir::ModuleOp module, int64_t numCores, int64_t arraysPerCore,
    llvm::StringRef topology, int64_t meshRows, int64_t meshCols,
    int64_t randomSeed, llvm::StringRef greedyHeuristic,
    llvm::StringRef annealingInitialSchedule, llvm::StringRef annealingMoveSet,
    int64_t annealingMoveRadius, double annealingInitialTemperature,
    double annealingFinalTemperature, double annealingCoolingRate,
    int64_t annealingStepsPerTemperature) {
  if (numCores <= 0) {
    module.emitError("expected Sculptor scheduling budget to have at least one "
                     "core");
    return mlir::failure();
  }

  if (arraysPerCore <= 0) {
    module.emitError("expected Sculptor scheduling budget to have at least one "
                     "array per core");
    return mlir::failure();
  }

  if (randomSeed < 0) {
    module.emitError("expected Sculptor random scheduling seed to be "
                     "non-negative");
    return mlir::failure();
  }

  auto greedyConfig = task_schedulers::parseGreedyScheduleConfig(
      module.getOperation(), greedyHeuristic);
  if (mlir::failed(greedyConfig))
    return mlir::failure();

  auto annealingConfig = task_schedulers::parseAnnealingScheduleConfig(
      module.getOperation(), annealingInitialSchedule, annealingMoveSet,
      annealingMoveRadius, annealingInitialTemperature,
      annealingFinalTemperature, annealingCoolingRate,
      annealingStepsPerTemperature);
  if (mlir::failed(annealingConfig))
    return mlir::failure();

  if (topology != "mesh") {
    module.emitError("unknown Sculptor scheduling topology '")
        << topology << "'";
    return mlir::failure();
  }

  if (meshRows <= 0) {
    module.emitError("expected mesh topology to have at least one row");
    return mlir::failure();
  }

  if (meshCols < 0) {
    module.emitError("expected mesh topology column count to be non-negative");
    return mlir::failure();
  }

  if (meshCols == 0) {
    if (numCores % meshRows != 0) {
      module.emitError("expected mesh topology rows to evenly divide the "
                       "number of cores when mesh-cols is inferred");
      return mlir::failure();
    }
    meshCols = numCores / meshRows;
  }

  if (meshCols <= 0) {
    module.emitError("expected mesh topology to have at least one column");
    return mlir::failure();
  }

  if (meshRows > std::numeric_limits<int64_t>::max() / meshCols) {
    module.emitError("Sculptor scheduling mesh topology overflows core count");
    return mlir::failure();
  }

  if (meshRows * meshCols != numCores) {
    module.emitError("expected mesh topology dimensions to match the number "
                     "of cores");
    return mlir::failure();
  }

  if (numCores > std::numeric_limits<int64_t>::max() / arraysPerCore) {
    module.emitError("Sculptor scheduling budget overflows total array count");
    return mlir::failure();
  }

  task_schedulers::HardwareBudget budget;
  budget.numCores = numCores;
  budget.arraysPerCore = arraysPerCore;
  budget.topology = topology.str();
  budget.meshRows = meshRows;
  budget.meshCols = meshCols;
  budget.numAnalogArrays = numCores * arraysPerCore;
  budget.randomSeed = randomSeed;
  budget.greedy = std::move(*greedyConfig);
  budget.annealing = std::move(*annealingConfig);
  budget.analogArrays.reserve(static_cast<size_t>(budget.numAnalogArrays));

  for (int64_t analogArray = 0; analogArray < budget.numAnalogArrays;
       ++analogArray)
    budget.analogArrays.push_back(analogArray);

  return budget;
}

mlir::ArrayAttr buildI64ArrayAttr(mlir::Builder &builder,
                                  llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

void attachBudgetAttrs(mlir::Operation *op, mlir::Builder &builder,
                       const task_schedulers::HardwareBudget &budget) {
  op->setAttr(schedule_attrs::kNumCoresAttrName,
              builder.getI64IntegerAttr(budget.numCores));
  op->setAttr(schedule_attrs::kArraysPerCoreAttrName,
              builder.getI64IntegerAttr(budget.arraysPerCore));
  op->setAttr(schedule_attrs::kTopologyAttrName,
              builder.getStringAttr(budget.topology));
  op->setAttr(schedule_attrs::kMeshRowsAttrName,
              builder.getI64IntegerAttr(budget.meshRows));
  op->setAttr(schedule_attrs::kMeshColsAttrName,
              builder.getI64IntegerAttr(budget.meshCols));
  op->setAttr(schedule_attrs::kNumAnalogArraysAttrName,
              builder.getI64IntegerAttr(budget.numAnalogArrays));
  op->setAttr(schedule_attrs::kAnalogArraysAttrName,
              buildI64ArrayAttr(builder, budget.analogArrays));
  op->setAttr(schedule_attrs::kGreedyLookaheadAttrName,
              builder.getI64IntegerAttr(budget.greedy.lookahead));
  op->setAttr(schedule_attrs::kGreedyBeamWidthAttrName,
              builder.getI64IntegerAttr(budget.greedy.beamWidth));
  op->setAttr(
      schedule_attrs::kGreedyCandidateScopeAttrName,
      builder.getStringAttr(task_schedulers::stringifyGreedyCandidateScope(
          budget.greedy.candidateScope)));
  op->setAttr(schedule_attrs::kGreedyHeuristicAttrName,
              builder.getStringAttr(budget.greedy.specification));
  op->setAttr(schedule_attrs::kAnnealingMoveSetAttrName,
              builder.getStringAttr(budget.annealing.moveSetSpecification));
  op->setAttr(schedule_attrs::kAnnealingMoveRadiusAttrName,
              builder.getI64IntegerAttr(budget.annealing.moveRadius));
}

} // namespace

namespace mlir {
namespace sculptor {

static FailureOr<int64_t> getIntegerSummaryAttr(func::FuncOp func,
                                                StringRef attrName) {
  auto attr = func->getAttrOfType<IntegerAttr>(attrName);
  if (!attr) {
    func.emitError("expected schedule summary attribute '") << attrName << "'";
    return failure();
  }
  return attr.getInt();
}

static FailureOr<double> getFloatSummaryAttr(func::FuncOp func,
                                             StringRef attrName) {
  auto attr = func->getAttrOfType<FloatAttr>(attrName);
  if (!attr) {
    func.emitError("expected schedule summary attribute '") << attrName << "'";
    return failure();
  }
  return attr.getValueAsDouble();
}

static void writeCsvString(llvm::raw_ostream &os, llvm::StringRef value) {
  if (!value.contains(',') && !value.contains('"') && !value.contains('\n')) {
    os << value;
    return;
  }

  os << '"';
  for (char c : value) {
    if (c == '"')
      os << "\"\"";
    else
      os << c;
  }
  os << '"';
}

static LogicalResult
appendScheduleSummary(func::FuncOp taskGraphFunc,
                      const task_schedulers::HardwareBudget &budget,
                      StringRef scheduleName, StringRef outputPath) {
  if (outputPath.empty())
    return success();

  auto taskCount =
      getIntegerSummaryAttr(taskGraphFunc, schedule_attrs::kTaskCountAttrName);
  auto dependencyCount = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kDependencyCountAttrName);
  auto numLogicalArrays = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kNumLogicalArraysAttrName);
  auto totalDigitalOps = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kTotalDigitalOpsAttrName);
  auto interCoreTransferBytes = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kInterCoreTransferBytesAttrName);
  auto totalTransferCost = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kTotalTransferCostAttrName);
  auto boundaryPenalty = getIntegerSummaryAttr(
      taskGraphFunc, schedule_attrs::kBoundaryPenaltyAttrName);
  auto graphScore =
      getIntegerSummaryAttr(taskGraphFunc, schedule_attrs::kGraphScoreAttrName);
  auto transferCostPerByte = getFloatSummaryAttr(
      taskGraphFunc, schedule_attrs::kTransferCostPerInterCoreByteAttrName);

  if (failed(taskCount) || failed(dependencyCount) ||
      failed(numLogicalArrays) || failed(totalDigitalOps) ||
      failed(interCoreTransferBytes) || failed(totalTransferCost) ||
      failed(boundaryPenalty) || failed(graphScore) ||
      failed(transferCostPerByte))
    return failure();

  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec, llvm::sys::fs::OF_Append);
  if (ec) {
    taskGraphFunc.emitError("failed to open schedule summary output '")
        << outputPath << "': " << ec.message();
    return failure();
  }

  writeCsvString(os, taskGraphFunc.getName());
  os << ',';
  writeCsvString(os, scheduleName);
  os << ',' << budget.numCores << ',' << budget.arraysPerCore << ','
     << budget.meshRows << ',' << budget.meshCols << ','
     << budget.greedy.lookahead << ',' << budget.greedy.beamWidth << ',';
  writeCsvString(os, task_schedulers::stringifyGreedyCandidateScope(
                         budget.greedy.candidateScope));
  os << ',';
  writeCsvString(os, budget.greedy.specification);
  os << ',' << *taskCount << ',' << *dependencyCount << ',' << *numLogicalArrays
     << ',' << *totalDigitalOps << ',' << *interCoreTransferBytes << ','
     << *totalTransferCost << ',';
  os << *transferCostPerByte << ',' << *boundaryPenalty << ',' << *graphScore
     << '\n';
  return success();
}

void ScheduleTaskGraphPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto budget = buildHardwareBudget(
      module, cores, arraysPerCore, topology, meshRows, meshCols, randomSeed,
      greedyHeuristic, annealingInitialSchedule, annealingMoveSet,
      annealingMoveRadius, annealingInitialTemperature,
      annealingFinalTemperature, annealingCoolingRate,
      annealingStepsPerTemperature);
  if (failed(budget)) {
    signalPassFailure();
    return;
  }

  task_schedulers::TaskGraphSchedulerRegistry registry;
  task_schedulers::registerRandomTaskScheduler(registry);
  task_schedulers::registerSnakeTaskScheduler(registry);
  task_schedulers::registerGreedyTaskScheduler(registry);
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
  attachBudgetAttrs(module.getOperation(), builder, *budget);

  bool foundTaskGraph = false;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;

    attachBudgetAttrs(func.getOperation(), builder, *budget);
    auto parsedDag = task_schedulers::parseTaskGraphDAG(func);
    if (failed(parsedDag)) {
      signalPassFailure();
      return;
    }

    task_schedulers::TaskGraphDAG dag = std::move(*parsedDag);
    auto islandGraph = task_schedulers::buildLogicalPlacementIslandGraph(dag);
    if (failed(islandGraph)) {
      func.emitError("failed to build logical placement islands");
      signalPassFailure();
      return;
    }

    task_schedulers::TaskGraphPlacementProblem placementProblem{
        func, *budget, dag, *islandGraph};
    auto placementPlan =
        selectedScheduler->buildPlacementPlan(placementProblem);
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

    if (failed(task_schedulers::fuseTaskGraphRoutines(module, func, *budget,
                                                      dag))) {
      func.emitError("failed to fuse task graph routines after schedule '")
          << selectedScheduler->getName() << "'";
      signalPassFailure();
      return;
    }

    task_schedulers::eraseUnusedTaskCallees(module);

    if (failed(task_schedulers::eraseUnusedTaskGraphTemporaryResources(func)) ||
        failed(mlir::sculptor::rebuildTaskGraphExecutionPlan(func))) {
      func.emitError("failed to compact task graph resources after schedule '")
          << selectedScheduler->getName() << "'";
      signalPassFailure();
      return;
    }

    auto scheduledDag = task_schedulers::parseTaskGraphDAG(func);
    if (failed(scheduledDag)) {
      signalPassFailure();
      return;
    }

    if (failed(task_schedulers::finalizeTaskGraphScheduleMetadata(
            module, func, *budget, *scheduledDag))) {
      func.emitError("failed to finalize task graph schedule metadata");
      signalPassFailure();
      return;
    }

    if (failed(appendScheduleSummary(
            func, *budget, selectedScheduler->getName(), summaryOutput))) {
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
