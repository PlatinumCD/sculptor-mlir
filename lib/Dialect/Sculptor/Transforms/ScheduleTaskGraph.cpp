#include "sculptor-mlir/Dialect/Sculptor/Transforms/ScheduleTaskGraph.h"

// ScheduleTaskGraph makes the target hardware budget explicit in IR and runs
// the selected task scheduler to assign task indices, cores, and arrays.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphExecutionPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphRoutineFuser.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleMetadata.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScorer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <optional>
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

mlir::FailureOr<task_schedulers::HardwareBudget>
buildHardwareBudget(mlir::ModuleOp module, int64_t numCores,
                    int64_t arraysPerCore, llvm::StringRef topology,
                    int64_t meshRows, int64_t meshCols, int64_t randomSeed,
                    int64_t greedyLookahead,
                    llvm::StringRef greedyCandidateScope) {
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

  if (greedyLookahead < 1) {
    module.emitError("expected Sculptor greedy lookahead to be at least one");
    return mlir::failure();
  }

  if (greedyCandidateScope != "cardinal" &&
      greedyCandidateScope != "diagonal" &&
      greedyCandidateScope != "producer-consumer") {
    module.emitError("unknown Sculptor greedy candidate scope '")
        << greedyCandidateScope
        << "'; expected 'cardinal', 'diagonal', or 'producer-consumer'";
    return mlir::failure();
  }

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
  budget.greedyLookahead = greedyLookahead;
  budget.greedyCandidateScope = greedyCandidateScope.str();
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
              builder.getI64IntegerAttr(budget.greedyLookahead));
  op->setAttr(schedule_attrs::kGreedyCandidateScopeAttrName,
              builder.getStringAttr(budget.greedyCandidateScope));
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_attrs = mlir::sculptor::task_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;

static bool isLogicalArrayResource(Value value) {
  auto resourceType = dyn_cast<sculptor::TaskResourceType>(value.getType());
  return resourceType &&
         isa<sculptor::LogicalArrayType>(resourceType.getValueType());
}

static FailureOr<func::FuncOp> lookupTaskCallee(ModuleOp module,
                                                sculptor::TaskCreateOp taskOp,
                                                StringRef placementKind) {
  auto callee =
      module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
  if (!callee) {
    return taskOp.emitError("expected task callee '")
           << taskOp.getCalleeAttr().getValue()
           << "' to resolve to a function for " << placementKind
           << " placement";
  }

  return callee;
}

static void attachCorePlacementAttrs(Operation *op, Builder &builder,
                                     int64_t coreId) {
  op->setAttr(runtime_attrs::kTaskCoreIdAttrName,
              builder.getI64IntegerAttr(coreId));
}

static void
attachAnalogArrayPlacementAttrs(Operation *op, Builder &builder,
                                const PhysicalArrayPlacement &placement) {
  attachCorePlacementAttrs(op, builder, placement.coreId);
  op->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
              builder.getI64IntegerAttr(placement.physicalArrayId));
  op->setAttr(runtime_attrs::kTaskLocalArrayIdAttrName,
              builder.getI64IntegerAttr(placement.localArrayId));
}

static LogicalResult
eraseUnusedTaskGraphTemporaryResources(func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected scheduled task graph function to have "
                            "one block");
    return failure();
  }

  SmallVector<Operation *> unusedResources;
  for (Operation &op : taskGraphFunc.getBody().front()) {
    auto temporaryOp = dyn_cast<sculptor::TaskGraphTemporaryOp>(&op);
    if (temporaryOp && temporaryOp.getResult().use_empty())
      unusedResources.push_back(&op);
  }

  for (Operation *op : unusedResources)
    op->erase();

  return success();
}

static bool isTaskGraphFunction(func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         isa<sculptor::TaskGraphType>(functionType.getResult(0));
}

static bool isGeneratedTaskCallee(func::FuncOp func) {
  if (func->hasAttr(task_attrs::kTaskKindAttrName))
    return true;

  return func.getSymName().starts_with("task_");
}

static bool callsGeneratedTaskCallee(ModuleOp module, func::FuncOp func) {
  bool found = false;
  func.walk([&](func::CallOp callOp) {
    if (found)
      return;

    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (callee && isGeneratedTaskCallee(callee))
      found = true;
  });
  return found;
}

static void eraseUnusedTaskCallees(ModuleOp module) {
  llvm::StringSet<> liveTaskCallees;
  module.walk([&](sculptor::TaskCreateOp taskOp) {
    liveTaskCallees.insert(taskOp.getCalleeAttr().getValue());
  });

  bool hasTaskGraph = false;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (isTaskGraphFunction(func)) {
      hasTaskGraph = true;
      break;
    }
  }

  SmallVector<func::FuncOp> staleEntryPoints;
  llvm::SmallPtrSet<Operation *, 4> staleEntryPointOps;
  if (hasTaskGraph) {
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.getName() != "forward" ||
          !callsGeneratedTaskCallee(module, func))
        continue;

      staleEntryPoints.push_back(func);
      staleEntryPointOps.insert(func.getOperation());
    }
  }

  llvm::StringSet<> calledTaskCallees;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (staleEntryPointOps.contains(func.getOperation()))
      continue;

    func.walk([&](func::CallOp callOp) {
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (callee && isGeneratedTaskCallee(callee))
        calledTaskCallees.insert(callee.getSymName());
    });
  }

  for (func::FuncOp func : staleEntryPoints)
    func.erase();

  SmallVector<func::FuncOp> deadTaskCallees;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!func.isPrivate() || !isGeneratedTaskCallee(func))
      continue;

    if (liveTaskCallees.contains(func.getSymName()) ||
        calledTaskCallees.contains(func.getSymName()))
      continue;

    deadTaskCallees.push_back(func);
  }

  for (func::FuncOp func : deadTaskCallees)
    func.erase();
}

FailureOr<TaskGraphDAG> parseTaskGraphDAG(func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected scheduled task graph function to have "
                            "one block");
    return failure();
  }

  TaskGraphDAG dag;
  Block &block = taskGraphFunc.getBody().front();

  for (Operation &op : block) {
    for (Value result : op.getResults()) {
      if (isLogicalArrayResource(result))
        dag.logicalArrayResources.push_back(result);
    }

    auto taskOp = dyn_cast<sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    TaskGraphNode node;
    node.op = taskOp;
    node.index = dag.nodes.size();
    dag.nodeIndexByTaskResult.try_emplace(taskOp.getResult(), node.index);
    dag.nodes.push_back(std::move(node));
  }

  for (TaskGraphNode &node : dag.nodes) {
    for (Value dependency : node.op.getDependencies()) {
      auto predecessorIt = dag.nodeIndexByTaskResult.find(dependency);
      if (predecessorIt == dag.nodeIndexByTaskResult.end()) {
        node.op.emitError("expected task dependency to reference an "
                          "sculptor.task.create result in the same task graph");
        return failure();
      }

      unsigned predecessorIndex = predecessorIt->second;
      if (predecessorIndex >= node.index) {
        node.op.emitError("expected task dependency to reference an earlier "
                          "task in the task graph");
        return failure();
      }

      node.predecessors.push_back(predecessorIndex);
      dag.nodes[predecessorIndex].successors.push_back(node.index);
      ++dag.dependencyCount;
    }
  }

  return dag;
}

FailureOr<PhysicalArrayPlacement>
resolvePhysicalArrayPlacement(Operation *diagnosticOp,
                              const HardwareBudget &budget,
                              int64_t physicalArrayId) {
  int64_t coreId = physicalArrayId / budget.arraysPerCore;
  int64_t localArrayId = physicalArrayId % budget.arraysPerCore;
  if (physicalArrayId < 0 || coreId < 0 || coreId >= budget.numCores ||
      localArrayId < 0 || localArrayId >= budget.arraysPerCore) {
    if (diagnosticOp)
      diagnosticOp->emitError("assigned analog array is outside the hardware "
                              "budget");
    return failure();
  }

  return PhysicalArrayPlacement{physicalArrayId, coreId, localArrayId};
}

LogicalResult attachTaskCorePlacement(ModuleOp module,
                                      sculptor::TaskCreateOp taskOp,
                                      const HardwareBudget &budget,
                                      int64_t coreId) {
  if (coreId < 0 || coreId >= budget.numCores) {
    taskOp.emitError("assigned core is outside the hardware budget");
    return failure();
  }

  auto callee = lookupTaskCallee(module, taskOp, "core");
  if (failed(callee))
    return failure();

  Builder builder(module.getContext());
  attachCorePlacementAttrs(taskOp.getOperation(), builder, coreId);
  attachCorePlacementAttrs(callee->getOperation(), builder, coreId);
  return success();
}

LogicalResult attachTaskAnalogArrayPlacement(ModuleOp module,
                                             sculptor::TaskCreateOp taskOp,
                                             const HardwareBudget &budget,
                                             int64_t physicalArrayId) {
  auto placement = resolvePhysicalArrayPlacement(taskOp.getOperation(), budget,
                                                 physicalArrayId);
  if (failed(placement))
    return failure();

  auto callee = lookupTaskCallee(module, taskOp, "analog array");
  if (failed(callee))
    return failure();

  Builder builder(module.getContext());
  attachAnalogArrayPlacementAttrs(taskOp.getOperation(), builder, *placement);
  attachAnalogArrayPlacementAttrs(callee->getOperation(), builder, *placement);
  return success();
}

LogicalResult
registerTaskGraphScheduler(TaskGraphSchedulerRegistry &registry,
                           std::unique_ptr<TaskGraphScheduler> scheduler) {
  if (!scheduler)
    return failure();

  StringRef name = scheduler->getName();
  if (name.empty() || registry.contains(name))
    return failure();

  registry.try_emplace(name, std::move(scheduler));
  return success();
}

const TaskGraphScheduler *
lookupTaskGraphScheduler(const TaskGraphSchedulerRegistry &registry,
                         StringRef name) {
  auto it = registry.find(name);
  if (it == registry.end())
    return nullptr;
  return it->second.get();
}

} // namespace task_schedulers

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
  auto boundaryPenalty =
      getIntegerSummaryAttr(taskGraphFunc, schedule_attrs::kBoundaryPenaltyAttrName);
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

  os << taskGraphFunc.getName() << ',' << scheduleName << ','
     << budget.numCores << ',' << budget.arraysPerCore << ','
     << budget.meshRows << ',' << budget.meshCols << ','
     << budget.greedyLookahead << ',' << budget.greedyCandidateScope << ','
     << *taskCount << ',' << *dependencyCount << ',' << *numLogicalArrays
     << ',' << *totalDigitalOps << ',' << *interCoreTransferBytes << ','
     << *totalTransferCost << ',';
  os << *transferCostPerByte << ',' << *boundaryPenalty << ',' << *graphScore
     << '\n';
  return success();
}

void ScheduleTaskGraphPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto budget = buildHardwareBudget(module, cores, arraysPerCore, topology,
                                    meshRows, meshCols, randomSeed,
                                    greedyLookahead, greedyCandidateScope);
  if (failed(budget)) {
    signalPassFailure();
    return;
  }

  task_schedulers::TaskGraphSchedulerRegistry registry;
  task_schedulers::registerRandomTaskScheduler(registry);
  task_schedulers::registerSnakeTaskScheduler(registry);
  task_schedulers::registerGreedyTaskScheduler(registry);
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

    if (failed(selectedScheduler->schedule(module, func, *budget, dag))) {
      func.emitError("failed to apply task graph schedule '")
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

    if (failed(appendScheduleSummary(func, *budget,
                                     selectedScheduler->getName(),
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
