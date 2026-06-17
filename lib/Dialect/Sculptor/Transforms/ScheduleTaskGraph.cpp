#include "sculptor-mlir/Dialect/Sculptor/Transforms/ScheduleTaskGraph.h"

// ScheduleTaskGraph makes the target hardware budget explicit in IR and runs
// the selected task scheduler to assign task indices, cores, and arrays.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
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
                    int64_t meshRows, int64_t meshCols) {
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
}

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
parsePlacementVector(mlir::Operation *diagnosticOp,
                     llvm::StringRef placementSpec) {
  llvm::SmallVector<int64_t, 8> placement;
  placementSpec = placementSpec.trim();
  if (placementSpec.empty())
    return placement;

  llvm::SmallVector<llvm::StringRef, 8> tokens;
  placementSpec.split(tokens, ",", -1, true);
  placement.reserve(tokens.size());

  for (llvm::StringRef token : tokens) {
    token = token.trim();
    if (token.empty()) {
      diagnosticOp->emitError("expected placement vector entries to be "
                              "non-empty integers");
      return mlir::failure();
    }

    int64_t coreId = 0;
    if (token.getAsInteger(10, coreId)) {
      diagnosticOp->emitError("expected placement vector entry '")
          << token << "' to be an integer core id";
      return mlir::failure();
    }

    placement.push_back(coreId);
  }

  return placement;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

static bool isLogicalArrayResource(Value value) {
  auto resourceType = dyn_cast<sculptor::TaskResourceType>(value.getType());
  return resourceType &&
         isa<sculptor::LogicalArrayType>(resourceType.getValueType());
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

void ScheduleTaskGraphPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto budget = buildHardwareBudget(module, cores, arraysPerCore, topology,
                                    meshRows, meshCols);
  if (failed(budget)) {
    signalPassFailure();
    return;
  }

  task_schedulers::TaskGraphScheduleOptions scheduleOptions;
  auto parsedPlacement = parsePlacementVector(module.getOperation(), placement);
  if (failed(parsedPlacement)) {
    signalPassFailure();
    return;
  }
  scheduleOptions.placement = std::move(*parsedPlacement);

  task_schedulers::TaskGraphSchedulerRegistry registry;
  const task_schedulers::TaskGraphScheduler *selectedScheduler =
      task_schedulers::lookupTaskGraphScheduler(registry, schedule);
  if (!selectedScheduler) {
    if (registry.empty()) {
      module.emitError("no task graph schedulers are registered");
    } else if (schedule.empty()) {
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
    auto dag = task_schedulers::parseTaskGraphDAG(func);
    if (failed(dag)) {
      signalPassFailure();
      return;
    }

    if (failed(selectedScheduler->schedule(module, func, *budget, *dag,
                                           scheduleOptions))) {
      func.emitError("failed to apply task graph schedule '")
          << selectedScheduler->getName() << "'";
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
