#include "sculptor-mlir/Dialect/Sculptor/Transforms/OptimizeTaskGraph.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphCleanup.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphOptimizer.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

#include <limits>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_graph = mlir::sculptor::task_graph;

static bool returnsTaskGraph(mlir::func::FuncOp func) {
  mlir::FunctionType type = func.getFunctionType();
  return type.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(type.getResult(0));
}

static mlir::LogicalResult
verifyScheduledAndUnfinalized(mlir::func::FuncOp taskGraphFunc,
                              const task_graph::TaskGraphDAG &dag) {
  if (taskGraphFunc->hasAttr(runtime_attrs::kTaskGraphResourceCountAttrName)) {
    return taskGraphFunc.emitError(
        "--sculptor-optimize-task-graph must run before "
        "--sculptor-finalize-task-graph-resources");
  }

  for (const task_graph::TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    if (!taskOp->getAttrOfType<mlir::IntegerAttr>(
            runtime_attrs::kTaskCoreIdAttrName)) {
      return taskOp.emitError("expected scheduled core placement; run "
                              "--sculptor-schedule-task-graph before "
                              "--sculptor-optimize-task-graph");
    }
  }
  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 4>>
parseSelectedPatterns(mlir::Operation *anchor, llvm::StringRef option,
                      llvm::SmallVectorImpl<std::string> &storage) {
  llvm::SmallVector<llvm::StringRef, 4> available;
  for (const task_graph::TaskGraphOptimizationPattern &pattern :
       task_graph::getTaskGraphOptimizationPatterns())
    available.push_back(pattern.name);

  llvm::StringRef trimmed = option.trim();
  if (trimmed.empty() || trimmed == "none")
    return llvm::SmallVector<llvm::StringRef, 4>{};
  if (trimmed == "all")
    return available;

  llvm::SmallSet<llvm::StringRef, 4> seen;
  llvm::SmallVector<llvm::StringRef, 4> selected;
  llvm::SmallVector<llvm::StringRef, 4> pieces;
  trimmed.split(pieces, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  storage.reserve(storage.size() + pieces.size());
  for (llvm::StringRef piece : pieces) {
    piece = piece.trim();
    if (piece.empty()) {
      anchor->emitError(
          "expected non-empty names in task-graph optimization pattern list");
      return mlir::failure();
    }
    if (!llvm::is_contained(available, piece)) {
      anchor->emitError("unknown task-graph optimization pattern '")
          << piece << "'";
      return mlir::failure();
    }
    if (!seen.insert(piece).second)
      continue;
    storage.push_back(piece.str());
    selected.push_back(storage.back());
  }
  return selected;
}

static void clearTimingMetadata(mlir::func::FuncOp taskGraphFunc) {
  auto removeTimingAttrs = [](mlir::Operation *op) {
    llvm::SmallVector<mlir::StringAttr, 16> names;
    for (mlir::NamedAttribute attr : op->getAttrs()) {
      if (attr.getName().getValue().starts_with("sculptor.timing."))
        names.push_back(attr.getName());
    }
    for (mlir::StringAttr name : names)
      op->removeAttr(name);
  };

  removeTimingAttrs(taskGraphFunc);
  for (mlir::sculptor::TaskCreateOp task :
       taskGraphFunc.getOps<mlir::sculptor::TaskCreateOp>())
    removeTimingAttrs(task);
}

static mlir::LogicalResult
refreshStructuralMetadata(mlir::func::FuncOp taskGraphFunc,
                          const task_graph::TaskGraphDAG &dag) {
  mlir::Builder builder(taskGraphFunc.getContext());
  taskGraphFunc->setAttr(
      schedule_attrs::kTaskCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag.nodes.size())));
  taskGraphFunc->setAttr(
      schedule_attrs::kDependencyCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag.dependencyCount)));

  int64_t totalDigitalOps = 0;
  for (const task_graph::TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp task = node.op;
    auto digitalOps = task->getAttrOfType<mlir::IntegerAttr>(
        runtime_attrs::kTaskDigitalOpsAttrName);
    if (!digitalOps)
      continue;
    int64_t value = digitalOps.getInt();
    if (value < 0 ||
        totalDigitalOps > std::numeric_limits<int64_t>::max() - value)
      return task.emitError(
          "invalid or overflowing digital operation count after task-graph "
          "optimization");
    totalDigitalOps += value;
  }
  taskGraphFunc->setAttr(schedule_attrs::kTotalDigitalOpsAttrName,
                         builder.getI64IntegerAttr(totalDigitalOps));
  return mlir::success();
}

} // namespace

namespace mlir {
namespace sculptor {

void OptimizeTaskGraphPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool foundTaskGraph = false;
  bool changedAnyGraph = false;

  llvm::SmallVector<std::string, 4> patternStorage;
  auto selectedPatterns =
      parseSelectedPatterns(module, patterns, patternStorage);
  if (failed(selectedPatterns)) {
    signalPassFailure();
    return;
  }

  for (func::FuncOp taskGraphFunc : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(taskGraphFunc))
      continue;
    foundTaskGraph = true;

    auto dag = task_graph::parseTaskGraphDAG(taskGraphFunc);
    if (failed(dag) ||
        failed(verifyScheduledAndUnfinalized(taskGraphFunc, *dag))) {
      signalPassFailure();
      return;
    }

    bool changed = false;
    if (failed(task_graph::optimizeTaskGraph(module, taskGraphFunc, *dag,
                                             *selectedPatterns, changed))) {
      signalPassFailure();
      return;
    }
    if (!changed)
      continue;

    changedAnyGraph = true;
    if (failed(task_graph::eraseUnusedTaskGraphIntermediateResources(
            taskGraphFunc))) {
      taskGraphFunc.emitError(
          "failed to erase dead resources after task-graph optimization");
      signalPassFailure();
      return;
    }

    auto optimizedDag = task_graph::parseTaskGraphDAG(taskGraphFunc);
    if (failed(optimizedDag)) {
      signalPassFailure();
      return;
    }
    if (failed(refreshStructuralMetadata(taskGraphFunc, *optimizedDag))) {
      signalPassFailure();
      return;
    }
    clearTimingMetadata(taskGraphFunc);
  }

  if (!foundTaskGraph) {
    module.emitError("expected at least one task graph function returning "
                     "!sculptor.task_graph");
    signalPassFailure();
    return;
  }
  if (requireChange && !changedAnyGraph) {
    module.emitError("no selected task-graph optimization pattern applied");
    signalPassFailure();
    return;
  }

  if (changedAnyGraph)
    task_graph::eraseUnusedTaskCallees(module);
}

void registerOptimizeTaskGraphPass() {
  PassRegistration<OptimizeTaskGraphPass>();
}

} // namespace sculptor
} // namespace mlir
