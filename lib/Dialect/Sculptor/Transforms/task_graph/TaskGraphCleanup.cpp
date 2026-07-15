#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphCleanup.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"

#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

namespace {

namespace task_attrs = mlir::sculptor::task_attrs;

static bool isTaskGraphFunction(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

static bool isGeneratedTaskCallee(mlir::func::FuncOp func) {
  return func->hasAttr(task_attrs::kTaskKindAttrName) ||
         func.getSymName().starts_with("task_");
}

static bool callsGeneratedTaskCallee(mlir::ModuleOp module,
                                     mlir::func::FuncOp func) {
  bool found = false;
  func.walk([&](mlir::func::CallOp callOp) {
    if (found)
      return;
    auto callee = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
    if (callee && isGeneratedTaskCallee(callee))
      found = true;
  });
  return found;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_graph {

LogicalResult
eraseUnusedTaskGraphIntermediateResources(func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected scheduled task graph function to have "
                            "one block");
    return failure();
  }

  SmallVector<Operation *> unusedResources;
  for (Operation &op : taskGraphFunc.getBody().front()) {
    auto intermediateOp = dyn_cast<sculptor::TaskGraphIntermediateOp>(&op);
    if (intermediateOp && intermediateOp.getResult().use_empty())
      unusedResources.push_back(&op);
  }
  for (Operation *op : unusedResources)
    op->erase();
  return success();
}

void eraseUnusedTaskCallees(ModuleOp module) {
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

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
