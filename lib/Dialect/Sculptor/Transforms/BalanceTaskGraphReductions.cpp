#include "sculptor-mlir/Dialect/Sculptor/Transforms/BalanceTaskGraphReductions.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphReductionBalancer.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "mlir/Pass/PassRegistry.h"

namespace {

bool returnsTaskGraph(mlir::func::FuncOp func) {
  mlir::FunctionType functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

bool hasEligibleReduction(mlir::func::FuncOp func, int64_t reductionWidth) {
  for (mlir::sculptor::TaskCreateOp task :
       func.getOps<mlir::sculptor::TaskCreateOp>()) {
    if (task->hasAttr(
            mlir::sculptor::task_graph_attrs::kTaskReductionAttrName) &&
        !task->hasAttr(
            mlir::sculptor::task_graph_attrs::kTaskReductionTreeIdAttrName) &&
        static_cast<int64_t>(task.getInputs().size()) > reductionWidth)
      return true;
  }
  return false;
}

} // namespace

namespace mlir {
namespace sculptor {

void BalanceTaskGraphReductionsPass::runOnOperation() {
  if (reductionWidth < 2) {
    getOperation().emitError(
        "expected task reduction width to be at least two");
    signalPassFailure();
    return;
  }

  bool foundTaskGraph = false;
  bool foundEligibleReduction = false;
  for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    foundTaskGraph = true;
    bool eligible = hasEligibleReduction(func, reductionWidth);
    foundEligibleReduction |= eligible;
    if (failed(task_graph::balanceTaskGraphReductions(getOperation(), func,
                                                      reductionWidth))) {
      signalPassFailure();
      return;
    }
    if (eligible)
      task_timing::invalidateTaskGraphStructure(func);
  }

  if (!foundTaskGraph) {
    getOperation().emitError(
        "expected a task graph function when balancing reductions");
    signalPassFailure();
    return;
  }

  if (requireChange && !foundEligibleReduction) {
    getOperation().emitError(
        "expected at least one eligible marked task reduction");
    signalPassFailure();
  }
}

void registerBalanceTaskGraphReductionsPass() {
  PassRegistration<BalanceTaskGraphReductionsPass>();
}

} // namespace sculptor
} // namespace mlir
