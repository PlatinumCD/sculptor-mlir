#include "sculptor-mlir/Dialect/Sculptor/Transforms/AnalyzeTaskGraphTiming.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "mlir/Pass/PassRegistry.h"

namespace {

bool returnsTaskGraph(mlir::func::FuncOp func) {
  mlir::FunctionType functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

} // namespace

namespace mlir {
namespace sculptor {

void AnalyzeTaskGraphTimingPass::runOnOperation() {
  task_timing::TimingModel model;
  model.analogMVMLatencyNs = analogMVMLatencyNs;
  model.analogIOBitsPerCycle = analogIOBitsPerCycle;
  model.analogIOShared = analogIOShared;
  model.digitalClockGHz = digitalClockGHz;
  model.digitalIssueWidth = digitalIssueWidth;
  model.digitalVectorBitsPerCycle = digitalVectorBitsPerCycle;
  model.networkLinkBitsPerCycle = networkLinkBitsPerCycle;
  model.networkHopLatencyCycles = networkHopLatencyCycles;
  model.networkPipelined = networkPipelined;

  if (failed(task_timing::validateTimingModel(getOperation(), model))) {
    signalPassFailure();
    return;
  }

  ModuleOp module = getOperation();
  bool foundTaskGraph = false;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;

    auto dag = task_graph::parseTaskGraphDAG(func);
    if (failed(dag)) {
      signalPassFailure();
      return;
    }

    auto executionGraph = task_graph::buildTaskExecutionGraph(func, *dag);
    if (failed(executionGraph)) {
      signalPassFailure();
      return;
    }

    auto islandGraph =
        task_graph::loadLogicalPlacementIslandGraph(*dag, *executionGraph);
    if (failed(islandGraph)) {
      func.emitError("failed to load logical placement islands for timing ")
          << "analysis";
      signalPassFailure();
      return;
    }

    auto analysis = task_timing::analyzeTaskGraphTiming(
        module, func, *dag, *executionGraph, *islandGraph, model);
    if (failed(analysis)) {
      signalPassFailure();
      return;
    }

    task_timing::attachTaskGraphTimingAnalysis(func, *dag, *analysis, model);
    foundTaskGraph = true;
  }

  if (!foundTaskGraph) {
    module.emitError("expected a task graph function for timing analysis");
    signalPassFailure();
  }
}

void registerAnalyzeTaskGraphTimingPass() {
  PassRegistration<AnalyzeTaskGraphTimingPass>();
}

} // namespace sculptor
} // namespace mlir
