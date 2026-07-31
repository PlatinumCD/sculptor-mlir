#include "sculptor-mlir/Dialect/Sculptor/Transforms/AnalyzeTaskGraphTiming.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "mlir/Pass/PassRegistry.h"

#ifndef SCULPTOR_COMPILER_REVISION
#define SCULPTOR_COMPILER_REVISION "unknown"
#endif

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
  std::optional<task_timing::MVMCostMode> parsedMVMCostMode =
      task_timing::symbolizeMVMCostMode(mvmCostMode);
  if (!parsedMVMCostMode) {
    getOperation().emitError("unknown MVM cost mode '")
        << mvmCostMode << "'; expected 'analog' or 'digital'";
    signalPassFailure();
    return;
  }

  task_timing::TimingModel model;
  model.compilerRevision = SCULPTOR_COMPILER_REVISION;
  model.timingBoundary = timingBoundary;
  model.runtimeTaskPolicy = runtimeTaskPolicy;
  model.runtimeTransmitPolicy = runtimeTransmitPolicy;
  model.memoryBackend = memoryBackend;
  model.analogMVMLatencyNs = analogMVMLatencyNs;
  model.analogIOBitsPerCycle = analogIOBitsPerCycle;
  model.analogIOShared = analogIOShared;
  model.digitalClockGHz = digitalClockGHz;
  model.digitalIssueWidth = digitalIssueWidth;
  model.digitalVectorBitsPerCycle = digitalVectorBitsPerCycle;
  model.fixedRuntimeDispatchCycles = fixedRuntimeDispatchCycles;
  model.fixedTaskEntryCycles = fixedTaskEntryCycles;
  model.fixedTaskExitCycles = fixedTaskExitCycles;
  model.networkLinkBitsPerCycle = networkLinkBitsPerCycle;
  model.networkHopLatencyCycles = networkHopLatencyCycles;
  model.networkPipelined = networkPipelined;
  model.networkLinkWordBits = networkLinkWordBits;
  model.protocolWordsPerRoute = protocolWordsPerRoute;
  model.nicInjectionWordsPerCycle = nicInjectionWordsPerCycle;
  model.rxDmaWordsPerCycle = rxDmaWordsPerCycle;
  model.routingPolicy = routingPolicy;

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
        module, func, *dag, *executionGraph, *islandGraph, model,
        *parsedMVMCostMode);
    if (failed(analysis)) {
      signalPassFailure();
      return;
    }

    task_timing::attachTaskGraphTimingAnalysis(func, *dag, *analysis, model,
                                               *parsedMVMCostMode);
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
