#include "sculptor-mlir/Dialect/Sculptor/Transforms/AssembleTaskGraph.h"

// AssembleTaskGraph turns the lowered forward function into an executable
// analog task graph.

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyStep.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

mlir::LogicalResult assembleTaskGraph(mlir::ModuleOp module,
                                      mlir::func::FuncOp forward,
                                      mlir::func::FuncOp &taskGraphFunc) {
  mlir::sculptor::TaskGraphAssemblySteps steps;
  mlir::sculptor::registerTaskGraphGeneratorAssembler(steps);
  mlir::sculptor::registerTaskGraphResourceAssembler(steps);
  mlir::sculptor::registerTaskGraphTaskAssembler(steps);

  taskGraphFunc = {};
  for (const auto &step : steps) {
    if (failed(step->assemble(module, forward))) {
      forward.emitError("failed to apply task graph assembly step '")
          << step->getName() << "'";
      return mlir::failure();
    }

    if (!taskGraphFunc) {
      taskGraphFunc =
          mlir::sculptor::assembler_utils::lookupGeneratedTaskGraphFunc(module,
                                                                      forward);
    }
  }

  if (!taskGraphFunc) {
    forward.emitError(
        "expected task graph assembly pipeline to create a generator function");
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult assembleExecutablePlan(mlir::ModuleOp module,
                                           mlir::func::FuncOp forward) {
  mlir::sculptor::TaskGraphAssemblySteps steps;
  mlir::sculptor::registerTaskGraphExecutionPlanAssembler(steps);

  for (const auto &step : steps) {
    if (failed(step->assemble(module, forward))) {
      forward.emitError("failed to apply task graph assembly step '")
          << step->getName() << "'";
      return mlir::failure();
    }
  }

  return mlir::success();
}

} // namespace

namespace mlir {
namespace sculptor {

void AssembleTaskGraphPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (func.getName() != "forward")
      continue;

    // Build the task graph structure from the lowered forward body.
    mlir::func::FuncOp taskGraphFunc;
    if (failed(assembleTaskGraph(module, func, taskGraphFunc))) {
      signalPassFailure();
      return;
    }

    // Add execution-plan metadata needed by downstream task-graph execution.
    if (failed(assembleExecutablePlan(module, func))) {
      signalPassFailure();
      return;
    }
  }
}

void registerAssembleTaskGraphPass() {
  PassRegistration<AssembleTaskGraphPass>();
}

} // namespace sculptor
} // namespace mlir
