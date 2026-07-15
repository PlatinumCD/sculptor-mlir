#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphIslandMap.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"

#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <system_error>

namespace {

namespace task_graph = mlir::sculptor::task_graph;

static bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

} // namespace

namespace mlir {
namespace sculptor {

void ExportTaskGraphIslandMapPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (output.empty()) {
    module.emitError("expected non-empty output path for "
                     "sculptor-export-task-graph-island-map");
    signalPassFailure();
    return;
  }

  std::error_code error;
  llvm::raw_fd_ostream os(output, error, llvm::sys::fs::OF_Text);
  if (error) {
    module.emitError("failed to open task graph island-map output file '")
        << output << "': " << error.message();
    signalPassFailure();
    return;
  }

  bool foundTaskGraph = false;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    foundTaskGraph = true;

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
      func.emitError("failed to load logical placement islands for export");
      signalPassFailure();
      return;
    }

    os << "# graph: " << func.getName() << '\n';
    os << "task_index,island_id\n";
    for (const task_graph::TaskGraphNode &node : dag->nodes) {
      auto islandIt = islandGraph->islandByTaskIndex.find(node.index);
      int64_t islandId = islandIt == islandGraph->islandByTaskIndex.end()
                             ? -1
                             : static_cast<int64_t>(islandIt->second);
      os << node.index << ',' << islandId << '\n';
    }
  }

  if (!foundTaskGraph) {
    module.emitError("expected at least one task graph function returning "
                     "!sculptor.task_graph");
    signalPassFailure();
  }
}

void registerExportTaskGraphIslandMapPass() {
  PassRegistration<ExportTaskGraphIslandMapPass>();
}

} // namespace sculptor
} // namespace mlir
