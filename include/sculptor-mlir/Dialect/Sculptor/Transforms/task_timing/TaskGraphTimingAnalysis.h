#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGANALYSIS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGANALYSIS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {

struct ExecutionEdge {
  unsigned producerTask = 0;
  unsigned consumerTask = 0;
  bool controlDependency = false;
  bool dataDependency = false;
  int64_t transferredBytes = 0;
  int64_t sourceCore = -1;
  int64_t destinationCore = -1;
  int64_t meshHops = 0;
  double transferStartNs = 0.0;
  double transferFinishNs = 0.0;
  double networkLatencyNs = 0.0;
  double contentionDelayNs = 0.0;
};

struct TimingAnalysis {
  llvm::SmallVector<ExecutionEdge, 16> executionEdges;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> incomingEdges;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> outgoingEdges;
  llvm::SmallVector<unsigned, 16> topologicalOrder;
  llvm::SmallVector<TaskTiming, 16> tasks;
  llvm::SmallVector<IslandTimingProfile, 16> islands;
  llvm::SmallVector<TimedIslandEdge, 16> timedIslandEdges;
  unsigned controlEdgeCount = 0;
  unsigned dataEdgeCount = 0;
  unsigned executionDepth = 0;
  int64_t totalDataBytes = 0;
  double criticalPathNs = 0.0;
  double totalNetworkLatencyNs = 0.0;
  double totalNetworkContentionDelayNs = 0.0;
  bool placementAware = false;
};

FailureOr<TimingAnalysis> analyzeTaskGraphTiming(
    ModuleOp module, func::FuncOp taskGraphFunc,
    const task_graph::TaskGraphDAG &dag,
    const task_graph::TaskExecutionGraph &executionGraph,
    const task_graph::LogicalPlacementIslandGraph &islandGraph,
    const TimingModel &model);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGANALYSIS_H
