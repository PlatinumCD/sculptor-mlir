#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGANALYSIS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGANALYSIS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingProfile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

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
  int64_t payloadWords = 0;
  int64_t protocolWords = 0;
  double transferStartNs = 0.0;
  double injectionStartNs = 0.0;
  double injectionFinishNs = 0.0;
  double routeArrivalNs = 0.0;
  double receiveStartNs = 0.0;
  double receiveCompleteNs = 0.0;
  double transferFinishNs = 0.0;
  double networkLatencyNs = 0.0;
  double contentionDelayNs = 0.0;
  double nicQueueDelayNs = 0.0;
  double linkQueueDelayNs = 0.0;
  double receiveQueueDelayNs = 0.0;
  int64_t causalParentTask = -1;
  int64_t causalParentEdge = -1;
  std::string causalResource;
};

struct CausalTimingEvent {
  unsigned id = 0;
  std::string kind;
  int64_t taskIndex = -1;
  int64_t edgeIndex = -1;
  int64_t coreId = -1;
  double startNs = 0.0;
  double finishNs = 0.0;
  int64_t parentEvent = -1;
  std::string resource;
};

struct TimingAnalysis {
  llvm::SmallVector<ExecutionEdge, 16> executionEdges;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> incomingEdges;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> outgoingEdges;
  llvm::SmallVector<unsigned, 16> topologicalOrder;
  llvm::SmallVector<TaskTiming, 16> tasks;
  llvm::SmallVector<IslandTimingProfile, 16> islands;
  llvm::SmallVector<TimedIslandEdge, 16> timedIslandEdges;
  llvm::SmallVector<CausalTimingEvent, 16> causalCriticalChain;
  unsigned controlEdgeCount = 0;
  unsigned dataEdgeCount = 0;
  unsigned executionDepth = 0;
  int64_t totalDataBytes = 0;
  int64_t totalDigitalReplacementOps = 0;
  double criticalPathNs = 0.0;
  double sumEdgeNetworkServiceNs = 0.0;
  double sumEdgeNetworkQueueDelayNs = 0.0;
  double sumTaskWorkNs = 0.0;
  double sumCoreQueueDelayNs = 0.0;
  double sumNicQueueDelayNs = 0.0;
  double sumLinkQueueDelayNs = 0.0;
  double sumReceiveQueueDelayNs = 0.0;
  double noContentionMakespanNs = 0.0;
  double zeroNetworkMakespanNs = 0.0;
  double exposedTransportNs = 0.0;
  double exposedContentionNs = 0.0;
  int64_t totalPayloadWords = 0;
  int64_t totalProtocolWords = 0;
  int64_t totalWordHops = 0;
  bool placementAware = false;
  MVMCostMode mvmCostMode = MVMCostMode::Analog;
};

FailureOr<TimingAnalysis> analyzeTaskGraphTiming(
    ModuleOp module, func::FuncOp taskGraphFunc,
    const task_graph::TaskGraphDAG &dag,
    const task_graph::TaskExecutionGraph &executionGraph,
    const task_graph::LogicalPlacementIslandGraph &islandGraph,
    const TimingModel &model, MVMCostMode mvmCostMode);

FailureOr<TimingAnalysis> evaluateTaskGraphPlacementTiming(
    func::FuncOp taskGraphFunc, const task_graph::TaskGraphDAG &dag,
    const TimingModel &model, const TimingAnalysis &baseAnalysis,
    llvm::ArrayRef<int64_t> coreByTask, int64_t meshRows, int64_t meshCols);

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGANALYSIS_H
