#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphNetworkTiming.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

namespace mlir {
namespace sculptor {
namespace task_timing {
namespace {

struct PlacementInfo {
  int64_t meshRows = 0;
  int64_t meshCols = 0;
  llvm::SmallVector<int64_t, 16> coreByTask;
};

struct TransferTiming {
  int64_t hops = 0;
  double finishNs = 0.0;
  double latencyNs = 0.0;
  double contentionDelayNs = 0.0;
};

struct LinkReservation {
  double startCycle = 0.0;
  double endCycle = 0.0;
};

struct TimingEvent {
  enum class Kind : unsigned { TaskFinish = 0, EdgeArrival = 1 };

  double timeNs = 0.0;
  Kind kind = Kind::TaskFinish;
  unsigned index = 0;
};

struct LaterTimingEvent {
  bool operator()(const TimingEvent &lhs, const TimingEvent &rhs) const {
    if (lhs.timeNs != rhs.timeNs)
      return lhs.timeNs > rhs.timeNs;
    if (lhs.kind != rhs.kind)
      return static_cast<unsigned>(lhs.kind) > static_cast<unsigned>(rhs.kind);
    return lhs.index > rhs.index;
  }
};

static uint64_t getDirectedEdgeKey(unsigned producer, unsigned consumer) {
  return (static_cast<uint64_t>(producer) << 32) |
         static_cast<uint64_t>(consumer);
}

static FailureOr<std::optional<PlacementInfo>>
collectPlacementInfo(func::FuncOp taskGraphFunc,
                     const task_graph::TaskGraphDAG &dag) {
  auto meshRows = taskGraphFunc->getAttrOfType<IntegerAttr>(
      schedule_attrs::kMeshRowsAttrName);
  auto meshCols = taskGraphFunc->getAttrOfType<IntegerAttr>(
      schedule_attrs::kMeshColsAttrName);
  if (!meshRows && !meshCols)
    return std::optional<PlacementInfo>{};
  if (!meshRows || !meshCols || meshRows.getInt() <= 0 ||
      meshCols.getInt() <= 0) {
    return taskGraphFunc.emitError(
        "expected positive mesh dimensions for placement-aware timing");
  }

  PlacementInfo placement;
  bool foundCore = false;
  placement.coreByTask.reserve(dag.nodes.size());
  for (const task_graph::TaskGraphNode &node : dag.nodes) {
    auto coreId =
        node.op->getAttrOfType<IntegerAttr>(runtime_attrs::kTaskCoreIdAttrName);
    if (!coreId) {
      placement.coreByTask.push_back(-1);
      continue;
    }
    foundCore = true;
    placement.coreByTask.push_back(coreId.getInt());
  }

  if (!foundCore)
    return std::optional<PlacementInfo>{};

  for (auto indexedCore : llvm::enumerate(placement.coreByTask)) {
    if (indexedCore.value() >= 0)
      continue;
    taskGraphFunc.emitError("expected every task to have a core ID for "
                            "placement-aware timing; task ")
        << indexedCore.index() << " is unassigned";
    return failure();
  }

  placement.meshRows = meshRows.getInt();
  placement.meshCols = meshCols.getInt();

  int64_t meshCapacity = placement.meshRows * placement.meshCols;
  for (auto indexedCore : llvm::enumerate(placement.coreByTask)) {
    if (indexedCore.value() >= 0 && indexedCore.value() < meshCapacity)
      continue;
    taskGraphFunc.emitError("task core ID is outside the configured mesh for "
                            "task ")
        << indexedCore.index();
    return failure();
  }
  return std::optional<PlacementInfo>{std::move(placement)};
}

class NetworkLinkScheduler {
public:
  NetworkLinkScheduler(const PlacementInfo &placement, const TimingModel &model)
      : placement(placement), model(model) {}

  TransferTiming schedule(int64_t sourceCore, int64_t destinationCore,
                          int64_t bytes, double readyNs) {
    TransferTiming timing;
    timing.finishNs = readyNs;
    if (sourceCore == destinationCore || bytes <= 0)
      return timing;

    llvm::SmallVector<std::pair<int64_t, int64_t>, 16> route =
        buildXYRoute(sourceCore, destinationCore);
    timing.hops = route.size();
    if (route.empty())
      return timing;

    double readyCycles = readyNs * model.digitalClockGHz;
    double flits = std::ceil(static_cast<double>(bytes) * 8.0 /
                             model.networkLinkBitsPerCycle);
    double hopLatency = model.networkHopLatencyCycles;
    double completionCycles = readyCycles;

    if (model.networkPipelined) {
      double firstFlitArrival = readyCycles;
      for (auto [from, to] : route) {
        uint64_t link = getDirectedEdgeKey(static_cast<unsigned>(from),
                                           static_cast<unsigned>(to));
        double start = reserveLink(link, firstFlitArrival, flits);
        firstFlitArrival = start + hopLatency;
        completionCycles = start + flits + hopLatency - 1.0;
      }
    } else {
      double current = readyCycles;
      for (auto [from, to] : route) {
        uint64_t link = getDirectedEdgeKey(static_cast<unsigned>(from),
                                           static_cast<unsigned>(to));
        double start = reserveLink(link, current, flits);
        completionCycles = start + flits + hopLatency - 1.0;
        current = completionCycles;
      }
    }

    double actualCycles = std::max(0.0, completionCycles - readyCycles);
    timing.finishNs = cyclesToNanoseconds(completionCycles, model);
    timing.latencyNs = cyclesToNanoseconds(actualCycles, model);
    double idealLatencyNs = estimateNetworkTransferLatencyNs(
        bytes, static_cast<int64_t>(route.size()), model);
    timing.contentionDelayNs =
        std::max(0.0, timing.latencyNs - idealLatencyNs);
    return timing;
  }

private:
  double reserveLink(uint64_t link, double earliestCycle,
                     double durationCycles) {
    llvm::SmallVector<LinkReservation, 8> &reservations =
        reservationsByLink[link];
    double start = earliestCycle;
    for (const LinkReservation &reservation : reservations) {
      if (start + durationCycles <= reservation.startCycle)
        break;
      if (start < reservation.endCycle)
        start = reservation.endCycle;
    }

    auto insertionPoint =
        llvm::lower_bound(reservations, start,
                          [](const LinkReservation &reservation, double cycle) {
                            return reservation.startCycle < cycle;
                          });
    reservations.insert(insertionPoint,
                        LinkReservation{start, start + durationCycles});
    return start;
  }

  llvm::SmallVector<std::pair<int64_t, int64_t>, 16>
  buildXYRoute(int64_t sourceCore, int64_t destinationCore) const {
    llvm::SmallVector<std::pair<int64_t, int64_t>, 16> route;
    int64_t current = sourceCore;
    int64_t currentRow = current / placement.meshCols;
    int64_t currentCol = current % placement.meshCols;
    int64_t destinationRow = destinationCore / placement.meshCols;
    int64_t destinationCol = destinationCore % placement.meshCols;

    while (currentCol != destinationCol) {
      int64_t next = current + (currentCol < destinationCol ? 1 : -1);
      route.push_back({current, next});
      current = next;
      currentCol += currentCol < destinationCol ? 1 : -1;
    }
    while (currentRow != destinationRow) {
      int64_t next =
          current + (currentRow < destinationRow ? placement.meshCols
                                                 : -placement.meshCols);
      route.push_back({current, next});
      current = next;
      currentRow += currentRow < destinationRow ? 1 : -1;
    }
    return route;
  }

  const PlacementInfo &placement;
  const TimingModel &model;
  llvm::DenseMap<uint64_t, llvm::SmallVector<LinkReservation, 8>>
      reservationsByLink;
};

static LogicalResult applyPlacementAwareNetworkTiming(
    func::FuncOp taskGraphFunc, const task_graph::TaskGraphDAG &dag,
    const PlacementInfo &placement, const TimingModel &model,
    TimingAnalysis &analysis) {
  analysis.placementAware = true;
  analysis.totalNetworkLatencyNs = 0.0;
  analysis.totalNetworkContentionDelayNs = 0.0;

  llvm::SmallVector<unsigned, 16> pendingEdges(dag.nodes.size(), 0);
  llvm::SmallVector<double, 16> readyTimeNs(dag.nodes.size(), 0.0);
  std::priority_queue<TimingEvent, std::vector<TimingEvent>, LaterTimingEvent>
      events;

  for (unsigned taskIndex = 0; taskIndex < dag.nodes.size(); ++taskIndex) {
    TaskTiming &timing = analysis.tasks[taskIndex];
    timing.earliestStartNs = 0.0;
    timing.earliestFinishNs = 0.0;
    timing.criticalPathRemainingNs = 0.0;
    timing.incomingNetworkDelayNs = 0.0;
    timing.isCritical = false;
    pendingEdges[taskIndex] = analysis.incomingEdges[taskIndex].size();
    if (pendingEdges[taskIndex] != 0)
      continue;
    timing.earliestFinishNs = timing.intrinsicLatencyNs;
    events.push(TimingEvent{timing.earliestFinishNs,
                            TimingEvent::Kind::TaskFinish, taskIndex});
  }

  for (ExecutionEdge &edge : analysis.executionEdges) {
    edge.sourceCore = placement.coreByTask[edge.producerTask];
    edge.destinationCore = placement.coreByTask[edge.consumerTask];
    edge.meshHops = 0;
    edge.transferStartNs = 0.0;
    edge.transferFinishNs = 0.0;
    edge.networkLatencyNs = 0.0;
    edge.contentionDelayNs = 0.0;
  }

  NetworkLinkScheduler network(placement, model);
  unsigned completedTasks = 0;
  unsigned arrivedEdges = 0;
  while (!events.empty()) {
    TimingEvent event = events.top();
    events.pop();

    if (event.kind == TimingEvent::Kind::TaskFinish) {
      ++completedTasks;
      for (unsigned edgeIndex : analysis.outgoingEdges[event.index]) {
        ExecutionEdge &edge = analysis.executionEdges[edgeIndex];
        edge.transferStartNs = event.timeNs;
        TransferTiming transfer = network.schedule(
            edge.sourceCore, edge.destinationCore,
            edge.dataDependency ? edge.transferredBytes : 0, event.timeNs);
        edge.meshHops = transfer.hops;
        edge.transferFinishNs = transfer.finishNs;
        edge.networkLatencyNs = transfer.latencyNs;
        edge.contentionDelayNs = transfer.contentionDelayNs;
        analysis.totalNetworkLatencyNs += transfer.latencyNs;
        analysis.totalNetworkContentionDelayNs += transfer.contentionDelayNs;
        events.push(TimingEvent{transfer.finishNs,
                                TimingEvent::Kind::EdgeArrival, edgeIndex});
      }
      continue;
    }

    ++arrivedEdges;
    const ExecutionEdge &edge = analysis.executionEdges[event.index];
    unsigned consumer = edge.consumerTask;
    readyTimeNs[consumer] = std::max(readyTimeNs[consumer], event.timeNs);
    if (--pendingEdges[consumer] != 0)
      continue;

    TaskTiming &timing = analysis.tasks[consumer];
    timing.earliestStartNs = readyTimeNs[consumer];
    double latestPredecessorFinishNs = 0.0;
    for (unsigned incomingEdgeIndex : analysis.incomingEdges[consumer]) {
      const ExecutionEdge &incoming =
          analysis.executionEdges[incomingEdgeIndex];
      latestPredecessorFinishNs =
          std::max(latestPredecessorFinishNs,
                   analysis.tasks[incoming.producerTask].earliestFinishNs);
    }
    timing.incomingNetworkDelayNs =
        std::max(0.0, timing.earliestStartNs - latestPredecessorFinishNs);
    timing.earliestFinishNs =
        timing.earliestStartNs + timing.intrinsicLatencyNs;
    events.push(TimingEvent{timing.earliestFinishNs,
                            TimingEvent::Kind::TaskFinish, consumer});
  }

  if (completedTasks != dag.nodes.size() ||
      arrivedEdges != analysis.executionEdges.size()) {
    return taskGraphFunc.emitError(
        "failed to complete placement-aware timing event simulation");
  }
  return success();
}

} // namespace

LogicalResult applyPlacementAwareNetworkTimingIfAvailable(
    func::FuncOp taskGraphFunc, const task_graph::TaskGraphDAG &dag,
    const TimingModel &model, TimingAnalysis &analysis) {
  auto placement = collectPlacementInfo(taskGraphFunc, dag);
  if (failed(placement))
    return failure();
  if (!*placement)
    return success();
  return applyPlacementAwareNetworkTiming(taskGraphFunc, dag, **placement,
                                          model, analysis);
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
