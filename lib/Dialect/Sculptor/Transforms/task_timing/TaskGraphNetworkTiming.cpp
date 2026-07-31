#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphNetworkTiming.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphRuntimeOrder.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TimingCostModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace sculptor {
namespace task_timing {
namespace {

constexpr double kTimingEpsilon = 1.0e-9;

enum class ReplayMode {
  Full,
  NoContention,
  ZeroNetwork,
};

struct PlacementInfo {
  int64_t meshRows = 0;
  int64_t meshCols = 0;
  llvm::SmallVector<int64_t, 16> coreByTask;
  llvm::SmallVector<unsigned, 16> localRuntimeIndexByTask;
};

struct CausalReference {
  int64_t task = -1;
  int64_t edge = -1;
  std::string resource;
};

struct TransferTiming {
  int64_t hops = 0;
  int64_t payloadWords = 0;
  int64_t protocolWords = 0;
  double injectionStartNs = 0.0;
  double injectionFinishNs = 0.0;
  double routeArrivalNs = 0.0;
  double receiveStartNs = 0.0;
  double receiveCompleteNs = 0.0;
  double idealServiceNs = 0.0;
  double nicQueueDelayNs = 0.0;
  double linkQueueDelayNs = 0.0;
  double receiveQueueDelayNs = 0.0;
  CausalReference parent;
};

struct ResourceAvailability {
  double cycle = 0.0;
  int64_t ownerEdge = -1;
};

struct LinkReservation {
  double startCycle = 0.0;
  double endCycle = 0.0;
  int64_t ownerEdge = -1;
};

struct ReservationResult {
  double startCycle = 0.0;
  int64_t blockerEdge = -1;
};

struct SimulationEvent {
  enum class Kind : unsigned {
    TaskFinish = 0,
    EdgeArrival = 1,
  };

  double timeNs = 0.0;
  Kind kind = Kind::TaskFinish;
  unsigned index = 0;
};

struct LaterSimulationEvent {
  bool operator()(const SimulationEvent &lhs,
                  const SimulationEvent &rhs) const {
    if (lhs.timeNs != rhs.timeNs)
      return lhs.timeNs > rhs.timeNs;
    if (lhs.kind != rhs.kind)
      return static_cast<unsigned>(lhs.kind) > static_cast<unsigned>(rhs.kind);
    return lhs.index > rhs.index;
  }
};

struct CoreState {
  bool busy = false;
  int64_t runningTask = -1;
  int64_t previousTask = -1;
  unsigned nextFixedExecutionRank = 0;
  double availableNs = 0.0;
  llvm::SmallVector<unsigned, 8> readyTasks;
};

struct SimulationResult {
  llvm::SmallVector<TaskTiming, 16> tasks;
  llvm::SmallVector<ExecutionEdge, 16> edges;
  llvm::SmallVector<CausalTimingEvent, 16> causalCriticalChain;
  double makespanNs = 0.0;
  double sumTaskWorkNs = 0.0;
  double sumCoreQueueDelayNs = 0.0;
  double sumEdgeNetworkServiceNs = 0.0;
  double sumEdgeNetworkQueueDelayNs = 0.0;
  double sumNicQueueDelayNs = 0.0;
  double sumLinkQueueDelayNs = 0.0;
  double sumReceiveQueueDelayNs = 0.0;
  int64_t totalPayloadWords = 0;
  int64_t totalProtocolWords = 0;
  int64_t totalWordHops = 0;
  llvm::SmallVector<unsigned, 16> executionRankByTask;
};

static uint64_t getDirectedEdgeKey(unsigned source, unsigned destination) {
  return (static_cast<uint64_t>(source) << 32) |
         static_cast<uint64_t>(destination);
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
  placement.meshRows = meshRows.getInt();
  placement.meshCols = meshCols.getInt();
  if (placement.meshRows >
      std::numeric_limits<int64_t>::max() / placement.meshCols)
    return taskGraphFunc.emitError("task timing mesh capacity overflow");
  int64_t meshCapacity = placement.meshRows * placement.meshCols;

  placement.coreByTask.reserve(dag.nodes.size());
  for (const task_graph::TaskGraphNode &node : dag.nodes) {
    auto coreId =
        node.op->getAttrOfType<IntegerAttr>(runtime_attrs::kTaskCoreIdAttrName);
    if (!coreId) {
      if (placement.coreByTask.empty())
        return std::optional<PlacementInfo>{};
      taskGraphFunc.emitError(
          "expected every task to have a core ID for placement-aware timing; "
          "task ")
          << node.index << " is unassigned";
      return failure();
    }
    if (coreId.getInt() < 0 || coreId.getInt() >= meshCapacity) {
      taskGraphFunc.emitError("task core ID is outside the configured timing "
                              "mesh for task ")
          << node.index;
      return failure();
    }
    placement.coreByTask.push_back(coreId.getInt());
  }

  placement.localRuntimeIndexByTask =
      task_graph::buildLocalRuntimeOrder(placement.coreByTask);
  return std::optional<PlacementInfo>{std::move(placement)};
}

class NetworkResourceScheduler {
public:
  NetworkResourceScheduler(const PlacementInfo &placement,
                           const TimingModel &model, ReplayMode mode)
      : placement(placement), model(model), mode(mode) {}

  FailureOr<TransferTiming> schedule(unsigned edgeIndex, int64_t sourceCore,
                                     int64_t destinationCore, int64_t bytes,
                                     double producerFinishNs) {
    TransferTiming timing;
    timing.injectionStartNs = producerFinishNs;
    timing.injectionFinishNs = producerFinishNs;
    timing.routeArrivalNs = producerFinishNs;
    timing.receiveStartNs = producerFinishNs;
    timing.receiveCompleteNs = producerFinishNs;

    if (sourceCore == destinationCore || bytes <= 0 ||
        mode == ReplayMode::ZeroNetwork)
      return timing;

    if (bytes > std::numeric_limits<int64_t>::max() / 8)
      return emitOverflow(edgeIndex, "route payload bit count");
    int64_t payloadBits = bytes * 8;
    timing.payloadWords = divideCeil(payloadBits, model.networkLinkWordBits);
    timing.protocolWords = model.protocolWordsPerRoute;
    if (timing.payloadWords >
        std::numeric_limits<int64_t>::max() - timing.protocolWords)
      return emitOverflow(edgeIndex, "route word count");
    int64_t totalWords = timing.payloadWords + timing.protocolWords;

    task_schedulers::HardwareBudget budget;
    budget.meshRows = placement.meshRows;
    budget.meshCols = placement.meshCols;
    llvm::SmallVector<std::pair<int64_t, int64_t>, 16> route =
        task_schedulers::buildMeshXYRoute(sourceCore, destinationCore, budget);
    timing.hops = route.size();
    if (route.empty())
      return timing;

    double readyCycle = producerFinishNs * model.digitalClockGHz;
    double injectionCycles = std::ceil(static_cast<double>(totalWords) /
                                       model.nicInjectionWordsPerCycle);
    double linkCycles =
        std::ceil(static_cast<double>(totalWords) * model.networkLinkWordBits /
                  model.networkLinkBitsPerCycle);
    double routeWordCycles = std::max(injectionCycles, linkCycles);
    double receiveCycles =
        std::ceil(static_cast<double>(totalWords) / model.rxDmaWordsPerCycle);
    double hopLatency = model.networkHopLatencyCycles;

    double injectionStart = readyCycle;
    if (mode == ReplayMode::Full) {
      ResourceAvailability &nic = nicByCore[sourceCore];
      if (nic.cycle > injectionStart + kTimingEpsilon) {
        timing.nicQueueDelayNs =
            cyclesToNanoseconds(nic.cycle - injectionStart, model);
        timing.parent.edge = nic.ownerEdge;
        timing.parent.resource = "source-nic:" + std::to_string(sourceCore);
        injectionStart = nic.cycle;
      }
      nic = ResourceAvailability{injectionStart + injectionCycles,
                                 static_cast<int64_t>(edgeIndex)};
    }
    double injectionFinish = injectionStart + injectionCycles;

    double currentHeadCycle = injectionStart;
    double routeArrivalCycle = injectionStart;
    for (auto [source, destination] : route) {
      double linkStart = currentHeadCycle;
      if (mode == ReplayMode::Full) {
        ReservationResult reservation =
            reserveLink(getDirectedEdgeKey(source, destination), linkStart,
                        routeWordCycles, edgeIndex);
        if (reservation.startCycle > linkStart + kTimingEpsilon) {
          timing.linkQueueDelayNs +=
              cyclesToNanoseconds(reservation.startCycle - linkStart, model);
          timing.parent.edge = reservation.blockerEdge;
          timing.parent.resource = "directed-link:" + std::to_string(source) +
                                   "->" + std::to_string(destination);
        }
        linkStart = reservation.startCycle;
      }
      routeArrivalCycle =
          linkStart + routeWordCycles + std::max(0.0, hopLatency - 1.0);
      currentHeadCycle =
          model.networkPipelined ? linkStart + hopLatency : routeArrivalCycle;
    }

    double receiveStart = routeArrivalCycle;
    if (mode == ReplayMode::Full) {
      ResourceAvailability &receiver = rxByCore[destinationCore];
      if (receiver.cycle > receiveStart + kTimingEpsilon) {
        timing.receiveQueueDelayNs =
            cyclesToNanoseconds(receiver.cycle - receiveStart, model);
        timing.parent.edge = receiver.ownerEdge;
        timing.parent.resource =
            "receive-dma:" + std::to_string(destinationCore);
        receiveStart = receiver.cycle;
      }
      receiver = ResourceAvailability{receiveStart + receiveCycles,
                                      static_cast<int64_t>(edgeIndex)};
    }
    double receiveComplete = receiveStart + receiveCycles;

    double idealRouteArrival = readyCycle + routeWordCycles +
                               static_cast<double>(route.size()) * hopLatency -
                               1.0;
    if (!model.networkPipelined)
      idealRouteArrival = readyCycle + static_cast<double>(route.size()) *
                                           (routeWordCycles + hopLatency - 1.0);
    double idealReceiveComplete =
        std::max(readyCycle + injectionCycles, idealRouteArrival) +
        receiveCycles;

    timing.injectionStartNs = cyclesToNanoseconds(injectionStart, model);
    timing.injectionFinishNs = cyclesToNanoseconds(injectionFinish, model);
    timing.routeArrivalNs = cyclesToNanoseconds(routeArrivalCycle, model);
    timing.receiveStartNs = cyclesToNanoseconds(receiveStart, model);
    timing.receiveCompleteNs = cyclesToNanoseconds(receiveComplete, model);
    timing.idealServiceNs =
        cyclesToNanoseconds(idealReceiveComplete - readyCycle, model);
    return timing;
  }

private:
  static int64_t divideCeil(int64_t numerator, int64_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
  }

  FailureOr<TransferTiming> emitOverflow(unsigned edgeIndex,
                                         StringRef quantity) const {
    anchor->emitError("timing overflow while calculating ")
        << quantity << " for execution edge " << edgeIndex;
    return failure();
  }

  ReservationResult reserveLink(uint64_t link, double earliestCycle,
                                double durationCycles, unsigned edgeIndex) {
    llvm::SmallVector<LinkReservation, 8> &reservations =
        reservationsByLink[link];
    double start = earliestCycle;
    int64_t blocker = -1;
    for (const LinkReservation &reservation : reservations) {
      if (start + durationCycles <= reservation.startCycle + kTimingEpsilon)
        break;
      if (start < reservation.endCycle - kTimingEpsilon) {
        start = reservation.endCycle;
        blocker = reservation.ownerEdge;
      }
    }

    auto insertionPoint =
        llvm::lower_bound(reservations, start,
                          [](const LinkReservation &reservation, double cycle) {
                            return reservation.startCycle < cycle;
                          });
    reservations.insert(insertionPoint,
                        LinkReservation{start, start + durationCycles,
                                        static_cast<int64_t>(edgeIndex)});
    return ReservationResult{start, blocker};
  }

public:
  void setDiagnosticAnchor(Operation *operation) { anchor = operation; }

private:
  const PlacementInfo &placement;
  const TimingModel &model;
  ReplayMode mode;
  Operation *anchor = nullptr;
  llvm::DenseMap<int64_t, ResourceAvailability> nicByCore;
  llvm::DenseMap<int64_t, ResourceAvailability> rxByCore;
  llvm::DenseMap<uint64_t, llvm::SmallVector<LinkReservation, 8>>
      reservationsByLink;
};

static void resetDynamicTaskTiming(TaskTiming &timing,
                                   unsigned localRuntimeIndex) {
  timing.localRuntimeIndex = localRuntimeIndex;
  timing.earliestStartNs = 0.0;
  timing.earliestFinishNs = 0.0;
  timing.criticalPathRemainingNs = 0.0;
  timing.slackNs = 0.0;
  timing.incomingNetworkDelayNs = 0.0;
  timing.coreQueueDelayNs = 0.0;
  timing.causalInputEdge = -1;
  timing.causalPreviousTask = -1;
  timing.isCritical = false;
}

static void resetDynamicEdgeTiming(ExecutionEdge &edge,
                                   const PlacementInfo &placement) {
  edge.sourceCore = placement.coreByTask[edge.producerTask];
  edge.destinationCore = placement.coreByTask[edge.consumerTask];
  edge.meshHops = 0;
  edge.payloadWords = 0;
  edge.protocolWords = 0;
  edge.transferStartNs = 0.0;
  edge.injectionStartNs = 0.0;
  edge.injectionFinishNs = 0.0;
  edge.routeArrivalNs = 0.0;
  edge.receiveStartNs = 0.0;
  edge.receiveCompleteNs = 0.0;
  edge.transferFinishNs = 0.0;
  edge.networkLatencyNs = 0.0;
  edge.contentionDelayNs = 0.0;
  edge.nicQueueDelayNs = 0.0;
  edge.linkQueueDelayNs = 0.0;
  edge.receiveQueueDelayNs = 0.0;
  edge.causalParentTask = edge.producerTask;
  edge.causalParentEdge = -1;
  edge.causalResource = "producer";
}

static void buildCausalCriticalChain(
    const PlacementInfo &placement,
    llvm::ArrayRef<llvm::SmallVector<unsigned, 4>> outgoingEdges,
    SimulationResult &result) {
  int64_t currentTask = -1;
  double latestFinish = -1.0;
  for (auto indexedTask : llvm::enumerate(result.tasks)) {
    if (!outgoingEdges[indexedTask.index()].empty())
      continue;
    if (indexedTask.value().earliestFinishNs > latestFinish) {
      latestFinish = indexedTask.value().earliestFinishNs;
      currentTask = indexedTask.index();
    }
  }
  if (currentTask < 0) {
    for (auto indexedTask : llvm::enumerate(result.tasks)) {
      if (indexedTask.value().earliestFinishNs > latestFinish) {
        latestFinish = indexedTask.value().earliestFinishNs;
        currentTask = indexedTask.index();
      }
    }
  }

  llvm::SmallVector<CausalTimingEvent, 16> reverseChain;
  llvm::DenseSet<uint64_t> visited;
  int64_t currentEdge = -1;
  while (currentTask >= 0 || currentEdge >= 0) {
    uint64_t key = currentTask >= 0
                       ? static_cast<uint64_t>(currentTask) << 1
                       : (static_cast<uint64_t>(currentEdge) << 1) | 1;
    if (!visited.insert(key).second)
      break;

    if (currentTask >= 0) {
      TaskTiming &task = result.tasks[currentTask];
      task.isCritical = true;
      task.slackNs = 0.0;
      task.criticalPathRemainingNs =
          std::max(0.0, result.makespanNs - task.earliestStartNs);
      reverseChain.push_back(CausalTimingEvent{
          0, "task", currentTask, -1, placement.coreByTask[currentTask],
          task.earliestStartNs, task.earliestFinishNs, -1,
          "core:" + std::to_string(placement.coreByTask[currentTask])});
      if (task.causalPreviousTask >= 0) {
        currentTask = task.causalPreviousTask;
        currentEdge = -1;
      } else {
        currentEdge = task.causalInputEdge;
        currentTask = -1;
      }
      continue;
    }

    ExecutionEdge &edge = result.edges[currentEdge];
    reverseChain.push_back(CausalTimingEvent{
        0, "route", -1, currentEdge, edge.destinationCore, edge.transferStartNs,
        edge.receiveCompleteNs, -1, edge.causalResource});
    if (edge.causalParentEdge >= 0) {
      currentEdge = edge.causalParentEdge;
      currentTask = -1;
    } else {
      currentTask = edge.causalParentTask;
      currentEdge = -1;
    }
  }

  auto reversed = llvm::reverse(reverseChain);
  result.causalCriticalChain.assign(reversed.begin(), reversed.end());
  for (auto indexedEvent : llvm::enumerate(result.causalCriticalChain)) {
    indexedEvent.value().id = indexedEvent.index();
    indexedEvent.value().parentEvent =
        indexedEvent.index() == 0 ? -1 : indexedEvent.index() - 1;
  }
}

static FailureOr<SimulationResult>
simulatePlacement(func::FuncOp taskGraphFunc,
                  const task_graph::TaskGraphDAG &dag,
                  const PlacementInfo &placement, const TimingModel &model,
                  const TimingAnalysis &baseAnalysis, ReplayMode mode,
                  llvm::ArrayRef<unsigned> fixedExecutionRankByTask = {}) {
  SimulationResult result;
  result.tasks = baseAnalysis.tasks;
  result.edges = baseAnalysis.executionEdges;
  result.executionRankByTask.assign(dag.nodes.size(),
                                    std::numeric_limits<unsigned>::max());
  if (!fixedExecutionRankByTask.empty() &&
      fixedExecutionRankByTask.size() != dag.nodes.size()) {
    taskGraphFunc.emitError(
        "expected one fixed execution rank per task during timing replay");
    return failure();
  }

  for (auto indexedTask : llvm::enumerate(result.tasks)) {
    resetDynamicTaskTiming(
        indexedTask.value(),
        placement.localRuntimeIndexByTask[indexedTask.index()]);
    result.sumTaskWorkNs += indexedTask.value().intrinsicLatencyNs;
  }
  for (ExecutionEdge &edge : result.edges)
    resetDynamicEdgeTiming(edge, placement);

  llvm::SmallVector<unsigned, 16> pendingEdges(dag.nodes.size(), 0);
  llvm::SmallVector<double, 16> readyTimeNs(dag.nodes.size(), 0.0);
  llvm::SmallVector<int64_t, 16> readyParentEdge(dag.nodes.size(), -1);
  llvm::DenseMap<int64_t, CoreState> coreStates;
  for (int64_t core : placement.coreByTask)
    coreStates.try_emplace(core);

  std::priority_queue<SimulationEvent, std::vector<SimulationEvent>,
                      LaterSimulationEvent>
      events;
  for (unsigned task = 0; task < dag.nodes.size(); ++task) {
    pendingEdges[task] = baseAnalysis.incomingEdges[task].size();
    if (pendingEdges[task] == 0)
      coreStates[placement.coreByTask[task]].readyTasks.push_back(task);
  }

  NetworkResourceScheduler network(placement, model, mode);
  network.setDiagnosticAnchor(taskGraphFunc);
  llvm::DenseMap<int64_t, unsigned> nextRecordedExecutionRank;

  auto dispatchReadyTasks = [&](double timeNs) {
    for (auto &coreEntry : coreStates) {
      CoreState &core = coreEntry.second;
      if (core.busy || core.readyTasks.empty())
        continue;
      auto selected = core.readyTasks.end();
      if (!fixedExecutionRankByTask.empty()) {
        selected = llvm::find_if(core.readyTasks, [&](unsigned task) {
          return fixedExecutionRankByTask[task] == core.nextFixedExecutionRank;
        });
        if (selected == core.readyTasks.end())
          continue;
      } else {
        selected =
            llvm::min_element(core.readyTasks, [&](unsigned lhs, unsigned rhs) {
              unsigned lhsIndex = placement.localRuntimeIndexByTask[lhs];
              unsigned rhsIndex = placement.localRuntimeIndexByTask[rhs];
              return lhsIndex != rhsIndex ? lhsIndex < rhsIndex : lhs < rhs;
            });
      }
      unsigned taskIndex = *selected;
      core.readyTasks.erase(selected);
      if (!fixedExecutionRankByTask.empty())
        ++core.nextFixedExecutionRank;
      result.executionRankByTask[taskIndex] =
          nextRecordedExecutionRank[placement.coreByTask[taskIndex]]++;
      TaskTiming &task = result.tasks[taskIndex];
      task.earliestStartNs = std::max(timeNs, readyTimeNs[taskIndex]);
      task.coreQueueDelayNs =
          std::max(0.0, task.earliestStartNs - readyTimeNs[taskIndex]);
      result.sumCoreQueueDelayNs += task.coreQueueDelayNs;

      if (core.previousTask >= 0 &&
          result.tasks[core.previousTask].earliestFinishNs + kTimingEpsilon >=
              readyTimeNs[taskIndex]) {
        task.causalPreviousTask = core.previousTask;
      } else {
        task.causalInputEdge = readyParentEdge[taskIndex];
      }
      if (task.causalInputEdge >= 0)
        task.incomingNetworkDelayNs =
            result.edges[task.causalInputEdge].networkLatencyNs;

      task.earliestFinishNs = task.earliestStartNs + task.intrinsicLatencyNs;
      core.busy = true;
      core.runningTask = taskIndex;
      core.availableNs = task.earliestFinishNs;
      events.push(SimulationEvent{
          task.earliestFinishNs, SimulationEvent::Kind::TaskFinish, taskIndex});
    }
  };

  dispatchReadyTasks(0.0);
  unsigned completedTasks = 0;
  unsigned arrivedEdges = 0;
  while (!events.empty()) {
    double eventTime = events.top().timeNs;
    do {
      llvm::SmallVector<SimulationEvent, 8> sameTimeEvents;
      while (!events.empty() &&
             std::abs(events.top().timeNs - eventTime) <= kTimingEpsilon) {
        sameTimeEvents.push_back(events.top());
        events.pop();
      }
      llvm::sort(sameTimeEvents,
                 [](const SimulationEvent &lhs, const SimulationEvent &rhs) {
                   if (lhs.kind != rhs.kind)
                     return static_cast<unsigned>(lhs.kind) <
                            static_cast<unsigned>(rhs.kind);
                   return lhs.index < rhs.index;
                 });

      for (const SimulationEvent &event : sameTimeEvents) {
        if (event.kind == SimulationEvent::Kind::TaskFinish) {
          ++completedTasks;
          CoreState &core = coreStates[placement.coreByTask[event.index]];
          core.busy = false;
          core.runningTask = -1;
          core.previousTask = event.index;
          core.availableNs = event.timeNs;

          for (unsigned edgeIndex : baseAnalysis.outgoingEdges[event.index]) {
            ExecutionEdge &edge = result.edges[edgeIndex];
            edge.transferStartNs = event.timeNs;
            auto transfer = network.schedule(
                edgeIndex, edge.sourceCore, edge.destinationCore,
                edge.dataDependency ? edge.transferredBytes : 0, event.timeNs);
            if (failed(transfer))
              return failure();
            edge.meshHops = transfer->hops;
            edge.payloadWords = transfer->payloadWords;
            edge.protocolWords = transfer->protocolWords;
            edge.injectionStartNs = transfer->injectionStartNs;
            edge.injectionFinishNs = transfer->injectionFinishNs;
            edge.routeArrivalNs = transfer->routeArrivalNs;
            edge.receiveStartNs = transfer->receiveStartNs;
            edge.receiveCompleteNs = transfer->receiveCompleteNs;
            edge.transferFinishNs = transfer->receiveCompleteNs;
            edge.networkLatencyNs =
                std::max(0.0, edge.receiveCompleteNs - event.timeNs);
            edge.nicQueueDelayNs = transfer->nicQueueDelayNs;
            edge.linkQueueDelayNs = transfer->linkQueueDelayNs;
            edge.receiveQueueDelayNs = transfer->receiveQueueDelayNs;
            edge.contentionDelayNs = edge.nicQueueDelayNs +
                                     edge.linkQueueDelayNs +
                                     edge.receiveQueueDelayNs;
            edge.causalParentTask = event.index;
            edge.causalParentEdge = transfer->parent.edge;
            if (edge.causalParentEdge >= 0)
              edge.causalParentTask = -1;
            edge.causalResource = transfer->parent.resource.empty()
                                      ? "producer"
                                      : transfer->parent.resource;

            result.sumEdgeNetworkServiceNs += transfer->idealServiceNs;
            result.sumEdgeNetworkQueueDelayNs += edge.contentionDelayNs;
            result.sumNicQueueDelayNs += edge.nicQueueDelayNs;
            result.sumLinkQueueDelayNs += edge.linkQueueDelayNs;
            result.sumReceiveQueueDelayNs += edge.receiveQueueDelayNs;
            auto accumulate = [&](int64_t &total, int64_t value,
                                  StringRef quantity) -> LogicalResult {
              if (value < 0 ||
                  total > std::numeric_limits<int64_t>::max() - value) {
                taskGraphFunc.emitError("timing overflow while accumulating ")
                    << quantity;
                return failure();
              }
              total += value;
              return success();
            };
            if (failed(accumulate(result.totalPayloadWords, edge.payloadWords,
                                  "payload words")) ||
                failed(accumulate(result.totalProtocolWords, edge.protocolWords,
                                  "protocol words")))
              return failure();
            if (edge.payloadWords >
                std::numeric_limits<int64_t>::max() - edge.protocolWords) {
              taskGraphFunc.emitError(
                  "timing overflow while calculating route word-hops");
              return failure();
            }
            int64_t routeWords = edge.payloadWords + edge.protocolWords;
            if (edge.meshHops > 0 &&
                routeWords >
                    std::numeric_limits<int64_t>::max() / edge.meshHops) {
              taskGraphFunc.emitError(
                  "timing overflow while calculating route word-hops");
              return failure();
            }
            if (failed(accumulate(result.totalWordHops,
                                  routeWords * edge.meshHops, "word-hops")))
              return failure();
            events.push(SimulationEvent{edge.receiveCompleteNs,
                                        SimulationEvent::Kind::EdgeArrival,
                                        edgeIndex});
          }
          continue;
        }

        ++arrivedEdges;
        const ExecutionEdge &edge = result.edges[event.index];
        unsigned consumer = edge.consumerTask;
        if (event.timeNs > readyTimeNs[consumer] + kTimingEpsilon ||
            (std::abs(event.timeNs - readyTimeNs[consumer]) <= kTimingEpsilon &&
             (readyParentEdge[consumer] < 0 ||
              static_cast<int64_t>(event.index) < readyParentEdge[consumer]))) {
          readyTimeNs[consumer] = event.timeNs;
          readyParentEdge[consumer] = event.index;
        }
        if (--pendingEdges[consumer] == 0)
          coreStates[placement.coreByTask[consumer]].readyTasks.push_back(
              consumer);
      }
    } while (!events.empty() &&
             std::abs(events.top().timeNs - eventTime) <= kTimingEpsilon);

    dispatchReadyTasks(eventTime);
  }

  if (completedTasks != dag.nodes.size() ||
      arrivedEdges != result.edges.size()) {
    taskGraphFunc.emitError(
        "failed to complete placement-aware resource timing simulation");
    return failure();
  }

  for (const TaskTiming &task : result.tasks)
    result.makespanNs = std::max(result.makespanNs, task.earliestFinishNs);
  if (mode == ReplayMode::Full)
    buildCausalCriticalChain(placement, baseAnalysis.outgoingEdges, result);
  return result;
}

static LogicalResult applyPlacementAwareResourceTiming(
    func::FuncOp taskGraphFunc, const task_graph::TaskGraphDAG &dag,
    const PlacementInfo &placement, const TimingModel &model,
    TimingAnalysis &analysis) {
  auto full = simulatePlacement(taskGraphFunc, dag, placement, model, analysis,
                                ReplayMode::Full);
  if (failed(full))
    return failure();
  auto noContention =
      simulatePlacement(taskGraphFunc, dag, placement, model, analysis,
                        ReplayMode::NoContention, full->executionRankByTask);
  auto zeroNetwork =
      simulatePlacement(taskGraphFunc, dag, placement, model, analysis,
                        ReplayMode::ZeroNetwork, full->executionRankByTask);
  if (failed(noContention) || failed(zeroNetwork))
    return failure();

  if (full->makespanNs + kTimingEpsilon < noContention->makespanNs ||
      noContention->makespanNs + kTimingEpsilon < zeroNetwork->makespanNs) {
    taskGraphFunc.emitError(
        "counterfactual timing replay violated monotonic makespan ordering");
    return failure();
  }

  analysis.placementAware = true;
  analysis.tasks = std::move(full->tasks);
  analysis.executionEdges = std::move(full->edges);
  analysis.causalCriticalChain = std::move(full->causalCriticalChain);
  analysis.criticalPathNs = full->makespanNs;
  analysis.sumTaskWorkNs = full->sumTaskWorkNs;
  analysis.sumCoreQueueDelayNs = full->sumCoreQueueDelayNs;
  analysis.sumEdgeNetworkServiceNs = full->sumEdgeNetworkServiceNs;
  analysis.sumEdgeNetworkQueueDelayNs = full->sumEdgeNetworkQueueDelayNs;
  analysis.sumNicQueueDelayNs = full->sumNicQueueDelayNs;
  analysis.sumLinkQueueDelayNs = full->sumLinkQueueDelayNs;
  analysis.sumReceiveQueueDelayNs = full->sumReceiveQueueDelayNs;
  analysis.noContentionMakespanNs = noContention->makespanNs;
  analysis.zeroNetworkMakespanNs = zeroNetwork->makespanNs;
  analysis.exposedContentionNs =
      std::max(0.0, full->makespanNs - noContention->makespanNs);
  analysis.exposedTransportNs =
      std::max(0.0, noContention->makespanNs - zeroNetwork->makespanNs);
  analysis.totalPayloadWords = full->totalPayloadWords;
  analysis.totalProtocolWords = full->totalProtocolWords;
  analysis.totalWordHops = full->totalWordHops;
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
  return applyPlacementAwareResourceTiming(taskGraphFunc, dag, **placement,
                                           model, analysis);
}

FailureOr<TimingAnalysis> evaluateTaskGraphPlacementTiming(
    func::FuncOp taskGraphFunc, const task_graph::TaskGraphDAG &dag,
    const TimingModel &model, const TimingAnalysis &baseAnalysis,
    llvm::ArrayRef<int64_t> coreByTask, int64_t meshRows, int64_t meshCols) {
  if (meshRows <= 0 || meshCols <= 0 ||
      meshRows > std::numeric_limits<int64_t>::max() / meshCols) {
    taskGraphFunc.emitError(
        "expected valid mesh dimensions for placement timing evaluation");
    return failure();
  }
  if (coreByTask.size() != dag.nodes.size()) {
    taskGraphFunc.emitError(
        "expected one core assignment per task for placement timing");
    return failure();
  }

  PlacementInfo placement;
  placement.meshRows = meshRows;
  placement.meshCols = meshCols;
  placement.coreByTask.append(coreByTask.begin(), coreByTask.end());
  int64_t meshCapacity = meshRows * meshCols;
  for (auto indexedCore : llvm::enumerate(coreByTask)) {
    if (indexedCore.value() < 0 || indexedCore.value() >= meshCapacity) {
      taskGraphFunc.emitError("task core ID is outside the timing mesh for "
                              "task ")
          << indexedCore.index();
      return failure();
    }
  }
  placement.localRuntimeIndexByTask =
      task_graph::buildLocalRuntimeOrder(placement.coreByTask);

  TimingAnalysis result = baseAnalysis;
  if (failed(applyPlacementAwareResourceTiming(taskGraphFunc, dag, placement,
                                               model, result)))
    return failure();
  return result;
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
