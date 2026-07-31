#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedySearch.h"

#include "GreedyHeuristic.h"
#include "GreedySearchEngine.h"
#include "GreedySearchInternals.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingAnalysis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;
namespace task_timing = mlir::sculptor::task_timing;
namespace task_graph = mlir::sculptor::task_graph;
namespace greedy = task_schedulers::greedy_detail;

using IslandAffinityEdge = task_schedulers::IslandAffinityEdge;
using IslandTimingProfile = task_timing::IslandTimingProfile;
using TimedIslandEdge = task_timing::TimedIslandEdge;

constexpr unsigned kMaxTimingLookaheadCandidates = 8;

static uint64_t getDirectedIslandEdgeKey(unsigned producer, unsigned consumer) {
  return (static_cast<uint64_t>(producer) << 32) |
         static_cast<uint64_t>(consumer);
}

struct TimingSearchModel {
  llvm::DenseMap<unsigned, const IslandTimingProfile *> timingByIsland;
  llvm::DenseMap<uint64_t, const TimedIslandEdge *> timingByDirectedEdge;
};

struct TimingPlacementScore {
  double completionTimeProxy = 0.0;
  double communicationProxy = 0.0;
  double resourceLoadProxy = 0.0;
  int64_t legacyScore = 0;
};

struct TimingPlacementState {
  greedy::PlacementState placement;
  llvm::DenseMap<unsigned, double> finishByIsland;
  llvm::DenseMap<int64_t, double> analogReadyByPhysicalArray;
  llvm::DenseMap<int64_t, double> analogWorkByPhysicalArray;
  llvm::SmallVector<double, 16> digitalReadyByCore;
  llvm::SmallVector<double, 16> digitalWorkByCore;
  TimingPlacementScore score;
};

struct TimingSearchContext {
  const task_schedulers::TaskGraphPlacementProblem &problem;
  const task_timing::SchedulingTimingProfile &timingProfile;
  const task_schedulers::GreedyScheduleConfig &config;
  const TimingSearchModel &model;
  const task_schedulers::GreedyHeuristic &legacyHeuristic;
  const greedy::CorePhysicalArraySlots &physicalArraysByCore;
  const task_schedulers::IslandPlacementResources &placementResources;
  llvm::ArrayRef<IslandAffinityEdge> affinityEdges;
  unsigned analogIslandCount = 0;
};

static mlir::FailureOr<TimingSearchModel> buildTimingSearchModel(
    const task_schedulers::TaskGraphPlacementProblem &problem,
    const task_timing::SchedulingTimingProfile &timingProfile) {
  TimingSearchModel model;
  for (const IslandTimingProfile &timing : timingProfile.islands) {
    if (!model.timingByIsland.try_emplace(timing.islandId, &timing).second) {
      problem.diagnosticOp->emitError(
          "expected one timing profile per logical placement island");
      return mlir::failure();
    }
  }

  for (const auto &island : problem.islandGraph.islands) {
    if (!model.timingByIsland.contains(island.islandIndex) ||
        (task_graph::isAnalogIsland(island) &&
         (!island.matrixSetupTaskIndex ||
          *island.matrixSetupTaskIndex >= problem.dag.nodes.size()))) {
      problem.diagnosticOp->emitError(
          "expected each logical island to have valid timing metadata");
      return mlir::failure();
    }
  }

  for (const TimedIslandEdge &edge : timingProfile.islandEdges) {
    model.timingByDirectedEdge[getDirectedIslandEdgeKey(
        edge.producerIsland, edge.consumerIsland)] = &edge;
  }
  return model;
}

static bool isHigherTimingPriority(const IslandTimingProfile &candidate,
                                   const IslandTimingProfile &current) {
  if (candidate.isCritical != current.isCritical)
    return candidate.isCritical;
  if (candidate.criticalPathRemainingNs != current.criticalPathRemainingNs)
    return candidate.criticalPathRemainingNs > current.criticalPathRemainingNs;
  if (candidate.slackNs != current.slackNs)
    return candidate.slackNs < current.slackNs;
  if (candidate.totalWorkNs != current.totalWorkNs)
    return candidate.totalWorkNs > current.totalWorkNs;
  if (candidate.earliestStartNs != current.earliestStartNs)
    return candidate.earliestStartNs < current.earliestStartNs;
  return candidate.islandId < current.islandId;
}

static const IslandTimingProfile *
selectNextIsland(const TimingPlacementState &state,
                 const TimingSearchContext &context) {
  const IslandTimingProfile *best = nullptr;
  for (const auto &island : context.problem.islandGraph.islands) {
    if (!task_graph::isAnalogIsland(island))
      continue;
    unsigned islandId = island.islandIndex;
    if (state.placement.coreByPlacedIsland.contains(islandId))
      continue;

    const IslandTimingProfile *timing =
        context.model.timingByIsland.lookup(islandId);
    if (!timing)
      continue;
    if (!best || isHigherTimingPriority(*timing, *best))
      best = timing;
  }

  // Placement order is not execution order. In particular, an island can own
  // both early analog work and a late digital fan-in, so requiring all of its
  // predecessors to be placed first would hide an important spatial anchor.
  return best;
}

static const TimedIslandEdge *
findTimedEdge(unsigned lhs, unsigned rhs, const TimingSearchContext &context) {
  return context.model.timingByDirectedEdge.lookup(
      getDirectedIslandEdgeKey(lhs, rhs));
}

static double estimateTransferLatencyNs(int64_t sourceCore,
                                        int64_t destinationCore,
                                        const TimedIslandEdge *edge,
                                        const TimingSearchContext &context) {
  if (!edge || edge->bytes <= 0 || sourceCore == destinationCore)
    return 0.0;

  int64_t hops = task_schedulers::getMeshDistance(sourceCore, destinationCore,
                                                  context.problem.budget);
  if (hops <= 0)
    return 0.0;

  return edge->estimatedTransferNsPerHop +
         edge->estimatedAdditionalHopNs * static_cast<double>(hops - 1);
}

static double getIncomingDataReadyTime(unsigned island, int64_t candidateCore,
                                       const TimingPlacementState &state,
                                       const TimingSearchContext &context) {
  const IslandTimingProfile *timing =
      context.model.timingByIsland.lookup(island);
  double dataReadyNs = timing ? timing->earliestStartNs : 0.0;
  const auto &predecessorMap =
      context.problem.islandGraph.executionGraph.predecessors;
  auto predecessors = predecessorMap.find(island);
  if (predecessors == predecessorMap.end())
    return dataReadyNs;

  for (unsigned predecessor : predecessors->second) {
    auto finishIt = state.finishByIsland.find(predecessor);
    auto coreIt = state.placement.coreByPlacedIsland.find(predecessor);
    if (finishIt == state.finishByIsland.end() ||
        coreIt == state.placement.coreByPlacedIsland.end())
      continue;
    double transferNs = estimateTransferLatencyNs(
        coreIt->second, candidateCore,
        findTimedEdge(predecessor, island, context), context);
    dataReadyNs = std::max(dataReadyNs, finishIt->second + transferNs);
  }
  return dataReadyNs;
}

static void addNewCriticalCommunication(unsigned island, int64_t candidateCore,
                                        TimingPlacementState &state,
                                        const TimingSearchContext &context) {
  for (const TimedIslandEdge &edge : context.timingProfile.islandEdges) {
    unsigned otherIsland = std::numeric_limits<unsigned>::max();
    if (edge.producerIsland == island)
      otherIsland = edge.consumerIsland;
    else if (edge.consumerIsland == island)
      otherIsland = edge.producerIsland;
    else
      continue;

    auto otherCore = state.placement.coreByPlacedIsland.find(otherIsland);
    if (otherCore == state.placement.coreByPlacedIsland.end())
      continue;

    double latencyNs = estimateTransferLatencyNs(otherCore->second,
                                                 candidateCore, &edge, context);
    const IslandTimingProfile *otherTiming =
        context.model.timingByIsland.lookup(otherIsland);
    const IslandTimingProfile *activeTiming =
        context.model.timingByIsland.lookup(island);
    double availableSlackNs =
        std::min(otherTiming ? otherTiming->slackNs : 0.0,
                 activeTiming ? activeTiming->slackNs : 0.0);
    double exposedLatencyNs = std::max(0.0, latencyNs - availableSlackNs);
    double pressure = std::max(edge.criticality, edge.consumerTimingPressure);
    state.score.communicationProxy += exposedLatencyNs * pressure;
  }
}

static TimingPlacementState
applyTimingEstimate(const TimingPlacementState &state,
                    greedy::PlacementState expandedPlacement, unsigned island,
                    const IslandTimingProfile &timing,
                    const TimingSearchContext &context) {
  TimingPlacementState expanded = state;
  expanded.placement = std::move(expandedPlacement);

  int64_t candidateCore = expanded.placement.currentCore;
  int64_t physicalArrayId =
      expanded.placement.islandPlacements.back().physicalArrayId;
  double dataReadyNs =
      getIncomingDataReadyTime(island, candidateCore, state, context);

  double analogFinishNs = dataReadyNs;
  if (timing.analogWorkNs > 0.0) {
    analogFinishNs =
        std::max(dataReadyNs,
                 expanded.analogReadyByPhysicalArray.lookup(physicalArrayId)) +
        timing.analogWorkNs;
    expanded.analogReadyByPhysicalArray[physicalArrayId] = analogFinishNs;
    double &analogWork = expanded.analogWorkByPhysicalArray[physicalArrayId];
    analogWork += timing.analogWorkNs;
    expanded.score.resourceLoadProxy =
        std::max(expanded.score.resourceLoadProxy, analogWork);
  }

  double digitalFinishNs = dataReadyNs;
  if (timing.digitalWorkNs > 0.0) {
    digitalFinishNs =
        std::max(dataReadyNs, expanded.digitalReadyByCore[candidateCore]) +
        timing.digitalWorkNs;
    expanded.digitalReadyByCore[candidateCore] = digitalFinishNs;
    expanded.digitalWorkByCore[candidateCore] += timing.digitalWorkNs;
    expanded.score.resourceLoadProxy =
        std::max(expanded.score.resourceLoadProxy,
                 expanded.digitalWorkByCore[candidateCore]);
  }

  double intrinsicSpanNs =
      std::max(0.0, timing.earliestFinishNs - timing.earliestStartNs);
  double finishNs = std::max(
      {analogFinishNs, digitalFinishNs, dataReadyNs + intrinsicSpanNs});
  expanded.finishByIsland[island] = finishNs;
  expanded.score.completionTimeProxy =
      std::max(expanded.score.completionTimeProxy, finishNs);
  expanded.score.legacyScore = expanded.placement.score;
  addNewCriticalCommunication(island, candidateCore, expanded, context);
  return expanded;
}

static bool isBetterTimingScore(const TimingPlacementScore &candidate,
                                const TimingPlacementScore &current) {
  if (candidate.completionTimeProxy != current.completionTimeProxy)
    return candidate.completionTimeProxy < current.completionTimeProxy;
  if (candidate.communicationProxy != current.communicationProxy)
    return candidate.communicationProxy < current.communicationProxy;
  if (candidate.resourceLoadProxy != current.resourceLoadProxy)
    return candidate.resourceLoadProxy < current.resourceLoadProxy;
  return candidate.legacyScore < current.legacyScore;
}

static bool isBetterTimingState(const TimingPlacementState &candidate,
                                const TimingPlacementState &current) {
  if (isBetterTimingScore(candidate.score, current.score))
    return true;
  if (isBetterTimingScore(current.score, candidate.score))
    return false;
  return candidate.placement.currentCore < current.placement.currentCore;
}

static llvm::SmallVector<TimingPlacementState, 8>
expandTimingState(const TimingPlacementState &state,
                  const TimingSearchContext &context) {
  llvm::SmallVector<TimingPlacementState, 8> timingStates;
  const IslandTimingProfile *timing = selectNextIsland(state, context);
  if (!timing)
    return timingStates;

  unsigned placementIndex = state.placement.islandPlacements.size();
  greedy::ExpansionRequest request{timing->islandId, placementIndex,
                                   context.analogIslandCount,
                                   /*pruneCandidates=*/false};

  llvm::SmallVector<greedy::PlacementState, 16> placementStates =
      greedy::expandState(state.placement, request, context.problem.budget,
                          context.config, context.legacyHeuristic,
                          context.physicalArraysByCore, context.affinityEdges,
                          context.problem.islandGraph.executionGraph.edges,
                          context.problem.constraints);
  timingStates.reserve(placementStates.size());
  for (greedy::PlacementState &placementState : placementStates) {
    timingStates.push_back(applyTimingEstimate(
        state, std::move(placementState), timing->islandId, *timing, context));
  }
  llvm::sort(timingStates, isBetterTimingState);
  if (timingStates.size() > kMaxTimingLookaheadCandidates)
    timingStates.resize(kMaxTimingLookaheadCandidates);
  return timingStates;
}

static mlir::FailureOr<task_schedulers::IslandPlacementPlan>
buildFinalPlacementPlan(const TimingPlacementState &state,
                        const TimingSearchContext &context,
                        llvm::ArrayRef<int64_t> physicalArrayOrder) {
  llvm::SmallVector<greedy::IslandPlacement, 8> placements =
      state.placement.islandPlacements;
  if (context.config.boundaryRegret) {
    greedy::repairBoundaryRegretPlacement(
        placements, context.problem.budget, physicalArrayOrder,
        context.affinityEdges, context.problem.constraints);
  }

  llvm::DenseMap<unsigned, int64_t> physicalArrayByIsland;
  for (const greedy::IslandPlacement &placement : placements)
    physicalArrayByIsland[placement.island] = placement.physicalArrayId;

  auto plan = task_schedulers::buildPlacementPlanFromAnalogPlacements(
      context.problem, context.placementResources, physicalArrayByIsland);
  if (mlir::failed(plan))
    return mlir::failure();
  plan->searchProxyObjective = task_schedulers::SearchProxyObjective{
      state.score.completionTimeProxy, state.score.communicationProxy,
      state.score.resourceLoadProxy};
  return plan;
}

static task_timing::TimingAnalysis buildBaseTimingAnalysis(
    const task_schedulers::TaskGraphPlacementProblem &problem,
    const task_timing::SchedulingTimingProfile &profile) {
  task_timing::TimingAnalysis analysis;
  analysis.tasks = profile.tasks;
  analysis.incomingEdges = problem.executionGraph.incomingEdges;
  analysis.outgoingEdges = problem.executionGraph.outgoingEdges;
  analysis.topologicalOrder = problem.executionGraph.topologicalOrder;
  analysis.controlEdgeCount = problem.executionGraph.controlEdgeCount;
  analysis.dataEdgeCount = problem.executionGraph.dataEdgeCount;
  analysis.executionDepth = 0;
  analysis.totalDataBytes = problem.executionGraph.totalDataBytes;
  analysis.mvmCostMode = profile.mvmCostMode;
  analysis.executionEdges.reserve(problem.executionGraph.edges.size());
  for (const task_graph::TaskExecutionEdge &edge :
       problem.executionGraph.edges) {
    analysis.executionEdges.push_back(task_timing::ExecutionEdge{
        edge.producerTask, edge.consumerTask, edge.controlDependency,
        edge.dataDependency, edge.transferredBytes});
  }
  return analysis;
}

static mlir::FailureOr<task_timing::TimingAnalysis>
evaluateCompletePlan(const task_schedulers::TaskGraphPlacementProblem &problem,
                     const task_schedulers::IslandPlacementPlan &plan,
                     const task_timing::TimingModel &model,
                     const task_timing::TimingAnalysis &baseAnalysis) {
  llvm::SmallVector<int64_t, 16> coreByTask(problem.dag.nodes.size(), -1);
  for (auto indexedIsland : llvm::enumerate(problem.islandGraph.islands)) {
    int64_t core = plan.placements[indexedIsland.index()].coreId;
    for (unsigned task : indexedIsland.value().taskIndices) {
      if (task >= coreByTask.size()) {
        problem.diagnosticOp->emitError(
            "logical island references an invalid task during final timing "
            "evaluation");
        return mlir::failure();
      }
      if (coreByTask[task] >= 0 && coreByTask[task] != core) {
        problem.diagnosticOp->emitError(
            "task belongs to conflicting logical island placements during "
            "final timing evaluation");
        return mlir::failure();
      }
      coreByTask[task] = core;
    }
  }
  for (auto indexedCore : llvm::enumerate(coreByTask)) {
    if (indexedCore.value() >= 0)
      continue;
    problem.diagnosticOp->emitError(
        "full-deployment timing score excludes task ")
        << indexedCore.index()
        << "; every analog, digital, and reduction task must belong to a "
           "placed island";
    return mlir::failure();
  }

  return task_timing::evaluateTaskGraphPlacementTiming(
      problem.taskGraphFunc, problem.dag, model, baseAnalysis, coreByTask,
      problem.budget.meshRows, problem.budget.meshCols);
}

static bool isBetterFullTiming(
    const task_schedulers::FullDeploymentTimingObjective &candidate,
    const task_schedulers::FullDeploymentTimingObjective &current) {
  if (candidate.makespanNs != current.makespanNs)
    return candidate.makespanNs < current.makespanNs;
  if (candidate.exposedContentionNs != current.exposedContentionNs)
    return candidate.exposedContentionNs < current.exposedContentionNs;
  if (candidate.exposedTransportNs != current.exposedTransportNs)
    return candidate.exposedTransportNs < current.exposedTransportNs;
  return candidate.totalWordHops < current.totalWordHops;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<IslandPlacementPlan> buildGreedyTimingPlacementPlan(
    const TaskGraphPlacementProblem &problem,
    const task_timing::SchedulingTimingProfile &timingProfile,
    llvm::ArrayRef<int64_t> physicalArrayOrder,
    const GreedyScheduleConfig &config) {
  if (problem.islandGraph.islands.empty())
    return IslandPlacementPlan{};
  if (physicalArrayOrder.empty()) {
    problem.diagnosticOp->emitError(
        "expected greedy-timing placement to have at least one physical "
        "analog array");
    return failure();
  }
  if (problem.budget.topology != "mesh" || problem.budget.meshRows <= 0 ||
      problem.budget.meshCols <= 0 || problem.budget.numCores <= 0 ||
      problem.budget.arraysPerCore <= 0) {
    problem.diagnosticOp->emitError(
        "expected greedy-timing placement to use a non-empty mesh");
    return failure();
  }

  auto model = buildTimingSearchModel(problem, timingProfile);
  if (failed(model))
    return failure();
  auto placementResources =
      buildIslandPlacementResources(problem, physicalArrayOrder);
  if (failed(placementResources))
    return failure();
  auto physicalArraysByCore = greedy::buildCorePhysicalArraySlots(
      problem.diagnosticOp, problem.budget,
      placementResources->analogPhysicalArrayOrder);
  if (failed(physicalArraysByCore))
    return failure();

  CompositeGreedyHeuristic legacyHeuristic(
      config.specification, config.boundaryRegret, config.compactRegion,
      config.spatialSharedLinkPressure);
  unsigned analogIslandCount = static_cast<unsigned>(
      llvm::count_if(problem.islandGraph.islands,
                     [](const task_graph::LogicalPlacementIsland &island) {
                       return task_graph::isAnalogIsland(island);
                     }));
  TimingSearchContext context{problem,
                              timingProfile,
                              config,
                              *model,
                              legacyHeuristic,
                              *physicalArraysByCore,
                              *placementResources,
                              problem.islandGraph.affinityGraph.edges,
                              analogIslandCount};
  TimingPlacementState initialState;
  initialState.placement.usedSlotsByCore.assign(
      static_cast<size_t>(problem.budget.numCores), 0);
  initialState.digitalReadyByCore.assign(
      static_cast<size_t>(problem.budget.numCores), 0.0);
  initialState.digitalWorkByCore.assign(
      static_cast<size_t>(problem.budget.numCores), 0.0);
  if (!placementResources->analogPhysicalArrayOrder.empty()) {
    initialState.placement.currentCore =
        placementResources->analogPhysicalArrayOrder.front() /
        problem.budget.arraysPerCore;
  }
  auto isComplete = [&](const TimingPlacementState &state) {
    return state.placement.islandPlacements.size() == analogIslandCount;
  };
  auto expand = [&](const TimingPlacementState &state, bool) {
    return expandTimingState(state, context);
  };
  llvm::SmallVector<TimingPlacementState, 8> finalStates;
  if (config.beamWidth > 1) {
    auto beamStates = greedy::runBeamSearchCandidates(
        std::move(initialState), static_cast<unsigned>(config.beamWidth),
        isComplete, expand,
        [](llvm::SmallVectorImpl<TimingPlacementState> &states,
           unsigned beamWidth) {
          llvm::sort(states, isBetterTimingState);
          if (states.size() > beamWidth)
            states.resize(beamWidth);
        });
    if (succeeded(beamStates))
      finalStates = std::move(*beamStates);
  } else {
    auto finalState =
        greedy::runLookaheadSearch<TimingPlacementState, TimingPlacementScore>(
            std::move(initialState), config.lookahead, isComplete, expand,
            [](const TimingPlacementState &state) { return state.score; },
            [](const TimingPlacementScore &candidateScore,
               const TimingPlacementState &candidate, bool hasBest,
               const TimingPlacementScore &bestScore,
               const TimingPlacementState &best, const TimingPlacementState &) {
              if (!hasBest || isBetterTimingScore(candidateScore, bestScore))
                return true;
              if (isBetterTimingScore(bestScore, candidateScore))
                return false;
              return candidate.placement.currentCore <
                     best.placement.currentCore;
            });
    if (succeeded(finalState))
      finalStates.push_back(std::move(*finalState));
  }
  if (finalStates.empty()) {
    problem.diagnosticOp->emitError(
        "failed to find a timing-aware greedy island placement");
    return failure();
  }
  llvm::sort(finalStates, isBetterTimingState);

  auto timingModel = task_timing::loadTimingModel(problem.taskGraphFunc);
  if (failed(timingModel))
    return failure();
  task_timing::TimingAnalysis baseAnalysis =
      buildBaseTimingAnalysis(problem, timingProfile);

  bool hasBest = false;
  unsigned selectedProxyRank = 0;
  IslandPlacementPlan bestPlan;
  TimingPlacementScore bestProxyScore;
  for (auto indexedState : llvm::enumerate(finalStates)) {
    const TimingPlacementState &state = indexedState.value();
    auto plan = buildFinalPlacementPlan(
        state, context, placementResources->analogPhysicalArrayOrder);
    if (failed(plan))
      return failure();
    auto exact =
        evaluateCompletePlan(problem, *plan, *timingModel, baseAnalysis);
    if (failed(exact))
      return failure();
    FullDeploymentTimingObjective objective{
        exact->criticalPathNs, exact->exposedContentionNs,
        exact->exposedTransportNs, exact->totalWordHops};
    plan->fullTimingObjective = objective;

    bool select = !hasBest;
    if (hasBest) {
      const FullDeploymentTimingObjective &bestObjective =
          *bestPlan.fullTimingObjective;
      if (isBetterFullTiming(objective, bestObjective))
        select = true;
      else if (!isBetterFullTiming(bestObjective, objective) &&
               isBetterTimingScore(state.score, bestProxyScore))
        select = true;
    }
    if (!select)
      continue;
    hasBest = true;
    bestPlan = std::move(*plan);
    bestProxyScore = state.score;
    selectedProxyRank = indexedState.index();
  }
  bestPlan.timingRerankCandidateCount =
      static_cast<unsigned>(finalStates.size());
  bestPlan.timingRerankSelectedProxyRank = selectedProxyRank;
  return bestPlan;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
