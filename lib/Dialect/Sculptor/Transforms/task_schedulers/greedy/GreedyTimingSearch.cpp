#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedySearch.h"

#include "GreedyHeuristic.h"
#include "GreedySearchEngine.h"
#include "GreedySearchInternals.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"

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
  double predictedMakespanNs = 0.0;
  double criticalCommunicationNs = 0.0;
  double maximumResourceWorkNs = 0.0;
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
  llvm::ArrayRef<IslandAffinityEdge> affinityEdges;
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
    if (island.matrixSetupTaskIndex >= problem.dag.nodes.size() ||
        !model.timingByIsland.contains(island.islandIndex)) {
      problem.diagnosticOp->emitError(
          "expected each logical island to have setup and timing metadata");
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
    state.score.criticalCommunicationNs += exposedLatencyNs * pressure;
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
    expanded.score.maximumResourceWorkNs =
        std::max(expanded.score.maximumResourceWorkNs, analogWork);
  }

  double digitalFinishNs = dataReadyNs;
  if (timing.digitalWorkNs > 0.0) {
    digitalFinishNs =
        std::max(dataReadyNs, expanded.digitalReadyByCore[candidateCore]) +
        timing.digitalWorkNs;
    expanded.digitalReadyByCore[candidateCore] = digitalFinishNs;
    expanded.digitalWorkByCore[candidateCore] += timing.digitalWorkNs;
    expanded.score.maximumResourceWorkNs =
        std::max(expanded.score.maximumResourceWorkNs,
                 expanded.digitalWorkByCore[candidateCore]);
  }

  double intrinsicSpanNs =
      std::max(0.0, timing.earliestFinishNs - timing.earliestStartNs);
  double finishNs = std::max(
      {analogFinishNs, digitalFinishNs, dataReadyNs + intrinsicSpanNs});
  expanded.finishByIsland[island] = finishNs;
  expanded.score.predictedMakespanNs =
      std::max(expanded.score.predictedMakespanNs, finishNs);
  expanded.score.legacyScore = expanded.placement.score;
  addNewCriticalCommunication(island, candidateCore, expanded, context);
  return expanded;
}

static bool isBetterTimingScore(const TimingPlacementScore &candidate,
                                const TimingPlacementScore &current) {
  if (candidate.predictedMakespanNs != current.predictedMakespanNs)
    return candidate.predictedMakespanNs < current.predictedMakespanNs;
  if (candidate.criticalCommunicationNs != current.criticalCommunicationNs)
    return candidate.criticalCommunicationNs < current.criticalCommunicationNs;
  if (candidate.maximumResourceWorkNs != current.maximumResourceWorkNs)
    return candidate.maximumResourceWorkNs < current.maximumResourceWorkNs;
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
  greedy::ExpansionRequest request{
      timing->islandId, placementIndex,
      static_cast<unsigned>(context.problem.islandGraph.islands.size()),
      /*pruneCandidates=*/false};

  llvm::SmallVector<greedy::PlacementState, 16> placementStates =
      greedy::expandState(state.placement, request, context.problem.budget,
                          context.config, context.legacyHeuristic,
                          context.physicalArraysByCore, context.affinityEdges,
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

  task_schedulers::IslandPlacementPlan plan;
  plan.physicalArrayByIsland.reserve(
      context.problem.islandGraph.islands.size());
  for (const auto &island : context.problem.islandGraph.islands) {
    auto physicalArray = physicalArrayByIsland.find(island.islandIndex);
    if (physicalArray == physicalArrayByIsland.end()) {
      context.problem.diagnosticOp->emitError(
          "expected greedy-timing to place every logical island");
      return mlir::failure();
    }
    plan.physicalArrayByIsland.push_back(physicalArray->second);
  }
  if (mlir::failed(
          task_schedulers::validatePlacementPlan(context.problem, plan)))
    return mlir::failure();
  return plan;
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
  auto physicalArraysByCore = greedy::buildCorePhysicalArraySlots(
      problem.diagnosticOp, problem.budget, physicalArrayOrder);
  if (failed(physicalArraysByCore))
    return failure();

  CompositeGreedyHeuristic legacyHeuristic(
      config.specification, config.boundaryRegret, config.compactRegion);
  TimingSearchContext context{problem,
                              timingProfile,
                              config,
                              *model,
                              legacyHeuristic,
                              *physicalArraysByCore,
                              problem.islandGraph.affinityGraph.edges};
  TimingPlacementState initialState;
  initialState.placement.usedSlotsByCore.assign(
      static_cast<size_t>(problem.budget.numCores), 0);
  initialState.digitalReadyByCore.assign(
      static_cast<size_t>(problem.budget.numCores), 0.0);
  initialState.digitalWorkByCore.assign(
      static_cast<size_t>(problem.budget.numCores), 0.0);

  auto isComplete = [&](const TimingPlacementState &state) {
    return state.placement.islandPlacements.size() ==
           problem.islandGraph.islands.size();
  };
  auto expand = [&](const TimingPlacementState &state, bool) {
    return expandTimingState(state, context);
  };
  FailureOr<TimingPlacementState> finalState =
      config.beamWidth > 1
          ? greedy::runBeamSearch(
                std::move(initialState),
                static_cast<unsigned>(config.beamWidth), isComplete, expand,
                [](llvm::SmallVectorImpl<TimingPlacementState> &states,
                   unsigned beamWidth) {
                  llvm::sort(states, isBetterTimingState);
                  if (states.size() > beamWidth)
                    states.resize(beamWidth);
                })
          : greedy::runLookaheadSearch<TimingPlacementState,
                                       TimingPlacementScore>(
                std::move(initialState), config.lookahead, isComplete, expand,
                [](const TimingPlacementState &state) { return state.score; },
                [](const TimingPlacementScore &candidateScore,
                   const TimingPlacementState &candidate, bool hasBest,
                   const TimingPlacementScore &bestScore,
                   const TimingPlacementState &best,
                   const TimingPlacementState &) {
                  if (!hasBest ||
                      isBetterTimingScore(candidateScore, bestScore))
                    return true;
                  if (isBetterTimingScore(bestScore, candidateScore))
                    return false;
                  return candidate.placement.currentCore <
                         best.placement.currentCore;
                });
  if (failed(finalState)) {
    problem.diagnosticOp->emitError(
        "failed to find a timing-aware greedy island placement");
    return failure();
  }
  return buildFinalPlacementPlan(*finalState, context, physicalArrayOrder);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
