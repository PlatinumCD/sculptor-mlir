#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedySearch.h"

#include "GreedySearchInternals.h"

#include "GreedyHeuristic.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;
namespace greedy = task_schedulers::greedy_detail;

using TaskGraphNode = task_schedulers::TaskGraphNode;
using IslandCommunicationEdge = task_schedulers::LogicalIslandCommunicationEdge;

struct GreedyLookaheadChoice {
  int64_t coreId = -1;
  int64_t score = 0;
};

static mlir::FailureOr<GreedyLookaheadChoice> chooseLookaheadPlacement(
    unsigned setupIndex, int64_t remainingLookahead,
    const greedy::PlacementState &state,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const task_schedulers::HardwareBudget &budget,
    const task_schedulers::GreedyHeuristic &heuristic,
    const greedy::CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland) {
  if (setupIndex >= matrixSetupTasks.size() || remainingLookahead <= 0) {
    return GreedyLookaheadChoice{
        state.currentCore,
        heuristic.evaluate(task_schedulers::GreedyHeuristicContext{
            budget, islandCommunicationEdges, state.coreByPlacedIsland,
            /*activeIsland=*/0, setupIndex,
            static_cast<unsigned>(matrixSetupTasks.size()), firstTaskIsland,
            lastTaskIsland, std::nullopt})};
  }

  llvm::SmallVector<greedy::PlacementState, 16> expandedStates =
      greedy::expandState(state, setupIndex, matrixSetupTasks, budget,
                          heuristic, physicalArraysByCore,
                          islandCommunicationEdges, firstTaskIsland,
                          lastTaskIsland,
                          /*pruneCandidates=*/remainingLookahead > 1);
  if (expandedStates.empty())
    return mlir::failure();

  bool hasBest = false;
  GreedyLookaheadChoice best;
  for (const greedy::PlacementState &expandedState : expandedStates) {
    int64_t candidateScore = expandedState.score;
    if (remainingLookahead > 1 && setupIndex + 1 < matrixSetupTasks.size()) {
      auto futureChoice = chooseLookaheadPlacement(
          setupIndex + 1, remainingLookahead - 1, expandedState,
          matrixSetupTasks, budget, heuristic, physicalArraysByCore,
          islandCommunicationEdges, firstTaskIsland, lastTaskIsland);
      if (mlir::failed(futureChoice))
        continue;
      candidateScore = futureChoice->score;
    }

    if (greedy::isBetterChoice(
            hasBest, candidateScore, expandedState.currentCore, best.score,
            best.coreId, state.currentCore, budget, state.coreByPlacedIsland)) {
      hasBest = true;
      best.coreId = expandedState.currentCore;
      best.score = candidateScore;
    }
  }

  if (!hasBest)
    return mlir::failure();
  return best;
}

static mlir::FailureOr<llvm::SmallVector<greedy::IslandPlacement, 8>>
buildBeamIslandPlacements(
    mlir::func::FuncOp taskGraphFunc,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const task_schedulers::HardwareBudget &budget,
    const task_schedulers::GreedyHeuristic &heuristic,
    const greedy::CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland) {
  unsigned beamWidth = static_cast<unsigned>(budget.greedy.beamWidth);
  llvm::SmallVector<greedy::PlacementState, 8> states;

  greedy::PlacementState initialState;
  initialState.usedSlotsByCore.assign(static_cast<size_t>(budget.numCores), 0);
  states.push_back(std::move(initialState));

  auto compareStates = [&](const greedy::PlacementState &lhs,
                           const greedy::PlacementState &rhs) {
    if (lhs.score != rhs.score)
      return lhs.score < rhs.score;

    int64_t lhsRegionDistance = greedy::minDistanceToPlacedRegion(
        lhs.currentCore, budget, lhs.coreByPlacedIsland);
    int64_t rhsRegionDistance = greedy::minDistanceToPlacedRegion(
        rhs.currentCore, budget, rhs.coreByPlacedIsland);
    if (lhsRegionDistance != rhsRegionDistance)
      return lhsRegionDistance < rhsRegionDistance;
    return lhs.currentCore < rhs.currentCore;
  };

  auto pruneStates =
      [&](llvm::SmallVectorImpl<greedy::PlacementState> &nextStates) {
        llvm::sort(nextStates, compareStates);
        if (nextStates.size() <= beamWidth)
          return;

        llvm::SmallVector<greedy::PlacementState, 8> prunedStates;
        llvm::DenseSet<int64_t> selectedCurrentCores;
        prunedStates.reserve(beamWidth);

        for (const greedy::PlacementState &state : nextStates) {
          if (prunedStates.size() >= beamWidth)
            break;
          if (!selectedCurrentCores.insert(state.currentCore).second)
            continue;
          prunedStates.push_back(state);
        }

        for (const greedy::PlacementState &state : nextStates) {
          if (prunedStates.size() >= beamWidth)
            break;
          prunedStates.push_back(state);
        }
        nextStates = std::move(prunedStates);
      };

  for (auto indexedSetup : llvm::enumerate(matrixSetupTasks)) {
    unsigned setupIndex = static_cast<unsigned>(indexedSetup.index());
    llvm::SmallVector<greedy::PlacementState, 16> nextStates;

    for (const greedy::PlacementState &state : states) {
      llvm::SmallVector<greedy::PlacementState, 16> expandedStates =
          greedy::expandState(state, setupIndex, matrixSetupTasks, budget,
                              heuristic, physicalArraysByCore,
                              islandCommunicationEdges, firstTaskIsland,
                              lastTaskIsland, /*pruneCandidates=*/true);
      nextStates.append(std::make_move_iterator(expandedStates.begin()),
                        std::make_move_iterator(expandedStates.end()));
    }

    if (nextStates.empty()) {
      taskGraphFunc.emitError("failed to find an available core for greedy "
                              "beam island placement");
      return mlir::failure();
    }

    pruneStates(nextStates);
    states = std::move(nextStates);
  }

  llvm::sort(states, compareStates);
  return states.front().islandPlacements;
}

static mlir::FailureOr<llvm::SmallVector<greedy::IslandPlacement, 8>>
buildIslandPlacements(
    mlir::func::FuncOp taskGraphFunc,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const task_schedulers::HardwareBudget &budget,
    const task_schedulers::GreedyHeuristic &heuristic,
    llvm::ArrayRef<int64_t> physicalArrayOrder,
    llvm::ArrayRef<IslandCommunicationEdge> islandCommunicationEdges,
    std::optional<unsigned> firstTaskIsland,
    std::optional<unsigned> lastTaskIsland) {
  if (matrixSetupTasks.empty())
    return llvm::SmallVector<greedy::IslandPlacement, 8>();

  if (budget.topology != "mesh" || budget.meshRows <= 0 ||
      budget.meshCols <= 0 || budget.numCores <= 0 ||
      budget.arraysPerCore <= 0) {
    taskGraphFunc.emitError("expected greedy island placement to use a "
                            "non-empty mesh hardware budget");
    return mlir::failure();
  }

  auto physicalArraysByCore = greedy::buildCorePhysicalArraySlots(
      taskGraphFunc.getOperation(), budget, physicalArrayOrder);
  if (mlir::failed(physicalArraysByCore))
    return mlir::failure();

  llvm::SmallVector<unsigned, 16> emptyUsedSlotsByCore(
      static_cast<size_t>(budget.numCores), 0);
  if ((*physicalArraysByCore)[0].empty()) {
    taskGraphFunc.emitError("expected greedy island placement to have an "
                            "available analog array on core 0");
    return mlir::failure();
  }

  bool repairBoundary = budget.greedy.boundaryRegret;
  if (budget.greedy.beamWidth > 1) {
    auto beamIslandPlacements = buildBeamIslandPlacements(
        taskGraphFunc, matrixSetupTasks, budget, heuristic,
        *physicalArraysByCore, islandCommunicationEdges, firstTaskIsland,
        lastTaskIsland);
    if (mlir::failed(beamIslandPlacements))
      return mlir::failure();

    if (repairBoundary)
      greedy::repairBoundaryRegretPlacement(
          *beamIslandPlacements, budget, physicalArrayOrder,
          islandCommunicationEdges, firstTaskIsland, lastTaskIsland);
    return *beamIslandPlacements;
  }

  greedy::PlacementState state;
  state.usedSlotsByCore.assign(static_cast<size_t>(budget.numCores), 0);
  state.islandPlacements.reserve(matrixSetupTasks.size());
  for (auto indexedSetup : llvm::enumerate(matrixSetupTasks)) {
    const TaskGraphNode *setupNode = indexedSetup.value();
    unsigned island = setupNode->index;

    auto selected = chooseLookaheadPlacement(
        static_cast<unsigned>(indexedSetup.index()), budget.greedy.lookahead,
        state, matrixSetupTasks, budget, heuristic, *physicalArraysByCore,
        islandCommunicationEdges, firstTaskIsland, lastTaskIsland);
    if (mlir::failed(selected)) {
      taskGraphFunc.emitError("failed to find an available core for greedy "
                              "island placement");
      return mlir::failure();
    }

    if (!greedy::applyCandidate(state, *setupNode, island, selected->coreId,
                                *physicalArraysByCore)) {
      taskGraphFunc.emitError("failed to commit selected greedy island "
                              "placement candidate");
      return mlir::failure();
    }
    state.score = selected->score;
  }

  if (repairBoundary)
    greedy::repairBoundaryRegretPlacement(
        state.islandPlacements, budget, physicalArrayOrder,
        islandCommunicationEdges, firstTaskIsland, lastTaskIsland);
  return state.islandPlacements;
}

static std::optional<unsigned> lookupIslandForTask(
    const task_schedulers::LogicalPlacementIslandGraph &islandGraph,
    unsigned taskIndex) {
  auto islandIt = islandGraph.islandByTaskIndex.find(taskIndex);
  if (islandIt == islandGraph.islandByTaskIndex.end())
    return std::nullopt;
  return islandIt->second;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<IslandPlacementPlan>
buildGreedyPlacementPlan(const TaskGraphPlacementProblem &problem,
                         llvm::ArrayRef<int64_t> physicalArrayOrder) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks =
      collectMatrixSetupTasks(problem.dag);
  if (!matrixSetupTasks.empty() && physicalArrayOrder.empty()) {
    problem.diagnosticOp->emitError(
        "expected matrix setup island placement to have at least one physical "
        "analog array");
    return failure();
  }

  CompositeGreedyHeuristic heuristic(problem.budget.greedy.specification,
                                     problem.budget.greedy.boundaryRegret,
                                     problem.budget.greedy.compactRegion);

  std::optional<unsigned> firstTaskIsland;
  std::optional<unsigned> lastTaskIsland;
  if (!problem.dag.nodes.empty()) {
    firstTaskIsland = lookupIslandForTask(problem.islandGraph,
                                          problem.dag.nodes.front().index);
    lastTaskIsland = lookupIslandForTask(problem.islandGraph,
                                         problem.dag.nodes.back().index);
  }

  auto islandPlacements = buildIslandPlacements(
      problem.taskGraphFunc, matrixSetupTasks, problem.budget, heuristic,
      physicalArrayOrder, problem.islandGraph.communicationEdges,
      firstTaskIsland, lastTaskIsland);
  if (failed(islandPlacements))
    return failure();

  if (islandPlacements->size() != problem.islandGraph.islands.size()) {
    problem.diagnosticOp->emitError(
        "expected greedy placement to assign every logical island");
    return failure();
  }

  IslandPlacementPlan plan;
  plan.physicalArrayByIsland.reserve(islandPlacements->size());
  for (auto indexedPlacement : llvm::enumerate(*islandPlacements)) {
    if (indexedPlacement.value().island !=
        problem.islandGraph.islands[indexedPlacement.index()].islandIndex) {
      problem.diagnosticOp->emitError(
          "expected greedy placement order to match logical island order");
      return failure();
    }
    plan.physicalArrayByIsland.push_back(
        indexedPlacement.value().physicalArrayId);
  }
  if (failed(validatePlacementPlan(problem, plan)))
    return failure();
  return plan;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
