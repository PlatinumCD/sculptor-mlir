#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedySearch.h"

#include "GreedySearchEngine.h"
#include "GreedySearchInternals.h"

#include "GreedyHeuristic.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;
namespace greedy = task_schedulers::greedy_detail;

using TaskGraphNode = task_schedulers::TaskGraphNode;
using IslandAffinityEdge = task_schedulers::IslandAffinityEdge;
using IslandExecutionEdge = mlir::sculptor::task_graph::IslandExecutionEdge;

static bool
comparePlacementStates(const greedy::PlacementState &lhs,
                       const greedy::PlacementState &rhs,
                       const task_schedulers::HardwareBudget &budget) {
  if (lhs.score != rhs.score)
    return lhs.score < rhs.score;

  int64_t lhsRegionDistance = greedy::minDistanceToPlacedRegion(
      lhs.currentCore, budget, lhs.coreByPlacedIsland);
  int64_t rhsRegionDistance = greedy::minDistanceToPlacedRegion(
      rhs.currentCore, budget, rhs.coreByPlacedIsland);
  if (lhsRegionDistance != rhsRegionDistance)
    return lhsRegionDistance < rhsRegionDistance;
  return lhs.currentCore < rhs.currentCore;
}

static void
prunePlacementBeam(llvm::SmallVectorImpl<greedy::PlacementState> &states,
                   unsigned beamWidth,
                   const task_schedulers::HardwareBudget &budget) {
  llvm::sort(states, [&](const greedy::PlacementState &lhs,
                         const greedy::PlacementState &rhs) {
    return comparePlacementStates(lhs, rhs, budget);
  });
  if (states.size() <= beamWidth)
    return;

  llvm::SmallVector<greedy::PlacementState, 8> selected;
  llvm::SmallVector<bool, 16> wasSelected(states.size(), false);
  llvm::DenseSet<int64_t> selectedCurrentCores;
  selected.reserve(beamWidth);
  for (auto indexedState : llvm::enumerate(states)) {
    if (selected.size() >= beamWidth)
      break;
    if (!selectedCurrentCores.insert(indexedState.value().currentCore).second)
      continue;
    selected.push_back(indexedState.value());
    wasSelected[indexedState.index()] = true;
  }
  for (auto indexedState : llvm::enumerate(states)) {
    if (selected.size() >= beamWidth)
      break;
    if (!wasSelected[indexedState.index()])
      selected.push_back(indexedState.value());
  }
  states = std::move(selected);
}

static mlir::FailureOr<llvm::SmallVector<greedy::IslandPlacement, 8>>
buildIslandPlacements(
    mlir::func::FuncOp taskGraphFunc,
    llvm::ArrayRef<const TaskGraphNode *> matrixSetupTasks,
    const task_schedulers::HardwareBudget &budget,
    const task_schedulers::GreedyScheduleConfig &config,
    const task_schedulers::GreedyHeuristic &heuristic,
    llvm::ArrayRef<int64_t> physicalArrayOrder,
    llvm::ArrayRef<IslandAffinityEdge> islandAffinityEdges,
    llvm::ArrayRef<IslandExecutionEdge> islandExecutionEdges,
    const task_schedulers::PlacementConstraints &constraints) {
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

  if (physicalArrayOrder.empty()) {
    taskGraphFunc.emitError("expected greedy island placement to have an "
                            "analog array");
    return mlir::failure();
  }

  auto isComplete = [&](const greedy::PlacementState &state) {
    return state.islandPlacements.size() == matrixSetupTasks.size();
  };
  auto expand = [&](const greedy::PlacementState &state,
                    bool recursivePruning) {
    unsigned placementIndex = state.islandPlacements.size();
    if (placementIndex >= matrixSetupTasks.size())
      return llvm::SmallVector<greedy::PlacementState, 16>{};
    unsigned island = matrixSetupTasks[placementIndex]->index;
    greedy::ExpansionRequest request{
        island, placementIndex, static_cast<unsigned>(matrixSetupTasks.size()),
        recursivePruning};
    return greedy::expandState(state, request, budget, config, heuristic,
                               *physicalArraysByCore, islandAffinityEdges,
                               islandExecutionEdges, constraints);
  };

  greedy::PlacementState initialState;
  initialState.usedSlotsByCore.assign(static_cast<size_t>(budget.numCores), 0);
  initialState.islandPlacements.reserve(matrixSetupTasks.size());
  initialState.currentCore = physicalArrayOrder.front() / budget.arraysPerCore;

  mlir::FailureOr<greedy::PlacementState> finalState =
      config.beamWidth > 1
          ? greedy::runBeamSearch(
                std::move(initialState),
                static_cast<unsigned>(config.beamWidth), isComplete, expand,
                [&](llvm::SmallVectorImpl<greedy::PlacementState> &states,
                    unsigned beamWidth) {
                  prunePlacementBeam(states, beamWidth, budget);
                })
          : greedy::runLookaheadSearch<greedy::PlacementState, int64_t>(
                std::move(initialState), config.lookahead, isComplete, expand,
                [](const greedy::PlacementState &state) { return state.score; },
                [&](int64_t candidateScore,
                    const greedy::PlacementState &candidate, bool hasBest,
                    int64_t bestScore, const greedy::PlacementState &best,
                    const greedy::PlacementState &parent) {
                  return greedy::isBetterChoice(
                      hasBest, candidateScore, candidate.currentCore, bestScore,
                      best.currentCore, parent.currentCore, budget,
                      parent.coreByPlacedIsland);
                });
  if (mlir::failed(finalState)) {
    taskGraphFunc.emitError(
        "failed to find an available core for greedy island placement");
    return mlir::failure();
  }

  if (config.boundaryRegret)
    greedy::repairBoundaryRegretPlacement(finalState->islandPlacements, budget,
                                          physicalArrayOrder,
                                          islandAffinityEdges, constraints);
  return finalState->islandPlacements;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<IslandPlacementPlan>
buildGreedyPlacementPlan(const TaskGraphPlacementProblem &problem,
                         llvm::ArrayRef<int64_t> physicalArrayOrder,
                         const GreedyScheduleConfig &config) {
  llvm::SmallVector<const TaskGraphNode *, 8> matrixSetupTasks =
      collectMatrixSetupTasks(problem.dag);
  if (!matrixSetupTasks.empty() && physicalArrayOrder.empty()) {
    problem.diagnosticOp->emitError(
        "expected matrix setup island placement to have at least one physical "
        "analog array");
    return failure();
  }

  CompositeGreedyHeuristic heuristic(
      config.specification, config.boundaryRegret, config.compactRegion,
      config.spatialSharedLinkPressure);

  auto resources = buildIslandPlacementResources(problem, physicalArrayOrder);
  if (failed(resources))
    return failure();

  auto islandPlacements = buildIslandPlacements(
      problem.taskGraphFunc, matrixSetupTasks, problem.budget, config,
      heuristic, resources->analogPhysicalArrayOrder,
      problem.islandGraph.affinityGraph.edges,
      problem.islandGraph.executionGraph.edges, problem.constraints);
  if (failed(islandPlacements))
    return failure();

  if (islandPlacements->size() != matrixSetupTasks.size()) {
    problem.diagnosticOp->emitError(
        "expected greedy placement to assign every logical island");
    return failure();
  }

  llvm::DenseMap<unsigned, int64_t> physicalArrayByAnalogIsland;
  for (const greedy::IslandPlacement &placement : *islandPlacements)
    physicalArrayByAnalogIsland[placement.island] = placement.physicalArrayId;
  return buildPlacementPlanFromAnalogPlacements(problem, *resources,
                                                physicalArrayByAnalogIsland);
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
