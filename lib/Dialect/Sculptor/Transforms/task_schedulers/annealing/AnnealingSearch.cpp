#include "AnnealingSearch.h"

#include "AnnealingMoves.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedySearch.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPhysicalArrayOrders.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementObjective.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;
namespace annealing = task_schedulers::annealing_detail;

struct PlacementScore {
  int64_t transferCost = 0;
  int64_t boundaryPenalty = 0;
  int64_t totalScore = 0;
};

struct SearchState {
  annealing::Placement placement;
  PlacementScore score;
};

static mlir::FailureOr<annealing::Placement> buildGreedyInitialPlacement(
    const task_schedulers::TaskGraphPlacementProblem &problem,
    const task_schedulers::GreedyScheduleConfig &greedyConfig) {
  auto plan = task_schedulers::buildGreedyPlacementPlan(
      problem, problem.budget.analogArrays, greedyConfig);
  if (mlir::failed(plan))
    return mlir::failure();

  llvm::DenseSet<int64_t> usedPhysicalArrays;
  annealing::Placement placement;
  placement.physicalArrayOrder.reserve(problem.budget.analogArrays.size());
  for (int64_t physicalArrayId : plan->physicalArrayByIsland) {
    placement.physicalArrayOrder.push_back(physicalArrayId);
    usedPhysicalArrays.insert(physicalArrayId);
  }

  for (int64_t physicalArrayId : problem.budget.analogArrays) {
    if (usedPhysicalArrays.contains(physicalArrayId))
      continue;
    placement.physicalArrayOrder.push_back(physicalArrayId);
  }
  return placement;
}

static mlir::FailureOr<annealing::Placement>
buildInitialPlacement(const task_schedulers::TaskGraphPlacementProblem &problem,
                      const task_schedulers::AnnealingScheduleConfig &config,
                      const task_schedulers::GreedyScheduleConfig &greedyConfig,
                      int64_t randomSeed) {
  if (problem.budget.analogArrays.empty()) {
    problem.diagnosticOp->emitError(
        "expected simulated annealing initial placement to have at least one "
        "analog array");
    return mlir::failure();
  }

  annealing::Placement placement;
  switch (config.initialSchedule) {
  case task_schedulers::AnnealingInitialSchedule::Identity:
    placement.physicalArrayOrder =
        task_schedulers::buildIdentityPhysicalArrayOrder(problem.budget);
    return placement;
  case task_schedulers::AnnealingInitialSchedule::Random:
    placement.physicalArrayOrder =
        task_schedulers::buildRandomPhysicalArrayOrder(problem.budget,
                                                       randomSeed);
    return placement;
  case task_schedulers::AnnealingInitialSchedule::Snake:
    placement.physicalArrayOrder =
        task_schedulers::buildSnakePhysicalArrayOrder(problem.budget);
    return placement;
  case task_schedulers::AnnealingInitialSchedule::Greedy:
    return buildGreedyInitialPlacement(problem, greedyConfig);
  }

  problem.diagnosticOp->emitError(
      "unknown simulated annealing initial schedule");
  return mlir::failure();
}

static mlir::FailureOr<PlacementScore> estimatePlacementScore(
    const annealing::Placement &placement,
    const task_schedulers::TaskGraphPlacementProblem &problem,
    const task_schedulers::IslandPlacementObjective &objective) {
  auto plan = task_schedulers::buildPlacementPlanFromPhysicalArrayOrder(
      problem, placement.physicalArrayOrder);
  if (mlir::failed(plan))
    return mlir::failure();

  auto sharedScore = objective.evaluate(*plan);
  if (mlir::failed(sharedScore))
    return mlir::failure();
  return PlacementScore{sharedScore->transferCost, sharedScore->boundaryPenalty,
                        sharedScore->total};
}

static bool shouldAcceptCandidate(const PlacementScore &currentScore,
                                  const PlacementScore &candidateScore,
                                  double temperature,
                                  std::mt19937 &randomEngine) {
  int64_t delta = candidateScore.totalScore - currentScore.totalScore;
  if (delta <= 0)
    return true;
  if (temperature <= 0.0)
    return false;

  double probability = std::exp(-static_cast<double>(delta) / temperature);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(randomEngine) < probability;
}

static double resolveInitialTemperature(
    const task_schedulers::AnnealingScheduleConfig &config,
    const PlacementScore &initialScore) {
  if (config.initialTemperature > 0.0)
    return config.initialTemperature;

  double scoreScale =
      std::max(1.0, static_cast<double>(initialScore.totalScore));
  return std::max(config.finalTemperature * 2.0, scoreScale);
}

static mlir::FailureOr<annealing::Placement>
runSearch(const task_schedulers::TaskGraphPlacementProblem &problem,
          const task_schedulers::AnnealingScheduleConfig &config,
          const task_schedulers::GreedyScheduleConfig &greedyConfig,
          int64_t randomSeed) {
  auto initialPlacement =
      buildInitialPlacement(problem, config, greedyConfig, randomSeed);
  if (mlir::failed(initialPlacement))
    return mlir::failure();

  task_schedulers::IslandPlacementObjective objective(problem);
  auto initialScore =
      estimatePlacementScore(*initialPlacement, problem, objective);
  if (mlir::failed(initialScore))
    return mlir::failure();

  SearchState current{std::move(*initialPlacement), *initialScore};
  SearchState best = current;
  std::mt19937 randomEngine(static_cast<uint32_t>(randomSeed));

  for (double temperature = resolveInitialTemperature(config, *initialScore);
       temperature > config.finalTemperature;
       temperature *= config.coolingRate) {
    for (int64_t step = 0; step < config.stepsPerTemperature; ++step) {
      auto candidatePlacement = annealing::perturbPlacement(
          problem.taskGraphFunc, current.placement, problem.dag,
          problem.islandGraph, config.moveKinds, config.moveRadius,
          randomEngine);
      if (mlir::failed(candidatePlacement))
        return mlir::failure();

      auto candidateScore =
          estimatePlacementScore(*candidatePlacement, problem, objective);
      if (mlir::failed(candidateScore))
        return mlir::failure();

      if (shouldAcceptCandidate(current.score, *candidateScore, temperature,
                                randomEngine)) {
        current.placement = std::move(*candidatePlacement);
        current.score = *candidateScore;
      }
      if (current.score.totalScore < best.score.totalScore)
        best = current;
    }
  }
  return best.placement;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace annealing_detail {

FailureOr<IslandPlacementPlan>
buildPlacementPlan(const TaskGraphPlacementProblem &problem,
                   const AnnealingScheduleConfig &config,
                   const GreedyScheduleConfig &greedyInitialPlacement,
                   int64_t randomSeed) {
  auto placement =
      runSearch(problem, config, greedyInitialPlacement, randomSeed);
  if (failed(placement))
    return failure();
  return buildPlacementPlanFromPhysicalArrayOrder(
      problem, placement->physicalArrayOrder);
}

} // namespace annealing_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
