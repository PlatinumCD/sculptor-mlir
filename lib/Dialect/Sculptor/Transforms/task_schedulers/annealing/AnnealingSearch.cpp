#include "AnnealingSearch.h"

#include "AnnealingMoves.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/GreedySearch.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPhysicalArrayOrders.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementObjective.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <chrono>
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

enum class StopReason {
  Plateau,
  MaximumEvaluations,
  MaximumRuntime,
};

struct SearchResult {
  annealing::Placement placement;
  int64_t epochs = 0;
  int64_t evaluations = 0;
  int64_t initialScore = 0;
  int64_t bestScore = 0;
  double uphillAcceptanceRate = 0.0;
  double searchSeconds = 0.0;
  StopReason stopReason = StopReason::Plateau;
};

static llvm::StringRef stringifyStopReason(StopReason reason) {
  switch (reason) {
  case StopReason::Plateau:
    return "plateau";
  case StopReason::MaximumEvaluations:
    return "maximum-evaluations";
  case StopReason::MaximumRuntime:
    return "maximum-runtime";
  }
  llvm_unreachable("unknown annealing stop reason");
}

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
  for (const auto &islandPlacement : plan->placements) {
    if (!islandPlacement.physicalArrayId)
      continue;
    placement.physicalArrayOrder.push_back(*islandPlacement.physicalArrayId);
    usedPhysicalArrays.insert(*islandPlacement.physicalArrayId);
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

static mlir::FailureOr<SearchResult>
runSearch(const task_schedulers::TaskGraphPlacementProblem &problem,
          const task_schedulers::AnnealingScheduleConfig &config,
          const task_schedulers::GreedyScheduleConfig &greedyConfig,
          int64_t randomSeed) {
  const auto searchStart = std::chrono::steady_clock::now();
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

  int64_t epochs = 0;
  int64_t evaluations = 0;
  int64_t epochsWithoutMeaningfulImprovement = 0;
  int64_t significantBestScore = best.score.totalScore;
  double lastUphillAcceptanceRate = 0.0;
  StopReason stopReason = StopReason::Plateau;
  double temperature = resolveInitialTemperature(config, *initialScore);

  auto elapsedSeconds = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         searchStart)
        .count();
  };
  auto reachedEvaluationLimit = [&]() {
    return config.maximumEvaluations > 0 &&
           evaluations >= config.maximumEvaluations;
  };
  auto reachedRuntimeLimit = [&]() {
    return config.maximumRuntimeSeconds > 0.0 &&
           elapsedSeconds() >= config.maximumRuntimeSeconds;
  };

  bool done = false;
  while (!done) {
    int64_t worseCandidates = 0;
    int64_t acceptedWorseCandidates = 0;
    for (int64_t step = 0; step < config.stepsPerTemperature; ++step) {
      if (reachedEvaluationLimit()) {
        stopReason = StopReason::MaximumEvaluations;
        done = true;
        break;
      }
      if (reachedRuntimeLimit()) {
        stopReason = StopReason::MaximumRuntime;
        done = true;
        break;
      }

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

      ++evaluations;
      bool isWorse = candidateScore->totalScore > current.score.totalScore;
      if (isWorse)
        ++worseCandidates;

      bool accepted = shouldAcceptCandidate(
          current.score, *candidateScore, temperature, randomEngine);
      if (accepted) {
        if (isWorse)
          ++acceptedWorseCandidates;
        current.placement = std::move(*candidatePlacement);
        current.score = *candidateScore;
      }
      if (current.score.totalScore < best.score.totalScore)
        best = current;
    }

    if (done)
      break;

    ++epochs;
    lastUphillAcceptanceRate =
        worseCandidates == 0
            ? 0.0
            : static_cast<double>(acceptedWorseCandidates) /
                  static_cast<double>(worseCandidates);

    int64_t scoreImprovement = significantBestScore - best.score.totalScore;
    double relativeImprovement =
        scoreImprovement <= 0
            ? 0.0
            : static_cast<double>(scoreImprovement) /
                  std::max(1.0,
                           std::abs(static_cast<double>(significantBestScore)));
    if (scoreImprovement > 0 &&
        relativeImprovement >= config.improvementThreshold) {
      significantBestScore = best.score.totalScore;
      epochsWithoutMeaningfulImprovement = 0;
    } else {
      ++epochsWithoutMeaningfulImprovement;
    }

    if (epochs >= config.minimumEpochs &&
        epochsWithoutMeaningfulImprovement >= config.plateauPatience &&
        lastUphillAcceptanceRate <= config.plateauAcceptanceRate) {
      stopReason = StopReason::Plateau;
      break;
    }

    if (reachedEvaluationLimit()) {
      stopReason = StopReason::MaximumEvaluations;
      break;
    }
    if (reachedRuntimeLimit()) {
      stopReason = StopReason::MaximumRuntime;
      break;
    }

    temperature =
        std::max(config.finalTemperature, temperature * config.coolingRate);
  }

  return SearchResult{std::move(best.placement),
                      epochs,
                      evaluations,
                      initialScore->totalScore,
                      best.score.totalScore,
                      lastUphillAcceptanceRate,
                      elapsedSeconds(),
                      stopReason};
}

static void attachSearchResult(mlir::func::FuncOp taskGraphFunc,
                               const SearchResult &result) {
  mlir::Builder builder(taskGraphFunc.getContext());
  taskGraphFunc->setAttr(
      mlir::sculptor::schedule_attrs::kAnnealingEpochsAttrName,
      builder.getI64IntegerAttr(result.epochs));
  taskGraphFunc->setAttr(
      mlir::sculptor::schedule_attrs::kAnnealingEvaluationsAttrName,
      builder.getI64IntegerAttr(result.evaluations));
  taskGraphFunc->setAttr(
      mlir::sculptor::schedule_attrs::kAnnealingInitialScoreAttrName,
      builder.getI64IntegerAttr(result.initialScore));
  taskGraphFunc->setAttr(
      mlir::sculptor::schedule_attrs::kAnnealingBestScoreAttrName,
      builder.getI64IntegerAttr(result.bestScore));
  taskGraphFunc->setAttr(
      mlir::sculptor::schedule_attrs::kAnnealingStopReasonAttrName,
      builder.getStringAttr(stringifyStopReason(result.stopReason)));
  taskGraphFunc->setAttr(
      mlir::sculptor::schedule_attrs::kAnnealingSearchSecondsAttrName,
      builder.getF64FloatAttr(result.searchSeconds));
  taskGraphFunc->setAttr(
      mlir::sculptor::schedule_attrs::kAnnealingUphillAcceptanceRateAttrName,
      builder.getF64FloatAttr(result.uphillAcceptanceRate));
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
  attachSearchResult(problem.taskGraphFunc, *placement);
  return buildPlacementPlanFromPhysicalArrayOrder(
      problem, placement->placement.physicalArrayOrder);
}

} // namespace annealing_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
