#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <tuple>
#include <vector>

namespace {

using namespace mlir;
using namespace mlir::sculptor::mapping;

struct PlacementSearchResult {
  SmallVector<int64_t> physicalTiles;
  int64_t initialScore = 0;
  int64_t evaluations = 0;
  std::optional<LogicalTileAnnealingTrace> annealingTrace;
};

struct GreedyPhysicalCandidate {
  int64_t physicalTile = -1;
  int64_t incrementalCost = 0;
  int64_t regionDistance = 0;
};

struct GreedyTilePriorityEntry {
  size_t tileIndex = 0;
  int64_t primary = 0;
  int64_t secondary = 0;
  int64_t logicalTileId = -1;
  uint64_t version = 0;
};

struct GreedyTilePriorityLess {
  bool operator()(const GreedyTilePriorityEntry &lhs,
                  const GreedyTilePriorityEntry &rhs) const {
    if (lhs.primary != rhs.primary)
      return lhs.primary < rhs.primary;
    if (lhs.secondary != rhs.secondary)
      return lhs.secondary < rhs.secondary;
    return lhs.logicalTileId > rhs.logicalTileId;
  }
};

using GreedyTilePriorityQueue =
    std::priority_queue<GreedyTilePriorityEntry,
                        std::vector<GreedyTilePriorityEntry>,
                        GreedyTilePriorityLess>;

struct GreedyPlacementState {
  explicit GreedyPlacementState(size_t tileCount)
      : physicalByTileIndex(tileCount, -1), placed(tileCount, false),
        frontierSum(tileCount, 0), frontierMax(tileCount, 0),
        frontierVersion(tileCount, 0) {}

  SmallVector<int64_t> physicalByTileIndex;
  SmallVector<bool> placed;
  llvm::DenseSet<int64_t> usedPhysicalTiles;
  std::set<int64_t> physicalFrontier;
  SmallVector<int64_t> frontierSum;
  SmallVector<int64_t> frontierMax;
  SmallVector<uint64_t> frontierVersion;
  GreedyTilePriorityQueue priorityQueue;
  int64_t currentPhysicalTile = 0;
  size_t placedCount = 0;
};

struct GreedyRolloutResult {
  GreedyPhysicalCandidate firstPlacement;
  int64_t totalCost = 0;
};

FailureOr<int64_t> getMeshCapacity(const LogicalTilePlacementProblem &problem) {
  std::optional<int64_t> capacity =
      llvm::checkedMul(problem.mesh.rows, problem.mesh.columns);
  if (!capacity) {
    problem.anchor->emitError("physical mesh capacity overflow");
    return failure();
  }
  return *capacity;
}

SmallVector<int64_t>
buildSnakePhysicalOrder(const LogicalTilePlacementProblem &problem) {
  SmallVector<int64_t> order;
  order.reserve(problem.mesh.rows * problem.mesh.columns);
  for (int64_t row = 0; row < problem.mesh.rows; ++row) {
    if (row % 2 == 0) {
      for (int64_t column = 0; column < problem.mesh.columns; ++column)
        order.push_back(row * problem.mesh.columns + column);
    } else {
      for (int64_t column = problem.mesh.columns; column-- > 0;)
        order.push_back(row * problem.mesh.columns + column);
    }
  }
  return order;
}

FailureOr<PlacementSearchResult>
buildRandomPlacement(const LogicalTilePlacementProblem &problem,
                     int64_t randomSeed) {
  FailureOr<int64_t> capacity = getMeshCapacity(problem);
  if (failed(capacity))
    return failure();
  SmallVector<int64_t> physicalTiles;
  physicalTiles.reserve(*capacity);
  for (int64_t tile = 0; tile < *capacity; ++tile)
    physicalTiles.push_back(tile);
  std::mt19937 engine(static_cast<uint32_t>(randomSeed));
  std::shuffle(physicalTiles.begin(), physicalTiles.end(), engine);
  physicalTiles.resize(problem.tileGraph.tiles.size());
  return PlacementSearchResult{std::move(physicalTiles), 0, 0};
}

PlacementSearchResult
buildSnakePlacement(const LogicalTilePlacementProblem &problem) {
  SmallVector<int64_t> physicalTiles = buildSnakePhysicalOrder(problem);
  physicalTiles.resize(problem.tileGraph.tiles.size());
  return {std::move(physicalTiles), 0, 0};
}

int64_t manhattanDistance(int64_t firstPhysicalTile, int64_t secondPhysicalTile,
                          const PhysicalMeshGeometry &mesh) {
  int64_t firstRow = firstPhysicalTile / mesh.columns;
  int64_t firstColumn = firstPhysicalTile % mesh.columns;
  int64_t secondRow = secondPhysicalTile / mesh.columns;
  int64_t secondColumn = secondPhysicalTile % mesh.columns;
  return std::abs(firstRow - secondRow) + std::abs(firstColumn - secondColumn);
}

FailureOr<int64_t> checkedWeightedDistance(int64_t bytes, int64_t distance,
                                           Operation *anchor) {
  std::optional<int64_t> cost = llvm::checkedMul(bytes, distance);
  if (!cost) {
    anchor->emitError("greedy logical-tile placement score overflow");
    return failure();
  }
  return *cost;
}

FailureOr<PlacementSearchResult>
buildGreedyPlacement(const LogicalTilePlacementProblem &problem,
                     const GreedyPlacementConfig &config) {
  FailureOr<int64_t> capacity = getMeshCapacity(problem);
  if (failed(capacity))
    return failure();
  const size_t tileCount = problem.tileGraph.tiles.size();
  GreedyPlacementState state(tileCount);
  SmallVector<llvm::DenseMap<unsigned, int64_t>> affinity(tileCount);

  for (const LogicalTileEdge &edge : problem.tileGraph.edges) {
    unsigned source = problem.tileGraph.tileIndexById.lookup(edge.sourceTileId);
    unsigned target = problem.tileGraph.tileIndexById.lookup(edge.targetTileId);
    if (source == target)
      continue;
    std::optional<int64_t> forward =
        llvm::checkedAdd(affinity[source].lookup(target), edge.byteSize);
    std::optional<int64_t> reverse =
        llvm::checkedAdd(affinity[target].lookup(source), edge.byteSize);
    if (!forward || !reverse) {
      problem.anchor->emitError("logical-tile affinity weight overflow");
      return failure();
    }
    affinity[source][target] = *forward;
    affinity[target][source] = *reverse;
  }

  SmallVector<size_t> tileOrder(tileCount);
  std::iota(tileOrder.begin(), tileOrder.end(), size_t{0});
  llvm::sort(tileOrder, [&](size_t lhs, size_t rhs) {
    return problem.tileGraph.tiles[lhs].id < problem.tileGraph.tiles[rhs].id;
  });

  auto visitPhysicalNeighbors = [&](int64_t physicalTile,
                                    bool includeDiagonals, auto &&visitor) {
    int64_t row = physicalTile / problem.mesh.columns;
    int64_t column = physicalTile % problem.mesh.columns;
    for (int64_t rowOffset = -1; rowOffset <= 1; ++rowOffset) {
      for (int64_t columnOffset = -1; columnOffset <= 1; ++columnOffset) {
        if (rowOffset == 0 && columnOffset == 0)
          continue;
        if (!includeDiagonals && std::abs(rowOffset) + std::abs(columnOffset) != 1)
          continue;
        int64_t candidateRow = row + rowOffset;
        int64_t candidateColumn = column + columnOffset;
        if (candidateRow < 0 || candidateRow >= problem.mesh.rows ||
            candidateColumn < 0 || candidateColumn >= problem.mesh.columns)
          continue;
        visitor(candidateRow * problem.mesh.columns + candidateColumn);
      }
    }
  };

  auto recordPhysicalPlacement = [&](GreedyPlacementState &target,
                                     int64_t physicalTile) {
    target.physicalFrontier.erase(physicalTile);
    target.usedPhysicalTiles.insert(physicalTile);
    visitPhysicalNeighbors(physicalTile, /*includeDiagonals=*/true,
                           [&](int64_t neighbor) {
                             if (!target.usedPhysicalTiles.contains(neighbor))
                               target.physicalFrontier.insert(neighbor);
                           });
  };

  size_t firstTile = tileOrder.front();
  state.physicalByTileIndex[firstTile] = 0;
  state.placed[firstTile] = true;
  state.placedCount = 1;
  recordPhysicalPlacement(state, 0);
  int64_t evaluations = 0;

  auto pushPriority = [&](GreedyPlacementState &target, size_t tile) {
    int64_t primary = config.priorityMode == GreedyPriorityMode::Sum
                          ? target.frontierSum[tile]
                          : target.frontierMax[tile];
    int64_t secondary = config.priorityMode == GreedyPriorityMode::Sum
                            ? target.frontierMax[tile]
                            : target.frontierSum[tile];
    target.priorityQueue.push({tile, primary, secondary,
                               problem.tileGraph.tiles[tile].id,
                               target.frontierVersion[tile]});
  };

  auto updatePriorityFrontier = [&](GreedyPlacementState &target,
                                    size_t newlyPlaced) -> LogicalResult {
    for (const auto &[neighbor, bytes] : affinity[newlyPlaced]) {
      if (target.placed[neighbor])
        continue;
      std::optional<int64_t> updated =
          llvm::checkedAdd(target.frontierSum[neighbor], bytes);
      if (!updated) {
        problem.anchor->emitError("greedy tile-order affinity overflow");
        return failure();
      }
      target.frontierSum[neighbor] = *updated;
      target.frontierMax[neighbor] =
          std::max(target.frontierMax[neighbor], bytes);
      ++target.frontierVersion[neighbor];
      pushPriority(target, neighbor);
    }
    return success();
  };

  auto selectPriorityTile = [&](GreedyPlacementState &target) {
    while (!target.priorityQueue.empty()) {
      GreedyTilePriorityEntry entry = target.priorityQueue.top();
      target.priorityQueue.pop();
      if (target.placed[entry.tileIndex] ||
          entry.version != target.frontierVersion[entry.tileIndex])
        continue;
      return entry.tileIndex;
    }
    for (size_t tile : tileOrder) {
      if (!target.placed[tile])
        return tile;
    }
    return tileCount;
  };

  if (config.tileOrder == GreedyTileOrder::Priority &&
      failed(updatePriorityFrontier(state, firstTile)))
    return failure();

  auto scoreCandidate = [&](const GreedyPlacementState &target, size_t tile,
                            int64_t physicalTile) -> FailureOr<int64_t> {
    int64_t cost = 0;
    for (const auto &[neighbor, bytes] : affinity[tile]) {
      if (!target.placed[neighbor])
        continue;
      int64_t distance = manhattanDistance(
          physicalTile, target.physicalByTileIndex[neighbor], problem.mesh);
      FailureOr<int64_t> contribution =
          checkedWeightedDistance(bytes, distance, problem.anchor);
      if (failed(contribution))
        return failure();
      std::optional<int64_t> updated = llvm::checkedAdd(cost, *contribution);
      if (!updated) {
        problem.anchor->emitError(
            "greedy logical-tile incremental score overflow");
        return failure();
      }
      cost = *updated;
    }
    return cost;
  };

  auto collectScopedCandidates = [&](const GreedyPlacementState &target) {
    SmallVector<int64_t, 8> candidates;
    if (config.candidateScope == GreedyCandidateScope::Frontier) {
      candidates.append(target.physicalFrontier.begin(),
                        target.physicalFrontier.end());
      return candidates;
    }
    bool includeDiagonals =
        config.candidateScope == GreedyCandidateScope::Diagonal;
    visitPhysicalNeighbors(target.currentPhysicalTile, includeDiagonals,
                           [&](int64_t candidate) {
                             if (!target.usedPhysicalTiles.contains(candidate))
                               candidates.push_back(candidate);
                           });
    return candidates;
  };

  auto collectNearestOpenCandidates =
      [&](const GreedyPlacementState &target) {
    SmallVector<int64_t> candidates;
    int64_t nearestDistance = std::numeric_limits<int64_t>::max();
    for (int64_t candidate = 0; candidate < *capacity; ++candidate) {
      if (target.usedPhysicalTiles.contains(candidate))
        continue;
      int64_t distance =
          manhattanDistance(target.currentPhysicalTile, candidate, problem.mesh);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        candidates.clear();
      }
      if (distance == nearestDistance)
        candidates.push_back(candidate);
    }
    return candidates;
  };

  auto selectNextLogicalTile = [&](GreedyPlacementState &target) {
    if (config.tileOrder == GreedyTileOrder::Priority)
      return selectPriorityTile(target);
    for (size_t tile : tileOrder) {
      if (!target.placed[tile])
        return tile;
    }
    return tileCount;
  };

  auto collectCandidates = [&](const GreedyPlacementState &target) {
    SmallVector<int64_t> candidates = collectScopedCandidates(target);
    if (candidates.empty())
      candidates = collectNearestOpenCandidates(target);
    return candidates;
  };

  auto chooseImmediatePlacement =
      [&](const GreedyPlacementState &target,
          size_t tile) -> FailureOr<GreedyPhysicalCandidate> {
    SmallVector<int64_t> candidates = collectCandidates(target);
    if (candidates.empty()) {
      problem.anchor->emitError(
          "greedy logical-tile placement exhausted the physical mesh");
      return failure();
    }

    GreedyPhysicalCandidate best;
    bool hasBest = false;
    for (int64_t candidate : candidates) {
      FailureOr<int64_t> incrementalCost =
          scoreCandidate(target, tile, candidate);
      if (failed(incrementalCost))
        return failure();
      ++evaluations;
      GreedyPhysicalCandidate scored{
          candidate, *incrementalCost,
          manhattanDistance(target.currentPhysicalTile, candidate,
                            problem.mesh)};
      if (!hasBest || std::tie(scored.incrementalCost, scored.regionDistance,
                               scored.physicalTile) <
                          std::tie(best.incrementalCost, best.regionDistance,
                                   best.physicalTile)) {
        best = scored;
        hasBest = true;
      }
    }
    return best;
  };

  auto commitPlacement = [&](GreedyPlacementState &target, size_t tile,
                             int64_t physicalTile) -> LogicalResult {
    if (target.placed[tile] ||
        target.usedPhysicalTiles.contains(physicalTile)) {
      problem.anchor->emitError("greedy rollout attempted duplicate placement");
      return failure();
    }
    target.physicalByTileIndex[tile] = physicalTile;
    target.placed[tile] = true;
    ++target.placedCount;
    recordPhysicalPlacement(target, physicalTile);
    target.currentPhysicalTile = physicalTile;
    if (config.tileOrder == GreedyTileOrder::Priority &&
        failed(updatePriorityFrontier(target, tile)))
      return failure();
    return success();
  };

  while (state.placedCount < tileCount) {
    size_t tile = selectNextLogicalTile(state);
    if (tile == tileCount) {
      problem.anchor->emitError(
          "greedy tile order exhausted before all tiles were placed");
      return failure();
    }
    SmallVector<int64_t> candidates = collectCandidates(state);
    if (candidates.empty()) {
      problem.anchor->emitError(
          "greedy logical-tile placement exhausted the physical mesh");
      return failure();
    }

    GreedyRolloutResult best;
    bool hasBest = false;
    for (int64_t candidate : candidates) {
      FailureOr<int64_t> incrementalCost =
          scoreCandidate(state, tile, candidate);
      if (failed(incrementalCost))
        return failure();
      ++evaluations;
      GreedyPhysicalCandidate firstPlacement{
          candidate, *incrementalCost,
          manhattanDistance(state.currentPhysicalTile, candidate,
                            problem.mesh)};

      GreedyPlacementState simulation = state;
      if (failed(commitPlacement(simulation, tile, candidate)))
        return failure();
      int64_t rolloutCost = *incrementalCost;
      int64_t rolloutDepth = std::min<int64_t>(
          config.lookahead,
          static_cast<int64_t>(tileCount - state.placedCount));
      for (int64_t depth = 1;
           depth < rolloutDepth && simulation.placedCount < tileCount;
           ++depth) {
        size_t futureTile = selectNextLogicalTile(simulation);
        if (futureTile == tileCount) {
          problem.anchor->emitError(
              "greedy rollout exhausted logical tiles unexpectedly");
          return failure();
        }
        FailureOr<GreedyPhysicalCandidate> futurePlacement =
            chooseImmediatePlacement(simulation, futureTile);
        if (failed(futurePlacement))
          return failure();
        std::optional<int64_t> updatedCost =
            llvm::checkedAdd(rolloutCost, futurePlacement->incrementalCost);
        if (!updatedCost) {
          problem.anchor->emitError("greedy rollout cost overflow");
          return failure();
        }
        rolloutCost = *updatedCost;
        if (failed(commitPlacement(simulation, futureTile,
                                   futurePlacement->physicalTile)))
          return failure();
      }

      GreedyRolloutResult scored{firstPlacement, rolloutCost};
      if (!hasBest ||
          std::tie(scored.totalCost, scored.firstPlacement.incrementalCost,
                   scored.firstPlacement.regionDistance,
                   scored.firstPlacement.physicalTile) <
              std::tie(best.totalCost, best.firstPlacement.incrementalCost,
                       best.firstPlacement.regionDistance,
                       best.firstPlacement.physicalTile)) {
        best = scored;
        hasBest = true;
      }
    }

    if (failed(
            commitPlacement(state, tile, best.firstPlacement.physicalTile)))
      return failure();
  }

  return PlacementSearchResult{std::move(state.physicalByTileIndex), 0,
                               evaluations};
}

FailureOr<PlacementSearchResult>
buildInitialPlacement(const LogicalTilePlacementProblem &problem,
                      LogicalTileScheduleKind schedule, int64_t randomSeed,
                      const GreedyPlacementConfig &greedyConfig) {
  switch (schedule) {
  case LogicalTileScheduleKind::Random:
    return buildRandomPlacement(problem, randomSeed);
  case LogicalTileScheduleKind::Snake:
    return buildSnakePlacement(problem);
  case LogicalTileScheduleKind::Greedy:
    return buildGreedyPlacement(problem, greedyConfig);
  case LogicalTileScheduleKind::Annealing:
    problem.anchor->emitError(
        "annealing cannot initialize logical-tile placement from itself");
    return failure();
  }
  llvm_unreachable("unknown logical-tile placement schedule");
}

FailureOr<PlacementSearchResult>
runAnnealing(const LogicalTilePlacementProblem &problem,
             const LogicalTilePlacementConfig &config) {
  FailureOr<PlacementSearchResult> initial = buildInitialPlacement(
      problem, config.annealingInitialSchedule, config.randomSeed,
      config.greedy);
  if (failed(initial))
    return failure();
  FailureOr<int64_t> initialScore =
      scoreLogicalTilePlacement(problem, initial->physicalTiles);
  if (failed(initialScore))
    return failure();

  SmallVector<int64_t> current = initial->physicalTiles;
  SmallVector<int64_t> best = current;
  int64_t currentScore = *initialScore;
  int64_t bestScore = currentScore;
  LogicalTileAnnealingTrace trace;
  trace.initialScore = currentScore;
  trace.finalScore = currentScore;
  trace.samples.push_back({0, currentScore, currentScore, currentScore, true});
  double temperature =
      config.annealingInitialTemperature > 0.0
          ? config.annealingInitialTemperature
          : std::max(1.0, static_cast<double>(currentScore) * 0.05);
  std::mt19937 engine(static_cast<uint32_t>(config.randomSeed));
  std::uniform_real_distribution<double> probability(0.0, 1.0);
  std::uniform_int_distribution<size_t> tileDistribution(0, current.size() - 1);
  FailureOr<int64_t> capacity = getMeshCapacity(problem);
  if (failed(capacity))
    return failure();
  std::uniform_int_distribution<int64_t> physicalDistribution(0, *capacity - 1);

  int64_t evaluations = 0;
  for (int64_t iteration = 0; iteration < config.annealingIterations;
       ++iteration) {
    SmallVector<int64_t> candidate = current;
    bool useRelocation = *capacity > static_cast<int64_t>(candidate.size()) &&
                         probability(engine) < 0.30;
    if (useRelocation) {
      size_t tile = tileDistribution(engine);
      std::set<int64_t> occupied(candidate.begin(), candidate.end());
      int64_t replacement = -1;
      for (int64_t attempt = 0; attempt < *capacity * 2; ++attempt) {
        int64_t value = physicalDistribution(engine);
        if (!occupied.contains(value)) {
          replacement = value;
          break;
        }
      }
      if (replacement < 0)
        continue;
      candidate[tile] = replacement;
    } else {
      size_t first = tileDistribution(engine);
      size_t second = tileDistribution(engine);
      if (first == second)
        second = (second + 1) % candidate.size();
      std::swap(candidate[first], candidate[second]);
    }

    FailureOr<int64_t> candidateScore =
        scoreLogicalTilePlacement(problem, candidate);
    if (failed(candidateScore))
      return failure();
    ++evaluations;
    int64_t delta = *candidateScore - currentScore;
    bool accept = delta <= 0;
    if (!accept && temperature > 0.0)
      accept = probability(engine) <
               std::exp(-static_cast<double>(delta) / temperature);
    if (accept) {
      current = std::move(candidate);
      currentScore = *candidateScore;
    }
    if (currentScore < bestScore) {
      best = current;
      bestScore = currentScore;
    }
    trace.samples.push_back(
        {evaluations, *candidateScore, currentScore, bestScore, accept});
    temperature = std::max(1.0e-9, temperature * config.annealingCoolingRate);
  }
  trace.finalScore = bestScore;
  trace.evaluations = evaluations;
  return PlacementSearchResult{std::move(best), *initialScore, evaluations,
                               std::move(trace)};
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<LogicalTilePlacementPlan>
scheduleLogicalTiles(const LogicalTilePlacementProblem &problem,
                     const LogicalTilePlacementConfig &config) {
  if (failed(validateLogicalTilePlacementProblem(problem)))
    return failure();
  if (config.greedy.lookahead <= 0) {
    problem.anchor->emitError("greedy lookahead must be positive");
    return failure();
  }
  if (config.annealingIterations < 0 ||
      config.annealingInitialTemperature < 0.0 ||
      config.annealingCoolingRate <= 0.0 || config.annealingCoolingRate > 1.0) {
    problem.anchor->emitError("invalid logical-tile annealing parameters");
    return failure();
  }

  FailureOr<PlacementSearchResult> search =
      config.schedule == LogicalTileScheduleKind::Annealing
          ? runAnnealing(problem, config)
          : buildInitialPlacement(problem, config.schedule, config.randomSeed,
                                  config.greedy);
  if (failed(search))
    return failure();
  FailureOr<int64_t> score =
      scoreLogicalTilePlacement(problem, search->physicalTiles);
  if (failed(score))
    return failure();
  int64_t initialScore = config.schedule == LogicalTileScheduleKind::Annealing
                             ? search->initialScore
                             : *score;
  FailureOr<LogicalTilePlacementPlan> plan = buildLogicalTilePlacementPlan(
      problem, search->physicalTiles,
      stringifyLogicalTileSchedule(config.schedule), initialScore,
      search->evaluations);
  if (failed(plan))
    return failure();
  plan->annealingTrace = std::move(search->annealingTrace);
  return plan;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
