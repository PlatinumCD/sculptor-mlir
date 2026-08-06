#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"

#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::sculptor::mapping;

using TilePair = std::pair<int64_t, int64_t>;

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

struct GreedyBeamState {
  SmallVector<std::pair<size_t, int64_t>, 4> decisions;
  int64_t cumulativeCost = 0;
  int64_t firstIncrementalCost = 0;
  int64_t firstRegionDistance = 0;
  int64_t firstPhysicalTile = -1;
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
buildGreedyPlacement(const LogicalTilePlacementProblem &problem) {
  FailureOr<int64_t> capacity = getMeshCapacity(problem);
  if (failed(capacity))
    return failure();
  const size_t tileCount = problem.tileGraph.tiles.size();
  SmallVector<int64_t> physicalByTileIndex(tileCount, -1);
  SmallVector<bool> placed(tileCount, false);
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

  std::set<int64_t> usedPhysicalTiles;
  size_t firstTile = tileOrder.front();
  physicalByTileIndex[firstTile] = 0;
  placed[firstTile] = true;
  usedPhysicalTiles.insert(0);
  int64_t currentPhysicalTile = 0;
  int64_t evaluations = 0;

  auto scoreCandidate = [&](size_t tile,
                            int64_t physicalTile) -> FailureOr<int64_t> {
    int64_t cost = 0;
    for (const auto &[neighbor, bytes] : affinity[tile]) {
      if (!placed[neighbor])
        continue;
      int64_t distance = manhattanDistance(
          physicalTile, physicalByTileIndex[neighbor], problem.mesh);
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

  auto collectLocalCandidates = [&]() {
    SmallVector<int64_t, 4> candidates;
    int64_t row = currentPhysicalTile / problem.mesh.columns;
    int64_t column = currentPhysicalTile % problem.mesh.columns;
    const std::pair<int64_t, int64_t> offsets[] = {
        {-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    for (auto [rowOffset, columnOffset] : offsets) {
      int64_t candidateRow = row + rowOffset;
      int64_t candidateColumn = column + columnOffset;
      if (candidateRow < 0 || candidateRow >= problem.mesh.rows ||
          candidateColumn < 0 || candidateColumn >= problem.mesh.columns)
        continue;
      int64_t candidate = candidateRow * problem.mesh.columns + candidateColumn;
      if (!usedPhysicalTiles.contains(candidate))
        candidates.push_back(candidate);
    }
    return candidates;
  };

  auto collectNearestOpenCandidates = [&]() {
    SmallVector<int64_t> candidates;
    int64_t nearestDistance = std::numeric_limits<int64_t>::max();
    for (int64_t candidate = 0; candidate < *capacity; ++candidate) {
      if (usedPhysicalTiles.contains(candidate))
        continue;
      int64_t distance =
          manhattanDistance(currentPhysicalTile, candidate, problem.mesh);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        candidates.clear();
      }
      if (distance == nearestDistance)
        candidates.push_back(candidate);
    }
    return candidates;
  };

  for (size_t orderIndex = 1; orderIndex < tileOrder.size(); ++orderIndex) {
    size_t tile = tileOrder[orderIndex];
    SmallVector<int64_t> candidates = collectLocalCandidates();
    if (candidates.empty())
      candidates = collectNearestOpenCandidates();
    if (candidates.empty()) {
      problem.anchor->emitError(
          "greedy logical-tile placement exhausted the physical mesh");
      return failure();
    }

    GreedyPhysicalCandidate best;
    bool hasBest = false;
    for (int64_t candidate : candidates) {
      FailureOr<int64_t> incrementalCost = scoreCandidate(tile, candidate);
      if (failed(incrementalCost))
        return failure();
      ++evaluations;
      GreedyPhysicalCandidate scored{
          candidate, *incrementalCost,
          manhattanDistance(currentPhysicalTile, candidate, problem.mesh)};
      if (!hasBest || std::tie(scored.incrementalCost, scored.regionDistance,
                               scored.physicalTile) <
                          std::tie(best.incrementalCost, best.regionDistance,
                                   best.physicalTile)) {
        best = scored;
        hasBest = true;
      }
    }

    physicalByTileIndex[tile] = best.physicalTile;
    placed[tile] = true;
    usedPhysicalTiles.insert(best.physicalTile);
    currentPhysicalTile = best.physicalTile;
  }

  return PlacementSearchResult{std::move(physicalByTileIndex), 0, evaluations};
}

FailureOr<PlacementSearchResult>
buildGreedyBeamPlacement(const LogicalTilePlacementProblem &problem,
                         int64_t lookahead, int64_t beamWidth) {
  FailureOr<int64_t> capacity = getMeshCapacity(problem);
  if (failed(capacity))
    return failure();
  const size_t tileCount = problem.tileGraph.tiles.size();
  SmallVector<int64_t> physicalByTileIndex(tileCount, -1);
  SmallVector<bool> placed(tileCount, false);
  std::set<int64_t> usedPhysicalTiles;

  std::map<TilePair, int64_t> affinity;
  SmallVector<int64_t> weightedDegree(tileCount, 0);
  for (const LogicalTileEdge &edge : problem.tileGraph.edges) {
    unsigned source = problem.tileGraph.tileIndexById.lookup(edge.sourceTileId);
    unsigned target = problem.tileGraph.tileIndexById.lookup(edge.targetTileId);
    TilePair pair =
        source < target ? TilePair{source, target} : TilePair{target, source};
    std::optional<int64_t> pairWeight =
        llvm::checkedAdd(affinity[pair], edge.byteSize);
    std::optional<int64_t> sourceDegree =
        llvm::checkedAdd(weightedDegree[source], edge.byteSize);
    std::optional<int64_t> targetDegree =
        llvm::checkedAdd(weightedDegree[target], edge.byteSize);
    if (!pairWeight || !sourceDegree || !targetDegree) {
      problem.anchor->emitError("logical-tile affinity weight overflow");
      return failure();
    }
    affinity[pair] = *pairWeight;
    weightedDegree[source] = *sourceDegree;
    weightedDegree[target] = *targetDegree;
  }

  auto communication = [&](size_t first, size_t second) {
    TilePair pair =
        first < second ? TilePair{first, second} : TilePair{second, first};
    auto found = affinity.find(pair);
    return found == affinity.end() ? int64_t{0} : found->second;
  };

  size_t firstTile = 0;
  for (size_t tile = 1; tile < tileCount; ++tile) {
    if (weightedDegree[tile] > weightedDegree[firstTile] ||
        (weightedDegree[tile] == weightedDegree[firstTile] &&
         problem.tileGraph.tiles[tile].id <
             problem.tileGraph.tiles[firstTile].id))
      firstTile = tile;
  }
  int64_t centerRow = problem.mesh.rows / 2;
  int64_t centerColumn = problem.mesh.columns / 2;
  int64_t centerPhysicalTile = centerRow * problem.mesh.columns + centerColumn;
  physicalByTileIndex[firstTile] = centerPhysicalTile;
  placed[firstTile] = true;
  usedPhysicalTiles.insert(centerPhysicalTile);

  int64_t evaluations = 0;
  size_t placedCount = 1;
  auto selectNextTile = [&]() {
    size_t nextTile = tileCount;
    int64_t bestPlacedAffinity = -1;
    int64_t bestDegree = -1;
    for (size_t tile = 0; tile < tileCount; ++tile) {
      if (placed[tile])
        continue;
      int64_t placedAffinity = 0;
      for (size_t other = 0; other < tileCount; ++other) {
        if (placed[other])
          placedAffinity += communication(tile, other);
      }
      if (nextTile == tileCount || placedAffinity > bestPlacedAffinity ||
          (placedAffinity == bestPlacedAffinity &&
           weightedDegree[tile] > bestDegree) ||
          (placedAffinity == bestPlacedAffinity &&
           weightedDegree[tile] == bestDegree &&
           problem.tileGraph.tiles[tile].id <
               problem.tileGraph.tiles[nextTile].id)) {
        nextTile = tile;
        bestPlacedAffinity = placedAffinity;
        bestDegree = weightedDegree[tile];
      }
    }
    return nextTile;
  };

  auto collectCandidates =
      [&](size_t nextTile) -> FailureOr<SmallVector<GreedyPhysicalCandidate>> {
    SmallVector<GreedyPhysicalCandidate> candidates;
    candidates.reserve(*capacity - usedPhysicalTiles.size());
    for (int64_t candidate = 0; candidate < *capacity; ++candidate) {
      if (usedPhysicalTiles.contains(candidate))
        continue;
      int64_t incrementalCost = 0;
      int64_t regionDistance = *capacity;
      for (size_t other = 0; other < tileCount; ++other) {
        if (!placed[other])
          continue;
        int64_t distance = manhattanDistance(
            candidate, physicalByTileIndex[other], problem.mesh);
        regionDistance = std::min(regionDistance, distance);
        int64_t bytes = communication(nextTile, other);
        FailureOr<int64_t> edgeCost =
            checkedWeightedDistance(bytes, distance, problem.anchor);
        if (failed(edgeCost))
          return failure();
        std::optional<int64_t> updated =
            llvm::checkedAdd(incrementalCost, *edgeCost);
        if (!updated) {
          problem.anchor->emitError(
              "greedy logical-tile incremental score overflow");
          return failure();
        }
        incrementalCost = *updated;
      }
      ++evaluations;
      candidates.push_back({candidate, incrementalCost, regionDistance});
    }

    llvm::sort(candidates, [](const GreedyPhysicalCandidate &lhs,
                              const GreedyPhysicalCandidate &rhs) {
      return std::tie(lhs.incrementalCost, lhs.regionDistance,
                      lhs.physicalTile) < std::tie(rhs.incrementalCost,
                                                   rhs.regionDistance,
                                                   rhs.physicalTile);
    });
    return candidates;
  };

  auto placeDecision = [&](size_t tile, int64_t physicalTile) {
    physicalByTileIndex[tile] = physicalTile;
    placed[tile] = true;
    usedPhysicalTiles.insert(physicalTile);
    ++placedCount;
  };
  auto unplaceDecision = [&](size_t tile, int64_t physicalTile) {
    --placedCount;
    usedPhysicalTiles.erase(physicalTile);
    placed[tile] = false;
    physicalByTileIndex[tile] = -1;
  };

  auto applyState = [&](const GreedyBeamState &state) {
    for (const auto &[tile, physicalTile] : state.decisions)
      placeDecision(tile, physicalTile);
  };
  auto unapplyState = [&](const GreedyBeamState &state) {
    for (auto decision = state.decisions.rbegin();
         decision != state.decisions.rend(); ++decision)
      unplaceDecision(decision->first, decision->second);
  };
  auto compareBeamStates = [](const GreedyBeamState &lhs,
                              const GreedyBeamState &rhs) {
    auto lhsKey = std::tie(lhs.cumulativeCost, lhs.firstIncrementalCost,
                           lhs.firstRegionDistance, lhs.firstPhysicalTile);
    auto rhsKey = std::tie(rhs.cumulativeCost, rhs.firstIncrementalCost,
                           rhs.firstRegionDistance, rhs.firstPhysicalTile);
    if (lhsKey != rhsKey)
      return lhsKey < rhsKey;
    return std::lexicographical_compare(
        lhs.decisions.begin(), lhs.decisions.end(), rhs.decisions.begin(),
        rhs.decisions.end());
  };

  while (placedCount < tileCount) {
    SmallVector<GreedyBeamState> beam(1);
    int64_t depthLimit = std::min<int64_t>(
        lookahead, static_cast<int64_t>(tileCount - placedCount));
    for (int64_t depth = 0; depth < depthLimit; ++depth) {
      SmallVector<GreedyBeamState> expanded;
      for (const GreedyBeamState &state : beam) {
        applyState(state);
        size_t nextTile = selectNextTile();
        FailureOr<SmallVector<GreedyPhysicalCandidate>> candidates =
            collectCandidates(nextTile);
        unapplyState(state);
        if (failed(candidates) || candidates->empty()) {
          problem.anchor->emitError(
              "greedy logical-tile beam exhausted the physical mesh");
          return failure();
        }

        for (const GreedyPhysicalCandidate &candidate : *candidates) {
          std::optional<int64_t> total =
              llvm::checkedAdd(state.cumulativeCost, candidate.incrementalCost);
          if (!total) {
            problem.anchor->emitError(
                "greedy logical-tile beam score overflow");
            return failure();
          }
          GreedyBeamState child = state;
          child.decisions.push_back({nextTile, candidate.physicalTile});
          child.cumulativeCost = *total;
          if (depth == 0) {
            child.firstIncrementalCost = candidate.incrementalCost;
            child.firstRegionDistance = candidate.regionDistance;
            child.firstPhysicalTile = candidate.physicalTile;
          }
          expanded.push_back(std::move(child));
        }
      }
      llvm::sort(expanded, compareBeamStates);
      if (expanded.size() > static_cast<size_t>(beamWidth))
        expanded.resize(static_cast<size_t>(beamWidth));
      beam = std::move(expanded);
    }
    if (beam.empty() || beam.front().decisions.empty()) {
      problem.anchor->emitError(
          "greedy logical-tile beam did not produce a placement");
      return failure();
    }
    const auto &[nextTile, physicalTile] = beam.front().decisions.front();
    placeDecision(nextTile, physicalTile);
  }
  return PlacementSearchResult{std::move(physicalByTileIndex), 0, evaluations};
}

FailureOr<PlacementSearchResult>
buildInitialPlacement(const LogicalTilePlacementProblem &problem,
                      LogicalTileScheduleKind schedule, int64_t randomSeed,
                      int64_t greedyLookahead, int64_t greedyBeamWidth) {
  switch (schedule) {
  case LogicalTileScheduleKind::Random:
    return buildRandomPlacement(problem, randomSeed);
  case LogicalTileScheduleKind::Snake:
    return buildSnakePlacement(problem);
  case LogicalTileScheduleKind::Greedy:
    return buildGreedyPlacement(problem);
  case LogicalTileScheduleKind::GreedyBeam:
    return buildGreedyBeamPlacement(problem, greedyLookahead, greedyBeamWidth);
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
      config.greedyLookahead, config.greedyBeamWidth);
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
  if (config.annealingIterations < 0 || config.greedyLookahead <= 0 ||
      config.greedyLookahead > 8 || config.greedyBeamWidth <= 0 ||
      config.annealingInitialTemperature < 0.0 ||
      config.annealingCoolingRate <= 0.0 || config.annealingCoolingRate > 1.0) {
    problem.anchor->emitError(
        "invalid logical-tile search parameters; greedy lookahead must be "
        "between 1 and 8 and beam width must be positive");
    return failure();
  }

  FailureOr<PlacementSearchResult> search =
      config.schedule == LogicalTileScheduleKind::Annealing
          ? runAnnealing(problem, config)
          : buildInitialPlacement(problem, config.schedule, config.randomSeed,
                                  config.greedyLookahead,
                                  config.greedyBeamWidth);
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
