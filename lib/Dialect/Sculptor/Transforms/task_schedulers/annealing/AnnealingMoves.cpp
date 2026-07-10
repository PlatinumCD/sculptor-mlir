#include "AnnealingMoves.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <random>

namespace {

namespace task_schedulers = mlir::sculptor::task_schedulers;
namespace annealing = task_schedulers::annealing_detail;

using AnnealingPlacement = annealing::Placement;
using AnnealingMoveKind = annealing::MoveKind;

struct AnnealingMove {
  AnnealingMoveKind kind = AnnealingMoveKind::None;
  unsigned firstIndex = 0;
  unsigned secondIndex = 0;
  unsigned segmentLength = 0;
  unsigned insertIndex = 0;
};

static unsigned getActivePlacementCount(
    const AnnealingPlacement &current,
    const task_schedulers::LogicalPlacementIslandGraph &islandGraph) {
  return static_cast<unsigned>(
      std::min(current.physicalArrayOrder.size(), islandGraph.islands.size()));
}

static unsigned chooseDifferentIndex(unsigned firstIndex, unsigned upperBound,
                                     std::mt19937 &randomEngine) {
  std::uniform_int_distribution<unsigned> distribution(0, upperBound - 2);
  unsigned selected = distribution(randomEngine);
  return selected < firstIndex ? selected : selected + 1;
}

static bool isMoveKindFeasible(AnnealingMoveKind moveKind,
                               unsigned activePlacementCount,
                               unsigned totalPlacementCount) {
  switch (moveKind) {
  case AnnealingMoveKind::None:
    return false;
  case AnnealingMoveKind::MoveOnePosition:
  case AnnealingMoveKind::MoveOneRelocation:
    return activePlacementCount >= 1 && totalPlacementCount >= 2;
  case AnnealingMoveKind::SwapTwoPositions:
  case AnnealingMoveKind::AdjacentSwap:
  case AnnealingMoveKind::SegmentReverse:
    return activePlacementCount >= 2;
  case AnnealingMoveKind::SegmentRelocation:
    return activePlacementCount >= 3;
  case AnnealingMoveKind::BlockSwap:
    return activePlacementCount >= 4;
  }
  return false;
}

static unsigned chooseDifferentIndexWithinRadius(unsigned firstIndex,
                                                 unsigned upperBound,
                                                 int64_t moveRadius,
                                                 std::mt19937 &randomEngine) {
  if (moveRadius <= 0 || static_cast<uint64_t>(moveRadius) >=
                             static_cast<uint64_t>(upperBound - 1))
    return chooseDifferentIndex(firstIndex, upperBound, randomEngine);

  unsigned radius = static_cast<unsigned>(moveRadius);
  unsigned lowerBound = firstIndex > radius ? firstIndex - radius : 0;
  unsigned upperBoundInclusive = std::min(firstIndex + radius, upperBound - 1);
  unsigned candidateCount = upperBoundInclusive - lowerBound;
  std::uniform_int_distribution<unsigned> distribution(0, candidateCount - 1);
  unsigned selected = lowerBound + distribution(randomEngine);
  return selected < firstIndex ? selected : selected + 1;
}

static AnnealingMove chooseMoveOnePosition(unsigned activePlacementCount,
                                           unsigned totalPlacementCount,
                                           int64_t moveRadius,
                                           std::mt19937 &randomEngine) {
  std::uniform_int_distribution<unsigned> activeIndexDistribution(
      0, activePlacementCount - 1);
  AnnealingMove move;
  move.kind = AnnealingMoveKind::MoveOnePosition;
  move.firstIndex = activeIndexDistribution(randomEngine);
  move.secondIndex = chooseDifferentIndexWithinRadius(
      move.firstIndex, totalPlacementCount, moveRadius, randomEngine);
  return move;
}

static AnnealingMove chooseMoveOneRelocation(unsigned activePlacementCount,
                                             unsigned totalPlacementCount,
                                             int64_t moveRadius,
                                             std::mt19937 &randomEngine) {
  std::uniform_int_distribution<unsigned> activeIndexDistribution(
      0, activePlacementCount - 1);
  AnnealingMove move;
  move.kind = AnnealingMoveKind::MoveOneRelocation;
  move.firstIndex = activeIndexDistribution(randomEngine);
  move.insertIndex = chooseDifferentIndexWithinRadius(
      move.firstIndex, totalPlacementCount, moveRadius, randomEngine);
  return move;
}

static AnnealingMove chooseSwapTwoPositions(unsigned activePlacementCount,
                                            std::mt19937 &randomEngine) {
  std::uniform_int_distribution<unsigned> activeIndexDistribution(
      0, activePlacementCount - 1);
  AnnealingMove move;
  move.kind = AnnealingMoveKind::SwapTwoPositions;
  move.firstIndex = activeIndexDistribution(randomEngine);
  move.secondIndex =
      chooseDifferentIndex(move.firstIndex, activePlacementCount, randomEngine);
  return move;
}

static AnnealingMove chooseAdjacentSwap(unsigned activePlacementCount,
                                        std::mt19937 &randomEngine) {
  std::uniform_int_distribution<unsigned> startDistribution(
      0, activePlacementCount - 2);
  AnnealingMove move;
  move.kind = AnnealingMoveKind::AdjacentSwap;
  move.firstIndex = startDistribution(randomEngine);
  move.secondIndex = move.firstIndex + 1;
  return move;
}

static AnnealingMove chooseSegmentReverse(unsigned activePlacementCount,
                                          std::mt19937 &randomEngine) {
  std::uniform_int_distribution<unsigned> lengthDistribution(
      2, activePlacementCount);
  unsigned length = lengthDistribution(randomEngine);
  std::uniform_int_distribution<unsigned> startDistribution(
      0, activePlacementCount - length);

  AnnealingMove move;
  move.kind = AnnealingMoveKind::SegmentReverse;
  move.firstIndex = startDistribution(randomEngine);
  move.secondIndex = move.firstIndex + length - 1;
  move.segmentLength = length;
  return move;
}

static AnnealingMove chooseSegmentRelocation(unsigned activePlacementCount,
                                             std::mt19937 &randomEngine) {
  unsigned maxLength = std::max(1u, activePlacementCount / 3);
  std::uniform_int_distribution<unsigned> lengthDistribution(1, maxLength);
  unsigned length = lengthDistribution(randomEngine);
  std::uniform_int_distribution<unsigned> startDistribution(
      0, activePlacementCount - length);

  AnnealingMove move;
  move.kind = AnnealingMoveKind::SegmentRelocation;
  move.firstIndex = startDistribution(randomEngine);
  move.segmentLength = length;

  unsigned reducedActiveCount = activePlacementCount - length;
  std::uniform_int_distribution<unsigned> insertDistribution(
      0, reducedActiveCount);
  do {
    move.insertIndex = insertDistribution(randomEngine);
  } while (move.insertIndex == move.firstIndex && reducedActiveCount > 0);
  return move;
}

static AnnealingMove chooseBlockSwap(unsigned activePlacementCount,
                                     std::mt19937 &randomEngine) {
  std::uniform_int_distribution<unsigned> lengthDistribution(
      1, activePlacementCount / 2);

  AnnealingMove move;
  move.kind = AnnealingMoveKind::BlockSwap;
  for (unsigned attempt = 0; attempt < 16; ++attempt) {
    unsigned length = lengthDistribution(randomEngine);
    std::uniform_int_distribution<unsigned> startDistribution(
        0, activePlacementCount - length);
    unsigned firstStart = startDistribution(randomEngine);
    unsigned secondStart = startDistribution(randomEngine);
    if (firstStart == secondStart)
      continue;
    if (firstStart > secondStart)
      std::swap(firstStart, secondStart);
    if (firstStart + length > secondStart)
      continue;

    move.firstIndex = firstStart;
    move.secondIndex = secondStart;
    move.segmentLength = length;
    return move;
  }

  move.firstIndex = 0;
  move.segmentLength = activePlacementCount / 2;
  move.secondIndex = move.segmentLength;
  return move;
}

static AnnealingMove choosePerturbationMove(
    const AnnealingPlacement &current,
    const task_schedulers::LogicalPlacementIslandGraph &islandGraph,
    llvm::ArrayRef<AnnealingMoveKind> enabledMoveKinds, int64_t moveRadius,
    std::mt19937 &randomEngine) {
  unsigned activePlacementCount = getActivePlacementCount(current, islandGraph);
  if (activePlacementCount == 0 || current.physicalArrayOrder.size() < 2)
    return AnnealingMove{};

  llvm::SmallVector<AnnealingMoveKind, 8> feasibleMoveKinds;
  for (AnnealingMoveKind moveKind : enabledMoveKinds) {
    if (isMoveKindFeasible(moveKind, activePlacementCount,
                           current.physicalArrayOrder.size()))
      feasibleMoveKinds.push_back(moveKind);
  }

  if (feasibleMoveKinds.empty())
    return AnnealingMove{};

  std::uniform_int_distribution<unsigned> moveKindDistribution(
      0, feasibleMoveKinds.size() - 1);
  switch (feasibleMoveKinds[moveKindDistribution(randomEngine)]) {
  case AnnealingMoveKind::None:
    return AnnealingMove{};
  case AnnealingMoveKind::MoveOnePosition:
    return chooseMoveOnePosition(
        activePlacementCount,
        static_cast<unsigned>(current.physicalArrayOrder.size()), moveRadius,
        randomEngine);
  case AnnealingMoveKind::MoveOneRelocation:
    return chooseMoveOneRelocation(
        activePlacementCount,
        static_cast<unsigned>(current.physicalArrayOrder.size()), moveRadius,
        randomEngine);
  case AnnealingMoveKind::SwapTwoPositions:
    return chooseSwapTwoPositions(activePlacementCount, randomEngine);
  case AnnealingMoveKind::AdjacentSwap:
    return chooseAdjacentSwap(activePlacementCount, randomEngine);
  case AnnealingMoveKind::SegmentReverse:
    return chooseSegmentReverse(activePlacementCount, randomEngine);
  case AnnealingMoveKind::SegmentRelocation:
    return chooseSegmentRelocation(activePlacementCount, randomEngine);
  case AnnealingMoveKind::BlockSwap:
    return chooseBlockSwap(activePlacementCount, randomEngine);
  }

  return AnnealingMove{};
}

static mlir::FailureOr<AnnealingPlacement>
applyPerturbationMove(mlir::func::FuncOp taskGraphFunc,
                      const AnnealingPlacement &current,
                      const AnnealingMove &move) {
  AnnealingPlacement candidate = current;
  switch (move.kind) {
  case AnnealingMoveKind::None:
    return candidate;
  case AnnealingMoveKind::MoveOnePosition:
  case AnnealingMoveKind::SwapTwoPositions:
  case AnnealingMoveKind::AdjacentSwap:
    if (move.firstIndex >= candidate.physicalArrayOrder.size() ||
        move.secondIndex >= candidate.physicalArrayOrder.size()) {
      taskGraphFunc.emitError("simulated annealing perturbation selected "
                              "an out-of-range placement index");
      return mlir::failure();
    }
    std::swap(candidate.physicalArrayOrder[move.firstIndex],
              candidate.physicalArrayOrder[move.secondIndex]);
    return candidate;
  case AnnealingMoveKind::MoveOneRelocation: {
    if (move.firstIndex >= candidate.physicalArrayOrder.size() ||
        move.insertIndex >= candidate.physicalArrayOrder.size()) {
      taskGraphFunc.emitError("simulated annealing one-position relocation "
                              "selected an out-of-range placement index");
      return mlir::failure();
    }
    int64_t physicalArray = candidate.physicalArrayOrder[move.firstIndex];
    candidate.physicalArrayOrder.erase(candidate.physicalArrayOrder.begin() +
                                       move.firstIndex);
    candidate.physicalArrayOrder.insert(
        candidate.physicalArrayOrder.begin() + move.insertIndex, physicalArray);
    return candidate;
  }
  case AnnealingMoveKind::SegmentReverse:
    if (move.firstIndex >= candidate.physicalArrayOrder.size() ||
        move.secondIndex >= candidate.physicalArrayOrder.size() ||
        move.firstIndex >= move.secondIndex) {
      taskGraphFunc.emitError("simulated annealing segment reverse selected "
                              "an invalid placement range");
      return mlir::failure();
    }
    std::reverse(candidate.physicalArrayOrder.begin() + move.firstIndex,
                 candidate.physicalArrayOrder.begin() + move.secondIndex + 1);
    return candidate;
  case AnnealingMoveKind::SegmentRelocation: {
    if (move.segmentLength == 0 ||
        move.firstIndex + move.segmentLength >
            candidate.physicalArrayOrder.size() ||
        move.insertIndex >
            candidate.physicalArrayOrder.size() - move.segmentLength) {
      taskGraphFunc.emitError("simulated annealing segment relocation selected "
                              "an invalid placement range");
      return mlir::failure();
    }

    llvm::SmallVector<int64_t, 8> segment(
        candidate.physicalArrayOrder.begin() + move.firstIndex,
        candidate.physicalArrayOrder.begin() + move.firstIndex +
            move.segmentLength);
    candidate.physicalArrayOrder.erase(
        candidate.physicalArrayOrder.begin() + move.firstIndex,
        candidate.physicalArrayOrder.begin() + move.firstIndex +
            move.segmentLength);
    candidate.physicalArrayOrder.insert(candidate.physicalArrayOrder.begin() +
                                            move.insertIndex,
                                        segment.begin(), segment.end());
    return candidate;
  }
  case AnnealingMoveKind::BlockSwap:
    if (move.segmentLength == 0 ||
        move.firstIndex + move.segmentLength >
            candidate.physicalArrayOrder.size() ||
        move.secondIndex + move.segmentLength >
            candidate.physicalArrayOrder.size() ||
        move.firstIndex + move.segmentLength > move.secondIndex) {
      taskGraphFunc.emitError("simulated annealing block swap selected an "
                              "invalid placement range");
      return mlir::failure();
    }
    std::swap_ranges(candidate.physicalArrayOrder.begin() + move.firstIndex,
                     candidate.physicalArrayOrder.begin() + move.firstIndex +
                         move.segmentLength,
                     candidate.physicalArrayOrder.begin() + move.secondIndex);
    return candidate;
  }

  taskGraphFunc.emitError("unknown simulated annealing perturbation move");
  return mlir::failure();
}

static mlir::FailureOr<AnnealingPlacement> perturbPlacementImpl(
    mlir::func::FuncOp taskGraphFunc, const AnnealingPlacement &current,
    const task_schedulers::TaskGraphDAG &dag,
    const task_schedulers::LogicalPlacementIslandGraph &islandGraph,
    llvm::ArrayRef<AnnealingMoveKind> enabledMoveKinds, int64_t moveRadius,
    std::mt19937 &randomEngine) {
  (void)dag;

  AnnealingMove move = choosePerturbationMove(
      current, islandGraph, enabledMoveKinds, moveRadius, randomEngine);
  return applyPerturbationMove(taskGraphFunc, current, move);
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace annealing_detail {

FailureOr<Placement>
perturbPlacement(func::FuncOp taskGraphFunc, const Placement &current,
                 const TaskGraphDAG &dag,
                 const LogicalPlacementIslandGraph &islandGraph,
                 llvm::ArrayRef<MoveKind> enabledMoveKinds, int64_t moveRadius,
                 std::mt19937 &randomEngine) {
  return perturbPlacementImpl(taskGraphFunc, current, dag, islandGraph,
                              enabledMoveKinds, moveRadius, randomEngine);
}

} // namespace annealing_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
