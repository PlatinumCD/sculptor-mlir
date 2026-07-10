#include "GreedyHeuristic.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementObjective.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

using mlir::sculptor::task_schedulers::HardwareBudget;

static int64_t clampToInt64(long double value) {
  if (value <= 0.0L)
    return 0;
  long double max =
      static_cast<long double>(std::numeric_limits<int64_t>::max());
  if (value >= max)
    return std::numeric_limits<int64_t>::max();
  return static_cast<int64_t>(std::llround(value));
}

static int64_t saturatingAdd(int64_t lhs, int64_t rhs) {
  if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs)
    return std::numeric_limits<int64_t>::max();
  return lhs + rhs;
}

static int64_t getDistanceToBoundaryMask(int64_t coreId, unsigned boundaryMask,
                                         const HardwareBudget &budget) {
  if (boundaryMask == 0)
    return 0;

  int64_t row = mlir::sculptor::task_schedulers::getMeshRow(coreId, budget);
  int64_t col = mlir::sculptor::task_schedulers::getMeshCol(coreId, budget);
  int64_t bestDistance = std::numeric_limits<int64_t>::max();
  if (boundaryMask & mlir::sculptor::task_schedulers::kMeshTopBoundary)
    bestDistance = std::min(bestDistance, row);
  if (boundaryMask & mlir::sculptor::task_schedulers::kMeshBottomBoundary)
    bestDistance = std::min(bestDistance, budget.meshRows - 1 - row);
  if (boundaryMask & mlir::sculptor::task_schedulers::kMeshLeftBoundary)
    bestDistance = std::min(bestDistance, col);
  if (boundaryMask & mlir::sculptor::task_schedulers::kMeshRightBoundary)
    bestDistance = std::min(bestDistance, budget.meshCols - 1 - col);

  return bestDistance == std::numeric_limits<int64_t>::max() ? 0 : bestDistance;
}

static int64_t getAverageIslandCommunicationBytes(
    const mlir::sculptor::task_schedulers::GreedyHeuristicContext &context) {
  int64_t totalBytes = 0;
  for (const auto &edge : context.islandCommunicationEdges)
    totalBytes = saturatingAdd(totalBytes, edge.byteSize);
  return std::max<int64_t>(
      1, totalBytes / std::max<unsigned>(1, context.totalPlacementCount));
}

static long double getPlacementProgressRamp(
    const mlir::sculptor::task_schedulers::GreedyHeuristicContext &context) {
  unsigned denominator = std::max<unsigned>(1, context.totalPlacementCount - 1);
  long double progress =
      static_cast<long double>(context.activePlacementIndex) /
      static_cast<long double>(denominator);
  return progress * progress * progress * progress;
}

static int64_t getCompactRegionShapePenalty(
    const mlir::sculptor::task_schedulers::GreedyHeuristicContext &context) {
  if (context.coreByPlacedIsland.empty())
    return 0;

  llvm::DenseSet<int64_t> occupiedCores;
  int64_t minRow = std::numeric_limits<int64_t>::max();
  int64_t maxRow = std::numeric_limits<int64_t>::min();
  int64_t minCol = std::numeric_limits<int64_t>::max();
  int64_t maxCol = std::numeric_limits<int64_t>::min();
  for (const auto &placedIsland : context.coreByPlacedIsland) {
    int64_t coreId = placedIsland.second;
    if (!occupiedCores.insert(coreId).second)
      continue;

    int64_t row =
        mlir::sculptor::task_schedulers::getMeshRow(coreId, context.budget);
    int64_t col =
        mlir::sculptor::task_schedulers::getMeshCol(coreId, context.budget);
    minRow = std::min(minRow, row);
    maxRow = std::max(maxRow, row);
    minCol = std::min(minCol, col);
    maxCol = std::max(maxCol, col);
  }

  if (occupiedCores.empty())
    return 0;

  int64_t height = maxRow - minRow + 1;
  int64_t width = maxCol - minCol + 1;
  int64_t area = height * width;
  int64_t occupied = static_cast<int64_t>(occupiedCores.size());
  int64_t shortSide = std::max<int64_t>(1, std::min(height, width));
  int64_t longSide = std::max(height, width);

  int64_t aspectUnits = std::abs(height - width) * 2;
  int64_t sparseUnits = std::max<int64_t>(0, area - occupied);
  int64_t stripUnits = std::max<int64_t>(0, longSide - 3 * shortSide) * 8;

  int64_t boundaryStripUnits = 0;
  if (minCol == 0 && height > width)
    boundaryStripUnits += (height - width) * 4;
  if (maxCol == context.budget.meshCols - 1 && height > width)
    boundaryStripUnits += (height - width) * 4;
  if (minRow == 0 && width > height)
    boundaryStripUnits += (width - height) * 4;
  if (maxRow == context.budget.meshRows - 1 && width > height)
    boundaryStripUnits += (width - height) * 4;

  int64_t shapeUnits =
      aspectUnits + sparseUnits + stripUnits + boundaryStripUnits;
  int64_t bytesScale =
      std::max<int64_t>(1, getAverageIslandCommunicationBytes(context) / 256);
  return clampToInt64(static_cast<long double>(shapeUnits) *
                      static_cast<long double>(bytesScale));
}

static int64_t getShapeRegretAllowance(
    const mlir::sculptor::task_schedulers::GreedyHeuristicContext &context,
    int64_t transferScore) {
  int64_t baseline = context.bestTransferCost
                         ? *context.bestTransferCost
                         : std::max<int64_t>(transferScore, 1);
  int64_t averageBytes = getAverageIslandCommunicationBytes(context);
  int64_t percentAllowance = std::max<int64_t>(1, baseline / 100);
  return std::max<int64_t>(averageBytes, percentAllowance);
}

static int64_t getBoundaryRegretPenalty(
    const mlir::sculptor::task_schedulers::GreedyHeuristicContext &context,
    int64_t transferScore) {
  if (!context.firstTaskIsland || !context.lastTaskIsland ||
      context.totalPlacementCount == 0)
    return 0;

  auto firstCoreIt = context.coreByPlacedIsland.find(*context.firstTaskIsland);
  auto activeCoreIt = context.coreByPlacedIsland.find(context.activeIsland);
  if (firstCoreIt == context.coreByPlacedIsland.end() ||
      activeCoreIt == context.coreByPlacedIsland.end())
    return 0;

  unsigned startBoundaryMask =
      mlir::sculptor::task_schedulers::getMeshBoundaryMask(firstCoreIt->second,
                                                           context.budget);
  if (startBoundaryMask == 0)
    return mlir::sculptor::task_schedulers::getBoundaryPenalty(
        std::max<int64_t>(transferScore, 1));

  unsigned activeBoundaryMask =
      mlir::sculptor::task_schedulers::getMeshBoundaryMask(activeCoreIt->second,
                                                           context.budget);
  bool boundaryCompatible = (activeBoundaryMask & startBoundaryMask) != 0;
  if (context.activeIsland == *context.lastTaskIsland) {
    if (boundaryCompatible)
      return 0;
    return mlir::sculptor::task_schedulers::getBoundaryPenalty(
        std::max<int64_t>(transferScore, 1));
  }

  if (!context.bestTransferCost)
    return 0;

  long double ramp = getPlacementProgressRamp(context);
  int64_t regretAllowance =
      clampToInt64(static_cast<long double>(
                       mlir::sculptor::task_schedulers::getBoundaryPenalty(
                           std::max<int64_t>(*context.bestTransferCost, 1))) *
                   ramp);
  if (regretAllowance <= 0)
    return 0;
  if (transferScore > saturatingAdd(*context.bestTransferCost, regretAllowance))
    return 0;

  int64_t distance = getDistanceToBoundaryMask(
      activeCoreIt->second, startBoundaryMask, context.budget);
  int64_t averageBytes = getAverageIslandCommunicationBytes(context);
  int64_t rawBoundaryPenalty =
      clampToInt64(static_cast<long double>(distance) *
                   static_cast<long double>(averageBytes) * ramp);
  return std::min(rawBoundaryPenalty, regretAllowance);
}

static int64_t getCompactRegionPenalty(
    const mlir::sculptor::task_schedulers::GreedyHeuristicContext &context,
    int64_t transferScore) {
  if (!context.bestTransferCost)
    return 0;

  int64_t regretAllowance = getShapeRegretAllowance(context, transferScore);
  if (transferScore > saturatingAdd(*context.bestTransferCost, regretAllowance))
    return 0;

  return std::min(getCompactRegionShapePenalty(context), regretAllowance);
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

int64_t TransferCostGreedyHeuristic::evaluate(
    const GreedyHeuristicContext &context) const {
  return computeIslandTransferCost(context.budget,
                                   context.islandCommunicationEdges,
                                   context.coreByPlacedIsland);
}

int64_t BoundaryRegretGreedyHeuristic::evaluate(
    const GreedyHeuristicContext &context) const {
  TransferCostGreedyHeuristic transferCost;
  int64_t transferScore = transferCost.evaluate(context);
  return saturatingAdd(transferScore,
                       getBoundaryRegretPenalty(context, transferScore));
}

int64_t CompactRegionGreedyHeuristic::evaluate(
    const GreedyHeuristicContext &context) const {
  TransferCostGreedyHeuristic transferCost;
  int64_t transferScore = transferCost.evaluate(context);
  return saturatingAdd(transferScore,
                       getCompactRegionPenalty(context, transferScore));
}

int64_t CompositeGreedyHeuristic::evaluate(
    const GreedyHeuristicContext &context) const {
  TransferCostGreedyHeuristic transferCost;
  int64_t transferScore = transferCost.evaluate(context);
  int64_t score = transferScore;

  if (boundaryRegret) {
    score =
        saturatingAdd(score, getBoundaryRegretPenalty(context, transferScore));
  }

  if (compactRegion)
    score =
        saturatingAdd(score, getCompactRegionPenalty(context, transferScore));

  return score;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
