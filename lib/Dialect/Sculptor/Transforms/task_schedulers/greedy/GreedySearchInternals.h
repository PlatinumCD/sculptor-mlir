#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYSEARCHINTERNALS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYSEARCHINTERNALS_H

#include "GreedyHeuristic.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleConfig.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace greedy_detail {

using CorePhysicalArraySlots =
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 16>;

struct IslandPlacement {
  unsigned island = 0;
  int64_t physicalArrayId = 0;
};

struct PlacementState {
  llvm::SmallVector<unsigned, 16> usedSlotsByCore;
  llvm::DenseMap<unsigned, int64_t> coreByPlacedIsland;
  llvm::SmallVector<IslandPlacement, 8> islandPlacements;
  int64_t currentCore = 0;
  int64_t score = 0;
};

struct ExpansionRequest {
  unsigned island = 0;
  unsigned placementIndex = 0;
  unsigned totalPlacementCount = 0;
  bool pruneCandidates = false;
};

FailureOr<CorePhysicalArraySlots>
buildCorePhysicalArraySlots(Operation *diagnosticOp,
                            const HardwareBudget &budget,
                            llvm::ArrayRef<int64_t> physicalArrayOrder);

int64_t minDistanceToPlacedRegion(
    int64_t candidateCore, const HardwareBudget &budget,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland);

bool isBetterChoice(
    bool hasBest, int64_t candidateScore, int64_t candidateCore,
    int64_t bestScore, int64_t bestCore, int64_t currentCore,
    const HardwareBudget &budget,
    const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland);

bool applyCandidate(PlacementState &state, unsigned island,
                    int64_t candidateCore,
                    const CorePhysicalArraySlots &physicalArraysByCore);

llvm::SmallVector<PlacementState, 16> expandState(
    const PlacementState &state, const ExpansionRequest &request,
    const HardwareBudget &budget, const GreedyScheduleConfig &config,
    const GreedyHeuristic &heuristic,
    const CorePhysicalArraySlots &physicalArraysByCore,
    llvm::ArrayRef<IslandAffinityEdge> islandAffinityEdges,
    const PlacementConstraints &constraints);

void repairBoundaryRegretPlacement(
    llvm::SmallVectorImpl<IslandPlacement> &islandPlacements,
    const HardwareBudget &budget, llvm::ArrayRef<int64_t> physicalArrayOrder,
    llvm::ArrayRef<IslandAffinityEdge> islandAffinityEdges,
    const PlacementConstraints &constraints);

} // namespace greedy_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYSEARCHINTERNALS_H
