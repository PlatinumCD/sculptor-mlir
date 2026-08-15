#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_LOGICALTILEPLACEMENT_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_LOGICALTILEPLACEMENT_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/PlacementObjective.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace sculptor {
namespace mapping {

struct ComputeGraph;
struct MappingCostProfile;
struct ResourceAllocationTree;

inline constexpr StringLiteral kLogicalTilePlacementAttrName =
    "sculptor.mapping.logical_tile_placement";
inline constexpr StringLiteral kLogicalTileAnnealingTraceAttrName =
    "sculptor.mapping.logical_tile_annealing_trace";
inline constexpr StringLiteral kLogicalTileGreedyTileOrderAttrName =
    "sculptor.mapping.logical_tile_greedy_tile_order";
inline constexpr StringLiteral kLogicalTileGreedyPriorityModeAttrName =
    "sculptor.mapping.logical_tile_greedy_priority_mode";
inline constexpr StringLiteral kLogicalTileGreedyCandidateScopeAttrName =
    "sculptor.mapping.logical_tile_greedy_candidate_scope";
inline constexpr StringLiteral kLogicalTileGreedyLookaheadAttrName =
    "sculptor.mapping.logical_tile_greedy_lookahead";
inline constexpr StringLiteral kLogicalTileMemoryEstimateMethodAttrName =
    "sculptor.mapping.memory_estimate_method";
inline constexpr StringLiteral kScheduleAwareMemoryEstimateMethod =
    "schedule-aware-lifetimes-v1";

struct PhysicalMeshGeometry {
  int64_t rows = 0;
  int64_t columns = 0;
  int64_t arraysPerCore = 0;
};

struct LogicalTileMemoryEstimate {
  int64_t logicalTileId = -1;
  int64_t persistentBytes = 0;
  // Transient bytes live at the peak-memory event. These are not cumulative
  // bytes touched over the complete schedule.
  int64_t producedBytes = 0;
  int64_t incomingBytes = 0;
  int64_t requiredBytes = 0;
  bool complete = false;
  std::string incompleteReason;
};

struct LogicalTilePlacementProblem {
  LogicalTilePlacementProblem(const LogicalTileGraph &tileGraph,
                              PhysicalMeshGeometry mesh, Operation *anchor)
      : tileGraph(tileGraph), mesh(mesh), anchor(anchor) {}

  const LogicalTileGraph &tileGraph;
  PhysicalMeshGeometry mesh;
  Operation *anchor = nullptr;
  const ComputeGraph *computeGraph = nullptr;
  const ResourceAllocationTree *raTree = nullptr;
  const MappingCostProfile *costProfile = nullptr;
  PlacementObjectiveKind objective = PlacementObjectiveKind::TransferCost;
  TemporalNetworkMode networkMode = TemporalNetworkMode::Finite;
  TemporalTimingScope timingScope = TemporalTimingScope::Warm;
  int64_t temporalCandidateLimit = 8;
  int64_t tileMemoryCapacityBytes = 0;
  SmallVector<LogicalTileMemoryEstimate, 0> memoryEstimates;
  DenseMap<int64_t, unsigned> memoryEstimateIndexByTileId;
};

enum class LogicalTileScheduleKind {
  Random,
  Snake,
  Greedy,
  Annealing,
};

enum class GreedyTileOrder {
  Sequential,
  Priority,
};

enum class GreedyPriorityMode {
  Sum,
  Max,
};

enum class GreedyCandidateScope {
  Cardinal,
  Diagonal,
  Frontier,
};

struct GreedyPlacementConfig {
  GreedyTileOrder tileOrder = GreedyTileOrder::Sequential;
  GreedyPriorityMode priorityMode = GreedyPriorityMode::Sum;
  GreedyCandidateScope candidateScope = GreedyCandidateScope::Cardinal;
  int64_t lookahead = 1;
};

struct LogicalTilePlacementConfig {
  LogicalTileScheduleKind schedule = LogicalTileScheduleKind::Greedy;
  GreedyPlacementConfig greedy;
  LogicalTileScheduleKind annealingInitialSchedule =
      LogicalTileScheduleKind::Greedy;
  int64_t randomSeed = 0;
  int64_t annealingIterations = 1000;
  double annealingInitialTemperature = 0.0;
  double annealingCoolingRate = 0.995;
  int64_t annealingTraceSampleInterval = 1;
  bool annealingIncrementalMakespan = true;
  int64_t annealingMakespanVerifyInterval = 0;
};

struct PhysicalTileLocation {
  int64_t physicalTileId = -1;
  int64_t row = -1;
  int64_t column = -1;
};

struct LogicalTilePhysicalAssignment {
  int64_t logicalTileId = -1;
  PhysicalTileLocation location;
};

struct PlacedLogicalTileEdge {
  int64_t edgeId = -1;
  int64_t sourceTileId = -1;
  int64_t targetTileId = -1;
  int64_t byteSize = 0;
  int64_t manhattanHops = 0;
  int64_t transferCost = 0;
};

struct LogicalTileAnnealingSample {
  int64_t iteration = 0;
  int64_t candidateScore = 0;
  int64_t currentScore = 0;
  int64_t bestScore = 0;
  bool accepted = false;
};

struct LogicalTileAnnealingTrace {
  int64_t version = 1;
  int64_t initialScore = 0;
  int64_t finalScore = 0;
  int64_t evaluations = 0;
  SmallVector<LogicalTileAnnealingSample, 0> samples;
};

struct LogicalTilePlacementPlan {
  int64_t version = 3;
  std::string schedule;
  PlacementObjectiveKind objective = PlacementObjectiveKind::TransferCost;
  TemporalNetworkMode networkMode = TemporalNetworkMode::Finite;
  TemporalTimingScope timingScope = TemporalTimingScope::Warm;
  std::string routePolicy = "xy";
  std::string costProfileName;
  std::string costProfileHash;
  PhysicalMeshGeometry mesh;
  int64_t tileMemoryCapacityBytes = 0;
  SmallVector<LogicalTileMemoryEstimate, 0> memoryEstimates;
  DenseMap<int64_t, unsigned> memoryEstimateIndexByTileId;
  int64_t initialScore = 0;
  int64_t objectiveScore = 0;
  int64_t totalTransferCost = 0;
  double predictedMakespanNs = 0.0;
  int64_t evaluations = 0;
  SmallVector<LogicalTilePhysicalAssignment, 0> assignments;
  SmallVector<PlacedLogicalTileEdge, 0> edges;
  DenseMap<int64_t, unsigned> assignmentIndexByTileId;
  std::optional<LogicalTileAnnealingTrace> annealingTrace;
};

LogicalResult initializeLogicalTilePlacementProblem(
    const ComputeGraph &computeGraph,
    const ResourceAllocationTree &resourceAllocationTree,
    MappingCostProfile &costProfileStorage,
    LogicalTilePlacementProblem &problem);

LogicalResult initializeLogicalTilePlacementProblemFromPlan(
    LogicalTilePlacementAttr placementAttr, const ComputeGraph &computeGraph,
    const ResourceAllocationTree &resourceAllocationTree,
    MappingCostProfile &costProfileStorage,
    LogicalTilePlacementProblem &problem);

FailureOr<LogicalTileScheduleKind>
parseLogicalTileSchedule(StringRef value, Operation *anchor,
                         bool allowAnnealing = true);

StringRef stringifyLogicalTileSchedule(LogicalTileScheduleKind schedule);

FailureOr<GreedyTileOrder> parseGreedyTileOrder(StringRef value,
                                                Operation *anchor);

StringRef stringifyGreedyTileOrder(GreedyTileOrder order);

FailureOr<GreedyPriorityMode> parseGreedyPriorityMode(StringRef value,
                                                      Operation *anchor);

StringRef stringifyGreedyPriorityMode(GreedyPriorityMode mode);

FailureOr<GreedyCandidateScope> parseGreedyCandidateScope(StringRef value,
                                                          Operation *anchor);

StringRef stringifyGreedyCandidateScope(GreedyCandidateScope scope);

LogicalResult
validateLogicalTilePlacementProblem(const LogicalTilePlacementProblem &problem);

FailureOr<int64_t>
scoreLogicalTilePlacement(const LogicalTilePlacementProblem &problem,
                          ArrayRef<int64_t> physicalTileByLogicalTileIndex);

FailureOr<LogicalTilePlacementPlan>
buildLogicalTilePlacementPlan(const LogicalTilePlacementProblem &problem,
                              ArrayRef<int64_t> physicalTileByLogicalTileIndex,
                              StringRef schedule, int64_t initialScore = 0,
                              int64_t evaluations = 0);

FailureOr<LogicalTilePlacementPlan>
scheduleLogicalTiles(const LogicalTilePlacementProblem &problem,
                     const LogicalTilePlacementConfig &config);

LogicalResult
verifyLogicalTilePlacementPlan(const LogicalTilePlacementProblem &problem,
                               const LogicalTilePlacementPlan &plan);

LogicalTilePlacementAttr
serializeLogicalTilePlacement(MLIRContext *context,
                              const LogicalTilePlacementPlan &plan);

FailureOr<LogicalTilePlacementPlan>
deserializeLogicalTilePlacement(LogicalTilePlacementAttr attr,
                                const LogicalTilePlacementProblem &problem);

LogicalTileAnnealingTraceAttr
serializeLogicalTileAnnealingTrace(MLIRContext *context,
                                   const LogicalTileAnnealingTrace &trace);

FailureOr<LogicalTileAnnealingTrace>
deserializeLogicalTileAnnealingTrace(LogicalTileAnnealingTraceAttr attr,
                                     const LogicalTilePlacementPlan &plan,
                                     Operation *anchor);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_LOGICALTILEPLACEMENT_H
