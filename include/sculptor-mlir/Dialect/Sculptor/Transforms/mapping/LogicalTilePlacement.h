#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_LOGICALTILEPLACEMENT_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_LOGICALTILEPLACEMENT_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"

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

inline constexpr StringLiteral kLogicalTilePlacementAttrName =
    "sculptor.mapping.logical_tile_placement";
inline constexpr StringLiteral kLogicalTileAnnealingTraceAttrName =
    "sculptor.mapping.logical_tile_annealing_trace";
inline constexpr StringLiteral kLogicalTileGreedyLookaheadAttrName =
    "sculptor.mapping.logical_tile_greedy_lookahead";
inline constexpr StringLiteral kLogicalTileGreedyBeamWidthAttrName =
    "sculptor.mapping.logical_tile_greedy_beam_width";

struct PhysicalMeshGeometry {
  int64_t rows = 0;
  int64_t columns = 0;
  int64_t arraysPerCore = 0;
};

struct LogicalTilePlacementProblem {
  const LogicalTileGraph &tileGraph;
  PhysicalMeshGeometry mesh;
  Operation *anchor = nullptr;
};

enum class LogicalTileScheduleKind {
  Random,
  Snake,
  Greedy,
  GreedyBeam,
  Annealing,
};

struct LogicalTilePlacementConfig {
  LogicalTileScheduleKind schedule = LogicalTileScheduleKind::Greedy;
  int64_t greedyLookahead = 1;
  int64_t greedyBeamWidth = 8;
  LogicalTileScheduleKind annealingInitialSchedule =
      LogicalTileScheduleKind::Greedy;
  int64_t randomSeed = 0;
  int64_t annealingIterations = 1000;
  double annealingInitialTemperature = 0.0;
  double annealingCoolingRate = 0.995;
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
  int64_t version = 1;
  std::string schedule;
  PhysicalMeshGeometry mesh;
  int64_t initialScore = 0;
  int64_t totalTransferCost = 0;
  int64_t evaluations = 0;
  SmallVector<LogicalTilePhysicalAssignment, 0> assignments;
  SmallVector<PlacedLogicalTileEdge, 0> edges;
  DenseMap<int64_t, unsigned> assignmentIndexByTileId;
  std::optional<LogicalTileAnnealingTrace> annealingTrace;
};

FailureOr<LogicalTileScheduleKind>
parseLogicalTileSchedule(StringRef value, Operation *anchor,
                         bool allowAnnealing = true);

StringRef stringifyLogicalTileSchedule(LogicalTileScheduleKind schedule);

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
