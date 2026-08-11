#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/PlacementObjective.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/TemporalPlacementModel.h"

#include <cmath>
#include <limits>

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<PlacementObjectiveKind> parsePlacementObjective(StringRef value,
                                                          Operation *anchor) {
  if (value == "transfer-cost")
    return PlacementObjectiveKind::TransferCost;
  if (value == "makespan")
    return PlacementObjectiveKind::Makespan;
  anchor->emitError("unknown logical-tile placement objective '")
      << value << "'";
  return failure();
}

FailureOr<TemporalNetworkMode> parseTemporalNetworkMode(StringRef value,
                                                        Operation *anchor) {
  if (value == "ideal")
    return TemporalNetworkMode::Ideal;
  if (value == "finite")
    return TemporalNetworkMode::Finite;
  if (value == "full")
    return TemporalNetworkMode::Full;
  anchor->emitError("unknown temporal network mode '") << value << "'";
  return failure();
}

FailureOr<TemporalTimingScope> parseTemporalTimingScope(StringRef value,
                                                        Operation *anchor) {
  if (value == "warm")
    return TemporalTimingScope::Warm;
  if (value == "cold")
    return TemporalTimingScope::Cold;
  anchor->emitError("unknown temporal timing scope '") << value << "'";
  return failure();
}

StringRef stringifyPlacementObjective(PlacementObjectiveKind value) {
  switch (value) {
  case PlacementObjectiveKind::TransferCost:
    return "transfer-cost";
  case PlacementObjectiveKind::Makespan:
    return "makespan";
  }
  llvm_unreachable("unknown placement objective");
}

StringRef stringifyTemporalNetworkMode(TemporalNetworkMode value) {
  switch (value) {
  case TemporalNetworkMode::Ideal:
    return "ideal";
  case TemporalNetworkMode::Finite:
    return "finite";
  case TemporalNetworkMode::Full:
    return "full";
  }
  llvm_unreachable("unknown temporal network mode");
}

StringRef stringifyTemporalTimingScope(TemporalTimingScope value) {
  switch (value) {
  case TemporalTimingScope::Warm:
    return "warm";
  case TemporalTimingScope::Cold:
    return "cold";
  }
  llvm_unreachable("unknown temporal timing scope");
}

FailureOr<PlacementObjectiveEvaluation>
evaluatePlacementObjective(const LogicalTilePlacementProblem &problem,
                           ArrayRef<int64_t> physicalTileByLogicalTileIndex) {
  FailureOr<int64_t> transfer =
      scoreLogicalTilePlacement(problem, physicalTileByLogicalTileIndex);
  if (failed(transfer))
    return failure();

  PlacementObjectiveEvaluation result;
  result.transferCost = *transfer;
  if (problem.objective == PlacementObjectiveKind::TransferCost) {
    result.score = *transfer;
    return result;
  }

  FailureOr<TemporalPlacementEvaluation> temporal =
      evaluateTemporalPlacement(problem, physicalTileByLogicalTileIndex);
  if (failed(temporal))
    return failure();
  if (!std::isfinite(temporal->makespanNs) || temporal->makespanNs < 0.0 ||
      temporal->makespanNs >
          static_cast<double>(std::numeric_limits<int64_t>::max())) {
    problem.anchor->emitError("temporal placement makespan is out of range");
    return failure();
  }
  result.makespanNs = temporal->makespanNs;
  result.score = static_cast<int64_t>(std::ceil(temporal->makespanNs));
  return result;
}

FailureOr<int64_t>
scorePlacementObjective(const LogicalTilePlacementProblem &problem,
                        ArrayRef<int64_t> physicalTileByLogicalTileIndex) {
  FailureOr<PlacementObjectiveEvaluation> evaluation =
      evaluatePlacementObjective(problem, physicalTileByLogicalTileIndex);
  if (failed(evaluation))
    return failure();
  return evaluation->score;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
