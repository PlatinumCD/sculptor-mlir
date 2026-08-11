#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_PLACEMENTOBJECTIVE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_PLACEMENTOBJECTIVE_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace mapping {

struct LogicalTilePlacementProblem;

enum class PlacementObjectiveKind { TransferCost, Makespan };
enum class TemporalNetworkMode { Ideal, Finite, Full };
enum class TemporalTimingScope { Warm, Cold };

struct PlacementObjectiveEvaluation {
  int64_t score = 0;
  int64_t transferCost = 0;
  double makespanNs = 0.0;
};

FailureOr<PlacementObjectiveKind> parsePlacementObjective(StringRef value,
                                                          Operation *anchor);
FailureOr<TemporalNetworkMode> parseTemporalNetworkMode(StringRef value,
                                                        Operation *anchor);
FailureOr<TemporalTimingScope> parseTemporalTimingScope(StringRef value,
                                                        Operation *anchor);

StringRef stringifyPlacementObjective(PlacementObjectiveKind value);
StringRef stringifyTemporalNetworkMode(TemporalNetworkMode value);
StringRef stringifyTemporalTimingScope(TemporalTimingScope value);

FailureOr<PlacementObjectiveEvaluation>
evaluatePlacementObjective(const LogicalTilePlacementProblem &problem,
                           ArrayRef<int64_t> physicalTileByLogicalTileIndex);

FailureOr<int64_t>
scorePlacementObjective(const LogicalTilePlacementProblem &problem,
                        ArrayRef<int64_t> physicalTileByLogicalTileIndex);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif
