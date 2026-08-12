#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_TEMPORALPLACEMENTMODEL_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_TEMPORALPLACEMENTMODEL_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>

namespace mlir {
namespace sculptor {
namespace mapping {

struct LogicalTilePlacementProblem;

struct TemporalTaskEvent {
  int64_t eventId = -1;
  int64_t leafId = -1;
  int64_t logicalTileId = -1;
  int64_t operationId = -1;
  int64_t workUnitId = -1;
  LogicalLaneKind laneKind = LogicalLaneKind::Digital;
  int64_t laneIndex = -1;
  double durationNs = 0.0;
};

struct TemporalPlacementEvaluation {
  double makespanNs = 0.0;
  double taskTimeOnCriticalChainNs = 0.0;
  double exposedTransportNs = 0.0;
  double exposedContentionNs = 0.0;
  double maximumTileLoadNs = 0.0;
  int64_t maximumDirectedLinkWords = 0;
  SmallVector<int64_t> criticalEventIds;
};

class IncrementalTemporalPlacementEvaluator {
public:
  static FailureOr<std::unique_ptr<IncrementalTemporalPlacementEvaluator>>
  create(const LogicalTilePlacementProblem &problem,
         ArrayRef<int64_t> initialPlacement);

  ~IncrementalTemporalPlacementEvaluator();
  IncrementalTemporalPlacementEvaluator(
      IncrementalTemporalPlacementEvaluator &&) noexcept;
  IncrementalTemporalPlacementEvaluator &
  operator=(IncrementalTemporalPlacementEvaluator &&) noexcept;

  const TemporalPlacementEvaluation &getCurrentEvaluation() const;

  FailureOr<TemporalPlacementEvaluation>
  evaluateCandidate(ArrayRef<int64_t> placement,
                    ArrayRef<unsigned> changedLogicalTileIndices);

  void commitCandidate();
  void discardCandidate();

  FailureOr<TemporalPlacementEvaluation>
  evaluateFromScratch(ArrayRef<int64_t> placement) const;

private:
  struct Impl;
  explicit IncrementalTemporalPlacementEvaluator(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl;
};

FailureOr<TemporalPlacementEvaluation>
evaluateTemporalPlacement(const LogicalTilePlacementProblem &problem,
                          ArrayRef<int64_t> physicalTileByLogicalTileIndex);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif
