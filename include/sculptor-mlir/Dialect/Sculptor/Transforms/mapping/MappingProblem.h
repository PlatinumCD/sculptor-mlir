#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGPROBLEM_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGPROBLEM_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

namespace mlir {
namespace sculptor {
namespace mapping {

struct MappingProblem {
  const ComputeGraph &graph;
  const ResourceAllocationTree &currentTree;
  const MappingHardwareModel &hardware;
  LogicalTileShape logicalTileShape;
  MappingObjective objective;
  bool mvmWaveColocation = false;
  bool balanceDigitalWork = false;
  /// Intermediate refinements need costs and structural feasibility, but only
  /// the final tree needs a concrete logical-tile realization.
  bool requireRealization = true;
  Operation *anchor = nullptr;
};

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGPROBLEM_H
