#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGREALIZATION_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGREALIZATION_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingProblem.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {
namespace mapping {

struct MappingAnalogLaneRef {
  int64_t tileId = -1;
  int64_t laneIndex = -1;

  bool operator==(const MappingAnalogLaneRef &other) const {
    return tileId == other.tileId && laneIndex == other.laneIndex;
  }
};

// The exact logical resources inherited by one RA-tree node. T-cut children
// inherit the same resources. S-cut children receive disjoint lane subsets.
struct MappingNodeResourceAllocation {
  int64_t nodeId = -1;
  SmallVector<int64_t> digitalTileIds;
  SmallVector<MappingAnalogLaneRef> analogLanes;
};

struct MappingLeafAssignment {
  int64_t leafId = -1;
  int64_t operationId = -1;
  int64_t tileId = -1;
  LogicalLaneKind laneKind = LogicalLaneKind::Digital;
  int64_t laneIndex = -1;
  double startNs = 0.0;
  double finishNs = 0.0;
};

// Compact logical-tile neighborhood available to one semantic layer region.
// Fine-grained leaf placement remains free inside this pool.
struct MappingLayerTilePool {
  int64_t layerRegionId = -1;
  SmallVector<int64_t> tileIds;
};

struct MappingRealization {
  bool feasible = true;
  std::string infeasibilityReason;
  int64_t logicalTileCount = 0;
  int64_t analogLanesPerTile = 0;
  double estimatedMakespanNs = 0.0;
  double estimatedCommunicationNs = 0.0;
  SmallVector<int64_t> digitalWorkPerTile;
  SmallVector<MappingLayerTilePool> layerTilePools;
  SmallVector<MappingNodeResourceAllocation> nodeAllocations;
  SmallVector<MappingLeafAssignment> leafAssignments;
};

FailureOr<MappingRealization>
realizeResourceAllocationTree(const MappingProblem &problem,
                              const ResourceAllocationTree &tree);

LogicalResult verifyMappingRealization(const MappingRealization &realization,
                                       const MappingProblem &problem,
                                       const ResourceAllocationTree &tree);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGREALIZATION_H
