#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_LOGICALTILE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_LOGICALTILE_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace sculptor {
namespace mapping {

enum class ComputeOperationKind;
struct ComputeGraph;
struct MappingHardwareModel;
struct MappingRealization;
struct ResourceAllocationTree;

inline constexpr StringLiteral kLogicalTileGraphAttrName =
    "sculptor.mapping.logical_tile_graph";

enum class LogicalLaneKind {
  Digital,
  Analog,
};

// A lane index is local to one logical tile. It does not identify a physical
// processor, analog array, mesh coordinate, or runtime resource.
struct LogicalLane {
  LogicalLaneKind kind = LogicalLaneKind::Digital;
  int64_t index = 0;
};

// The resource-capacity shape available while constructing an RA mapping.
// It is a template, not one realized logical tile.
struct LogicalTileShape {
  LogicalLane digitalLane{LogicalLaneKind::Digital, 0};
  SmallVector<LogicalLane, 0> analogLanes;
};

FailureOr<LogicalTileShape> buildLogicalTileShape(int64_t analogLaneCount,
                                                  Operation *anchor);

// One RA leaf or tiled work unit assigned to a lane in a logical tile. The
// RA-node path preserves its temporal/spatial hierarchy without duplicating
// the ResourceAllocationTree.
struct LogicalTileAssignment {
  int64_t leafId = -1;
  int64_t operationId = -1;
  int64_t workUnitId = -1;
  LogicalLaneKind laneKind = LogicalLaneKind::Digital;
  int64_t laneIndex = -1;
  SmallVector<int64_t> raNodePath;
  double startNs = 0.0;
  double finishNs = 0.0;
};

// One abstract analog lane inside a logical tile. A lane-binding group ties a
// matrix setup to every MVM that consumes that programmed array.
struct LogicalTileAnalogLane {
  int64_t laneIndex = -1;
  std::optional<int64_t> laneBindingGroup;
  SmallVector<LogicalTileAssignment> assignments;
};

// One exact ST-graph dependency. A tensor ID of -1 denotes an explicit
// work-unit edge carried by the RA tree rather than a whole tensor edge.
struct LogicalTileDependency {
  int64_t sourceOperationId = -1;
  int64_t sourceWorkUnitId = -1;
  int64_t targetOperationId = -1;
  int64_t targetWorkUnitId = -1;
  int64_t tensorId = -1;
  int64_t byteSize = 0;
};

// A concrete RA-planned placement unit. It owns one digital lane and the
// analog lanes used by its assignments, but has no physical mesh coordinate.
struct LogicalTile {
  int64_t id = -1;
  int64_t digitalWork = 0;
  SmallVector<LogicalTileAssignment> digitalAssignments;
  SmallVector<LogicalTileAnalogLane> analogLanes;
  SmallVector<int64_t> modelInputTensorIds;
  SmallVector<int64_t> modelOutputTensorIds;
  SmallVector<LogicalTileDependency> internalDependencies;
};

// Communication that a physical placer must account for between two logical
// tiles. Details retain the original ST endpoints; byteSize is their checked
// aggregate.
struct LogicalTileEdge {
  int64_t id = -1;
  int64_t sourceTileId = -1;
  int64_t targetTileId = -1;
  int64_t byteSize = 0;
  SmallVector<LogicalTileDependency> dependencies;
};

struct LogicalTileGraph {
  int64_t version = 2;
  int64_t plannedMeshRows = 0;
  int64_t plannedMeshCols = 0;
  int64_t logicalTileCapacity = 0;
  int64_t analogLanesPerTile = 0;
  SmallVector<LogicalTile, 0> tiles;
  SmallVector<LogicalTileEdge> edges;
  DenseMap<int64_t, unsigned> tileIndexById;
};

FailureOr<LogicalTileGraph>
buildLogicalTileGraph(const ComputeGraph &graph,
                      const ResourceAllocationTree &tree,
                      const MappingRealization &realization,
                      const MappingHardwareModel &hardware,
                      Operation *anchor);

LogicalResult verifyLogicalTileGraph(const LogicalTileGraph &tileGraph,
                                     const ComputeGraph &graph,
                                     const ResourceAllocationTree &tree,
                                     Operation *anchor);

LogicalTileGraphAttr serializeLogicalTileGraph(MLIRContext *context,
                                               const LogicalTileGraph &graph);

FailureOr<LogicalTileGraph> deserializeLogicalTileGraph(
    LogicalTileGraphAttr attr, const ComputeGraph &graph,
    const ResourceAllocationTree &tree, Operation *anchor);

std::optional<LogicalLaneKind>
classifyLogicalLaneRequirement(ComputeOperationKind operationKind);

StringRef stringifyLogicalLaneKind(LogicalLaneKind kind);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_LOGICALTILE_H
