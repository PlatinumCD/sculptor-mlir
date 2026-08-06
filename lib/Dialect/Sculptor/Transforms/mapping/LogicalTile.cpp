#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<LogicalTileShape> buildLogicalTileShape(int64_t analogLaneCount,
                                                  Operation *anchor) {
  if (analogLaneCount <= 0) {
    anchor->emitError("expected a positive logical analog lane count");
    return failure();
  }

  LogicalTileShape tile;
  tile.analogLanes.reserve(analogLaneCount);
  for (int64_t lane = 0; lane < analogLaneCount; ++lane)
    tile.analogLanes.push_back({LogicalLaneKind::Analog, lane});
  return tile;
}

std::optional<LogicalLaneKind>
classifyLogicalLaneRequirement(ComputeOperationKind operationKind) {
  switch (operationKind) {
  case ComputeOperationKind::MatrixSetup:
  case ComputeOperationKind::PhysicalMVM:
    return LogicalLaneKind::Analog;
  case ComputeOperationKind::Structured:
  case ComputeOperationKind::DigitalStage:
  case ComputeOperationKind::VectorTile:
  case ComputeOperationKind::TileRecombine:
    return LogicalLaneKind::Digital;
  case ComputeOperationKind::LogicalMVM:
    return std::nullopt;
  }
  llvm_unreachable("unknown compute operation kind");
}

StringRef stringifyLogicalLaneKind(LogicalLaneKind kind) {
  switch (kind) {
  case LogicalLaneKind::Digital:
    return "digital";
  case LogicalLaneKind::Analog:
    return "analog";
  }
  llvm_unreachable("unknown logical lane kind");
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
