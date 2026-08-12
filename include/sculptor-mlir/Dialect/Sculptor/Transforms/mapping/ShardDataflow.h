#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_SHARDDATAFLOW_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_SHARDDATAFLOW_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace mlir {
namespace sculptor {
namespace mapping {

inline constexpr StringLiteral kShardDataflowModeAttrName =
    "sculptor.mapping.dataflow_mode";
inline constexpr StringLiteral kShardWorkUnitEdgesAttrName =
    "sculptor.mapping.shard_work_unit_edges";
inline constexpr StringLiteral kShardGroupCountAttrName =
    "sculptor.mapping.shard_group_count";
inline constexpr StringLiteral kShardEdgeCountAttrName =
    "sculptor.mapping.shard_edge_count";
inline constexpr StringLiteral kAssemblyBoundaryCountAttrName =
    "sculptor.mapping.assembly_boundary_count";

enum class DigitalDataflowMode { Bulk, Sharded };

struct StaticTileRegion {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
};

std::optional<StaticTileRegion>
mapIterationTileThroughIndexingMap(AffineMap indexingMap,
                                   ArrayRef<int64_t> iterationOffsets,
                                   ArrayRef<int64_t> iterationSizes);

FailureOr<DigitalDataflowMode> parseDigitalDataflowMode(StringRef value,
                                                        Operation *anchor);
StringRef stringifyDigitalDataflowMode(DigitalDataflowMode mode);

LogicalResult planShardDataflow(func::FuncOp function,
                                const ComputeGraph &graph,
                                DigitalDataflowMode mode,
                                int64_t propagationDepth,
                                bool requireCompleteChain);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif
