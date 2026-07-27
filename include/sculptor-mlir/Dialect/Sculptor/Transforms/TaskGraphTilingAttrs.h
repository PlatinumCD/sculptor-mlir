#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTILINGATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTILINGATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace tiling_attrs {

inline constexpr llvm::StringLiteral
    kSourceResourceAttrName("sculptor.source_resource");
inline constexpr llvm::StringLiteral kTileAttrName("sculptor.tile");
inline constexpr llvm::StringLiteral kTileGridAttrName("sculptor.tile_grid");
inline constexpr llvm::StringLiteral
    kTilePhysicalShapeAttrName("sculptor.tile_physical_shape");
inline constexpr llvm::StringLiteral
    kTileValidShapeAttrName("sculptor.tile_valid_shape");
inline constexpr llvm::StringLiteral
    kVectorTileAttrName("sculptor.vector_tile");
inline constexpr llvm::StringLiteral
    kVectorTileGridAttrName("sculptor.vector_tile_grid");
inline constexpr llvm::StringLiteral
    kVectorTilePhysicalColsAttrName("sculptor.vector_tile_physical_cols");
inline constexpr llvm::StringLiteral
    kVectorTileValidColsAttrName("sculptor.vector_tile_valid_cols");

} // namespace tiling_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTILINGATTRS_H
