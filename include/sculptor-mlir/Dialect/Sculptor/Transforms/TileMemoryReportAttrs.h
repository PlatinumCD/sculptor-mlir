#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILEMEMORYREPORTATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILEMEMORYREPORTATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::sculptor::tile_memory_report_attrs {

inline constexpr llvm::StringLiteral
    kReportsAttrName("sculptor.memory.reports");
inline constexpr llvm::StringLiteral kSchemaVersionFieldName("schema_version");
inline constexpr llvm::StringLiteral kStageFieldName("stage");

} // namespace mlir::sculptor::tile_memory_report_attrs

#endif
