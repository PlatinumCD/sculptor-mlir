#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace task_attrs {

inline constexpr llvm::StringLiteral kTaskDomainAttrName("sculptor.task_domain");
inline constexpr llvm::StringLiteral kTaskKindAttrName("sculptor.task_kind");
inline constexpr llvm::StringLiteral kTaskNameAttrName("sculptor.task_name");
inline constexpr llvm::StringLiteral
    kSourceLayerAttrName("sculptor.source_layer");
inline constexpr llvm::StringLiteral
    kSourceTaskOrdinalAttrName("sculptor.source_task_ordinal");

} // namespace task_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKATTRS_H
