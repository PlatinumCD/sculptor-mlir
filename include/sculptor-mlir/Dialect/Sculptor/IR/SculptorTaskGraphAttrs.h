#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORTASKGRAPHATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORTASKGRAPHATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace task_graph_attrs {

inline constexpr llvm::StringLiteral
    kTaskReductionAttrName("sculptor.task.reduction");
inline constexpr llvm::StringLiteral
    kTaskReductionHelperAttrName("sculptor.task.reduction_helper");
inline constexpr llvm::StringLiteral
    kTaskReductionTreeIdAttrName("sculptor.task.reduction_tree_id");
inline constexpr llvm::StringLiteral
    kTaskReductionLevelAttrName("sculptor.task.reduction_level");
inline constexpr llvm::StringLiteral
    kTaskReductionLaneAttrName("sculptor.task.reduction_lane");
inline constexpr llvm::StringLiteral
    kTaskReductionWidthAttrName("sculptor.task.reduction_width");
inline constexpr llvm::StringLiteral
    kTaskDistributionAttrName("sculptor.task.distribution");

} // namespace task_graph_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORTASKGRAPHATTRS_H
