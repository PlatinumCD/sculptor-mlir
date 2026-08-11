#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_REDUCTIONTREE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_REDUCTIONTREE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sculptor {
namespace mapping {

inline constexpr StringLiteral kReductionTreePolicyAttrName =
    "sculptor.mapping.reduction_tree_policy";
inline constexpr StringLiteral kReductionTreeFingerprintAttrName =
    "sculptor.mapping.reduction_tree_fingerprint";
inline constexpr StringLiteral kReductionTreeIdAttrName =
    "sculptor.mapping.reduction_tree_id";
inline constexpr StringLiteral kReductionNodeIdAttrName =
    "sculptor.mapping.reduction_node_id";
inline constexpr StringLiteral kReductionLevelAttrName =
    "sculptor.mapping.reduction_level";
inline constexpr StringLiteral kReductionOrdinalAttrName =
    "sculptor.mapping.reduction_ordinal";
inline constexpr StringLiteral kReductionWidthAttrName =
    "sculptor.mapping.reduction_width";

enum class ReductionTreePolicy { None, Balanced };

FailureOr<ReductionTreePolicy> parseReductionTreePolicy(StringRef value,
                                                        Operation *anchor);
StringRef stringifyReductionTreePolicy(ReductionTreePolicy policy);

FailureOr<int64_t> buildReductionTrees(func::FuncOp function,
                                       ReductionTreePolicy policy,
                                       int64_t fanIn, int64_t minimumWidth);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif
