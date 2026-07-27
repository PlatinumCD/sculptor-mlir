#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHOPTIMIZATIONATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHOPTIMIZATIONATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace optimization_attrs {

inline constexpr llvm::StringLiteral
    kStreamingConvolutionAttrName("sculptor.optimize.streaming_convolution");
inline constexpr llvm::StringLiteral kInputShapeFieldName("input_shape");
inline constexpr llvm::StringLiteral kOutputShapeFieldName("output_shape");
inline constexpr llvm::StringLiteral kKernelShapeFieldName("kernel_shape");
inline constexpr llvm::StringLiteral kStrideFieldName("stride");
inline constexpr llvm::StringLiteral kPaddingFieldName("padding");
inline constexpr llvm::StringLiteral kDilationFieldName("dilation");
inline constexpr llvm::StringLiteral kHasBiasFieldName("has_bias");
inline constexpr llvm::StringLiteral kBiasFieldName("bias");

inline constexpr llvm::StringLiteral
    kSourceIslandIdsAttrName("sculptor.optimize.source_island_ids");

} // namespace optimization_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHOPTIMIZATIONATTRS_H
