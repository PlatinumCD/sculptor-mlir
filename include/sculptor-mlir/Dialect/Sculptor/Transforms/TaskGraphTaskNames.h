#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKNAMES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKNAMES_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace task_graph_names {

inline constexpr llvm::StringLiteral kAnalogDomain("analog");
inline constexpr llvm::StringLiteral kDigitalDomain("digital");

inline constexpr llvm::StringLiteral kAnalogTaskKindPrefix("sculptor.");
inline constexpr llvm::StringLiteral
    kMatrixSetupTaskKind("sculptor.matrix_setup");
inline constexpr llvm::StringLiteral kMVMTaskKind("sculptor.mvm");
inline constexpr llvm::StringLiteral
    kMVMSequenceTaskKind("sculptor.mvm_sequence");
inline constexpr llvm::StringLiteral
    kConvTileMVMTaskKind("sculptor.conv_tile_mvm");

inline constexpr llvm::StringLiteral kConvPatchTaskKind("digital.conv_patch");
inline constexpr llvm::StringLiteral kVectorTileTaskKind("digital.vector_tile");
inline constexpr llvm::StringLiteral
    kTileRecombineTaskKind("digital.tile_recombine");
inline constexpr llvm::StringLiteral kBiasAddTaskKind("digital.bias_add");
inline constexpr llvm::StringLiteral kReductionTaskKind("digital.reduction");
inline constexpr llvm::StringLiteral
    kStreamingConvolutionTaskKind("mixed.streaming_conv_mvm");
inline constexpr llvm::StringLiteral kDigitalMatmulTaskKind("digital.matmul");
inline constexpr llvm::StringLiteral
    kDigitalMatmulPartitionTaskKind("digital.matmul_partition");
inline constexpr llvm::StringLiteral
    kDigitalMatmulShardTaskKind("digital.matmul_shard");
inline constexpr llvm::StringLiteral
    kDigitalMatmulAssemblyTaskKind("digital.matmul_assembly");
inline constexpr llvm::StringLiteral
    kDigitalAttentionScoresTaskKind("digital.attention_scores");
inline constexpr llvm::StringLiteral
    kDigitalAttentionApplyTaskKind("digital.attention_apply");

inline constexpr llvm::StringLiteral kForwardSourceLayer("forward");

} // namespace task_graph_names
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKNAMES_H
