#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHSCHEDULEATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHSCHEDULEATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace schedule_attrs {

inline constexpr llvm::StringLiteral
    kNumCoresAttrName("sculptor.schedule.num_cores");
inline constexpr llvm::StringLiteral
    kArraysPerCoreAttrName("sculptor.schedule.arrays_per_core");
inline constexpr llvm::StringLiteral
    kTopologyAttrName("sculptor.schedule.topology");
inline constexpr llvm::StringLiteral
    kMeshRowsAttrName("sculptor.schedule.mesh_rows");
inline constexpr llvm::StringLiteral
    kMeshColsAttrName("sculptor.schedule.mesh_cols");
inline constexpr llvm::StringLiteral
    kNumAnalogArraysAttrName("sculptor.schedule.num_analog_arrays");
inline constexpr llvm::StringLiteral
    kAnalogArraysAttrName("sculptor.schedule.analog_arrays");
inline constexpr llvm::StringLiteral
    kGreedyLookaheadAttrName("sculptor.schedule.greedy_lookahead");
inline constexpr llvm::StringLiteral
    kGreedyBeamWidthAttrName("sculptor.schedule.greedy_beam_width");
inline constexpr llvm::StringLiteral
    kGreedyCandidateScopeAttrName("sculptor.schedule.greedy_candidate_scope");
inline constexpr llvm::StringLiteral
    kGreedyHeuristicAttrName("sculptor.schedule.greedy_heuristic");
inline constexpr llvm::StringLiteral
    kAnnealingMoveSetAttrName("sculptor.schedule.annealing_move_set");
inline constexpr llvm::StringLiteral
    kAnnealingMoveRadiusAttrName("sculptor.schedule.annealing_move_radius");
inline constexpr llvm::StringLiteral
    kTaskCountAttrName("sculptor.schedule.task_count");
inline constexpr llvm::StringLiteral
    kDependencyCountAttrName("sculptor.schedule.dependency_count");
inline constexpr llvm::StringLiteral
    kCoreTransferBytesAttrName("sculptor.schedule.core_transfer_bytes");
inline constexpr llvm::StringLiteral kInterCoreTransferBytesAttrName(
    "sculptor.schedule.inter_core_transfer_bytes");
inline constexpr llvm::StringLiteral
    kCoreTransferCostAttrName("sculptor.schedule.core_transfer_cost");
inline constexpr llvm::StringLiteral
    kTotalTransferCostAttrName("sculptor.schedule.total_transfer_cost");
inline constexpr llvm::StringLiteral kTransferCostPerInterCoreByteAttrName(
    "sculptor.schedule.transfer_cost_per_inter_core_byte");
inline constexpr llvm::StringLiteral
    kGraphScoreAttrName("sculptor.schedule.graph_score");
inline constexpr llvm::StringLiteral
    kBoundaryPenaltyAttrName("sculptor.schedule.boundary_penalty");
inline constexpr llvm::StringLiteral
    kTotalDigitalOpsAttrName("sculptor.schedule.total_digital_ops");
inline constexpr llvm::StringLiteral
    kNumLogicalArraysAttrName("sculptor.schedule.num_logical_arrays");
inline constexpr llvm::StringLiteral
    kLogicalArrayIndexAttrName("sculptor.schedule.logical_array_index");
inline constexpr llvm::StringLiteral kLogicalArrayToAnalogArrayAttrName(
    "sculptor.schedule.logical_array_to_analog_array");
inline constexpr llvm::StringLiteral
    kIslandIndexAttrName("sculptor.schedule.island");

} // namespace schedule_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHSCHEDULEATTRS_H
