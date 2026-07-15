#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTIMINGATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTIMINGATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace timing_attrs {

inline constexpr llvm::StringLiteral
    kTopologicalIndexAttrName("sculptor.timing.topological_index");
inline constexpr llvm::StringLiteral
    kDependencyDepthAttrName("sculptor.timing.dependency_depth");
inline constexpr llvm::StringLiteral kControlPredecessorCountAttrName(
    "sculptor.timing.control_predecessor_count");
inline constexpr llvm::StringLiteral
    kDataPredecessorCountAttrName("sculptor.timing.data_predecessor_count");
inline constexpr llvm::StringLiteral
    kIncomingDataBytesAttrName("sculptor.timing.incoming_data_bytes");
inline constexpr llvm::StringLiteral
    kOutgoingDataBytesAttrName("sculptor.timing.outgoing_data_bytes");
inline constexpr llvm::StringLiteral
    kDigitalOpsAttrName("sculptor.timing.digital_ops");
inline constexpr llvm::StringLiteral
    kAnalogLoadLatencyNsAttrName("sculptor.timing.analog_load_latency_ns");
inline constexpr llvm::StringLiteral kAnalogExecuteLatencyNsAttrName(
    "sculptor.timing.analog_execute_latency_ns");
inline constexpr llvm::StringLiteral
    kAnalogStoreLatencyNsAttrName("sculptor.timing.analog_store_latency_ns");
inline constexpr llvm::StringLiteral
    kIntrinsicLatencyNsAttrName("sculptor.timing.intrinsic_latency_ns");
inline constexpr llvm::StringLiteral
    kEarliestStartNsAttrName("sculptor.timing.earliest_start_ns");
inline constexpr llvm::StringLiteral
    kEarliestFinishNsAttrName("sculptor.timing.earliest_finish_ns");
inline constexpr llvm::StringLiteral kCriticalPathRemainingNsAttrName(
    "sculptor.timing.critical_path_remaining_ns");
inline constexpr llvm::StringLiteral
    kSlackNsAttrName("sculptor.timing.slack_ns");
inline constexpr llvm::StringLiteral
    kIsCriticalAttrName("sculptor.timing.is_critical");
inline constexpr llvm::StringLiteral kIncomingNetworkDelayNsAttrName(
    "sculptor.timing.incoming_network_delay_ns");

inline constexpr llvm::StringLiteral
    kTaskCountAttrName("sculptor.timing.task_count");
inline constexpr llvm::StringLiteral
    kExecutionEdgeCountAttrName("sculptor.timing.execution_edge_count");
inline constexpr llvm::StringLiteral
    kControlEdgeCountAttrName("sculptor.timing.control_edge_count");
inline constexpr llvm::StringLiteral
    kDataEdgeCountAttrName("sculptor.timing.data_edge_count");
inline constexpr llvm::StringLiteral
    kExecutionDepthAttrName("sculptor.timing.execution_depth");
inline constexpr llvm::StringLiteral
    kCriticalPathNsAttrName("sculptor.timing.critical_path_ns");
inline constexpr llvm::StringLiteral
    kTotalDataBytesAttrName("sculptor.timing.total_data_bytes");
inline constexpr llvm::StringLiteral
    kPlacementAwareAttrName("sculptor.timing.placement_aware");
inline constexpr llvm::StringLiteral
    kTotalNetworkLatencyNsAttrName("sculptor.timing.total_network_latency_ns");
inline constexpr llvm::StringLiteral kTotalNetworkContentionDelayNsAttrName(
    "sculptor.timing.total_network_contention_delay_ns");

inline constexpr llvm::StringLiteral
    kTimingModelAttrName("sculptor.timing.model");
inline constexpr llvm::StringLiteral
    kNetworkEdgesAttrName("sculptor.timing.network_edges");
inline constexpr llvm::StringLiteral
    kIslandProfilesAttrName("sculptor.timing.islands");
inline constexpr llvm::StringLiteral
    kTimedIslandEdgesAttrName("sculptor.timing.island_edges");

} // namespace timing_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTIMINGATTRS_H
