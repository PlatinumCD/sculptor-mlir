#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTIMINGATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTIMINGATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace timing_attrs {

inline constexpr llvm::StringLiteral
    kGraphGenerationAttrName("sculptor.task_graph.generation");
inline constexpr llvm::StringLiteral
    kAnalysisGenerationAttrName("sculptor.timing.generation");
inline constexpr llvm::StringLiteral
    kTopologicalIndexAttrName("sculptor.timing.topological_index");
inline constexpr llvm::StringLiteral
    kDependencyDepthAttrName("sculptor.timing.dependency_depth");
inline constexpr llvm::StringLiteral
    kLocalRuntimeIndexAttrName("sculptor.timing.local_runtime_index");
inline constexpr llvm::StringLiteral kControlPredecessorCountAttrName(
    "sculptor.timing.control_predecessor_count");
inline constexpr llvm::StringLiteral
    kDataPredecessorCountAttrName("sculptor.timing.data_predecessor_count");
inline constexpr llvm::StringLiteral kScalarInstructionEstimateAttrName(
    "sculptor.timing.scalar_instruction_estimate");
inline constexpr llvm::StringLiteral kVectorInstructionEstimateAttrName(
    "sculptor.timing.vector_instruction_estimate");
inline constexpr llvm::StringLiteral kLoadInstructionEstimateAttrName(
    "sculptor.timing.load_instruction_estimate");
inline constexpr llvm::StringLiteral kStoreInstructionEstimateAttrName(
    "sculptor.timing.store_instruction_estimate");
inline constexpr llvm::StringLiteral kControlInstructionEstimateAttrName(
    "sculptor.timing.control_instruction_estimate");
inline constexpr llvm::StringLiteral
    kRuntimeDispatchCyclesAttrName("sculptor.timing.runtime_dispatch_cycles");
inline constexpr llvm::StringLiteral
    kTaskEntryCyclesAttrName("sculptor.timing.task_entry_cycles");
inline constexpr llvm::StringLiteral
    kTaskExitCyclesAttrName("sculptor.timing.task_exit_cycles");
inline constexpr llvm::StringLiteral
    kPredictedCpuCyclesAttrName("sculptor.timing.predicted_cpu_cycles");
inline constexpr llvm::StringLiteral
    kCostSourceAttrName("sculptor.timing.cost_source");
inline constexpr llvm::StringLiteral
    kCostConfidenceAttrName("sculptor.timing.cost_confidence");
inline constexpr llvm::StringLiteral
    kAnalogLoadLatencyNsAttrName("sculptor.timing.analog_load_latency_ns");
inline constexpr llvm::StringLiteral kAnalogExecuteLatencyNsAttrName(
    "sculptor.timing.analog_execute_latency_ns");
inline constexpr llvm::StringLiteral
    kAnalogStoreLatencyNsAttrName("sculptor.timing.analog_store_latency_ns");
inline constexpr llvm::StringLiteral kAnalogPipelineLatencyNsAttrName(
    "sculptor.timing.analog_pipeline_latency_ns");
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
    kCoreQueueDelayNsAttrName("sculptor.timing.core_queue_delay_ns");
inline constexpr llvm::StringLiteral
    kCausalInputEdgeAttrName("sculptor.timing.causal_input_edge");
inline constexpr llvm::StringLiteral
    kCausalPreviousTaskAttrName("sculptor.timing.causal_previous_task");

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
inline constexpr llvm::StringLiteral kTotalDigitalReplacementOpsAttrName(
    "sculptor.timing.total_digital_replacement_ops");
inline constexpr llvm::StringLiteral
    kMVMCostModeAttrName("sculptor.timing.mvm_cost_mode");
inline constexpr llvm::StringLiteral
    kPlacementAwareAttrName("sculptor.timing.placement_aware");
inline constexpr llvm::StringLiteral kSumEdgeNetworkServiceNsAttrName(
    "sculptor.timing.sum_edge_network_service_ns");
inline constexpr llvm::StringLiteral kSumEdgeNetworkQueueDelayNsAttrName(
    "sculptor.timing.sum_edge_network_queue_delay_ns");
inline constexpr llvm::StringLiteral
    kSumTaskWorkNsAttrName("sculptor.timing.sum_task_work_ns");
inline constexpr llvm::StringLiteral
    kSumCoreQueueDelayNsAttrName("sculptor.timing.sum_core_queue_delay_ns");
inline constexpr llvm::StringLiteral
    kSumNicQueueDelayNsAttrName("sculptor.timing.sum_nic_queue_delay_ns");
inline constexpr llvm::StringLiteral
    kSumLinkQueueDelayNsAttrName("sculptor.timing.sum_link_queue_delay_ns");
inline constexpr llvm::StringLiteral kSumReceiveQueueDelayNsAttrName(
    "sculptor.timing.sum_receive_queue_delay_ns");
inline constexpr llvm::StringLiteral kNoContentionMakespanNsAttrName(
    "sculptor.timing.no_contention_makespan_ns");
inline constexpr llvm::StringLiteral
    kZeroNetworkMakespanNsAttrName("sculptor.timing.zero_network_makespan_ns");
inline constexpr llvm::StringLiteral
    kExposedTransportNsAttrName("sculptor.timing.exposed_transport_ns");
inline constexpr llvm::StringLiteral
    kExposedContentionNsAttrName("sculptor.timing.exposed_contention_ns");
inline constexpr llvm::StringLiteral
    kTotalPayloadWordsAttrName("sculptor.timing.total_payload_words");
inline constexpr llvm::StringLiteral
    kTotalProtocolWordsAttrName("sculptor.timing.total_protocol_words");
inline constexpr llvm::StringLiteral
    kTotalWordHopsAttrName("sculptor.timing.total_word_hops");

inline constexpr llvm::StringLiteral
    kTimingModelAttrName("sculptor.timing.model");
inline constexpr llvm::StringLiteral
    kNetworkEdgesAttrName("sculptor.timing.network_edges");
inline constexpr llvm::StringLiteral
    kIslandProfilesAttrName("sculptor.timing.islands");
inline constexpr llvm::StringLiteral
    kTimedIslandEdgesAttrName("sculptor.timing.island_edges");
inline constexpr llvm::StringLiteral
    kCausalCriticalChainAttrName("sculptor.timing.causal_critical_chain");

} // namespace timing_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTIMINGATTRS_H
