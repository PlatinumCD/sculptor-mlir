#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHWORKLOADATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHWORKLOADATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace workload_attrs {

inline constexpr llvm::StringLiteral
    kIncomingDataBytesAttrName("sculptor.workload.incoming_data_bytes");
inline constexpr llvm::StringLiteral
    kOutgoingDataBytesAttrName("sculptor.workload.outgoing_data_bytes");
inline constexpr llvm::StringLiteral
    kDigitalOpsAttrName("sculptor.workload.digital_ops");
inline constexpr llvm::StringLiteral
    kDigitalReplacementOpsAttrName("sculptor.workload.digital_replacement_ops");
inline constexpr llvm::StringLiteral
    kAnalogLoadBytesAttrName("sculptor.workload.analog_load_bytes");
inline constexpr llvm::StringLiteral
    kAnalogExecutionCountAttrName("sculptor.workload.analog_execution_count");
inline constexpr llvm::StringLiteral
    kAnalogStoreBytesAttrName("sculptor.workload.analog_store_bytes");
inline constexpr llvm::StringLiteral
    kStaticElementsAttrName("sculptor.workload.static_elements");
inline constexpr llvm::StringLiteral
    kLocalBytesReadAttrName("sculptor.workload.local_bytes_read");
inline constexpr llvm::StringLiteral
    kLocalBytesWrittenAttrName("sculptor.workload.local_bytes_written");
inline constexpr llvm::StringLiteral
    kLoopIterationsAttrName("sculptor.workload.loop_iterations");

} // namespace workload_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHWORKLOADATTRS_H
