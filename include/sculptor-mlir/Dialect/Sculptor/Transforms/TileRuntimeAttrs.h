#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILERUNTIMEATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILERUNTIMEATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace tile_runtime_attrs {

inline constexpr llvm::StringLiteral
    kTaskGraphResourceCountAttrName("sculptor.runtime.resource_count");
inline constexpr llvm::StringLiteral
    kTaskGraphInputSlotsAttrName("sculptor.runtime.input_slots");
inline constexpr llvm::StringLiteral
    kTaskGraphOutputSlotsAttrName("sculptor.runtime.output_slots");
inline constexpr llvm::StringLiteral
    kTaskGraphRouteInputSlotsAttrName("sculptor.runtime.route_input_slots");
inline constexpr llvm::StringLiteral
    kTaskGraphRouteOutputSlotsAttrName("sculptor.runtime.route_output_slots");
inline constexpr llvm::StringLiteral
    kTaskGraphTempOffsetsAttrName("sculptor.runtime.temp_offsets");
inline constexpr llvm::StringLiteral
    kTaskGraphTempBaseSlotAttrName("sculptor.runtime.temp_base_slot");
inline constexpr llvm::StringLiteral
    kTaskGraphTempCountAttrName("sculptor.runtime.temp_count");
inline constexpr llvm::StringLiteral
    kTaskGraphWorkspaceSizeAttrName("sculptor.runtime.workspace_size");

inline constexpr llvm::StringLiteral
    kResourceSlotAttrName("sculptor.runtime.slot");
inline constexpr llvm::StringLiteral
    kResourceByteSizeAttrName("sculptor.runtime.byte_size");
inline constexpr llvm::StringLiteral
    kResourceTempIndexAttrName("sculptor.runtime.temp_index");
inline constexpr llvm::StringLiteral
    kResourceTempOffsetAttrName("sculptor.runtime.temp_offset");

inline constexpr llvm::StringLiteral
    kTaskIndexAttrName("sculptor.runtime.task_index");
inline constexpr llvm::StringLiteral
    kTaskCoreIdAttrName("sculptor.runtime.core_id");
inline constexpr llvm::StringLiteral
    kTaskPhysicalArrayIdAttrName("sculptor.runtime.physical_array_id");
inline constexpr llvm::StringLiteral
    kTaskLocalArrayIdAttrName("sculptor.runtime.local_array_id");
inline constexpr llvm::StringLiteral
    kTaskArrayBindingsAttrName("sculptor.runtime.array_bindings");
inline constexpr llvm::StringLiteral
    kArrayBindingInputIndexFieldName("input_index");
inline constexpr llvm::StringLiteral
    kArrayBindingPhysicalIdFieldName("physical_array_id");
inline constexpr llvm::StringLiteral
    kArrayBindingLocalIdFieldName("local_array_id");
inline constexpr llvm::StringLiteral
    kTaskInputSlotsAttrName("sculptor.runtime.input_slots");
inline constexpr llvm::StringLiteral
    kTaskOutputSlotsAttrName("sculptor.runtime.output_slots");
inline constexpr llvm::StringLiteral
    kTaskResultIndicesAttrName("sculptor.runtime.result_indices");

} // namespace tile_runtime_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILERUNTIMEATTRS_H
