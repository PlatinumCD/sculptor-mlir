#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHSCRATCHPADATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHSCRATCHPADATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::sculptor::scratchpad_attrs {

inline constexpr llvm::StringLiteral
    kStorageClassAttrName("sculptor.runtime.storage_class");
inline constexpr llvm::StringLiteral kScratchpadStorageClass("scratchpad");
inline constexpr llvm::StringLiteral
    kScratchpadOffsetAttrName("sculptor.runtime.scratchpad_offset");
inline constexpr llvm::StringLiteral kScratchpadRequiredBytesAttrName(
    "sculptor.runtime.scratchpad_required_bytes");
inline constexpr llvm::StringLiteral
    kScratchpadAlignmentAttrName("sculptor.runtime.scratchpad_alignment");
inline constexpr llvm::StringLiteral kScratchpadDMADescriptorsAttrName(
    "sculptor.runtime.scratchpad_dma_descriptors");
inline constexpr llvm::StringLiteral
    kScratchpadABIVersionAttrName("sculptor.runtime.scratchpad_abi_version");
inline constexpr llvm::StringLiteral
    kScratchpadFeatureBitsAttrName("sculptor.runtime.scratchpad_feature_bits");

inline constexpr llvm::StringLiteral kDMAIdFieldName("descriptor_id");
inline constexpr llvm::StringLiteral kDMADirectionFieldName("direction");
inline constexpr llvm::StringLiteral
    kDMAGlobalResourceIdFieldName("global_resource_id");
inline constexpr llvm::StringLiteral kDMARouteIdFieldName("route_id");
inline constexpr llvm::StringLiteral
    kDMAScratchpadOffsetFieldName("scratchpad_offset");
inline constexpr llvm::StringLiteral kDMAByteSizeFieldName("byte_size");
inline constexpr llvm::StringLiteral
    kDMACompletionTokenFieldName("completion_token_id");
inline constexpr llvm::StringLiteral kDMATriggerKindFieldName("trigger_kind");
inline constexpr llvm::StringLiteral kDMATriggerIdFieldName("trigger_id");
inline constexpr llvm::StringLiteral kDMAFlagsFieldName("flags");
inline constexpr llvm::StringLiteral
    kDMASourceStorageFieldName("source_storage");
inline constexpr llvm::StringLiteral
    kDMADestinationStorageFieldName("destination_storage");
inline constexpr llvm::StringLiteral kDMAReservedFieldName("reserved");

inline constexpr uint32_t kDirectionBackingToScratchpad = 0;
inline constexpr uint32_t kDirectionScratchpadToBacking = 1;
inline constexpr uint32_t kDirectionNicToScratchpad = 2;
inline constexpr uint32_t kDirectionScratchpadToNic = 3;
inline constexpr uint32_t kStorageBacking = 0;
inline constexpr uint32_t kStorageScratchpad = 1;
inline constexpr uint32_t kStorageNic = 2;
inline constexpr uint32_t kTriggerBoot = 0;
inline constexpr uint32_t kTriggerResourceReady = 1;
inline constexpr uint32_t kTriggerRouteArrival = 2;
inline constexpr uint32_t kTriggerTaskComplete = 3;
inline constexpr uint32_t kDMAAsynchronous = 1U << 0;
inline constexpr uint32_t kInvalidU32 = UINT32_MAX;

} // namespace mlir::sculptor::scratchpad_attrs

#endif
