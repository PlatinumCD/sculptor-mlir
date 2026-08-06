#pragma once

#include <stddef.h>
#include <stdint.h>

namespace golem::runtime {

constexpr uint32_t ScratchpadDMAFeature = UINT32_C(1) << 0;
constexpr uint64_t ScratchpadPhysicalBase = UINT64_C(0x90000000);
constexpr uint64_t ScratchpadMaximumBytes = UINT64_C(16) << 20;
constexpr uint64_t ScratchpadDMARegisterBase = UINT64_C(0x10011000);

enum class ScratchpadDMADirection : uint32_t {
    BackingToScratchpad = 0,
    ScratchpadToBacking = 1,
    NicToScratchpad = 2,
    ScratchpadToNic = 3,
};

enum class ScratchpadStorage : uint32_t {
    Backing = 0,
    Scratchpad = 1,
    Nic = 2,
};

enum class ScratchpadDMATrigger : uint32_t {
    Boot = 0,
    ResourceReady = 1,
    RouteArrival = 2,
    TaskComplete = 3,
};

enum ScratchpadDMAFlags : uint32_t {
    ScratchpadDMAAsynchronous = UINT32_C(1) << 0,
    ScratchpadDMABlocking = UINT32_C(1) << 1,
    ScratchpadDMAPingPongZero = UINT32_C(1) << 2,
    ScratchpadDMAPingPongOne = UINT32_C(1) << 3,
};

struct alignas(8) ScratchpadDMADescriptor {
    uint32_t descriptor_id;
    ScratchpadDMADirection direction;
    uint32_t local_slot;
    uint32_t route_id;
    uint64_t scratchpad_offset;
    uint64_t byte_size;
    uint32_t completion_token_id;
    ScratchpadDMATrigger trigger_kind;
    uint32_t trigger_id;
    uint32_t flags;
    ScratchpadStorage source_storage;
    ScratchpadStorage destination_storage;
    uint64_t reserved;

    bool valid(uint64_t scratchpad_bytes) const noexcept;
};

struct ScratchpadABI {
    uint32_t features;
    uint64_t required_bytes;
    const ScratchpadDMADescriptor* descriptors;
    uint32_t descriptor_count;

    bool valid(uint64_t available_bytes) const noexcept;
};

static_assert(offsetof(ScratchpadDMADescriptor, scratchpad_offset) == 16);
static_assert(offsetof(ScratchpadDMADescriptor, byte_size) == 24);
static_assert(offsetof(ScratchpadDMADescriptor, completion_token_id) == 32);
static_assert(offsetof(ScratchpadDMADescriptor, source_storage) == 48);
static_assert(offsetof(ScratchpadDMADescriptor, reserved) == 56);
static_assert(sizeof(ScratchpadDMADescriptor) == 64);

extern "C" uint32_t golem_tile_abi_features() __attribute__((weak));
extern "C" uint64_t golem_tile_scratchpad_required_bytes()
    __attribute__((weak));
extern "C" const ScratchpadDMADescriptor* golem_tile_dma_descriptors()
    __attribute__((weak));
extern "C" uint32_t golem_tile_dma_descriptor_count()
    __attribute__((weak));

inline ScratchpadABI linkedScratchpadABI() noexcept {
    const uint32_t features = golem_tile_abi_features != nullptr
        ? golem_tile_abi_features()
        : 0;
    if ((features & ScratchpadDMAFeature) == 0) {
        return {features, 0, nullptr, 0};
    }
    return {
        features,
        golem_tile_scratchpad_required_bytes != nullptr
            ? golem_tile_scratchpad_required_bytes()
            : 0,
        golem_tile_dma_descriptors != nullptr
            ? golem_tile_dma_descriptors()
            : nullptr,
        golem_tile_dma_descriptor_count != nullptr
            ? golem_tile_dma_descriptor_count()
            : 0,
    };
}

}  // namespace golem::runtime
