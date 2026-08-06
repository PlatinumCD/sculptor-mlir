#include "golem/runtime/scratchpad_abi.h"

namespace golem::runtime {

namespace {

bool validDirection(ScratchpadDMADirection direction) {
    return direction >= ScratchpadDMADirection::BackingToScratchpad &&
           direction <= ScratchpadDMADirection::ScratchpadToNic;
}

bool validStorage(ScratchpadStorage storage) {
    return storage >= ScratchpadStorage::Backing &&
           storage <= ScratchpadStorage::Nic;
}

bool validTrigger(ScratchpadDMATrigger trigger) {
    return trigger >= ScratchpadDMATrigger::Boot &&
           trigger <= ScratchpadDMATrigger::TaskComplete;
}

}  // namespace

bool ScratchpadDMADescriptor::valid(
    uint64_t scratchpad_bytes
) const noexcept {
    constexpr uint32_t known_flags =
        ScratchpadDMAAsynchronous |
        ScratchpadDMABlocking |
        ScratchpadDMAPingPongZero |
        ScratchpadDMAPingPongOne;
    const bool asynchronous =
        (flags & ScratchpadDMAAsynchronous) != 0;
    const bool blocking = (flags & ScratchpadDMABlocking) != 0;
    const bool ping_zero = (flags & ScratchpadDMAPingPongZero) != 0;
    const bool ping_one = (flags & ScratchpadDMAPingPongOne) != 0;

    if (!validDirection(direction) ||
        !validStorage(source_storage) ||
        !validStorage(destination_storage) ||
        !validTrigger(trigger_kind) ||
        byte_size == 0 ||
        reserved != 0 ||
        (flags & ~known_flags) != 0 ||
        asynchronous == blocking ||
        (ping_zero && ping_one) ||
        scratchpad_offset > scratchpad_bytes ||
        byte_size > scratchpad_bytes - scratchpad_offset) {
        return false;
    }

    switch (direction) {
    case ScratchpadDMADirection::BackingToScratchpad:
        return source_storage == ScratchpadStorage::Backing &&
               destination_storage == ScratchpadStorage::Scratchpad;
    case ScratchpadDMADirection::ScratchpadToBacking:
        return source_storage == ScratchpadStorage::Scratchpad &&
               destination_storage == ScratchpadStorage::Backing;
    case ScratchpadDMADirection::NicToScratchpad:
        return source_storage == ScratchpadStorage::Nic &&
               destination_storage == ScratchpadStorage::Scratchpad;
    case ScratchpadDMADirection::ScratchpadToNic:
        return source_storage == ScratchpadStorage::Scratchpad &&
               destination_storage == ScratchpadStorage::Nic;
    }
    return false;
}

bool ScratchpadABI::valid(uint64_t available_bytes) const noexcept {
    if ((features & ~ScratchpadDMAFeature) != 0 ||
        required_bytes > ScratchpadMaximumBytes ||
        required_bytes > available_bytes ||
        (descriptor_count != 0 && descriptors == nullptr)) {
        return false;
    }
    if ((features & ScratchpadDMAFeature) == 0) {
        return required_bytes == 0 &&
               descriptors == nullptr &&
               descriptor_count == 0;
    }
    for (uint32_t index = 0; index < descriptor_count; ++index) {
        if (descriptors[index].descriptor_id != index ||
            !descriptors[index].valid(required_bytes)) {
            return false;
        }
    }
    return true;
}

}  // namespace golem::runtime
