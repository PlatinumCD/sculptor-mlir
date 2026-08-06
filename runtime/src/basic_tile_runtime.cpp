#include "golem/runtime/basic_tile_runtime.h"

namespace golem::runtime {

namespace {

void storeWord(uint8_t* bytes, uint32_t word) {
    bytes[0] = static_cast<uint8_t>(word);
    bytes[1] = static_cast<uint8_t>(word >> 8U);
    bytes[2] = static_cast<uint8_t>(word >> 16U);
    bytes[3] = static_cast<uint8_t>(word >> 24U);
}

}  // namespace

BasicTileRuntime::BasicTileRuntime(
    TileABI abi,
    WordTransport transport
) noexcept
    : abi_(abi),
      transport_(transport),
      booted_(false) {
}

bool BasicTileRuntime::valid() const noexcept {
    const bool has_routes =
        abi_.incoming_route_count != 0 ||
        abi_.outgoing_route_count != 0;
    return abi_.valid() && (!has_routes || transport_.valid());
}

bool BasicTileRuntime::boot() noexcept {
    if (booted_) {
        return true;
    }
    if (!valid()) {
        return false;
    }

    for (uint32_t index = 0; index < abi_.boot_task_count; ++index) {
        const Task& task = abi_.boot_tasks[index];
        if (task.execute(nullptr, 0, nullptr, 0) != TaskStatus::Success) {
            return false;
        }
    }

    booted_ = true;
    return true;
}

bool BasicTileRuntime::booted() const noexcept {
    return booted_;
}

bool BasicTileRuntime::execute(
    uint32_t task_id,
    const Tensor* inputs,
    uint32_t input_count,
    Tensor* outputs,
    uint32_t output_count
) const noexcept {
    if (!booted_) {
        return false;
    }

    const Task* task = abi_.dispatch_tasks.find(task_id);
    if (task == nullptr ||
        task->input_count != input_count ||
        task->output_count != output_count ||
        (input_count != 0 && inputs == nullptr) ||
        (output_count != 0 && outputs == nullptr)) {
        return false;
    }

    return task->execute(
        inputs,
        input_count,
        outputs,
        output_count
    ) == TaskStatus::Success;
}

bool BasicTileRuntime::sendRoute(
    uint32_t route_id,
    const void* data,
    uint64_t byte_size
) const noexcept {
    const Route* route = abi_.findOutgoingRoute(route_id);
    if (!booted_ ||
        !transport_.valid() ||
        route == nullptr ||
        data == nullptr ||
        route->byte_size != byte_size ||
        byte_size % sizeof(uint32_t) != 0) {
        return false;
    }

    const auto* words = static_cast<const uint32_t*>(data);
    const uint64_t word_count = byte_size / sizeof(uint32_t);
    for (uint64_t offset = 0; offset < word_count;) {
        const uint32_t burst_words = static_cast<uint32_t>(
            word_count - offset < kTransportBurstWords
                ? word_count - offset
                : kTransportBurstWords
        );
        while (!transport_.trySendWords(
            route->destination_core,
            words + offset,
            burst_words
        )) {
        }
        offset += burst_words;
    }
    return true;
}

bool BasicTileRuntime::receiveRoute(
    uint32_t route_id,
    void* data,
    uint64_t byte_size
) const noexcept {
    const Route* route = abi_.findIncomingRoute(route_id);
    if (!booted_ ||
        !transport_.valid() ||
        route == nullptr ||
        data == nullptr ||
        route->byte_size != byte_size ||
        byte_size % sizeof(uint32_t) != 0) {
        return false;
    }

    auto* bytes = static_cast<uint8_t*>(data);
    for (uint64_t offset = 0; offset < byte_size; offset += sizeof(uint32_t)) {
        uint32_t word = 0;
        while (!transport_.tryReceive(&word)) {
        }
        storeWord(bytes + offset, word);
    }
    return true;
}

const TileABI& BasicTileRuntime::abi() const noexcept {
    return abi_;
}

}  // namespace golem::runtime
