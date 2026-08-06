#pragma once

#include <stdint.h>

namespace golem::runtime {

inline constexpr uint32_t kTransportBurstWords = 4096;

using TrySendWord = bool (*)(
    void* context,
    uint32_t destination_tile,
    uint32_t word
);

using TryReceiveWord = bool (*)(void* context, uint32_t* word);

using TrySendWords = bool (*)(
    void* context,
    uint32_t destination_tile,
    const uint32_t* words,
    uint32_t word_count
);

struct WordTransport {
    void* context;
    TrySendWord try_send;
    TryReceiveWord try_receive;
    TrySendWords try_send_words = nullptr;

    bool valid() const noexcept;
    bool trySend(uint32_t destination_tile, uint32_t word) const noexcept;
    bool trySendWords(
        uint32_t destination_tile,
        const uint32_t* words,
        uint32_t word_count
    ) const noexcept;
    bool tryReceive(uint32_t* word) const noexcept;
};

}  // namespace golem::runtime
