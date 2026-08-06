#pragma once

#include <stdint.h>

#include "golem/runtime/transport.h"

namespace golem::runtime {

struct RoutedWord {
    uint32_t source_tile;
    uint32_t payload;
};

using TryReceiveRoutedWord = bool (*)(void* context, RoutedWord* word);

using TryStartReceiveWords = bool (*)(
    void* context,
    uint32_t source_tile,
    uint32_t route_id,
    void* destination,
    uint32_t word_count
);

using TryReceiveWordsCompletion = bool (*)(
    void* context,
    uint32_t* source_tile,
    uint32_t* route_id
);

struct RoutedWordTransport {
    void* context;
    TrySendWord try_send;
    TryReceiveRoutedWord try_receive;
    TrySendWords try_send_words = nullptr;
    TryStartReceiveWords try_start_receive_words = nullptr;
    TryReceiveWordsCompletion try_receive_words_completion = nullptr;

    bool valid() const noexcept;
    bool receiveDMAAvailable() const noexcept;
    bool trySend(uint32_t destination_tile, uint32_t word) const noexcept;
    bool trySendWords(
        uint32_t destination_tile,
        const uint32_t* words,
        uint32_t word_count
    ) const noexcept;
    bool tryReceive(RoutedWord* word) const noexcept;
    bool tryStartReceiveWords(
        uint32_t source_tile,
        uint32_t route_id,
        void* destination,
        uint32_t word_count
    ) const noexcept;
    bool tryReceiveWordsCompletion(
        uint32_t* source_tile,
        uint32_t* route_id
    ) const noexcept;
};

}  // namespace golem::runtime
