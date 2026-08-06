#include "golem/runtime/transport.h"

namespace golem::runtime {

bool WordTransport::valid() const noexcept {
    return try_send != nullptr && try_receive != nullptr;
}

bool WordTransport::trySend(
    uint32_t destination_tile,
    uint32_t word
) const noexcept {
    return try_send != nullptr &&
           try_send(context, destination_tile, word);
}

bool WordTransport::trySendWords(
    uint32_t destination_tile,
    const uint32_t* words,
    uint32_t word_count
) const noexcept {
    if (words == nullptr || word_count == 0) {
        return false;
    }
    if (try_send_words != nullptr) {
        return try_send_words(
            context, destination_tile, words, word_count);
    }
    if (try_send == nullptr) {
        return false;
    }
    for (uint32_t index = 0; index < word_count; ++index) {
        while (!trySend(destination_tile, words[index])) {
        }
    }
    return true;
}

bool WordTransport::tryReceive(uint32_t* word) const noexcept {
    return try_receive != nullptr &&
           word != nullptr &&
           try_receive(context, word);
}

}  // namespace golem::runtime
