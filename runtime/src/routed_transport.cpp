#include "golem/runtime/routed_transport.h"

namespace golem::runtime {

bool RoutedWordTransport::valid() const noexcept {
    return try_send != nullptr && try_receive != nullptr;
}

bool RoutedWordTransport::receiveDMAAvailable() const noexcept {
    return try_start_receive_words != nullptr &&
           try_receive_words_completion != nullptr;
}

bool RoutedWordTransport::trySend(
    uint32_t destination_tile,
    uint32_t word
) const noexcept {
    return try_send != nullptr &&
           try_send(context, destination_tile, word);
}

bool RoutedWordTransport::trySendWords(
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

bool RoutedWordTransport::tryReceive(RoutedWord* word) const noexcept {
    return try_receive != nullptr &&
           word != nullptr &&
           try_receive(context, word);
}

bool RoutedWordTransport::tryStartReceiveWords(
    uint32_t source_tile,
    uint32_t route_id,
    void* destination,
    uint32_t word_count
) const noexcept {
    return try_start_receive_words != nullptr &&
           destination != nullptr &&
           word_count != 0 &&
           try_start_receive_words(
               context,
               source_tile,
               route_id,
               destination,
               word_count
           );
}

bool RoutedWordTransport::tryReceiveWordsCompletion(
    uint32_t* source_tile,
    uint32_t* route_id
) const noexcept {
    return try_receive_words_completion != nullptr &&
           source_tile != nullptr &&
           route_id != nullptr &&
           try_receive_words_completion(
               context, source_tile, route_id);
}

}  // namespace golem::runtime
