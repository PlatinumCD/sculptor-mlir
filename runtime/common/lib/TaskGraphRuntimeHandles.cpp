#include "TaskGraphRuntimeCommon.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace sculptor::runtime {
namespace {

constexpr uint32_t kPersistentHandleMagic = 0x414E474C;

using PersistentHandleDestroyFn = void (*)(void *);

struct PersistentHandleHeader {
  uint32_t magic;
  uint32_t reserved;
  PersistentHandleDestroyFn destroy;
};

} // namespace

void destroyPersistentHandle(void *handle) {
  if (!handle)
    return;

  auto *header = reinterpret_cast<PersistentHandleHeader *>(
      static_cast<uint8_t *>(handle) - sizeof(PersistentHandleHeader));
  if (header->magic != kPersistentHandleMagic) {
    std::free(header);
    return;
  }

  if (header->destroy)
    header->destroy(handle);
  std::free(header);
}

void *createPersistentHandle(uint64_t payload_size, void *destroy_fn) {
  size_t allocationSize =
      sizeof(PersistentHandleHeader) + static_cast<size_t>(payload_size);
  auto *header =
      static_cast<PersistentHandleHeader *>(std::calloc(1, allocationSize));
  if (!header)
    return nullptr;

  header->magic = kPersistentHandleMagic;
  header->destroy = reinterpret_cast<PersistentHandleDestroyFn>(destroy_fn);
  return reinterpret_cast<uint8_t *>(header) + sizeof(PersistentHandleHeader);
}

} // namespace sculptor::runtime

extern "C" {

void *sculptor_runtime_persistent_handle_create(uint64_t payload_size,
                                              void *destroy_fn) {
  return sculptor::runtime::createPersistentHandle(payload_size, destroy_fn);
}

} // extern "C"
