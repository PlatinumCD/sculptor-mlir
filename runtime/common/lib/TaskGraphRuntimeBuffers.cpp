#include "TaskGraphRuntimeCommon.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

const TaskCall *castTaskCall(void *opaque) {
  return static_cast<const TaskCall *>(opaque);
}

bool pointerAliasesRuntimeBuffer(const TaskCall *call, void *ptr) {
  if (!call || !call->graph || !call->exec || !call->exec->slots || !ptr)
    return false;

  std::uintptr_t pointer = reinterpret_cast<std::uintptr_t>(ptr);
  uint32_t slotCount = sculptor::runtime::computeSlotCount(call->graph);
  for (uint32_t i = 0; i < slotCount; ++i) {
    const SlotValue &slot = call->exec->slots[i];
    if (slot.kind != RES_BUFFER || !slot.as.buffer.data ||
        slot.as.buffer.byte_size == 0) {
      continue;
    }

    std::uintptr_t begin =
        reinterpret_cast<std::uintptr_t>(slot.as.buffer.data);
    std::uintptr_t end =
        begin + static_cast<std::uintptr_t>(slot.as.buffer.byte_size);
    if (pointer >= begin && pointer < end)
      return true;
  }

  return false;
}

} // namespace

extern "C" {

void sculptor_runtime_copy_to_buffer(void *dst, const void *src, uint64_t size) {
  if (!dst || !src || size == 0)
    return;
  std::memcpy(dst, src, static_cast<size_t>(size));
}

void sculptor_runtime_free_result_buffer(void *opaque, void *ptr) {
  if (!ptr)
    return;

  if (pointerAliasesRuntimeBuffer(castTaskCall(opaque), ptr))
    return;

  std::free(ptr);
}

} // extern "C"
