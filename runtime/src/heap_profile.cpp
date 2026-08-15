#include "golem/runtime/heap_profile.h"

namespace golem::runtime {

namespace {

struct alignas(max_align_t) ProfiledAllocationHeader {
  uint64_t byte_count;
};

static_assert(sizeof(ProfiledAllocationHeader) % alignof(max_align_t) == 0);

HeapProfile *active_profile = nullptr;

extern "C" void *malloc(size_t size);
extern "C" void free(void *allocation);

} // namespace

void resetHeapProfile(HeapProfile &profile) noexcept { profile = {}; }

void recordHeapAllocation(HeapProfile &profile, uint64_t byte_count,
                          bool succeeded) noexcept {
  if (!succeeded) {
    ++profile.failed_allocation_count;
    profile.failed_allocation_size = byte_count;
    return;
  }

  ++profile.allocation_count;
  if (profile.current_live_bytes > UINT64_MAX - byte_count) {
    profile.current_live_bytes = UINT64_MAX;
  } else {
    profile.current_live_bytes += byte_count;
  }
  if (profile.current_live_bytes > profile.peak_live_bytes) {
    profile.peak_live_bytes = profile.current_live_bytes;
  }
}

void recordHeapDeallocation(HeapProfile &profile,
                            uint64_t byte_count) noexcept {
  if (byte_count > profile.current_live_bytes) {
    profile.current_live_bytes = 0;
    return;
  }
  profile.current_live_bytes -= byte_count;
}

extern "C" void golem_runtime_set_heap_profile(HeapProfile *profile) {
  active_profile = profile;
}

extern "C" void *golem_runtime_profiled_malloc(size_t byte_count) {
  if (byte_count > SIZE_MAX - sizeof(ProfiledAllocationHeader)) {
    if (active_profile != nullptr) {
      recordHeapAllocation(*active_profile, byte_count, false);
    }
    return nullptr;
  }

  auto *header = static_cast<ProfiledAllocationHeader *>(
      malloc(byte_count + sizeof(ProfiledAllocationHeader)));
  if (active_profile != nullptr) {
    recordHeapAllocation(*active_profile, byte_count, header != nullptr);
  }
  if (header == nullptr) {
    return nullptr;
  }
  header->byte_count = byte_count;
  return header + 1;
}

extern "C" void golem_runtime_profiled_free(void *allocation) {
  if (allocation == nullptr) {
    return;
  }
  auto *header = static_cast<ProfiledAllocationHeader *>(allocation) - 1;
  if (active_profile != nullptr) {
    recordHeapDeallocation(*active_profile, header->byte_count);
  }
  free(header);
}

} // namespace golem::runtime
