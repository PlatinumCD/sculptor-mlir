#pragma once

#include <stddef.h>
#include <stdint.h>

namespace golem::runtime {

// Optional process-wide heap counters for one bare-metal tile. Normal runtime
// allocations are unchanged when no profile is active.
struct HeapProfile {
  uint64_t allocation_count = 0;
  uint64_t current_live_bytes = 0;
  uint64_t peak_live_bytes = 0;
  uint64_t failed_allocation_count = 0;
  uint64_t failed_allocation_size = 0;
};

void resetHeapProfile(HeapProfile &profile) noexcept;
void recordHeapAllocation(HeapProfile &profile, uint64_t byte_count,
                          bool succeeded) noexcept;
void recordHeapDeallocation(HeapProfile &profile, uint64_t byte_count) noexcept;

// Generated routines use these wrappers only when the optional
// --sculptor-instrument-tile-heap pass rewrites their malloc/free calls.
extern "C" void golem_runtime_set_heap_profile(HeapProfile *profile);
extern "C" void *golem_runtime_profiled_malloc(size_t byte_count);
extern "C" void golem_runtime_profiled_free(void *allocation);

} // namespace golem::runtime
