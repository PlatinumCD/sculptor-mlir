#ifndef SCULPTOR_TASK_GRAPH_RUNTIME_COMMON_H
#define SCULPTOR_TASK_GRAPH_RUNTIME_COMMON_H

#include "TaskGraphRuntimeBackend.h"

#include <algorithm>
#include <cstdlib>

namespace sculptor::runtime {

template <typename T> T *allocateZeroedArray(uint32_t count) {
  if (count == 0)
    return nullptr;

  return static_cast<T *>(std::calloc(count, sizeof(T)));
}

inline uint32_t computeSlotCount(const GraphImage *graph) {
  uint32_t slotCount = 0;
  if (!graph)
    return slotCount;

  for (uint32_t i = 0; i < graph->resource_count; ++i)
    slotCount = std::max(slotCount, graph->resources[i].slot + 1);
  return slotCount;
}

void destroyPersistentHandle(void *handle);

} // namespace sculptor::runtime

#endif // SCULPTOR_TASK_GRAPH_RUNTIME_COMMON_H
