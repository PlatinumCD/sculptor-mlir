#ifndef SCULPTOR_TASK_GRAPH_RUNTIME_BACKEND_H
#define SCULPTOR_TASK_GRAPH_RUNTIME_BACKEND_H

#include "TaskGraphRuntime.h"

#include <cstdint>

namespace sculptor::runtime {

constexpr int32_t kTaskCoreIdNone = -1;
constexpr int32_t kRuntimeErrorInvalidArgument = -1;
constexpr int32_t kRuntimeErrorInvalidGraph = -2;

struct RuntimeState {
  GraphImage *graph = nullptr;
  ExecContext exec{};
  RuntimeTask *runtimeTasks = nullptr;
  TaskCall *taskCalls = nullptr;
  uint32_t slotCount = 0;
};

bool validateGraph(const GraphImage *graph);
bool initializeRuntimeState(RuntimeState *runtime, GraphImage *graph);
void destroyRuntimeState(RuntimeState *runtime);
int32_t bindRuntimeIO(RuntimeState *runtime, const void *const *inputs,
                      void *const *outputs);
int32_t executeTaskRangeSerial(RuntimeState *runtime, uint32_t begin,
                               uint32_t end);

} // namespace sculptor::runtime

#endif // SCULPTOR_TASK_GRAPH_RUNTIME_BACKEND_H
