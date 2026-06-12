#include "TaskGraphRuntimeCommon.h"

#include <cstdint>
#include <cstdlib>

namespace sculptor::runtime {

bool validateGraph(const GraphImage *graph) {
  if (!graph)
    return false;

  uint32_t slotCount = computeSlotCount(graph);
  for (uint32_t i = 0; i < graph->task_count; ++i) {
    const TaskDesc &task = graph->tasks[i];
    if (task.callable_index >= graph->callable_count)
      return false;
    if (task.arg_begin + task.arg_count > graph->binding_count)
      return false;
    if (task.dep_begin + task.dep_count > graph->dep_count)
      return false;

    for (uint32_t depIndex = 0; depIndex < task.dep_count; ++depIndex) {
      if (graph->deps[task.dep_begin + depIndex] >= i)
        return false;
    }

    for (uint32_t argIndex = 0; argIndex < task.arg_count; ++argIndex) {
      const ArgBinding &binding = graph->bindings[task.arg_begin + argIndex];
      if (binding.source == SRC_SLOT && binding.index >= slotCount)
        return false;
    }
  }

  return true;
}

static bool initializeResourceSlots(RuntimeState *runtime) {
  GraphImage *graph = runtime->graph;

  for (uint32_t i = 0; i < graph->resource_count; ++i) {
    const ResourceDesc &resource = graph->resources[i];
    if (resource.slot >= runtime->slotCount)
      return false;

    SlotValue &slot = runtime->exec.slots[resource.slot];
    slot.kind = resource.kind;
    if (resource.kind == RES_BUFFER) {
      slot.as.buffer.data = nullptr;
      slot.as.buffer.byte_size = resource.byte_size;
      if (resource.storage == STORAGE_TEMP) {
        if (resource.workspace_offset + resource.byte_size >
            graph->workspace_size) {
          return false;
        }

        slot.as.buffer.data =
            runtime->exec.workspace + resource.workspace_offset;
      }
    } else {
      slot.as.handle = nullptr;
    }
  }

  return true;
}

static void bindRuntimeTaskCalls(RuntimeState *runtime) {
  GraphImage *graph = runtime->graph;

  for (uint32_t i = 0; i < graph->task_count; ++i) {
    const TaskDesc &task = graph->tasks[i];
    runtime->runtimeTasks[i].fn = graph->callables[task.callable_index].entry;
    runtime->taskCalls[i].graph = graph;
    runtime->taskCalls[i].task = &graph->tasks[i];
    runtime->taskCalls[i].exec = &runtime->exec;
    runtime->runtimeTasks[i].opaque = &runtime->taskCalls[i];
  }
}

static void destroyPersistentRuntimeHandles(RuntimeState *runtime) {
  if (!runtime->graph || !runtime->exec.slots)
    return;

  for (uint32_t i = 0; i < runtime->graph->resource_count; ++i) {
    const ResourceDesc &resource = runtime->graph->resources[i];
    if (resource.storage != STORAGE_PERSISTENT || resource.kind != RES_HANDLE ||
        resource.slot >= runtime->slotCount) {
      continue;
    }

    destroyPersistentHandle(runtime->exec.slots[resource.slot].as.handle);
    runtime->exec.slots[resource.slot].as.handle = nullptr;
  }
}

bool initializeRuntimeState(RuntimeState *runtime, GraphImage *graph) {
  if (!runtime || !validateGraph(graph))
    return false;

  runtime->graph = graph;
  runtime->slotCount = computeSlotCount(graph);
  runtime->exec.workspace = allocateZeroedArray<uint8_t>(
      static_cast<uint32_t>(graph->workspace_size));
  runtime->exec.slots = allocateZeroedArray<SlotValue>(runtime->slotCount);
  runtime->runtimeTasks = allocateZeroedArray<RuntimeTask>(graph->task_count);
  runtime->taskCalls = allocateZeroedArray<TaskCall>(graph->task_count);

  if ((graph->workspace_size && !runtime->exec.workspace) ||
      (runtime->slotCount && !runtime->exec.slots) ||
      (graph->task_count && !runtime->runtimeTasks) ||
      (graph->task_count && !runtime->taskCalls)) {
    return false;
  }

  if (!initializeResourceSlots(runtime))
    return false;

  bindRuntimeTaskCalls(runtime);
  return true;
}

void destroyRuntimeState(RuntimeState *runtime) {
  if (!runtime)
    return;

  destroyPersistentRuntimeHandles(runtime);
  std::free(runtime->runtimeTasks);
  std::free(runtime->taskCalls);
  std::free(runtime->exec.slots);
  std::free(runtime->exec.workspace);
  sculptor_runtime_graph_destroy(runtime->graph);

  runtime->graph = nullptr;
  runtime->exec = {};
  runtime->runtimeTasks = nullptr;
  runtime->taskCalls = nullptr;
  runtime->slotCount = 0;
}

int32_t bindRuntimeIO(RuntimeState *runtime, const void *const *inputs,
                      void *const *outputs) {
  if (!runtime || !runtime->graph)
    return kRuntimeErrorInvalidArgument;

  uint32_t inputIndex = 0;
  uint32_t outputIndex = 0;
  for (uint32_t i = 0; i < runtime->graph->resource_count; ++i) {
    const ResourceDesc &resource = runtime->graph->resources[i];
    if (resource.slot >= runtime->slotCount)
      return kRuntimeErrorInvalidGraph;

    SlotValue &slot = runtime->exec.slots[resource.slot];
    if (resource.storage == STORAGE_INPUT) {
      if (resource.kind != RES_BUFFER || !inputs || !inputs[inputIndex])
        return kRuntimeErrorInvalidArgument;
      slot.as.buffer.data = const_cast<void *>(inputs[inputIndex++]);
      slot.as.buffer.byte_size = resource.byte_size;
    } else if (resource.storage == STORAGE_OUTPUT) {
      if (resource.kind != RES_BUFFER || !outputs || !outputs[outputIndex])
        return kRuntimeErrorInvalidArgument;
      slot.as.buffer.data = outputs[outputIndex++];
      slot.as.buffer.byte_size = resource.byte_size;
    }
  }

  return 0;
}

int32_t executeTaskRangeSerial(RuntimeState *runtime, uint32_t begin,
                               uint32_t end) {
  if (!runtime || !runtime->runtimeTasks)
    return kRuntimeErrorInvalidArgument;

  for (uint32_t i = begin; i < end; ++i) {
    RuntimeTask &task = runtime->runtimeTasks[i];
    if (!task.fn)
      return kRuntimeErrorInvalidGraph;

    int32_t rc = task.fn(task.opaque);
    if (rc != 0)
      return rc;
  }

  return 0;
}

} // namespace sculptor::runtime
