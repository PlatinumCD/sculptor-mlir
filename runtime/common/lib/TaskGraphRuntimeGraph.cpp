#include "TaskGraphRuntimeCommon.h"

#include <cstdlib>
#include <cstring>

namespace {

constexpr int32_t kTaskCoreIdNone = -1;

} // namespace

using sculptor::runtime::allocateZeroedArray;

extern "C" {

GraphImage *
sculptor_runtime_graph_create(uint32_t resource_count, uint32_t callable_count,
                            uint32_t task_count, uint32_t binding_count,
                            uint32_t dep_count, uint32_t payload_blob_size,
                            uint32_t const_blob_size, uint64_t workspace_size) {
  auto *graph = static_cast<GraphImage *>(std::calloc(1, sizeof(GraphImage)));
  if (!graph)
    return nullptr;

  graph->resource_count = resource_count;
  graph->callable_count = callable_count;
  graph->task_count = task_count;
  graph->binding_count = binding_count;
  graph->dep_count = dep_count;
  graph->payload_blob_size = payload_blob_size;
  graph->const_blob_size = const_blob_size;
  graph->workspace_size = workspace_size;

  graph->resources = allocateZeroedArray<ResourceDesc>(resource_count);
  graph->callables = allocateZeroedArray<CallableDesc>(callable_count);
  graph->tasks = allocateZeroedArray<TaskDesc>(task_count);
  graph->bindings = allocateZeroedArray<ArgBinding>(binding_count);
  graph->deps = allocateZeroedArray<uint32_t>(dep_count);
  graph->payload_blob = allocateZeroedArray<uint8_t>(payload_blob_size);
  graph->const_blob = allocateZeroedArray<uint8_t>(const_blob_size);

  if ((resource_count && !graph->resources) ||
      (callable_count && !graph->callables) || (task_count && !graph->tasks) ||
      (binding_count && !graph->bindings) || (dep_count && !graph->deps) ||
      (payload_blob_size && !graph->payload_blob) ||
      (const_blob_size && !graph->const_blob)) {
    sculptor_runtime_graph_destroy(graph);
    return nullptr;
  }

  for (uint32_t i = 0; i < task_count; ++i)
    const_cast<TaskDesc *>(graph->tasks)[i].core_id = kTaskCoreIdNone;

  return graph;
}

void sculptor_runtime_graph_destroy(GraphImage *graph) {
  if (!graph)
    return;

  std::free(const_cast<ResourceDesc *>(graph->resources));
  std::free(const_cast<CallableDesc *>(graph->callables));
  std::free(const_cast<TaskDesc *>(graph->tasks));
  std::free(const_cast<ArgBinding *>(graph->bindings));
  std::free(const_cast<uint32_t *>(graph->deps));
  std::free(const_cast<uint8_t *>(graph->payload_blob));
  std::free(const_cast<uint8_t *>(graph->const_blob));
  std::free(graph);
}

void sculptor_runtime_graph_set_resource(GraphImage *graph, uint32_t index,
                                       int32_t kind, int32_t storage,
                                       uint32_t slot, uint64_t byte_size,
                                       uint64_t workspace_offset) {
  if (!graph || index >= graph->resource_count)
    return;

  auto &resource = const_cast<ResourceDesc *>(graph->resources)[index];
  resource.kind = kind;
  resource.storage = storage;
  resource.slot = slot;
  resource.byte_size = byte_size;
  resource.workspace_offset = workspace_offset;
}

void sculptor_runtime_graph_set_callable(GraphImage *graph, uint32_t index,
                                       uint32_t symbol_id, TaskEntry entry,
                                       uint32_t signature_id) {
  if (!graph || index >= graph->callable_count)
    return;

  auto &callable = const_cast<CallableDesc *>(graph->callables)[index];
  callable.symbol_id = symbol_id;
  callable.entry = entry;
  callable.signature_id = signature_id;
}

void sculptor_runtime_graph_set_task(GraphImage *graph, uint32_t index,
                                   uint32_t callable_index, uint32_t arg_begin,
                                   uint16_t arg_count, uint32_t dep_begin,
                                   uint16_t dep_count, uint32_t payload_offset,
                                   uint32_t payload_size, int32_t core_id) {
  if (!graph || index >= graph->task_count)
    return;

  auto &task = const_cast<TaskDesc *>(graph->tasks)[index];
  task.callable_index = callable_index;
  task.arg_begin = arg_begin;
  task.arg_count = arg_count;
  task.dep_begin = dep_begin;
  task.dep_count = dep_count;
  task.payload_offset = payload_offset;
  task.payload_size = payload_size;
  task.core_id = core_id;
}

void sculptor_runtime_graph_set_binding(GraphImage *graph, uint32_t index,
                                      int32_t kind, uint16_t flags,
                                      int32_t source, uint32_t source_index,
                                      uint32_t byte_offset,
                                      uint32_t byte_size) {
  if (!graph || index >= graph->binding_count)
    return;

  auto &binding = const_cast<ArgBinding *>(graph->bindings)[index];
  binding.kind = kind;
  binding.flags = flags;
  binding.source = source;
  binding.index = source_index;
  binding.byte_offset = byte_offset;
  binding.byte_size = byte_size;
}

void sculptor_runtime_graph_set_dep(GraphImage *graph, uint32_t index,
                                  uint32_t dependency) {
  if (!graph || index >= graph->dep_count)
    return;

  const_cast<uint32_t *>(graph->deps)[index] = dependency;
}

void sculptor_runtime_graph_copy_payload(GraphImage *graph, uint32_t offset,
                                       const void *data, uint32_t size) {
  if (!graph || !data || offset + size > graph->payload_blob_size)
    return;

  std::memcpy(const_cast<uint8_t *>(graph->payload_blob) + offset, data, size);
}

void sculptor_runtime_graph_copy_const_blob(GraphImage *graph, uint32_t offset,
                                          const void *data, uint32_t size) {
  if (!graph || !data || offset + size > graph->const_blob_size)
    return;

  std::memcpy(const_cast<uint8_t *>(graph->const_blob) + offset, data, size);
}

} // extern "C"
