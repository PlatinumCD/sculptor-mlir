#ifndef SCULPTOR_TASK_GRAPH_RUNTIME_H
#define SCULPTOR_TASK_GRAPH_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t (*TaskEntry)(void *opaque);

typedef struct {
  TaskEntry fn;
  void *opaque;
} RuntimeTask;

typedef enum { RES_BUFFER = 0, RES_HANDLE = 1 } ResourceKind;

typedef enum {
  STORAGE_INPUT = 0,
  STORAGE_OUTPUT = 1,
  STORAGE_TEMP = 2,
  STORAGE_PERSISTENT = 3
} StorageClass;

typedef struct {
  int32_t kind;
  int32_t storage;
  uint32_t slot;
  uint32_t reserved;
  uint64_t byte_size;
  uint64_t workspace_offset;
} ResourceDesc;

typedef enum {
  ARG_BUFFER = 0,
  ARG_HANDLE = 1,
  ARG_I32 = 2,
  ARG_I64 = 3,
  ARG_F32 = 4,
  ARG_F64 = 5,
  ARG_BYTES = 6
} ArgKind;

typedef enum { SRC_SLOT = 0, SRC_INLINE = 1, SRC_CONST_BLOB = 2 } ArgSource;

enum { ARG_IN = 1 << 0, ARG_OUT = 1 << 1 };

typedef struct {
  int32_t kind;
  uint16_t flags;
  uint16_t reserved0;
  int32_t source;
  uint32_t index;
  uint32_t byte_offset;
  uint32_t byte_size;
} ArgBinding;

typedef struct {
  uint32_t symbol_id;
  TaskEntry entry;
  uint32_t signature_id;
  uint32_t reserved;
} CallableDesc;

typedef struct {
  uint32_t callable_index;
  uint32_t arg_begin;
  uint16_t arg_count;
  uint16_t reserved0;
  uint32_t dep_begin;
  uint16_t dep_count;
  uint16_t reserved;
  uint32_t payload_offset;
  uint32_t payload_size;
  int32_t core_id;
} TaskDesc;

typedef struct {
  const ResourceDesc *resources;
  uint32_t resource_count;

  const CallableDesc *callables;
  uint32_t callable_count;

  const TaskDesc *tasks;
  uint32_t task_count;

  const ArgBinding *bindings;
  uint32_t binding_count;

  const uint32_t *deps;
  uint32_t dep_count;

  const uint8_t *payload_blob;
  uint32_t payload_blob_size;

  const uint8_t *const_blob;
  uint32_t const_blob_size;

  uint64_t workspace_size;
} GraphImage;

typedef struct {
  void *data;
  uint64_t byte_size;
} BufferView;

typedef struct {
  int32_t kind;
  int32_t reserved;
  union {
    BufferView buffer;
    void *handle;
  } as;
} SlotValue;

typedef struct {
  uint8_t *workspace;
  SlotValue *slots;
  void *services;
} ExecContext;

typedef struct {
  const GraphImage *graph;
  const TaskDesc *task;
  ExecContext *exec;
} TaskCall;

typedef struct RuntimeHandle RuntimeHandle;

GraphImage *
sculptor_runtime_graph_create(uint32_t resource_count, uint32_t callable_count,
                            uint32_t task_count, uint32_t binding_count,
                            uint32_t dep_count, uint32_t payload_blob_size,
                            uint32_t const_blob_size,
                            uint64_t workspace_size);
void sculptor_runtime_graph_destroy(GraphImage *graph);
void sculptor_runtime_graph_set_resource(GraphImage *graph, uint32_t index,
                                       int32_t kind, int32_t storage,
                                       uint32_t slot, uint64_t byte_size,
                                       uint64_t workspace_offset);
void sculptor_runtime_graph_set_callable(GraphImage *graph, uint32_t index,
                                       uint32_t symbol_id, TaskEntry entry,
                                       uint32_t signature_id);
void sculptor_runtime_graph_set_task(GraphImage *graph, uint32_t index,
                                   uint32_t callable_index, uint32_t arg_begin,
                                   uint16_t arg_count, uint32_t dep_begin,
                                   uint16_t dep_count,
                                   uint32_t payload_offset,
                                   uint32_t payload_size, int32_t core_id);
void sculptor_runtime_graph_set_binding(GraphImage *graph, uint32_t index,
                                      int32_t kind, uint16_t flags,
                                      int32_t source, uint32_t source_index,
                                      uint32_t byte_offset, uint32_t byte_size);
void sculptor_runtime_graph_set_dep(GraphImage *graph, uint32_t index,
                                  uint32_t dependency);
void sculptor_runtime_graph_copy_payload(GraphImage *graph, uint32_t offset,
                                       const void *data, uint32_t size);
void sculptor_runtime_graph_copy_const_blob(GraphImage *graph, uint32_t offset,
                                          const void *data, uint32_t size);

RuntimeHandle *sculptor_runtime_init(GraphImage *graph);
int32_t sculptor_runtime_execute(RuntimeHandle *runtime,
                               const void *const *inputs, void *const *outputs);
void sculptor_runtime_destroy(RuntimeHandle *runtime);

BufferView sculptor_runtime_task_arg_buffer(void *opaque, uint32_t index);
void *sculptor_runtime_task_arg_handle(void *opaque, uint32_t index);
int32_t sculptor_runtime_task_set_arg_handle(void *opaque, uint32_t index,
                                           void *handle);
int32_t sculptor_runtime_task_require_handle(void *opaque, uint32_t index);
int32_t sculptor_runtime_task_arg_i32(void *opaque, uint32_t index);
int64_t sculptor_runtime_task_arg_i64(void *opaque, uint32_t index);
float sculptor_runtime_task_arg_f32(void *opaque, uint32_t index);
double sculptor_runtime_task_arg_f64(void *opaque, uint32_t index);
const void *sculptor_runtime_task_arg_bytes(void *opaque, uint32_t index);
uint32_t sculptor_runtime_task_arg_size(void *opaque, uint32_t index);
int32_t sculptor_runtime_task_core_id(void *opaque);
void sculptor_runtime_copy_to_buffer(void *dst, const void *src, uint64_t size);
void sculptor_runtime_free_result_buffer(void *opaque, void *ptr);
void *sculptor_runtime_persistent_handle_create(uint64_t payload_size,
                                              void *destroy_fn);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SCULPTOR_TASK_GRAPH_RUNTIME_H
