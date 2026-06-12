#include "TaskGraphRuntimeCommon.h"

#include <cstdint>
#include <cstring>

namespace {

constexpr int32_t kRuntimeErrorInvalidBinding = -4;
constexpr int32_t kRuntimeErrorMissingHandle = -5;

const TaskCall *castTaskCall(void *opaque) {
  return static_cast<const TaskCall *>(opaque);
}

const ArgBinding *taskBinding(const TaskCall *call, uint32_t index) {
  if (!call || !call->graph || !call->task || !call->exec)
    return nullptr;
  if (index >= call->task->arg_count)
    return nullptr;

  uint32_t bindingIndex = call->task->arg_begin + index;
  if (bindingIndex >= call->graph->binding_count)
    return nullptr;
  return &call->graph->bindings[bindingIndex];
}

const uint8_t *bindingData(const TaskCall *call, const ArgBinding *binding) {
  if (!call || !binding || !call->graph)
    return nullptr;

  switch (binding->source) {
  case SRC_INLINE: {
    uint64_t begin = static_cast<uint64_t>(call->task->payload_offset) +
                     binding->byte_offset;
    uint64_t end = begin + binding->byte_size;
    if (end > call->graph->payload_blob_size)
      return nullptr;
    return call->graph->payload_blob + begin;
  }
  case SRC_CONST_BLOB: {
    uint64_t begin = static_cast<uint64_t>(binding->index);
    uint64_t end = begin + binding->byte_size;
    if (end > call->graph->const_blob_size)
      return nullptr;
    return call->graph->const_blob + begin;
  }
  default:
    return nullptr;
  }
}

template <typename T>
T readScalarTaskArg(void *opaque, uint32_t index, int32_t expectedKind,
                    T defaultValue) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  const uint8_t *data = bindingData(call, binding);
  if (!binding || binding->kind != expectedKind || !data ||
      binding->byte_size < sizeof(defaultValue)) {
    return defaultValue;
  }

  T value = defaultValue;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

} // namespace

extern "C" {

BufferView sculptor_runtime_task_arg_buffer(void *opaque, uint32_t index) {
  BufferView empty{nullptr, 0};
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding || binding->kind != ARG_BUFFER || binding->source != SRC_SLOT)
    return empty;
  if (binding->index >= sculptor::runtime::computeSlotCount(call->graph))
    return empty;

  const SlotValue &slot = call->exec->slots[binding->index];
  if (slot.kind != RES_BUFFER)
    return empty;
  return slot.as.buffer;
}

void *sculptor_runtime_task_arg_handle(void *opaque, uint32_t index) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding || binding->kind != ARG_HANDLE || binding->source != SRC_SLOT)
    return nullptr;
  if (binding->index >= sculptor::runtime::computeSlotCount(call->graph))
    return nullptr;

  const SlotValue &slot = call->exec->slots[binding->index];
  if (slot.kind != RES_HANDLE)
    return nullptr;
  return slot.as.handle;
}

int32_t sculptor_runtime_task_set_arg_handle(void *opaque, uint32_t index,
                                           void *handle) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding || binding->kind != ARG_HANDLE || binding->source != SRC_SLOT)
    return kRuntimeErrorInvalidBinding;
  if (binding->index >= sculptor::runtime::computeSlotCount(call->graph))
    return kRuntimeErrorInvalidBinding;

  SlotValue &slot = call->exec->slots[binding->index];
  slot.kind = RES_HANDLE;
  slot.as.handle = handle;
  return 0;
}

int32_t sculptor_runtime_task_require_handle(void *opaque, uint32_t index) {
  return sculptor_runtime_task_arg_handle(opaque, index)
             ? 0
             : kRuntimeErrorMissingHandle;
}

int32_t sculptor_runtime_task_arg_i32(void *opaque, uint32_t index) {
  return readScalarTaskArg<int32_t>(opaque, index, ARG_I32, 0);
}

int64_t sculptor_runtime_task_arg_i64(void *opaque, uint32_t index) {
  return readScalarTaskArg<int64_t>(opaque, index, ARG_I64, 0);
}

float sculptor_runtime_task_arg_f32(void *opaque, uint32_t index) {
  return readScalarTaskArg<float>(opaque, index, ARG_F32, 0.0f);
}

double sculptor_runtime_task_arg_f64(void *opaque, uint32_t index) {
  return readScalarTaskArg<double>(opaque, index, ARG_F64, 0.0);
}

const void *sculptor_runtime_task_arg_bytes(void *opaque, uint32_t index) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding || binding->kind != ARG_BYTES)
    return nullptr;
  return bindingData(call, binding);
}

uint32_t sculptor_runtime_task_arg_size(void *opaque, uint32_t index) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding)
    return 0;
  return binding->byte_size;
}

} // extern "C"
