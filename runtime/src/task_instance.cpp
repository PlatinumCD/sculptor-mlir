#include "golem/runtime/task_instance.h"

namespace golem::runtime {

namespace {

void resetInstance(TaskInstance& instance) {
    instance.execution_id = 0;
    instance.task_id = 0;
    instance.inputs = nullptr;
    instance.received_input_count = 0;
    instance.outputs = nullptr;
    instance.state = TaskInstanceState::Free;
}

}  // namespace

void TaskInstancePool::initialize() noexcept {
    for (TaskInstance& instance : instances_) {
        resetInstance(instance);
    }
}

TaskInstance* TaskInstancePool::find(
    ExecutionId execution_id,
    uint32_t task_id
) noexcept {
    for (TaskInstance& instance : instances_) {
        if (instance.state != TaskInstanceState::Free &&
            instance.execution_id == execution_id &&
            instance.task_id == task_id) {
            return &instance;
        }
    }

    return nullptr;
}

const TaskInstance* TaskInstancePool::find(
    ExecutionId execution_id,
    uint32_t task_id
) const noexcept {
    for (const TaskInstance& instance : instances_) {
        if (instance.state != TaskInstanceState::Free &&
            instance.execution_id == execution_id &&
            instance.task_id == task_id) {
            return &instance;
        }
    }

    return nullptr;
}

TaskInstance* TaskInstancePool::acquire(
    ExecutionId execution_id,
    uint32_t task_id,
    Tensor* inputs,
    Tensor* outputs
) noexcept {
    if (find(execution_id, task_id) != nullptr) {
        return nullptr;
    }

    for (TaskInstance& instance : instances_) {
        if (instance.state == TaskInstanceState::Free) {
            instance.execution_id = execution_id;
            instance.task_id = task_id;
            instance.inputs = inputs;
            instance.received_input_count = 0;
            instance.outputs = outputs;
            instance.state = TaskInstanceState::WaitingForInputs;
            return &instance;
        }
    }

    return nullptr;
}

bool TaskInstancePool::release(TaskInstance* instance) noexcept {
    const uint32_t index = indexOf(instance);
    if (index == kTaskInstanceCapacity) {
        return false;
    }

    TaskInstance& owned = instances_[index];
    if (owned.state != TaskInstanceState::Complete &&
        owned.state != TaskInstanceState::Failed) {
        return false;
    }

    resetInstance(owned);
    return true;
}

TaskInstance* TaskInstancePool::get(uint32_t index) noexcept {
    return index < kTaskInstanceCapacity ? &instances_[index] : nullptr;
}

const TaskInstance* TaskInstancePool::get(uint32_t index) const noexcept {
    return index < kTaskInstanceCapacity ? &instances_[index] : nullptr;
}

uint32_t TaskInstancePool::indexOf(
    const TaskInstance* instance
) const noexcept {
    if (instance == nullptr) {
        return kTaskInstanceCapacity;
    }

    for (uint32_t index = 0; index < kTaskInstanceCapacity; ++index) {
        if (instance == &instances_[index]) {
            return index;
        }
    }

    return kTaskInstanceCapacity;
}

}  // namespace golem::runtime
