#pragma once

#include <stdint.h>

#include "golem/runtime/execution.h"
#include "golem/runtime/tensor.h"

namespace golem::runtime {

enum class TaskInstanceState : uint32_t {
    Free = 0,
    WaitingForInputs = 1,
    Ready = 2,
    Running = 3,
    ForwardingOutputs = 4,
    Complete = 5,
    Failed = 6,
};

struct TaskInstance {
    ExecutionId execution_id;
    uint32_t task_id;
    Tensor* inputs;
    uint32_t received_input_count;
    Tensor* outputs;
    TaskInstanceState state;
};

constexpr uint32_t kTaskInstanceCapacity = 16;

class TaskInstancePool {
public:
    void initialize() noexcept;

    TaskInstance* find(ExecutionId execution_id, uint32_t task_id) noexcept;
    const TaskInstance* find(
        ExecutionId execution_id,
        uint32_t task_id
    ) const noexcept;

    TaskInstance* acquire(
        ExecutionId execution_id,
        uint32_t task_id,
        Tensor* inputs,
        Tensor* outputs
    ) noexcept;

    bool release(TaskInstance* instance) noexcept;

    TaskInstance* get(uint32_t index) noexcept;
    const TaskInstance* get(uint32_t index) const noexcept;
    uint32_t indexOf(const TaskInstance* instance) const noexcept;

private:
    TaskInstance instances_[kTaskInstanceCapacity];
};

}  // namespace golem::runtime
