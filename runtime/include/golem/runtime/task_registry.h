#pragma once

#include <stdint.h>

#include "golem/runtime/task.h"

namespace golem::runtime {

struct TaskRegistry {
    const Task* tasks;
    uint32_t task_count;

    const Task* find(uint32_t task_id) const noexcept;
    bool valid() const noexcept;
};

}  // namespace golem::runtime
