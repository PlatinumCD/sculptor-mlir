#pragma once

#include <stdint.h>

#include "golem/runtime/task_instance.h"

namespace golem::runtime {

struct ReadyQueue {
    uint32_t slots[kTaskInstanceCapacity];
    uint32_t next_read;
    uint32_t next_write;

    void initialize() noexcept;
    bool empty() const noexcept;
    bool full() const noexcept;
    uint32_t size() const noexcept;
    bool push(uint32_t task_instance_index) noexcept;
    bool pop(uint32_t* task_instance_index) noexcept;
};

}  // namespace golem::runtime
