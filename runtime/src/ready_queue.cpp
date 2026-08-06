#include "golem/runtime/ready_queue.h"

namespace golem::runtime {

void ReadyQueue::initialize() noexcept {
    for (uint32_t& slot : slots) {
        slot = 0;
    }
    next_read = 0;
    next_write = 0;
}

bool ReadyQueue::empty() const noexcept {
    return next_read == next_write;
}

bool ReadyQueue::full() const noexcept {
    return size() >= kTaskInstanceCapacity;
}

uint32_t ReadyQueue::size() const noexcept {
    return next_write - next_read;
}

bool ReadyQueue::push(uint32_t task_instance_index) noexcept {
    if (task_instance_index >= kTaskInstanceCapacity || full()) {
        return false;
    }

    slots[next_write % kTaskInstanceCapacity] = task_instance_index;
    ++next_write;
    return true;
}

bool ReadyQueue::pop(uint32_t* task_instance_index) noexcept {
    if (task_instance_index == nullptr || empty()) {
        return false;
    }

    *task_instance_index = slots[next_read % kTaskInstanceCapacity];
    ++next_read;
    return true;
}

}  // namespace golem::runtime
