#include "golem/runtime/task_registry.h"

namespace golem::runtime {

const Task* TaskRegistry::find(uint32_t task_id) const noexcept {
    if (tasks == nullptr) {
        return nullptr;
    }

    uint32_t first = 0;
    uint32_t last = task_count;

    while (first < last) {
        const uint32_t middle = first + (last - first) / 2U;
        const Task& candidate = tasks[middle];

        if (candidate.id < task_id) {
            first = middle + 1U;
        } else if (candidate.id > task_id) {
            last = middle;
        } else {
            return &candidate;
        }
    }

    return nullptr;
}

bool TaskRegistry::valid() const noexcept {
    if (task_count != 0 && tasks == nullptr) {
        return false;
    }

    for (uint32_t index = 0; index < task_count; ++index) {
        if (tasks[index].execute == nullptr) {
            return false;
        }
        if (index != 0 && tasks[index - 1U].id >= tasks[index].id) {
            return false;
        }
    }

    return true;
}

}  // namespace golem::runtime
