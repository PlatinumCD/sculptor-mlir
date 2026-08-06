#pragma once

#include <stdint.h>

#include "golem/runtime/tensor.h"

namespace golem::runtime {

enum class TaskStatus : uint32_t {
    Success = 0,
    Failure = 1,
};

using TaskExecute = TaskStatus (*)(
    const Tensor* inputs,
    uint32_t input_count,
    Tensor* outputs,
    uint32_t output_count
);

struct Task {
    uint32_t id;
    TaskExecute execute;
    uint32_t input_count;
    uint32_t output_count;
};

}  // namespace golem::runtime
