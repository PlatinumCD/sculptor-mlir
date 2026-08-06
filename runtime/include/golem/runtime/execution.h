#pragma once

#include <stdint.h>

namespace golem::runtime {

using ExecutionId = uint64_t;

constexpr uint32_t kMaxInFlightExecutions = 2;

}  // namespace golem::runtime
