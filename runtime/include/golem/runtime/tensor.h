#pragma once

#include <stdint.h>

namespace golem::runtime {

enum class ElementType : uint32_t {
    Invalid = 0,
    Float32 = 1,
    Int8 = 2,
    UInt8 = 3,
    Int16 = 4,
    UInt16 = 5,
    Int32 = 6,
    UInt32 = 7,
    Int64 = 8,
    UInt64 = 9,
};

struct Tensor {
    ElementType element_type;
    int64_t rank;
    void* descriptor;
};

constexpr uint32_t elementSizeBytes(ElementType element_type) {
    switch (element_type) {
        case ElementType::Int8:
        case ElementType::UInt8:
            return 1;
        case ElementType::Int16:
        case ElementType::UInt16:
            return 2;
        case ElementType::Float32:
        case ElementType::Int32:
        case ElementType::UInt32:
            return 4;
        case ElementType::Int64:
        case ElementType::UInt64:
            return 8;
        case ElementType::Invalid:
            return 0;
    }

    return 0;
}

constexpr bool validTensorRank(int64_t rank) {
    return rank >= 0;
}

constexpr bool validTensor(const Tensor& tensor) {
    return tensor.element_type != ElementType::Invalid &&
           elementSizeBytes(tensor.element_type) != 0 &&
           validTensorRank(tensor.rank) &&
           tensor.descriptor != nullptr;
}

}  // namespace golem::runtime
