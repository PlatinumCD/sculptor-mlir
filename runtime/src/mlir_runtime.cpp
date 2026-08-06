// Minimal bare-metal implementation of MLIR's memrefCopy ABI.
//
// The descriptor layout and traversal match MLIR CRunnerUtils, but this
// version has no hosted C++ runtime dependency and allocates no scratch data.

#include <stddef.h>
#include <stdint.h>

namespace {

struct UnrankedMemRef {
    int64_t rank;
    void* descriptor;
};

struct RankedMemRefHeader {
    void* allocated;
    void* aligned;
    int64_t offset;
};

typedef uint64_t AliasedUInt64 __attribute__((__may_alias__));
typedef uint32_t AliasedUInt32 __attribute__((__may_alias__));

#if defined(MITTENS_MEMREF_COPY_TEST_HOOK)
extern "C" void mittensMemrefCopyTestHook(size_t byte_count);
#endif

void copyBytes(
    uint8_t* destination,
    const uint8_t* source,
    size_t byte_count
) {
#if defined(MITTENS_MEMREF_COPY_TEST_HOOK)
    mittensMemrefCopyTestHook(byte_count);
#endif
    if ((reinterpret_cast<uintptr_t>(destination) |
         reinterpret_cast<uintptr_t>(source)) %
            alignof(AliasedUInt64) ==
        0) {
        while (byte_count >= sizeof(AliasedUInt64)) {
            *reinterpret_cast<AliasedUInt64*>(destination) =
                *reinterpret_cast<const AliasedUInt64*>(source);
            destination += sizeof(AliasedUInt64);
            source += sizeof(AliasedUInt64);
            byte_count -= sizeof(AliasedUInt64);
        }
    }
    if ((reinterpret_cast<uintptr_t>(destination) |
         reinterpret_cast<uintptr_t>(source)) %
            alignof(AliasedUInt32) ==
        0) {
        while (byte_count >= sizeof(AliasedUInt32)) {
            *reinterpret_cast<AliasedUInt32*>(destination) =
                *reinterpret_cast<const AliasedUInt32*>(source);
            destination += sizeof(AliasedUInt32);
            source += sizeof(AliasedUInt32);
            byte_count -= sizeof(AliasedUInt32);
        }
    }
    while (byte_count != 0) {
        *destination++ = *source++;
        --byte_count;
    }
}

}  // namespace

extern "C" void memrefCopy(
    int64_t element_size,
    UnrankedMemRef* source,
    UnrankedMemRef* destination
) {
    if (element_size <= 0 ||
        source == nullptr ||
        destination == nullptr ||
        source->rank < 0 ||
        source->rank != destination->rank ||
        source->descriptor == nullptr ||
        destination->descriptor == nullptr) {
        return;
    }

    const int64_t rank = source->rank;
    const auto* source_header =
        static_cast<const RankedMemRefHeader*>(source->descriptor);
    auto* destination_header =
        static_cast<RankedMemRefHeader*>(destination->descriptor);
    const auto* source_sizes = reinterpret_cast<const int64_t*>(
        source_header + 1
    );
    const auto* destination_sizes = reinterpret_cast<const int64_t*>(
        destination_header + 1
    );
    const auto* source_strides = source_sizes + rank;
    const auto* destination_strides = destination_sizes + rank;

    for (int64_t axis = 0; axis < rank; ++axis) {
        if (source_sizes[axis] <= 0 ||
            source_sizes[axis] != destination_sizes[axis]) {
            return;
        }
    }

    const auto* source_base =
        static_cast<const uint8_t*>(source_header->aligned);
    auto* destination_base =
        static_cast<uint8_t*>(destination_header->aligned);
    if (source_base == nullptr || destination_base == nullptr) {
        return;
    }

    const auto* source_pointer =
        source_base + source_header->offset * element_size;
    auto* destination_pointer =
        destination_base + destination_header->offset * element_size;
    if (rank == 0) {
        copyBytes(
            destination_pointer,
            source_pointer,
            static_cast<size_t>(element_size)
        );
        return;
    }

    size_t contiguous_element_count = 1;
    bool source_is_contiguous = true;
    bool destination_is_contiguous = true;
    for (int64_t axis = rank; axis-- > 0;) {
        const size_t axis_size =
            static_cast<size_t>(source_sizes[axis]);
        if (axis_size != 1) {
            if (contiguous_element_count >
                static_cast<size_t>(INT64_MAX)) {
                source_is_contiguous = false;
                destination_is_contiguous = false;
            } else {
                const int64_t expected_stride =
                    static_cast<int64_t>(contiguous_element_count);
                source_is_contiguous =
                    source_is_contiguous &&
                    source_strides[axis] == expected_stride;
                destination_is_contiguous =
                    destination_is_contiguous &&
                    destination_strides[axis] == expected_stride;
            }
        }

        if (axis_size > SIZE_MAX / contiguous_element_count) {
            return;
        }
        contiguous_element_count *= axis_size;
    }
    if (source_is_contiguous && destination_is_contiguous) {
        const size_t element_bytes =
            static_cast<size_t>(element_size);
        if (element_bytes >
            SIZE_MAX / contiguous_element_count) {
            return;
        }
        copyBytes(
            destination_pointer,
            source_pointer,
            contiguous_element_count * element_bytes
        );
        return;
    }

    size_t suffix_element_count = 1;
    int64_t suffix_start = rank;
    for (int64_t axis = rank; axis-- > 0;) {
        const size_t axis_size =
            static_cast<size_t>(source_sizes[axis]);
        if (axis_size != 1) {
            if (suffix_element_count >
                static_cast<size_t>(INT64_MAX)) {
                break;
            }
            const int64_t expected_stride =
                static_cast<int64_t>(suffix_element_count);
            if (source_strides[axis] != expected_stride ||
                destination_strides[axis] != expected_stride) {
                break;
            }
        }
        if (axis_size > SIZE_MAX / suffix_element_count) {
            return;
        }
        suffix_element_count *= axis_size;
        suffix_start = axis;
    }

    int64_t iteration_rank = rank;
    size_t copy_byte_count = static_cast<size_t>(element_size);
    if (suffix_element_count > 1) {
        if (copy_byte_count > SIZE_MAX / suffix_element_count) {
            return;
        }
        copy_byte_count *= suffix_element_count;
        iteration_rank = suffix_start;
    }
    if (iteration_rank == 0) {
        copyBytes(
            destination_pointer,
            source_pointer,
            copy_byte_count
        );
        return;
    }

    auto* indices = static_cast<int64_t*>(
        __builtin_alloca(
            static_cast<size_t>(iteration_rank) * sizeof(int64_t)
        )
    );
    auto* source_byte_strides = static_cast<int64_t*>(
        __builtin_alloca(
            static_cast<size_t>(iteration_rank) * sizeof(int64_t)
        )
    );
    auto* destination_byte_strides = static_cast<int64_t*>(
        __builtin_alloca(
            static_cast<size_t>(iteration_rank) * sizeof(int64_t)
        )
    );
    for (int64_t axis = 0; axis < iteration_rank; ++axis) {
        indices[axis] = 0;
        source_byte_strides[axis] =
            source_strides[axis] * element_size;
        destination_byte_strides[axis] =
            destination_strides[axis] * element_size;
    }

    int64_t source_index = 0;
    int64_t destination_index = 0;
    for (;;) {
        copyBytes(
            destination_pointer + destination_index,
            source_pointer + source_index,
            copy_byte_count
        );

        for (int64_t axis = iteration_rank; axis-- > 0;) {
            const int64_t next_index = ++indices[axis];
            source_index += source_byte_strides[axis];
            destination_index += destination_byte_strides[axis];
            if (source_sizes[axis] != next_index) {
                break;
            }
            if (axis == 0) {
                return;
            }
            indices[axis] = 0;
            source_index -=
                source_sizes[axis] * source_byte_strides[axis];
            destination_index -=
                destination_sizes[axis] *
                destination_byte_strides[axis];
        }
    }
}
