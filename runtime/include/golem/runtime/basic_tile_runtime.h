#pragma once

#include <stdint.h>

#include "golem/runtime/tile_abi.h"
#include "golem/runtime/transport.h"

namespace golem::runtime {

// A deliberately small, single-execution runtime for the first compiler-to-
// mesh proof. Tensor descriptors and storage remain owned by the tile program.
class BasicTileRuntime {
public:
    BasicTileRuntime(TileABI abi, WordTransport transport) noexcept;

    bool valid() const noexcept;
    bool boot() noexcept;
    bool booted() const noexcept;

    bool execute(
        uint32_t task_id,
        const Tensor* inputs,
        uint32_t input_count,
        Tensor* outputs,
        uint32_t output_count
    ) const noexcept;

    bool sendRoute(
        uint32_t route_id,
        const void* data,
        uint64_t byte_size
    ) const noexcept;

    bool receiveRoute(
        uint32_t route_id,
        void* data,
        uint64_t byte_size
    ) const noexcept;

    const TileABI& abi() const noexcept;

private:
    TileABI abi_;
    WordTransport transport_;
    bool booted_;
};

}  // namespace golem::runtime
