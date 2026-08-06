#pragma once

#include <stddef.h>
#include <stdint.h>

#include "golem/runtime/execution.h"
#include "golem/runtime/routed_transport.h"
#include "golem/runtime/tile_abi.h"

namespace golem::runtime {

enum class DeploymentStep : uint32_t {
    Idle = 0,
    Progress = 1,
    Complete = 2,
    Failed = 3,
    WaitForReceive = 4,
    WaitForTransmit = 5,
};

enum class DeploymentError : uint32_t {
    None = 0,
    InvalidABI = 1,
    InvalidTransport = 2,
    AllocationFailed = 3,
    InvalidModelIO = 4,
    BootTaskFailed = 5,
    InvalidFrame = 6,
    TaskFailed = 7,
    RouteSendFailed = 8,
};

enum class DeploymentTransmitPolicy : uint32_t {
    Blocking = 0,
    OverlapReadyTasks = 1,
};

enum class InvalidFrameReason : uint32_t {
    None = 0,
    UnknownSource = 1,
    InvalidMagic = 2,
    UnknownRoute = 3,
    SourceMismatch = 4,
    ExecutionMismatch = 5,
    WordCountMismatch = 6,
    DuplicateRoute = 7,
    UnexpectedWordDuringDMA = 8,
    DMACompletionUnknownSource = 9,
    DMACompletionPhaseMismatch = 10,
    DMACompletionRouteMismatch = 11,
    DMACompletionUnknownRoute = 12,
    DMACompletionSourceMismatch = 13,
    DMACompletionDuplicateRoute = 14,
    MissingPayloadRoute = 15,
    InvalidReceivePhase = 16,
};

struct InvalidFrameDiagnostic {
    InvalidFrameReason reason = InvalidFrameReason::None;
    uint32_t source_tile = UINT32_MAX;
    uint32_t route_id = UINT32_MAX;
    uint32_t receive_phase = UINT32_MAX;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

using ReadRuntimeCycle = uint64_t (*)(void* context);

struct DeploymentProfile {
    void* context = nullptr;
    ReadRuntimeCycle read_cycle = nullptr;
    uint64_t receive_steps = 0;
    uint64_t receive_cycles = 0;
    uint64_t transmit_steps = 0;
    uint64_t transmit_cycles = 0;
    uint64_t transmit_blocked_steps = 0;
    uint64_t transmit_blocked_cycles = 0;
    uint64_t execute_steps = 0;
    uint64_t execute_cycles = 0;
    uint64_t idle_steps = 0;
    uint64_t idle_cycles = 0;
    uint64_t receive_wait_steps = 0;
    uint64_t receive_wait_cycles = 0;
    uint64_t complete_steps = 0;
    uint64_t complete_cycles = 0;
    uint64_t failed_steps = 0;
    uint64_t failed_cycles = 0;
};

enum class TaskTraceEvent : uint32_t {
    Start = 1,
    Finish = 2,
};

using EmitTaskTrace = void (*)(
    void* context,
    TaskTraceEvent event,
    uint32_t task_id,
    ExecutionId execution_id
);

struct DeploymentTrace {
    void* context = nullptr;
    EmitTaskTrace emit = nullptr;
};

// A single-execution dataflow runtime for compiler-generated tile ELFs.
//
// The runtime owns descriptor and workspace storage, while callers provide
// model input/output buffers. Intermediate and route resources are placed at
// compiler-assigned offsets in the workspace. Every inter-tile frame still
// moves as individual 32-bit words.
class DeploymentRuntime {
public:
    DeploymentRuntime(
        TileABI abi,
        RoutedWordTransport transport,
        DeploymentProfile* profile = nullptr,
        DeploymentTrace* trace = nullptr,
        DeploymentTransmitPolicy transmit_policy =
            DeploymentTransmitPolicy::Blocking
    ) noexcept;

    bool initialize() noexcept;
    bool boot() noexcept;

    bool bindModelInput(uint32_t model_index, void* data) noexcept;
    bool bindModelOutput(uint32_t model_index, void* data) noexcept;

    DeploymentStep step() noexcept;
    bool complete() const noexcept;
    bool failed() const noexcept;
    DeploymentError error() const noexcept;
    const InvalidFrameDiagnostic& invalidFrameDiagnostic() const noexcept;
    ExecutionId executionId() const noexcept;

    const Tensor* resourceTensor(uint32_t local_slot) const noexcept;
    const TileABI& abi() const noexcept;

private:
    enum class ProfileActivity : uint32_t {
        Receive,
        Transmit,
        TransmitBlocked,
        Execute,
        Idle,
        ReceiveWait,
        Complete,
        Failed,
    };

    struct ReceiveState {
        uint32_t source_tile;
        uint32_t phase;
        uint32_t route_id;
        ExecutionId execution_id;
        uint32_t word_count;
        uint32_t received_words;
        uint8_t* destination;
    };

    bool allocateState() noexcept;
    bool initializeResources() noexcept;
    bool setResourceData(uint32_t local_slot, void* data) noexcept;
    bool consumeWord(const RoutedWord& word) noexcept;
    bool progressReceiveDMA() noexcept;
    bool executeReadyTask() noexcept;
    bool executeReadyTaskWithoutTransmitConflict() noexcept;
    bool executeTask(uint32_t task_index) noexcept;
    bool taskReady(uint32_t task_index) const noexcept;
    bool taskOutputConflictsWithPendingTransmit(
        uint32_t task_index) const noexcept;
    bool beginTaskRoutes(uint32_t task_id) noexcept;
    bool scheduleTaskRoutes(uint32_t task_id) noexcept;
    bool beginNextQueuedTaskRoutes() noexcept;
    bool taskHasOutgoingRoutes(uint32_t task_id) const noexcept;
    bool progressTaskRoutes() noexcept;
    bool allOutgoingRoutesSent() const noexcept;
    bool hasPendingIncomingRoute() const noexcept;
    ReceiveState* receiveState(uint32_t source_tile) noexcept;
    uint32_t taskIndex(uint32_t task_id) const noexcept;
    void setInvalidFrame(
        InvalidFrameReason reason,
        uint32_t source_tile,
        uint32_t route_id,
        uint32_t receive_phase,
        uint64_t expected,
        uint64_t actual
    ) noexcept;
    void setError(DeploymentError error) noexcept;
    uint64_t profileCycle() const noexcept;
    void recordProfileStep(
        ProfileActivity activity,
        uint64_t start_cycle
    ) noexcept;
    void emitTaskTrace(
        TaskTraceEvent event,
        uint32_t task_id
    ) noexcept;

    TileABI abi_;
    RoutedWordTransport transport_;
    ExecutionId execution_id_;
    DeploymentError error_;
    InvalidFrameDiagnostic invalid_frame_diagnostic_;
    bool initialized_;
    bool booted_;
    uint32_t complete_task_count_;

    void* workspace_allocation_;
    uint8_t* workspace_;
    void* descriptor_allocation_;
    uint8_t* descriptors_;
    Tensor* resource_tensors_;
    bool* resource_bound_;
    bool* resource_ready_;
    bool* task_complete_;
    bool* outgoing_route_sent_;
    bool* incoming_route_received_;
    Tensor* task_inputs_;
    Tensor* task_outputs_;
    ReceiveState* receive_states_;
    uint32_t receive_state_count_;

    DeploymentTransmitPolicy transmit_policy_;
    uint32_t* pending_transmit_tasks_;
    uint32_t pending_transmit_read_;
    uint32_t pending_transmit_write_;
    bool transmit_active_;
    uint32_t transmit_task_id_;
    uint32_t transmit_route_index_;
    uint32_t transmit_phase_;
    uint64_t transmit_word_offset_;
    DeploymentProfile* profile_;
    DeploymentTrace* trace_;
};

}  // namespace golem::runtime
