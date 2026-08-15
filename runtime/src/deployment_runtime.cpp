#include "golem/runtime/deployment_runtime.h"

#include "golem/runtime/scratchpad_abi.h"

namespace golem::runtime {

namespace {

constexpr uint32_t kFrameMagic = UINT32_C(0x474f4c4d);
constexpr uintptr_t kDataAlignment = 64;
constexpr uint32_t kFrameHeaderWords = 5;
constexpr uint32_t kInvalidIndex = UINT32_MAX;

enum ReceivePhase : uint32_t {
  ReceiveMagic = 0,
  ReceiveRoute = 1,
  ReceiveExecutionLow = 2,
  ReceiveExecutionHigh = 3,
  ReceiveWordCount = 4,
  ReceivePayloadWords = 5,
  ReceiveDMASubmit = 6,
  ReceiveDMAActive = 7,
};

struct MemRefDescriptorHeader {
  void *allocated;
  void *aligned;
  int64_t offset;
};

static_assert(sizeof(MemRefDescriptorHeader) == 24);

extern "C" void *malloc(size_t size);

void zeroBytes(void *memory, size_t byte_count) {
  auto *bytes = static_cast<uint8_t *>(memory);
  for (size_t index = 0; index < byte_count; ++index) {
    bytes[index] = 0;
  }
}

void *allocateBytes(size_t byte_count, HeapProfile *heap_profile) {
  if (byte_count == 0) {
    return nullptr;
  }
  void *memory = malloc(byte_count);
  if (heap_profile != nullptr) {
    recordHeapAllocation(*heap_profile, byte_count, memory != nullptr);
  }
  if (memory != nullptr) {
    zeroBytes(memory, byte_count);
  }
  return memory;
}

uint8_t *alignPointer(void *allocation, uintptr_t alignment) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(allocation);
  const uintptr_t aligned = (raw + alignment - 1U) & ~(alignment - 1U);
  return reinterpret_cast<uint8_t *>(aligned);
}

void storeWord(uint8_t *bytes, uint32_t word) {
  bytes[0] = static_cast<uint8_t>(word);
  bytes[1] = static_cast<uint8_t>(word >> 8U);
  bytes[2] = static_cast<uint8_t>(word >> 16U);
  bytes[3] = static_cast<uint8_t>(word >> 24U);
}

uint64_t descriptorSize(uint32_t rank) {
  return sizeof(MemRefDescriptorHeader) +
         static_cast<uint64_t>(rank) * 2U * sizeof(int64_t);
}

} // namespace

DeploymentRuntime::DeploymentRuntime(TileABI abi, RoutedWordTransport transport,
                                     DeploymentProfile *profile,
                                     DeploymentTrace *trace,
                                     DeploymentTransmitPolicy transmit_policy,
                                     HeapProfile *heap_profile) noexcept
    : abi_(abi), transport_(transport), execution_id_(0),
      error_(DeploymentError::None), invalid_frame_diagnostic_(),
      initialized_(false), booted_(false), complete_task_count_(0),
      workspace_allocation_(nullptr), workspace_(nullptr),
      descriptor_allocation_(nullptr), descriptors_(nullptr),
      resource_tensors_(nullptr), resource_bound_(nullptr),
      resource_ready_(nullptr), task_complete_(nullptr),
      outgoing_route_sent_(nullptr), incoming_route_received_(nullptr),
      task_inputs_(nullptr), task_outputs_(nullptr), receive_states_(nullptr),
      receive_state_count_(0), transmit_policy_(transmit_policy),
      pending_transmit_tasks_(nullptr), pending_transmit_read_(0),
      pending_transmit_write_(0), transmit_active_(false), transmit_task_id_(0),
      transmit_route_index_(0), transmit_phase_(0), transmit_word_offset_(0),
      transmit_segment_index_(0), transmit_segment_word_offset_(0),
      transmit_route_limit_(0), complete_outgoing_route_count_(0),
      complete_incoming_route_count_(0), task_readiness_(nullptr),
      task_queued_(nullptr), ready_tasks_(nullptr), ready_task_count_(0),
      task_ready_cycles_(nullptr), task_ready_from_route_(nullptr),
      task_finish_cycles_(nullptr), task_nic_submitted_(nullptr),
      profile_(profile), trace_(trace), heap_profile_(heap_profile) {
  golem_runtime_set_heap_profile(heap_profile_);
}

bool DeploymentRuntime::initialize() noexcept {
  if (initialized_) {
    return !failed();
  }
  if (!abi_.valid() || !abi_.hasDeploymentPlan()) {
    setError(DeploymentError::InvalidABI);
    return false;
  }
  if ((abi_.incoming_route_count != 0 || abi_.outgoing_route_count != 0) &&
      !transport_.valid()) {
    setError(DeploymentError::InvalidTransport);
    return false;
  }
  if (!allocateState() || !initializeResources()) {
    if (!failed()) {
      setError(DeploymentError::AllocationFailed);
    }
    return false;
  }
  initialized_ = true;
  return true;
}

bool DeploymentRuntime::allocateState() noexcept {
  uint64_t descriptor_bytes = 0;
  uint32_t max_inputs = 0;
  uint32_t max_outputs = 0;
  for (uint32_t slot = 0; slot < abi_.resource_count; ++slot) {
    const uint64_t size = descriptorSize(abi_.resources[slot].rank);
    if (descriptor_bytes > UINT64_MAX - size) {
      return false;
    }
    descriptor_bytes += size;
  }
  for (uint32_t index = 0; index < abi_.task_binding_count; ++index) {
    const TaskBinding &binding = abi_.task_bindings[index];
    if (binding.input_count > max_inputs) {
      max_inputs = binding.input_count;
    }
    if (binding.output_count > max_outputs) {
      max_outputs = binding.output_count;
    }
  }
  if (descriptor_bytes > SIZE_MAX ||
      abi_.workspace_size > SIZE_MAX - (kDataAlignment - 1U)) {
    return false;
  }

  if (abi_.workspace_size != 0) {
    workspace_allocation_ = allocateBytes(
        static_cast<size_t>(abi_.workspace_size + kDataAlignment - 1U),
        heap_profile_);
    if (workspace_allocation_ == nullptr) {
      return false;
    }
    workspace_ = alignPointer(workspace_allocation_, kDataAlignment);
  }
  descriptor_allocation_ =
      allocateBytes(static_cast<size_t>(descriptor_bytes), heap_profile_);
  descriptors_ = static_cast<uint8_t *>(descriptor_allocation_);
  resource_tensors_ = static_cast<Tensor *>(
      allocateBytes(sizeof(Tensor) * abi_.resource_count, heap_profile_));
  resource_bound_ = static_cast<bool *>(
      allocateBytes(sizeof(bool) * abi_.resource_count, heap_profile_));
  resource_ready_ = static_cast<bool *>(
      allocateBytes(sizeof(bool) * abi_.resource_count, heap_profile_));
  task_complete_ = static_cast<bool *>(
      allocateBytes(sizeof(bool) * abi_.task_binding_count, heap_profile_));
  pending_transmit_tasks_ = static_cast<uint32_t *>(
      allocateBytes(sizeof(uint32_t) * abi_.task_binding_count, heap_profile_));
  if (staticRuntimeEnabled()) {
    task_readiness_ = static_cast<uint32_t *>(allocateBytes(
        sizeof(uint32_t) * abi_.task_binding_count, heap_profile_));
    task_queued_ = static_cast<bool *>(allocateBytes(
        sizeof(bool) * abi_.task_binding_count, heap_profile_));
    ready_tasks_ = static_cast<uint32_t *>(allocateBytes(
        sizeof(uint32_t) * abi_.task_binding_count, heap_profile_));
    if (profile_ != nullptr) {
      task_ready_cycles_ = static_cast<uint64_t *>(allocateBytes(
          sizeof(uint64_t) * abi_.task_binding_count, heap_profile_));
      task_ready_from_route_ = static_cast<bool *>(allocateBytes(
          sizeof(bool) * abi_.task_binding_count, heap_profile_));
      task_finish_cycles_ = static_cast<uint64_t *>(allocateBytes(
          sizeof(uint64_t) * abi_.task_binding_count, heap_profile_));
      task_nic_submitted_ = static_cast<bool *>(allocateBytes(
          sizeof(bool) * abi_.task_binding_count, heap_profile_));
    }
  }
  outgoing_route_sent_ = static_cast<bool *>(
      allocateBytes(sizeof(bool) * abi_.outgoing_route_count, heap_profile_));
  incoming_route_received_ = static_cast<bool *>(
      allocateBytes(sizeof(bool) * abi_.incoming_route_count, heap_profile_));
  task_inputs_ = static_cast<Tensor *>(
      allocateBytes(sizeof(Tensor) * max_inputs, heap_profile_));
  task_outputs_ = static_cast<Tensor *>(
      allocateBytes(sizeof(Tensor) * max_outputs, heap_profile_));
  receive_states_ = static_cast<ReceiveState *>(allocateBytes(
      sizeof(ReceiveState) * abi_.incoming_route_count, heap_profile_));

  if ((descriptor_bytes != 0 && descriptors_ == nullptr) ||
      (abi_.resource_count != 0 &&
       (resource_tensors_ == nullptr || resource_bound_ == nullptr ||
        resource_ready_ == nullptr)) ||
      (abi_.task_binding_count != 0 &&
       (task_complete_ == nullptr || pending_transmit_tasks_ == nullptr ||
        (staticRuntimeEnabled() &&
         (task_readiness_ == nullptr || task_queued_ == nullptr ||
          ready_tasks_ == nullptr ||
          (profile_ != nullptr &&
           (task_ready_cycles_ == nullptr || task_ready_from_route_ == nullptr ||
            task_finish_cycles_ == nullptr ||
            task_nic_submitted_ == nullptr)))))) ||
      (abi_.outgoing_route_count != 0 && outgoing_route_sent_ == nullptr) ||
      (abi_.incoming_route_count != 0 &&
       (incoming_route_received_ == nullptr || receive_states_ == nullptr)) ||
      (max_inputs != 0 && task_inputs_ == nullptr) ||
      (max_outputs != 0 && task_outputs_ == nullptr)) {
    return false;
  }

  if (staticRuntimeEnabled()) {
    for (uint32_t task_index = 0; task_index < abi_.task_binding_count;
         ++task_index)
      task_readiness_[task_index] =
          abi_.static_tasks[task_index].initial_readiness;
  }

  for (uint32_t route_index = 0; route_index < abi_.incoming_route_count;
       ++route_index) {
    const uint32_t source = abi_.incoming_routes[route_index].source_core;
    bool known = false;
    for (uint32_t state_index = 0; state_index < receive_state_count_;
         ++state_index) {
      if (receive_states_[state_index].source_tile == source) {
        known = true;
        break;
      }
    }
    if (!known) {
      receive_states_[receive_state_count_].source_tile = source;
      ++receive_state_count_;
    }
  }
  return true;
}

bool DeploymentRuntime::initializeResources() noexcept {
  uint8_t *descriptor = descriptors_;
  for (uint32_t slot = 0; slot < abi_.resource_count; ++slot) {
    const Resource &resource = abi_.resources[slot];
    resource_tensors_[slot] = {
        resource.element_type,
        static_cast<int64_t>(resource.rank),
        descriptor,
    };
    descriptor += descriptorSize(resource.rank);

    if ((resource.flags & ResourceWorkspace) != 0) {
      if (!setResourceData(slot, workspace_ + resource.workspace_offset)) {
        return false;
      }
    } else if ((resource.flags & ResourceScratchpad) != 0) {
      const uint64_t address =
          ScratchpadPhysicalBase + resource.workspace_offset;
      if (address > UINTPTR_MAX ||
          !setResourceData(slot, reinterpret_cast<void *>(
                                     static_cast<uintptr_t>(address)))) {
        return false;
      }
    }
  }
  if (staticRuntimeEnabled()) {
    for (uint32_t task_index = 0; task_index < abi_.task_binding_count;
         ++task_index) {
      if (task_readiness_[task_index] == 0 &&
          !enqueueReadyTask(task_index))
        return false;
    }
  }
  return true;
}

bool DeploymentRuntime::setResourceData(uint32_t local_slot,
                                        void *data) noexcept {
  const Resource *resource = abi_.findResource(local_slot);
  if (resource == nullptr || data == nullptr) {
    return false;
  }

  const bool was_bound = resource_bound_[local_slot];
  Tensor &tensor = resource_tensors_[local_slot];
  auto *header = static_cast<MemRefDescriptorHeader *>(tensor.descriptor);
  header->allocated = data;
  header->aligned = data;
  header->offset = 0;

  auto *sizes =
      reinterpret_cast<int64_t *>(static_cast<uint8_t *>(tensor.descriptor) +
                                  sizeof(MemRefDescriptorHeader));
  auto *strides = sizes + resource->rank;
  uint64_t stride = 1;
  for (uint32_t reverse = 0; reverse < resource->rank; ++reverse) {
    const uint32_t dimension = resource->rank - reverse - 1U;
    const int64_t size =
        abi_.resource_dimensions[resource->dimension_offset + dimension];
    sizes[dimension] = size;
    strides[dimension] = static_cast<int64_t>(stride);
    stride *= static_cast<uint64_t>(size);
  }
  resource_bound_[local_slot] = true;
  if (!was_bound && staticRuntimeEnabled()) {
    const StaticResource &entry = abi_.static_resources[local_slot];
    if (!notifyConsumers(entry.bound_consumer_offset,
                         entry.bound_consumer_count))
      return false;
  }
  return true;
}

bool DeploymentRuntime::bindRouteInputView(const Route &route) noexcept {
  const RouteView *view = abi_.findRouteView(route.id);
  if (view == nullptr || (view->flags & RouteViewIncoming) == 0) {
    return true;
  }
  if (view->movement_mode == MemoryMovementMode::Segmented)
    return view->owner_slot < abi_.resource_count &&
           resource_bound_[view->owner_slot];
  if (view->local_resource_slot != route.local_slot ||
      view->owner_slot >= abi_.resource_count ||
      !resource_bound_[view->owner_slot]) {
    return false;
  }
  auto *owner = static_cast<MemRefDescriptorHeader *>(
      resource_tensors_[view->owner_slot].descriptor);
  auto *local = static_cast<MemRefDescriptorHeader *>(
      resource_tensors_[route.local_slot].descriptor);
  if (owner == nullptr || local == nullptr || owner->aligned == nullptr ||
      view->byte_offset >
          UINTPTR_MAX - reinterpret_cast<uintptr_t>(owner->aligned)) {
    return false;
  }
  if (route.local_slot != view->owner_slot) {
    local->allocated = owner->allocated;
    local->aligned = static_cast<uint8_t *>(owner->aligned) + view->byte_offset;
    local->offset = 0;
    resource_bound_[route.local_slot] = true;
  }
  return true;
}

uint8_t *
DeploymentRuntime::incomingRouteDestination(const Route &route) noexcept {
  const RouteView *view = abi_.findRouteView(route.id);
  if (view == nullptr || (view->flags & RouteViewIncoming) == 0) {
    auto *descriptor = static_cast<MemRefDescriptorHeader *>(
        resource_tensors_[route.local_slot].descriptor);
    return descriptor == nullptr ? nullptr
                                 : static_cast<uint8_t *>(descriptor->aligned);
  }
  if (!bindRouteInputView(route)) {
    return nullptr;
  }
  if (view->movement_mode == MemoryMovementMode::Segmented)
    return nullptr;
  auto *owner = static_cast<MemRefDescriptorHeader *>(
      resource_tensors_[view->owner_slot].descriptor);
  return static_cast<uint8_t *>(owner->aligned) + view->byte_offset;
}

uint8_t *DeploymentRuntime::incomingRouteSegmentDestination(
    const Route &route, const MemorySegment &segment) noexcept {
  const RouteView *view = abi_.findRouteView(route.id);
  if (view == nullptr || (view->flags & RouteViewIncoming) == 0 ||
      view->movement_mode != MemoryMovementMode::Segmented ||
      view->owner_slot >= abi_.resource_count ||
      !resource_bound_[view->owner_slot])
    return nullptr;
  auto *owner = static_cast<MemRefDescriptorHeader *>(
      resource_tensors_[view->owner_slot].descriptor);
  if (owner == nullptr || owner->aligned == nullptr ||
      segment.destination_byte_offset >
          UINTPTR_MAX - reinterpret_cast<uintptr_t>(owner->aligned))
    return nullptr;
  return static_cast<uint8_t *>(owner->aligned) +
         segment.destination_byte_offset;
}

const uint8_t *
DeploymentRuntime::outgoingRouteSource(const Route &route) const noexcept {
  const RouteView *view = abi_.findRouteView(route.id);
  uint32_t slot = route.local_slot;
  uint64_t offset = 0;
  if (view != nullptr && (view->flags & RouteViewOutgoing) != 0) {
    if (view->movement_mode == MemoryMovementMode::Segmented)
      return nullptr;
    slot = view->owner_slot;
    offset = view->byte_offset;
  }
  if (slot >= abi_.resource_count || !resource_bound_[slot]) {
    return nullptr;
  }
  const auto *descriptor = static_cast<const MemRefDescriptorHeader *>(
      resource_tensors_[slot].descriptor);
  if (descriptor == nullptr || descriptor->aligned == nullptr ||
      offset > UINTPTR_MAX - reinterpret_cast<uintptr_t>(descriptor->aligned)) {
    return nullptr;
  }
  return static_cast<const uint8_t *>(descriptor->aligned) + offset;
}

const uint8_t *DeploymentRuntime::outgoingRouteSegmentSource(
    const Route &route, const MemorySegment &segment) const noexcept {
  const RouteView *view = abi_.findRouteView(route.id);
  if (view == nullptr || (view->flags & RouteViewOutgoing) == 0 ||
      view->movement_mode != MemoryMovementMode::Segmented ||
      view->owner_slot >= abi_.resource_count ||
      !resource_bound_[view->owner_slot])
    return nullptr;
  const auto *owner = static_cast<const MemRefDescriptorHeader *>(
      resource_tensors_[view->owner_slot].descriptor);
  if (owner == nullptr || owner->aligned == nullptr ||
      segment.source_byte_offset >
          UINTPTR_MAX - reinterpret_cast<uintptr_t>(owner->aligned))
    return nullptr;
  return static_cast<const uint8_t *>(owner->aligned) +
         segment.source_byte_offset;
}

void DeploymentRuntime::updateAssemblyReadiness(uint32_t route_id) noexcept {
  const RouteView *route_view = abi_.findRouteView(route_id);
  if (route_view == nullptr || route_view->assembly_id == InvalidTileABIId) {
    return;
  }
  const Assembly *assembly = abi_.findAssembly(route_view->assembly_id);
  if (assembly == nullptr) {
    return;
  }
  for (uint32_t index = 0; index < assembly->contribution_count; ++index) {
    const AssemblyContribution &contribution =
        abi_.assembly_contributions[assembly->contribution_offset + index];
    if (contribution.route_id == InvalidTileABIId) {
      return;
    }
    const Route *route = abi_.findIncomingRoute(contribution.route_id);
    if (route == nullptr) {
      return;
    }
    const uint32_t route_index =
        static_cast<uint32_t>(route - abi_.incoming_routes);
    if (!incoming_route_received_[route_index]) {
      return;
    }
  }
  markResourceReady(assembly->owner_slot, true);
}

void DeploymentRuntime::completeIncomingRoute(const Route &route) noexcept {
  const uint32_t route_index =
      static_cast<uint32_t>(&route - abi_.incoming_routes);
  incoming_route_received_[route_index] = true;
  ++complete_incoming_route_count_;
  if (!markResourceReady(route.local_slot, true))
    return;
  updateAssemblyReadiness(route.id);
}

bool DeploymentRuntime::boot() noexcept {
  if (booted_) {
    return !failed();
  }
  if (!initialize()) {
    return false;
  }
  for (uint32_t index = 0; index < abi_.boot_task_count; ++index) {
    if (abi_.boot_tasks[index].execute(nullptr, 0, nullptr, 0) !=
        TaskStatus::Success) {
      setError(DeploymentError::BootTaskFailed);
      return false;
    }
  }
  booted_ = true;
  return true;
}

bool DeploymentRuntime::bindModelInput(uint32_t model_index,
                                       void *data) noexcept {
  if (!initialize()) {
    return false;
  }
  for (uint32_t index = 0; index < abi_.model_input_count; ++index) {
    const ModelIO &input = abi_.model_inputs[index];
    if (input.model_index == model_index) {
      if (!setResourceData(input.local_slot, data)) {
        setError(DeploymentError::InvalidModelIO);
        return false;
      }
      if (!markResourceReady(input.local_slot))
        return false;
      return true;
    }
  }
  setError(DeploymentError::InvalidModelIO);
  return false;
}

bool DeploymentRuntime::bindModelOutput(uint32_t model_index,
                                        void *data) noexcept {
  if (!initialize()) {
    return false;
  }
  for (uint32_t index = 0; index < abi_.model_output_count; ++index) {
    const ModelIO &output = abi_.model_outputs[index];
    if (output.model_index == model_index) {
      if (!setResourceData(output.local_slot, data)) {
        setError(DeploymentError::InvalidModelIO);
        return false;
      }
      return true;
    }
  }
  setError(DeploymentError::InvalidModelIO);
  return false;
}

DeploymentStep DeploymentRuntime::step() noexcept {
#if defined(GOLEM_RUNTIME_ENABLE_PROFILE)
  const uint64_t start_cycle = profileCycle();
#define GOLEM_RECORD_PROFILE_STEP(activity)                                    \
  recordProfileStep((activity), start_cycle)
#else
#define GOLEM_RECORD_PROFILE_STEP(activity) ((void)0)
#endif
  if (failed()) {
    GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Failed);
    return DeploymentStep::Failed;
  }
  if (!boot()) {
    GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Failed);
    return DeploymentStep::Failed;
  }
  if (complete()) {
    GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Complete);
    return DeploymentStep::Complete;
  }

  if (transport_.receiveDMAAvailable()) {
    const bool progressed = progressReceiveDMA();
    if (failed()) {
      GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Failed);
      return DeploymentStep::Failed;
    }
    if (progressed) {
      const DeploymentStep result =
          complete() ? DeploymentStep::Complete : DeploymentStep::Progress;
      GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Receive);
      return result;
    }
  }

  RoutedWord word{};
  if (transport_.tryReceive(&word)) {
    if (!consumeWord(word)) {
      if (!failed()) {
        setInvalidFrame(InvalidFrameReason::InvalidReceivePhase,
                        word.source_tile, UINT32_MAX, UINT32_MAX, 0,
                        word.payload);
      }
      GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Failed);
      return DeploymentStep::Failed;
    }
    const DeploymentStep result =
        complete() ? DeploymentStep::Complete : DeploymentStep::Progress;
    GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Receive);
    return result;
  }

  if (transmit_active_) {
    const bool progressed = progressTaskRoutes();
    if (failed()) {
      GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Failed);
      return DeploymentStep::Failed;
    }
    if (!transmit_active_ && !beginNextQueuedTaskRoutes()) {
      setError(DeploymentError::RouteSendFailed);
      GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Failed);
      return DeploymentStep::Failed;
    }
    if (complete()) {
      GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Transmit);
      return DeploymentStep::Complete;
    }
    if (!progressed &&
        transmit_policy_ == DeploymentTransmitPolicy::OverlapReadyTasks &&
        executeReadyTaskWithoutTransmitConflict()) {
      GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Execute);
      return DeploymentStep::Progress;
    }
    const DeploymentStep result =
        progressed ? DeploymentStep::Progress : DeploymentStep::WaitForTransmit;
    GOLEM_RECORD_PROFILE_STEP(progressed ? ProfileActivity::Transmit
                                         : ProfileActivity::TransmitBlocked);
    return result;
  }

  if (executeReadyTask()) {
    const DeploymentStep result =
        complete() ? DeploymentStep::Complete : DeploymentStep::Progress;
    GOLEM_RECORD_PROFILE_STEP(ProfileActivity::Execute);
    return result;
  }
  const bool receive_wait = !failed() && hasPendingIncomingRoute();
  const DeploymentStep result =
      failed() ? DeploymentStep::Failed
               : (receive_wait ? DeploymentStep::WaitForReceive
                               : DeploymentStep::Idle);
  GOLEM_RECORD_PROFILE_STEP(failed()
                                ? ProfileActivity::Failed
                                : (receive_wait ? ProfileActivity::ReceiveWait
                                                : ProfileActivity::Idle));
#undef GOLEM_RECORD_PROFILE_STEP
  return result;
}

bool DeploymentRuntime::staticRuntimeEnabled() const noexcept {
  return (abi_.abi_features & TileABIStaticRuntime) != 0;
}

bool DeploymentRuntime::notifyConsumers(uint32_t offset, uint32_t count,
                                        bool route_arrival) noexcept {
  if (!staticRuntimeEnabled())
    return true;
  if (offset > abi_.static_runtime_data_count ||
      count > abi_.static_runtime_data_count - offset) {
    setError(DeploymentError::InvalidReadinessEvent);
    return false;
  }
  for (uint32_t consumer = 0; consumer < count; ++consumer) {
    const uint32_t task_index =
        abi_.static_runtime_data[offset + consumer];
    if (task_index >= abi_.task_binding_count ||
        task_complete_[task_index] || task_readiness_[task_index] == 0) {
      setError(DeploymentError::InvalidReadinessEvent);
      return false;
    }
    --task_readiness_[task_index];
    if (profile_ != nullptr)
      ++profile_->readiness_decrements;
    if (task_readiness_[task_index] == 0 &&
        !enqueueReadyTask(task_index, route_arrival)) {
      setError(DeploymentError::InvalidReadinessEvent);
      return false;
    }
  }
  return true;
}

bool DeploymentRuntime::markResourceReady(uint32_t local_slot,
                                          bool route_arrival) noexcept {
  if (local_slot >= abi_.resource_count) {
    setError(DeploymentError::InvalidReadinessEvent);
    return false;
  }
  if (resource_ready_[local_slot])
    return true;
  resource_ready_[local_slot] = true;
  if (!staticRuntimeEnabled())
    return true;
  const StaticResource &resource = abi_.static_resources[local_slot];
  return notifyConsumers(resource.ready_consumer_offset,
                         resource.ready_consumer_count, route_arrival);
}

bool DeploymentRuntime::notifyTaskComplete(uint32_t task_index) noexcept {
  if (!staticRuntimeEnabled())
    return true;
  if (task_index >= abi_.static_task_count) {
    setError(DeploymentError::InvalidReadinessEvent);
    return false;
  }
  const StaticTask &task = abi_.static_tasks[task_index];
  return notifyConsumers(task.dependent_offset, task.dependent_count);
}

bool DeploymentRuntime::enqueueReadyTask(uint32_t task_index,
                                         bool route_arrival) noexcept {
  if (task_index >= abi_.task_binding_count || task_complete_[task_index])
    return false;
  if (task_queued_[task_index])
    return true;
  if (ready_task_count_ >= abi_.task_binding_count)
    return false;
  uint32_t index = ready_task_count_++;
  while (index != 0) {
    const uint32_t parent = (index - 1U) / 2U;
    if (ready_tasks_[parent] <= task_index)
      break;
    ready_tasks_[index] = ready_tasks_[parent];
    index = parent;
  }
  ready_tasks_[index] = task_index;
  task_queued_[task_index] = true;
  if (profile_ != nullptr) {
    task_ready_cycles_[task_index] = profileCycle();
    task_ready_from_route_[task_index] = route_arrival;
    ++profile_->ready_queue_pushes;
    if (ready_task_count_ > profile_->maximum_ready_queue_occupancy)
      profile_->maximum_ready_queue_occupancy = ready_task_count_;
  }
  return true;
}

void DeploymentRuntime::restoreReadyHeap(uint32_t index) noexcept {
  if (index >= ready_task_count_)
    return;
  while (index != 0) {
    const uint32_t parent = (index - 1U) / 2U;
    if (ready_tasks_[parent] <= ready_tasks_[index])
      break;
    const uint32_t value = ready_tasks_[parent];
    ready_tasks_[parent] = ready_tasks_[index];
    ready_tasks_[index] = value;
    index = parent;
  }
  while (true) {
    const uint32_t left = index * 2U + 1U;
    if (left >= ready_task_count_)
      break;
    const uint32_t right = left + 1U;
    const uint32_t child =
        right < ready_task_count_ && ready_tasks_[right] < ready_tasks_[left]
            ? right
            : left;
    if (ready_tasks_[index] <= ready_tasks_[child])
      break;
    const uint32_t value = ready_tasks_[index];
    ready_tasks_[index] = ready_tasks_[child];
    ready_tasks_[child] = value;
    index = child;
  }
}

bool DeploymentRuntime::popReadyTask(uint32_t *task_index) noexcept {
  if (task_index == nullptr || ready_task_count_ == 0)
    return false;
  *task_index = ready_tasks_[0];
  if (profile_ != nullptr)
    ++profile_->ready_queue_pops;
  task_queued_[*task_index] = false;
  --ready_task_count_;
  if (ready_task_count_ != 0) {
    ready_tasks_[0] = ready_tasks_[ready_task_count_];
    restoreReadyHeap(0);
  }
  return true;
}

bool DeploymentRuntime::popReadyTaskWithoutTransmitConflict(
    uint32_t *task_index) noexcept {
  if (task_index == nullptr || ready_task_count_ == 0)
    return false;
  uint32_t selected = ready_task_count_;
  uint32_t selected_task = UINT32_MAX;
  for (uint32_t index = 0; index < ready_task_count_; ++index) {
    const uint32_t candidate = ready_tasks_[index];
    if (candidate < selected_task &&
        !taskOutputConflictsWithPendingTransmit(candidate)) {
      selected = index;
      selected_task = candidate;
    }
  }
  if (selected == ready_task_count_)
    return false;
  *task_index = selected_task;
  if (profile_ != nullptr)
    ++profile_->ready_queue_pops;
  task_queued_[selected_task] = false;
  --ready_task_count_;
  if (selected != ready_task_count_) {
    ready_tasks_[selected] = ready_tasks_[ready_task_count_];
    restoreReadyHeap(selected);
  }
  return true;
}

bool DeploymentRuntime::taskReady(uint32_t task_index) const noexcept {
  if (task_index >= abi_.task_binding_count || task_complete_[task_index]) {
    return false;
  }
  if (staticRuntimeEnabled())
    return task_readiness_[task_index] == 0;
  const TaskBinding &binding = abi_.task_bindings[task_index];
  for (uint32_t input = 0; input < binding.input_count; ++input) {
    const uint32_t slot = abi_.task_binding_data[binding.input_offset + input];
    if (!resource_bound_[slot] || !resource_ready_[slot]) {
      return false;
    }
  }
  for (uint32_t output = 0; output < binding.output_count; ++output) {
    const uint32_t slot =
        abi_.task_binding_data[binding.output_offset + output];
    if (!resource_bound_[slot]) {
      return false;
    }
  }
  for (uint32_t dependency = 0; dependency < binding.dependency_count;
       ++dependency) {
    const uint32_t dependency_id =
        abi_.task_binding_data[binding.dependency_offset + dependency];
    const uint32_t dependency_index = taskIndex(dependency_id);
    if (dependency_index == kInvalidIndex ||
        !task_complete_[dependency_index]) {
      return false;
    }
  }
  return true;
}

bool DeploymentRuntime::executeReadyTask() noexcept {
  if (staticRuntimeEnabled()) {
    uint32_t task_index = 0;
    return popReadyTask(&task_index) && executeTask(task_index);
  }
  for (uint32_t task_index = 0; task_index < abi_.task_binding_count;
       ++task_index) {
    if (taskReady(task_index)) {
      return executeTask(task_index);
    }
  }
  return false;
}

bool DeploymentRuntime::executeReadyTaskWithoutTransmitConflict() noexcept {
  if (staticRuntimeEnabled()) {
    uint32_t task_index = 0;
    return popReadyTaskWithoutTransmitConflict(&task_index) &&
           executeTask(task_index);
  }
  for (uint32_t task_index = 0; task_index < abi_.task_binding_count;
       ++task_index) {
    if (taskReady(task_index) &&
        !taskOutputConflictsWithPendingTransmit(task_index)) {
      return executeTask(task_index);
    }
  }
  return false;
}

bool DeploymentRuntime::executeTask(uint32_t task_index) noexcept {
  if (!taskReady(task_index)) {
    return false;
  }
  if (staticRuntimeEnabled())
    task_queued_[task_index] = false;
  if (staticRuntimeEnabled() && profile_ != nullptr &&
      task_ready_from_route_[task_index]) {
    const uint64_t start_cycle = profileCycle();
    if (start_cycle >= task_ready_cycles_[task_index]) {
      ++profile_->route_ready_to_task_start_events;
      profile_->route_ready_to_task_start_cycles +=
          start_cycle - task_ready_cycles_[task_index];
    }
  }
  const TaskBinding &binding = abi_.task_bindings[task_index];
  const Task &task = abi_.dispatch_tasks.tasks[task_index];
  for (uint32_t input = 0; input < binding.input_count; ++input) {
    const uint32_t slot = abi_.task_binding_data[binding.input_offset + input];
    task_inputs_[input] = resource_tensors_[slot];
  }
  for (uint32_t output = 0; output < binding.output_count; ++output) {
    const uint32_t slot =
        abi_.task_binding_data[binding.output_offset + output];
    task_outputs_[output] = resource_tensors_[slot];
  }
  emitTaskTrace(TaskTraceEvent::Start, task.id);
  const TaskStatus status = task.execute(task_inputs_, binding.input_count,
                                         task_outputs_, binding.output_count);
  emitTaskTrace(TaskTraceEvent::Finish, task.id);
  if (status != TaskStatus::Success) {
    setError(DeploymentError::TaskFailed);
    return false;
  }
  if (staticRuntimeEnabled() && profile_ != nullptr) {
    task_finish_cycles_[task_index] = profileCycle();
    task_nic_submitted_[task_index] = false;
  }
  for (uint32_t output = 0; output < binding.output_count; ++output) {
    const uint32_t slot =
        abi_.task_binding_data[binding.output_offset + output];
    if (!markResourceReady(slot))
      return false;
  }
  task_complete_[task_index] = true;
  ++complete_task_count_;
  if (!notifyTaskComplete(task_index))
    return false;
  if (!scheduleTaskRoutes(task.id)) {
    setError(DeploymentError::RouteSendFailed);
    return false;
  }
  return true;
}

bool DeploymentRuntime::taskOutputConflictsWithPendingTransmit(
    uint32_t task_index) const noexcept {
  const TaskBinding &binding = abi_.task_bindings[task_index];
  for (uint32_t output = 0; output < binding.output_count; ++output) {
    const uint32_t output_slot =
        abi_.task_binding_data[binding.output_offset + output];
    const Resource *output_resource = abi_.findResource(output_slot);
    const auto *output_descriptor = static_cast<const MemRefDescriptorHeader *>(
        resource_tensors_[output_slot].descriptor);
    if (output_resource == nullptr || output_descriptor == nullptr ||
        output_descriptor->aligned == nullptr) {
      return true;
    }
    const uintptr_t output_start =
        reinterpret_cast<uintptr_t>(output_descriptor->aligned);
    if (output_resource->byte_size > UINTPTR_MAX - output_start) {
      return true;
    }
    const uintptr_t output_finish =
        output_start + static_cast<uintptr_t>(output_resource->byte_size);

    auto taskRoutesConflict = [&](uint32_t task_id) {
      const uint32_t source_index = taskIndex(task_id);
      if (source_index == kInvalidIndex || !task_complete_[source_index])
        return false;
      const StaticTask &task = abi_.static_tasks[source_index];
      for (uint32_t ordinal = 0; ordinal < task.outgoing_route_count;
           ++ordinal) {
        const uint32_t route_index = task.outgoing_route_offset + ordinal;
        if (!outgoing_route_sent_[route_index] &&
            routeConflictsWithOutput(abi_.outgoing_routes[route_index],
                                     output_slot, output_start, output_finish))
          return true;
      }
      return false;
    };

    if (staticRuntimeEnabled()) {
      if (transmit_active_ && taskRoutesConflict(transmit_task_id_))
        return true;
      for (uint32_t pending = pending_transmit_read_;
           pending < pending_transmit_write_; ++pending) {
        const uint32_t task_id =
            pending_transmit_tasks_[pending % abi_.task_binding_count];
        if (taskRoutesConflict(task_id))
          return true;
      }
    } else {
      for (uint32_t route_index = 0; route_index < abi_.outgoing_route_count;
           ++route_index) {
        if (outgoing_route_sent_[route_index])
          continue;
        const Route &route = abi_.outgoing_routes[route_index];
        const uint32_t source_index = taskIndex(route.source_task);
        if (source_index != kInvalidIndex && task_complete_[source_index] &&
            routeConflictsWithOutput(route, output_slot, output_start,
                                     output_finish))
          return true;
      }
    }
  }
  return false;
}

bool DeploymentRuntime::routeConflictsWithOutput(
    const Route &route, uint32_t output_slot, uintptr_t output_start,
    uintptr_t output_finish) const noexcept {
  if (route.local_slot == output_slot)
    return true;
  const SegmentedMovement *segmented =
      abi_.findSegmentedMovement(route.id, RouteViewOutgoing);
  if (segmented != nullptr) {
    for (uint32_t segment_index = 0;
         segment_index < segmented->segment_count; ++segment_index) {
      const MemorySegment &segment =
          abi_.memory_segments[segmented->segment_offset + segment_index];
      const uint8_t *segment_data =
          outgoingRouteSegmentSource(route, segment);
      if (segment_data == nullptr)
        return true;
      const uintptr_t route_start =
          reinterpret_cast<uintptr_t>(segment_data);
      if (segment.byte_size > UINTPTR_MAX - route_start)
        return true;
      const uintptr_t route_finish =
          route_start + static_cast<uintptr_t>(segment.byte_size);
      if (output_start < route_finish && route_start < output_finish)
        return true;
    }
    return false;
  }
  const uint8_t *route_data = outgoingRouteSource(route);
  if (route_data == nullptr)
    return true;
  const uintptr_t route_start = reinterpret_cast<uintptr_t>(route_data);
  if (route.byte_size > UINTPTR_MAX - route_start)
    return true;
  const uintptr_t route_finish =
      route_start + static_cast<uintptr_t>(route.byte_size);
  return output_start < route_finish && route_start < output_finish;
}

bool DeploymentRuntime::taskHasOutgoingRoutes(uint32_t task_id) const noexcept {
  if (staticRuntimeEnabled()) {
    const uint32_t task_index = taskIndex(task_id);
    if (task_index == kInvalidIndex)
      return false;
    const StaticTask &task = abi_.static_tasks[task_index];
    for (uint32_t ordinal = 0; ordinal < task.outgoing_route_count; ++ordinal) {
      if (!outgoing_route_sent_[task.outgoing_route_offset + ordinal])
        return true;
    }
    return false;
  }
  for (uint32_t index = 0; index < abi_.outgoing_route_count; ++index) {
    if (!outgoing_route_sent_[index] &&
        abi_.outgoing_routes[index].source_task == task_id) {
      return true;
    }
  }
  return false;
}

bool DeploymentRuntime::scheduleTaskRoutes(uint32_t task_id) noexcept {
  if (!taskHasOutgoingRoutes(task_id)) {
    return true;
  }
  if (transmit_policy_ == DeploymentTransmitPolicy::Blocking) {
    return beginTaskRoutes(task_id);
  }
  if (pending_transmit_write_ - pending_transmit_read_ >=
      abi_.task_binding_count) {
    return false;
  }
  pending_transmit_tasks_[pending_transmit_write_ % abi_.task_binding_count] =
      task_id;
  ++pending_transmit_write_;
  return beginNextQueuedTaskRoutes();
}

bool DeploymentRuntime::beginNextQueuedTaskRoutes() noexcept {
  if (transmit_active_ || pending_transmit_read_ == pending_transmit_write_) {
    return true;
  }
  const uint32_t task_id =
      pending_transmit_tasks_[pending_transmit_read_ % abi_.task_binding_count];
  ++pending_transmit_read_;
  return beginTaskRoutes(task_id);
}

bool DeploymentRuntime::beginTaskRoutes(uint32_t task_id) noexcept {
  if (transmit_active_) {
    return false;
  }
  transmit_task_id_ = task_id;
  if (staticRuntimeEnabled()) {
    const uint32_t task_index = taskIndex(task_id);
    if (task_index == kInvalidIndex)
      return false;
    const StaticTask &task = abi_.static_tasks[task_index];
    transmit_route_index_ = task.outgoing_route_offset;
    transmit_route_limit_ =
        task.outgoing_route_offset + task.outgoing_route_count;
  } else {
    transmit_route_index_ = 0;
    transmit_route_limit_ = abi_.outgoing_route_count;
  }
  transmit_phase_ = 0;
  transmit_word_offset_ = 0;
  transmit_segment_index_ = 0;
  transmit_segment_word_offset_ = 0;
  transmit_active_ = true;
  selectNextTaskRoute();
  return true;
}

bool DeploymentRuntime::selectNextTaskRoute() noexcept {
  while (transmit_route_index_ < transmit_route_limit_) {
    const Route &route = abi_.outgoing_routes[transmit_route_index_];
    if (!outgoing_route_sent_[transmit_route_index_] &&
        (staticRuntimeEnabled() || route.source_task == transmit_task_id_))
      return true;
    ++transmit_route_index_;
  }
  transmit_active_ = false;
  return false;
}

bool DeploymentRuntime::progressTaskRoutes() noexcept {
  if (!transmit_active_) {
    return false;
  }

  if (!selectNextTaskRoute()) {
    return true;
  }

  const Route &route = abi_.outgoing_routes[transmit_route_index_];
  const Resource *resource = abi_.findResource(route.local_slot);
  if (resource == nullptr || !resource_ready_[route.local_slot] ||
      route.byte_size == 0 || route.byte_size % sizeof(uint32_t) != 0 ||
      route.byte_size / sizeof(uint32_t) > UINT32_MAX) {
    setError(DeploymentError::RouteSendFailed);
    transmit_active_ = false;
    return false;
  }
  const SegmentedMovement *segmented =
      abi_.findSegmentedMovement(route.id, RouteViewOutgoing);
  const uint8_t *route_data =
      segmented == nullptr ? outgoingRouteSource(route) : nullptr;
  if (segmented == nullptr && route_data == nullptr) {
    setError(DeploymentError::RouteSendFailed);
    transmit_active_ = false;
    return false;
  }

  const uint32_t atomic_limit =
      transport_.try_send_words == nullptr ? 1U : kTransportBurstWords;

  if (transmit_phase_ == 0) {
    const uint32_t header[kFrameHeaderWords] = {
        kFrameMagic,
        route.id,
        static_cast<uint32_t>(execution_id_),
        static_cast<uint32_t>(execution_id_ >> 32U),
        static_cast<uint32_t>(route.byte_size / sizeof(uint32_t)),
    };
    const uint32_t remaining =
        kFrameHeaderWords - static_cast<uint32_t>(transmit_word_offset_);
    const uint32_t send_words =
        remaining < atomic_limit ? remaining : atomic_limit;
    if (!transport_.trySendWords(route.destination_core,
                                 header + transmit_word_offset_, send_words)) {
      return false;
    }
    if (transmit_word_offset_ == 0 && staticRuntimeEnabled() &&
        profile_ != nullptr) {
      const uint32_t task_index = taskIndex(transmit_task_id_);
      if (task_index != kInvalidIndex && !task_nic_submitted_[task_index]) {
        const uint64_t submit_cycle = profileCycle();
        if (submit_cycle >= task_finish_cycles_[task_index]) {
          ++profile_->task_finish_to_nic_submit_events;
          profile_->task_finish_to_nic_submit_cycles +=
              submit_cycle - task_finish_cycles_[task_index];
        }
        task_nic_submitted_[task_index] = true;
      }
    }
    transmit_word_offset_ += send_words;
    if (transmit_word_offset_ == kFrameHeaderWords) {
      transmit_phase_ = 1;
      transmit_word_offset_ = 0;
      transmit_segment_index_ = 0;
      transmit_segment_word_offset_ = 0;
    }
    return true;
  }

  if (segmented != nullptr) {
    if (transmit_segment_index_ >= segmented->segment_count) {
      setError(DeploymentError::RouteSendFailed);
      transmit_active_ = false;
      return false;
    }
    const MemorySegment &segment =
        abi_.memory_segments[segmented->segment_offset +
                             transmit_segment_index_];
    const uint8_t *segment_data = outgoingRouteSegmentSource(route, segment);
    if (segment_data == nullptr ||
        segment.byte_size / sizeof(uint32_t) > UINT32_MAX) {
      setError(DeploymentError::RouteSendFailed);
      transmit_active_ = false;
      return false;
    }
    const auto *words = reinterpret_cast<const uint32_t *>(segment_data);
    const uint32_t segment_words =
        static_cast<uint32_t>(segment.byte_size / sizeof(uint32_t));
    const uint32_t remaining = segment_words - transmit_segment_word_offset_;
    const uint32_t send_words =
        remaining < atomic_limit ? remaining : atomic_limit;
    if (!transport_.trySendWords(route.destination_core,
                                 words + transmit_segment_word_offset_,
                                 send_words))
      return false;
    transmit_segment_word_offset_ += send_words;
    if (transmit_segment_word_offset_ == segment_words) {
      ++transmit_segment_index_;
      transmit_segment_word_offset_ = 0;
    }
    if (transmit_segment_index_ == segmented->segment_count) {
      outgoing_route_sent_[transmit_route_index_] = true;
      ++complete_outgoing_route_count_;
      ++transmit_route_index_;
      transmit_phase_ = 0;
      transmit_word_offset_ = 0;
      transmit_segment_index_ = 0;
    }
    return true;
  }

  const auto *words = reinterpret_cast<const uint32_t *>(route_data);
  const uint64_t word_count = route.byte_size / sizeof(uint32_t);
  const uint64_t remaining = word_count - transmit_word_offset_;
  const uint32_t send_words = static_cast<uint32_t>(
      remaining < atomic_limit ? remaining : atomic_limit);
  if (!transport_.trySendWords(route.destination_core,
                               words + transmit_word_offset_, send_words)) {
    return false;
  }
  transmit_word_offset_ += send_words;
  if (transmit_word_offset_ == word_count) {
    outgoing_route_sent_[transmit_route_index_] = true;
    ++complete_outgoing_route_count_;
    ++transmit_route_index_;
    transmit_phase_ = 0;
    transmit_word_offset_ = 0;
    transmit_segment_index_ = 0;
    transmit_segment_word_offset_ = 0;
  }
  return true;
}

bool DeploymentRuntime::allOutgoingRoutesSent() const noexcept {
  if (staticRuntimeEnabled())
    return complete_outgoing_route_count_ == abi_.outgoing_route_count;
  for (uint32_t index = 0; index < abi_.outgoing_route_count; ++index) {
    if (!outgoing_route_sent_[index]) {
      return false;
    }
  }
  return true;
}

bool DeploymentRuntime::hasPendingIncomingRoute() const noexcept {
  if (staticRuntimeEnabled())
    return complete_incoming_route_count_ != abi_.incoming_route_count;
  for (uint32_t index = 0; index < abi_.incoming_route_count; ++index) {
    if (!incoming_route_received_[index]) {
      return true;
    }
  }
  return false;
}

DeploymentRuntime::ReceiveState *
DeploymentRuntime::receiveState(uint32_t source_tile) noexcept {
  for (uint32_t index = 0; index < receive_state_count_; ++index) {
    if (receive_states_[index].source_tile == source_tile) {
      return &receive_states_[index];
    }
  }
  return nullptr;
}

bool DeploymentRuntime::progressReceiveDMA() noexcept {
  if (!transport_.receiveDMAAvailable()) {
    return false;
  }

  uint32_t completed_source = 0;
  uint32_t completed_route = 0;
  if (transport_.tryReceiveWordsCompletion(&completed_source,
                                           &completed_route)) {
    ReceiveState *state = receiveState(completed_source);
    if (state == nullptr) {
      setInvalidFrame(InvalidFrameReason::DMACompletionUnknownSource,
                      completed_source, completed_route, UINT32_MAX, 0, 0);
      return true;
    }
    if (state->phase != ReceiveDMAActive) {
      setInvalidFrame(InvalidFrameReason::DMACompletionPhaseMismatch,
                      completed_source, completed_route, state->phase,
                      ReceiveDMAActive, state->phase);
      return true;
    }
    if (state->route_id != completed_route) {
      setInvalidFrame(InvalidFrameReason::DMACompletionRouteMismatch,
                      completed_source, completed_route, state->phase,
                      state->route_id, completed_route);
      return true;
    }
    const Route *route = abi_.findIncomingRoute(completed_route);
    if (route == nullptr) {
      setInvalidFrame(InvalidFrameReason::DMACompletionUnknownRoute,
                      completed_source, completed_route, state->phase, 0, 0);
      return true;
    }
    if (route->source_core != completed_source) {
      setInvalidFrame(InvalidFrameReason::DMACompletionSourceMismatch,
                      completed_source, completed_route, state->phase,
                      route->source_core, completed_source);
      return true;
    }
    const uint32_t route_index =
        static_cast<uint32_t>(route - abi_.incoming_routes);
    if (incoming_route_received_[route_index]) {
      setInvalidFrame(InvalidFrameReason::DMACompletionDuplicateRoute,
                      completed_source, completed_route, state->phase, 0, 1);
      return true;
    }
    completeIncomingRoute(*route);
    state->phase = ReceiveMagic;
    state->route_id = 0;
    state->execution_id = 0;
    state->word_count = 0;
    state->received_words = 0;
    state->segment_index = 0;
    state->segment_word_offset = 0;
    state->destination = nullptr;
    return true;
  }

  for (uint32_t index = 0; index < receive_state_count_; ++index) {
    ReceiveState &state = receive_states_[index];
    if (state.phase != ReceiveDMASubmit) {
      continue;
    }
    if (!transport_.tryStartReceiveWords(state.source_tile, state.route_id,
                                         state.destination, state.word_count)) {
      return false;
    }
    state.phase = ReceiveDMAActive;
    return true;
  }
  return false;
}

bool DeploymentRuntime::consumeWord(const RoutedWord &word) noexcept {
  ReceiveState *state = receiveState(word.source_tile);
  if (state == nullptr) {
    setInvalidFrame(InvalidFrameReason::UnknownSource, word.source_tile,
                    UINT32_MAX, UINT32_MAX, 0, word.payload);
    return false;
  }
  switch (state->phase) {
  case ReceiveMagic:
    if (word.payload != kFrameMagic) {
      setInvalidFrame(InvalidFrameReason::InvalidMagic, word.source_tile,
                      UINT32_MAX, state->phase, kFrameMagic, word.payload);
      return false;
    }
    state->phase = ReceiveRoute;
    return true;
  case ReceiveRoute:
    state->route_id = word.payload;
    state->phase = ReceiveExecutionLow;
    return true;
  case ReceiveExecutionLow:
    state->execution_id = word.payload;
    state->phase = ReceiveExecutionHigh;
    return true;
  case ReceiveExecutionHigh:
    state->execution_id |= static_cast<ExecutionId>(word.payload) << 32U;
    state->phase = ReceiveWordCount;
    return true;
  case ReceiveWordCount: {
    const Route *route = abi_.findIncomingRoute(state->route_id);
    if (route == nullptr) {
      setInvalidFrame(InvalidFrameReason::UnknownRoute, word.source_tile,
                      state->route_id, state->phase, 0, 0);
      return false;
    }
    if (route->source_core != word.source_tile) {
      setInvalidFrame(InvalidFrameReason::SourceMismatch, word.source_tile,
                      state->route_id, state->phase, route->source_core,
                      word.source_tile);
      return false;
    }
    if (state->execution_id != execution_id_) {
      setInvalidFrame(InvalidFrameReason::ExecutionMismatch, word.source_tile,
                      state->route_id, state->phase, execution_id_,
                      state->execution_id);
      return false;
    }
    const uint64_t expected_words = route->byte_size / sizeof(uint32_t);
    if (expected_words != word.payload) {
      setInvalidFrame(InvalidFrameReason::WordCountMismatch, word.source_tile,
                      state->route_id, state->phase, expected_words,
                      word.payload);
      return false;
    }
    const uint32_t route_index =
        static_cast<uint32_t>(route - abi_.incoming_routes);
    if (incoming_route_received_[route_index]) {
      setInvalidFrame(InvalidFrameReason::DuplicateRoute, word.source_tile,
                      state->route_id, state->phase, 0, 1);
      return false;
    }
    state->word_count = word.payload;
    state->received_words = 0;
    state->segment_index = 0;
    state->segment_word_offset = 0;
    const SegmentedMovement *segmented =
        abi_.findSegmentedMovement(route->id, RouteViewIncoming);
    state->destination =
        segmented == nullptr ? incomingRouteDestination(*route) : nullptr;
    if (segmented == nullptr && state->destination == nullptr) {
      setInvalidFrame(InvalidFrameReason::MissingPayloadRoute, word.source_tile,
                      state->route_id, state->phase, 1, 0);
      return false;
    }
    state->phase = segmented == nullptr && transport_.receiveDMAAvailable()
                       ? ReceiveDMASubmit
                       : ReceivePayloadWords;
    return state->word_count != 0;
  }
  case ReceivePayloadWords: {
    const SegmentedMovement *segmented =
        abi_.findSegmentedMovement(state->route_id, RouteViewIncoming);
    if (segmented != nullptr) {
      if (state->segment_index >= segmented->segment_count) {
        setInvalidFrame(InvalidFrameReason::MissingPayloadRoute,
                        word.source_tile, state->route_id, state->phase,
                        segmented->segment_count, state->segment_index);
        return false;
      }
      const MemorySegment &segment =
          abi_.memory_segments[segmented->segment_offset +
                               state->segment_index];
      const Route *payloadRoute = abi_.findIncomingRoute(state->route_id);
      uint8_t *destination =
          payloadRoute == nullptr
              ? nullptr
              : incomingRouteSegmentDestination(*payloadRoute, segment);
      if (destination == nullptr) {
        setInvalidFrame(InvalidFrameReason::MissingPayloadRoute,
                        word.source_tile, state->route_id, state->phase, 1, 0);
        return false;
      }
      storeWord(destination +
                    static_cast<uint64_t>(state->segment_word_offset) *
                        sizeof(uint32_t),
                word.payload);
      ++state->segment_word_offset;
      if (static_cast<uint64_t>(state->segment_word_offset) *
              sizeof(uint32_t) ==
          segment.byte_size) {
        ++state->segment_index;
        state->segment_word_offset = 0;
      }
    } else {
      storeWord(state->destination +
                    static_cast<uint64_t>(state->received_words) *
                        sizeof(uint32_t),
                word.payload);
    }
    ++state->received_words;
    if (state->received_words == state->word_count) {
      if (segmented != nullptr &&
          (state->segment_index != segmented->segment_count ||
           state->segment_word_offset != 0)) {
        setInvalidFrame(InvalidFrameReason::WordCountMismatch, word.source_tile,
                        state->route_id, state->phase, segmented->segment_count,
                        state->segment_index);
        return false;
      }
      const Route *route = abi_.findIncomingRoute(state->route_id);
      if (route == nullptr) {
        setInvalidFrame(InvalidFrameReason::MissingPayloadRoute,
                        word.source_tile, state->route_id, state->phase, 0, 0);
        return false;
      }
      completeIncomingRoute(*route);
      state->phase = ReceiveMagic;
      state->segment_index = 0;
      state->segment_word_offset = 0;
    }
    return true;
  }
  case ReceiveDMASubmit:
  case ReceiveDMAActive:
    setInvalidFrame(InvalidFrameReason::UnexpectedWordDuringDMA,
                    word.source_tile, state->route_id, state->phase, 0,
                    word.payload);
    return false;
  default:
    setInvalidFrame(InvalidFrameReason::InvalidReceivePhase, word.source_tile,
                    state->route_id, state->phase, 0, word.payload);
    return false;
  }
}

uint32_t DeploymentRuntime::taskIndex(uint32_t task_id) const noexcept {
  if (staticRuntimeEnabled()) {
    const TaskBinding *binding = abi_.findTaskBinding(task_id);
    if (profile_ != nullptr && binding != nullptr)
      ++profile_->direct_index_hits;
    return binding == nullptr
               ? kInvalidIndex
               : static_cast<uint32_t>(binding - abi_.task_bindings);
  }
  if (profile_ != nullptr)
    ++profile_->compatibility_search_calls;
  for (uint32_t index = 0; index < abi_.task_binding_count; ++index) {
    if (abi_.task_bindings[index].task_id == task_id) {
      return index;
    }
  }
  return kInvalidIndex;
}

bool DeploymentRuntime::complete() const noexcept {
  return initialized_ && !failed() &&
         complete_task_count_ == abi_.task_binding_count && !transmit_active_ &&
         pending_transmit_read_ == pending_transmit_write_ &&
         allOutgoingRoutesSent();
}

bool DeploymentRuntime::failed() const noexcept {
  return error_ != DeploymentError::None;
}

DeploymentError DeploymentRuntime::error() const noexcept { return error_; }

const InvalidFrameDiagnostic &
DeploymentRuntime::invalidFrameDiagnostic() const noexcept {
  return invalid_frame_diagnostic_;
}

ExecutionId DeploymentRuntime::executionId() const noexcept {
  return execution_id_;
}

const Tensor *
DeploymentRuntime::resourceTensor(uint32_t local_slot) const noexcept {
  return initialized_ && local_slot < abi_.resource_count
             ? &resource_tensors_[local_slot]
             : nullptr;
}

const TileABI &DeploymentRuntime::abi() const noexcept { return abi_; }

void DeploymentRuntime::setError(DeploymentError error) noexcept {
  if (error_ == DeploymentError::None) {
    error_ = error;
  }
}

void DeploymentRuntime::setInvalidFrame(InvalidFrameReason reason,
                                        uint32_t source_tile, uint32_t route_id,
                                        uint32_t receive_phase,
                                        uint64_t expected,
                                        uint64_t actual) noexcept {
  if (error_ != DeploymentError::None) {
    return;
  }
  invalid_frame_diagnostic_ = {
      reason, source_tile, route_id, receive_phase, expected, actual,
  };
  setError(DeploymentError::InvalidFrame);
}

uint64_t DeploymentRuntime::profileCycle() const noexcept {
  return profile_ != nullptr && profile_->read_cycle != nullptr
             ? profile_->read_cycle(profile_->context)
             : 0;
}

void DeploymentRuntime::recordProfileStep(ProfileActivity activity,
                                          uint64_t start_cycle) noexcept {
  if (profile_ == nullptr || profile_->read_cycle == nullptr) {
    return;
  }
  const uint64_t cycles = profile_->read_cycle(profile_->context) - start_cycle;
  switch (activity) {
  case ProfileActivity::Receive:
    ++profile_->receive_steps;
    profile_->receive_cycles += cycles;
    return;
  case ProfileActivity::Transmit:
    ++profile_->transmit_steps;
    profile_->transmit_cycles += cycles;
    return;
  case ProfileActivity::TransmitBlocked:
    ++profile_->transmit_blocked_steps;
    profile_->transmit_blocked_cycles += cycles;
    return;
  case ProfileActivity::Execute:
    ++profile_->execute_steps;
    profile_->execute_cycles += cycles;
    return;
  case ProfileActivity::Idle:
    ++profile_->idle_steps;
    profile_->idle_cycles += cycles;
    return;
  case ProfileActivity::ReceiveWait:
    ++profile_->receive_wait_steps;
    profile_->receive_wait_cycles += cycles;
    return;
  case ProfileActivity::Complete:
    ++profile_->complete_steps;
    profile_->complete_cycles += cycles;
    return;
  case ProfileActivity::Failed:
    ++profile_->failed_steps;
    profile_->failed_cycles += cycles;
    return;
  }
}

void DeploymentRuntime::emitTaskTrace(TaskTraceEvent event,
                                      uint32_t task_id) noexcept {
  if (trace_ != nullptr && trace_->emit != nullptr) {
    trace_->emit(trace_->context, event, task_id, execution_id_);
  }
}

} // namespace golem::runtime
