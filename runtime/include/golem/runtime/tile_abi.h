#pragma once

#include <stddef.h>
#include <stdint.h>

#include "golem/runtime/task.h"
#include "golem/runtime/task_registry.h"

namespace golem::runtime {

// These records are the C++ view of the tables emitted by
// --sculptor-emit-golem-tile-abi.
struct Route {
  uint32_t id;
  uint32_t source_core;
  uint32_t source_task;
  uint32_t source_output;
  uint32_t destination_core;
  uint32_t destination_task;
  uint32_t destination_input;
  uint32_t global_resource_id;
  uint32_t local_slot;
  uint64_t byte_size;
};

struct ModelIO {
  uint32_t model_index;
  uint32_t owner_core;
  uint32_t global_resource_id;
  uint32_t local_slot;
  uint64_t byte_size;
};

enum class ResourceKind : uint32_t {
  ModelInput = 0,
  ModelOutput = 1,
  Intermediate = 2,
  Persistent = 3,
  RouteInput = 4,
  RouteOutput = 5,
};

enum ResourceFlags : uint32_t {
  ResourceWorkspace = UINT32_C(1) << 0,
  ResourceExternal = UINT32_C(1) << 1,
  ResourceScratchpad = UINT32_C(1) << 2,
  ResourceSpill = UINT32_C(1) << 3,
};

struct Resource {
  uint32_t global_resource_id;
  uint32_t route_id;
  uint32_t local_slot;
  ResourceKind kind;
  ElementType element_type;
  uint32_t rank;
  uint32_t dimension_offset;
  uint32_t flags;
  uint64_t byte_size;
  uint64_t workspace_offset;
};

struct TaskBinding {
  uint32_t task_id;
  uint32_t input_offset;
  uint32_t input_count;
  uint32_t output_offset;
  uint32_t output_count;
  uint32_t dependency_offset;
  uint32_t dependency_count;
  uint32_t reserved;
};

enum class MemoryOwnerKind : uint32_t {
  ModelInput = 0,
  ModelOutput = 1,
  Intermediate = 2,
  Persistent = 3,
  RouteInput = 4,
  RouteOutput = 5,
  Assembly = 6,
  LocalTemporary = 7,
};

enum class MemoryContiguity : uint32_t {
  Contiguous = 0,
  Strided = 1,
  NonContiguous = 2,
};

enum class MemoryMovementMode : uint32_t {
  LocalAlias = 0,
  Contiguous = 1,
  Packed = 2,
  Segmented = 3,
  Assembly = 4,
};

enum TileABIFeatures : uint32_t {
  TileABIScratchpadDMA = UINT32_C(1) << 0,
  TileABIMemoryViews = UINT32_C(1) << 1,
  TileABIAssemblyJoins = UINT32_C(1) << 2,
  TileABISegmentedMovement = UINT32_C(1) << 3,
  TileABIStaticRuntime = UINT32_C(1) << 4,
};

constexpr uint32_t InvalidTileABIId = UINT32_MAX;

struct MemoryOwner {
  uint32_t id;
  uint32_t global_resource_id;
  uint32_t local_slot;
  MemoryOwnerKind kind;
  uint64_t byte_size;
};

struct MemoryView {
  uint32_t id;
  uint32_t owner_id;
  uint32_t owner_slot;
  uint32_t rank;
  uint32_t geometry_offset;
  MemoryContiguity contiguity;
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t byte_offset;
  uint64_t byte_size;
};

enum RouteViewFlags : uint32_t {
  RouteViewIncoming = UINT32_C(1) << 0,
  RouteViewOutgoing = UINT32_C(1) << 1,
  RouteViewAssemblyDestination = UINT32_C(1) << 2,
};

struct RouteView {
  uint32_t route_id;
  uint32_t local_resource_slot;
  uint32_t owner_slot;
  uint32_t view_id;
  MemoryMovementMode movement_mode;
  uint32_t completion_event_id;
  uint32_t assembly_id;
  uint32_t flags;
  uint64_t byte_offset;
  uint64_t byte_size;
};

struct Assembly {
  uint32_t id;
  uint32_t owner_slot;
  uint32_t contribution_offset;
  uint32_t contribution_count;
  uint32_t readiness_event_id;
  uint32_t reserved;
};

struct AssemblyContribution {
  uint32_t assembly_id;
  uint32_t source_view_id;
  uint32_t destination_view_id;
  uint32_t completion_event_id;
  uint32_t route_id;
  uint32_t flags;
};

struct SegmentedMovement {
  uint32_t movement_id;
  uint32_t route_id;
  uint32_t segment_offset;
  uint32_t segment_count;
  uint32_t completion_event_id;
  uint32_t assembly_id;
  uint32_t flags;
  uint32_t reserved;
  uint64_t byte_size;
};

struct MemorySegment {
  uint64_t source_byte_offset;
  uint64_t destination_byte_offset;
  uint64_t byte_size;
};

// Compact compiler-generated indexes for the static deployment runtime.
// Global task and route IDs remain stable deployment identities. Index values
// address this tile's local ABI tables.
struct TileABIIdIndex {
  uint32_t id;
  uint32_t index;
};

struct StaticTask {
  uint32_t initial_readiness;
  uint32_t outgoing_route_offset;
  uint32_t outgoing_route_count;
  uint32_t dependent_offset;
  uint32_t dependent_count;
};

struct StaticResource {
  uint32_t ready_consumer_offset;
  uint32_t ready_consumer_count;
  uint32_t bound_consumer_offset;
  uint32_t bound_consumer_count;
};

struct TileABI {
  uint32_t core_id;
  const Task *boot_tasks;
  uint32_t boot_task_count;
  TaskRegistry dispatch_tasks;
  const Route *incoming_routes;
  uint32_t incoming_route_count;
  const Route *outgoing_routes;
  uint32_t outgoing_route_count;
  const ModelIO *model_inputs;
  uint32_t model_input_count;
  const ModelIO *model_outputs;
  uint32_t model_output_count;
  const Resource *resources;
  uint32_t resource_count;
  const int64_t *resource_dimensions;
  uint32_t resource_dimension_count;
  uint64_t workspace_size;
  const TaskBinding *task_bindings;
  uint32_t task_binding_count;
  const uint32_t *task_binding_data;
  uint32_t task_binding_data_count;
  const MemoryOwner *memory_owners;
  uint32_t memory_owner_count;
  const MemoryView *memory_views;
  uint32_t memory_view_count;
  const int64_t *memory_view_geometry;
  uint32_t memory_view_geometry_count;
  const RouteView *route_views;
  uint32_t route_view_count;
  const Assembly *assemblies;
  uint32_t assembly_count;
  const AssemblyContribution *assembly_contributions;
  uint32_t assembly_contribution_count;
  uint32_t abi_features;
  const SegmentedMovement *segmented_movements;
  uint32_t segmented_movement_count;
  const MemorySegment *memory_segments;
  uint32_t memory_segment_count;
  const TileABIIdIndex *task_id_index;
  uint32_t task_id_index_count;
  const TileABIIdIndex *incoming_route_id_index;
  uint32_t incoming_route_id_index_count;
  const TileABIIdIndex *outgoing_route_id_index;
  uint32_t outgoing_route_id_index_count;
  const StaticTask *static_tasks;
  uint32_t static_task_count;
  const StaticResource *static_resources;
  uint32_t static_resource_count;
  const uint32_t *static_runtime_data;
  uint32_t static_runtime_data_count;

  bool valid() const noexcept;
  bool hasDeploymentPlan() const noexcept;
  bool validDeploymentPlan() const noexcept;
  const Route *findIncomingRoute(uint32_t route_id) const noexcept;
  const Route *findOutgoingRoute(uint32_t route_id) const noexcept;
  const Resource *findResource(uint32_t local_slot) const noexcept;
  const TaskBinding *findTaskBinding(uint32_t task_id) const noexcept;
  const MemoryOwner *findMemoryOwner(uint32_t owner_id) const noexcept;
  const MemoryView *findMemoryView(uint32_t view_id) const noexcept;
  const RouteView *findRouteView(uint32_t route_id) const noexcept;
  const Assembly *findAssembly(uint32_t assembly_id) const noexcept;
  const SegmentedMovement *
  findSegmentedMovement(uint32_t route_id,
                        uint32_t direction_flag) const noexcept;
};

static_assert(offsetof(Task, execute) == 8);
static_assert(sizeof(Task) == 24);
static_assert(offsetof(Tensor, rank) == 8);
static_assert(offsetof(Tensor, descriptor) == 16);
static_assert(sizeof(Tensor) == 24);
static_assert(offsetof(Route, byte_size) == 40);
static_assert(sizeof(Route) == 48);
static_assert(offsetof(ModelIO, byte_size) == 16);
static_assert(sizeof(ModelIO) == 24);
static_assert(offsetof(Resource, byte_size) == 32);
static_assert(offsetof(Resource, workspace_offset) == 40);
static_assert(sizeof(Resource) == 48);
static_assert(offsetof(TaskBinding, reserved) == 28);
static_assert(sizeof(TaskBinding) == 32);
static_assert(sizeof(MemoryOwner) == 24);
static_assert(sizeof(MemoryView) == 48);
static_assert(sizeof(RouteView) == 48);
static_assert(sizeof(Assembly) == 24);
static_assert(sizeof(AssemblyContribution) == 24);
static_assert(sizeof(SegmentedMovement) == 40);
static_assert(sizeof(MemorySegment) == 24);
static_assert(sizeof(TileABIIdIndex) == 8);
static_assert(sizeof(StaticTask) == 20);
static_assert(sizeof(StaticResource) == 16);

extern "C" uint32_t golem_tile_core_id();

extern "C" const Task *golem_tile_boot_tasks();
extern "C" uint32_t golem_tile_boot_task_count();
extern "C" const Task *golem_tile_dispatch_tasks();
extern "C" uint32_t golem_tile_dispatch_task_count();

extern "C" const Route *golem_tile_incoming_routes();
extern "C" uint32_t golem_tile_incoming_route_count();
extern "C" const Route *golem_tile_outgoing_routes();
extern "C" uint32_t golem_tile_outgoing_route_count();

extern "C" const ModelIO *golem_tile_model_inputs();
extern "C" uint32_t golem_tile_model_input_count();
extern "C" const ModelIO *golem_tile_model_outputs();
extern "C" uint32_t golem_tile_model_output_count();

extern "C" const Resource *golem_tile_resources();
extern "C" uint32_t golem_tile_resource_count();
extern "C" const int64_t *golem_tile_resource_dimensions();
extern "C" uint32_t golem_tile_resource_dimension_count();
extern "C" uint64_t golem_tile_workspace_size();

extern "C" const TaskBinding *golem_tile_task_bindings();
extern "C" uint32_t golem_tile_task_binding_count();
extern "C" const uint32_t *golem_tile_task_binding_data();
extern "C" uint32_t golem_tile_task_binding_data_count();

extern "C" const MemoryOwner *golem_tile_memory_owners() __attribute__((weak));
extern "C" uint32_t golem_tile_memory_owner_count() __attribute__((weak));
extern "C" const MemoryView *golem_tile_memory_views() __attribute__((weak));
extern "C" uint32_t golem_tile_memory_view_count() __attribute__((weak));
extern "C" const int64_t *golem_tile_memory_view_geometry()
    __attribute__((weak));
extern "C" uint32_t golem_tile_memory_view_geometry_count()
    __attribute__((weak));
extern "C" const RouteView *golem_tile_route_views() __attribute__((weak));
extern "C" uint32_t golem_tile_route_view_count() __attribute__((weak));
extern "C" const Assembly *golem_tile_assemblies() __attribute__((weak));
extern "C" uint32_t golem_tile_assembly_count() __attribute__((weak));
extern "C" const AssemblyContribution *golem_tile_assembly_contributions()
    __attribute__((weak));
extern "C" uint32_t golem_tile_assembly_contribution_count()
    __attribute__((weak));
extern "C" const SegmentedMovement *golem_tile_segmented_movements()
    __attribute__((weak));
extern "C" uint32_t golem_tile_segmented_movement_count() __attribute__((weak));
extern "C" const MemorySegment *golem_tile_memory_segments()
    __attribute__((weak));
extern "C" uint32_t golem_tile_memory_segment_count() __attribute__((weak));
extern "C" uint32_t golem_tile_abi_features() __attribute__((weak));
extern "C" const TileABIIdIndex *golem_tile_task_id_index()
    __attribute__((weak));
extern "C" uint32_t golem_tile_task_id_index_count() __attribute__((weak));
extern "C" const TileABIIdIndex *golem_tile_incoming_route_id_index()
    __attribute__((weak));
extern "C" uint32_t golem_tile_incoming_route_id_index_count()
    __attribute__((weak));
extern "C" const TileABIIdIndex *golem_tile_outgoing_route_id_index()
    __attribute__((weak));
extern "C" uint32_t golem_tile_outgoing_route_id_index_count()
    __attribute__((weak));
extern "C" const StaticTask *golem_tile_static_tasks() __attribute__((weak));
extern "C" uint32_t golem_tile_static_task_count() __attribute__((weak));
extern "C" const StaticResource *golem_tile_static_resources()
    __attribute__((weak));
extern "C" uint32_t golem_tile_static_resource_count() __attribute__((weak));
extern "C" const uint32_t *golem_tile_static_runtime_data()
    __attribute__((weak));
extern "C" uint32_t golem_tile_static_runtime_data_count()
    __attribute__((weak));

// Call this only from an ELF that links one generated Sculptor tile module.
inline TileABI linkedTileABI() noexcept {
  return {
      golem_tile_core_id(),
      golem_tile_boot_tasks(),
      golem_tile_boot_task_count(),
      {
          golem_tile_dispatch_tasks(),
          golem_tile_dispatch_task_count(),
      },
      golem_tile_incoming_routes(),
      golem_tile_incoming_route_count(),
      golem_tile_outgoing_routes(),
      golem_tile_outgoing_route_count(),
      golem_tile_model_inputs(),
      golem_tile_model_input_count(),
      golem_tile_model_outputs(),
      golem_tile_model_output_count(),
      golem_tile_resources(),
      golem_tile_resource_count(),
      golem_tile_resource_dimensions(),
      golem_tile_resource_dimension_count(),
      golem_tile_workspace_size(),
      golem_tile_task_bindings(),
      golem_tile_task_binding_count(),
      golem_tile_task_binding_data(),
      golem_tile_task_binding_data_count(),
      golem_tile_memory_owners != nullptr ? golem_tile_memory_owners()
                                          : nullptr,
      golem_tile_memory_owner_count != nullptr ? golem_tile_memory_owner_count()
                                               : 0,
      golem_tile_memory_views != nullptr ? golem_tile_memory_views() : nullptr,
      golem_tile_memory_view_count != nullptr ? golem_tile_memory_view_count()
                                              : 0,
      golem_tile_memory_view_geometry != nullptr
          ? golem_tile_memory_view_geometry()
          : nullptr,
      golem_tile_memory_view_geometry_count != nullptr
          ? golem_tile_memory_view_geometry_count()
          : 0,
      golem_tile_route_views != nullptr ? golem_tile_route_views() : nullptr,
      golem_tile_route_view_count != nullptr ? golem_tile_route_view_count()
                                             : 0,
      golem_tile_assemblies != nullptr ? golem_tile_assemblies() : nullptr,
      golem_tile_assembly_count != nullptr ? golem_tile_assembly_count() : 0,
      golem_tile_assembly_contributions != nullptr
          ? golem_tile_assembly_contributions()
          : nullptr,
      golem_tile_assembly_contribution_count != nullptr
          ? golem_tile_assembly_contribution_count()
          : 0,
      golem_tile_abi_features != nullptr ? golem_tile_abi_features() : 0,
      golem_tile_segmented_movements != nullptr
          ? golem_tile_segmented_movements()
          : nullptr,
      golem_tile_segmented_movement_count != nullptr
          ? golem_tile_segmented_movement_count()
          : 0,
      golem_tile_memory_segments != nullptr ? golem_tile_memory_segments()
                                            : nullptr,
      golem_tile_memory_segment_count != nullptr
          ? golem_tile_memory_segment_count()
          : 0,
      golem_tile_task_id_index != nullptr ? golem_tile_task_id_index()
                                          : nullptr,
      golem_tile_task_id_index_count != nullptr
          ? golem_tile_task_id_index_count()
          : 0,
      golem_tile_incoming_route_id_index != nullptr
          ? golem_tile_incoming_route_id_index()
          : nullptr,
      golem_tile_incoming_route_id_index_count != nullptr
          ? golem_tile_incoming_route_id_index_count()
          : 0,
      golem_tile_outgoing_route_id_index != nullptr
          ? golem_tile_outgoing_route_id_index()
          : nullptr,
      golem_tile_outgoing_route_id_index_count != nullptr
          ? golem_tile_outgoing_route_id_index_count()
          : 0,
      golem_tile_static_tasks != nullptr ? golem_tile_static_tasks() : nullptr,
      golem_tile_static_task_count != nullptr ? golem_tile_static_task_count()
                                              : 0,
      golem_tile_static_resources != nullptr ? golem_tile_static_resources()
                                             : nullptr,
      golem_tile_static_resource_count != nullptr
          ? golem_tile_static_resource_count()
          : 0,
      golem_tile_static_runtime_data != nullptr
          ? golem_tile_static_runtime_data()
          : nullptr,
      golem_tile_static_runtime_data_count != nullptr
          ? golem_tile_static_runtime_data_count()
          : 0,
  };
}

} // namespace golem::runtime
