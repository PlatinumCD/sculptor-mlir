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

struct TileABI {
    uint32_t core_id;
    const Task* boot_tasks;
    uint32_t boot_task_count;
    TaskRegistry dispatch_tasks;
    const Route* incoming_routes;
    uint32_t incoming_route_count;
    const Route* outgoing_routes;
    uint32_t outgoing_route_count;
    const ModelIO* model_inputs;
    uint32_t model_input_count;
    const ModelIO* model_outputs;
    uint32_t model_output_count;
    const Resource* resources;
    uint32_t resource_count;
    const int64_t* resource_dimensions;
    uint32_t resource_dimension_count;
    uint64_t workspace_size;
    const TaskBinding* task_bindings;
    uint32_t task_binding_count;
    const uint32_t* task_binding_data;
    uint32_t task_binding_data_count;

    bool valid() const noexcept;
    bool hasDeploymentPlan() const noexcept;
    bool validDeploymentPlan() const noexcept;
    const Route* findIncomingRoute(uint32_t route_id) const noexcept;
    const Route* findOutgoingRoute(uint32_t route_id) const noexcept;
    const Resource* findResource(uint32_t local_slot) const noexcept;
    const TaskBinding* findTaskBinding(uint32_t task_id) const noexcept;
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

extern "C" uint32_t golem_tile_core_id();

extern "C" const Task* golem_tile_boot_tasks();
extern "C" uint32_t golem_tile_boot_task_count();
extern "C" const Task* golem_tile_dispatch_tasks();
extern "C" uint32_t golem_tile_dispatch_task_count();

extern "C" const Route* golem_tile_incoming_routes();
extern "C" uint32_t golem_tile_incoming_route_count();
extern "C" const Route* golem_tile_outgoing_routes();
extern "C" uint32_t golem_tile_outgoing_route_count();

extern "C" const ModelIO* golem_tile_model_inputs();
extern "C" uint32_t golem_tile_model_input_count();
extern "C" const ModelIO* golem_tile_model_outputs();
extern "C" uint32_t golem_tile_model_output_count();

extern "C" const Resource* golem_tile_resources();
extern "C" uint32_t golem_tile_resource_count();
extern "C" const int64_t* golem_tile_resource_dimensions();
extern "C" uint32_t golem_tile_resource_dimension_count();
extern "C" uint64_t golem_tile_workspace_size();

extern "C" const TaskBinding* golem_tile_task_bindings();
extern "C" uint32_t golem_tile_task_binding_count();
extern "C" const uint32_t* golem_tile_task_binding_data();
extern "C" uint32_t golem_tile_task_binding_data_count();

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
    };
}

}  // namespace golem::runtime
