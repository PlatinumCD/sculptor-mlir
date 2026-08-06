#include "golem/runtime/tile_abi.h"

namespace golem::runtime {

namespace {

template <typename Entry>
bool validTable(const Entry* entries, uint32_t count) {
    return count == 0 || entries != nullptr;
}

const Route* findRoute(
    const Route* routes,
    uint32_t route_count,
    uint32_t route_id
) {
    for (uint32_t index = 0; index < route_count; ++index) {
        if (routes[index].id == route_id) {
            return &routes[index];
        }
    }
    return nullptr;
}

bool validRange(uint32_t offset, uint32_t count, uint32_t table_count) {
    return offset <= table_count && count <= table_count - offset;
}

bool validResourceKind(ResourceKind kind) {
    return kind >= ResourceKind::ModelInput &&
           kind <= ResourceKind::RouteOutput;
}

bool validRouteTable(
    const Route* routes,
    uint32_t route_count,
    uint32_t core_id,
    bool incoming
) {
    if (!validTable(routes, route_count)) {
        return false;
    }

    for (uint32_t index = 0; index < route_count; ++index) {
        const Route& route = routes[index];
        if (route.byte_size == 0 || route.byte_size % sizeof(uint32_t) != 0) {
            return false;
        }
        if ((incoming && route.destination_core != core_id) ||
            (!incoming && route.source_core != core_id)) {
            return false;
        }
        for (uint32_t prior = 0; prior < index; ++prior) {
            if (routes[prior].id == route.id) {
                return false;
            }
        }
    }
    return true;
}

bool validModelIOTable(
    const ModelIO* entries,
    uint32_t count,
    uint32_t core_id
) {
    if (!validTable(entries, count)) {
        return false;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if (entries[index].owner_core != core_id ||
            entries[index].byte_size == 0) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool TileABI::valid() const noexcept {
    if (!validTable(boot_tasks, boot_task_count) ||
        !dispatch_tasks.valid() ||
        !validRouteTable(
            incoming_routes,
            incoming_route_count,
            core_id,
            true
        ) ||
        !validRouteTable(
            outgoing_routes,
            outgoing_route_count,
            core_id,
            false
        ) ||
        !validModelIOTable(model_inputs, model_input_count, core_id) ||
        !validModelIOTable(model_outputs, model_output_count, core_id)) {
        return false;
    }

    for (uint32_t index = 0; index < boot_task_count; ++index) {
        const Task& task = boot_tasks[index];
        if (task.execute == nullptr ||
            task.input_count != 0 ||
            task.output_count != 0) {
            return false;
        }
    }
    return !hasDeploymentPlan() || validDeploymentPlan();
}

bool TileABI::hasDeploymentPlan() const noexcept {
    return resources != nullptr ||
           resource_count != 0 ||
           resource_dimensions != nullptr ||
           resource_dimension_count != 0 ||
           workspace_size != 0 ||
           task_bindings != nullptr ||
           task_binding_count != 0 ||
           task_binding_data != nullptr ||
           task_binding_data_count != 0;
}

bool TileABI::validDeploymentPlan() const noexcept {
    if (!validTable(resources, resource_count) ||
        !validTable(resource_dimensions, resource_dimension_count) ||
        !validTable(task_bindings, task_binding_count) ||
        !validTable(task_binding_data, task_binding_data_count) ||
        task_binding_count != dispatch_tasks.task_count) {
        return false;
    }

    for (uint32_t slot = 0; slot < resource_count; ++slot) {
        const Resource& resource = resources[slot];
        if (resource.local_slot != slot ||
            !validResourceKind(resource.kind) ||
            resource.element_type == ElementType::Invalid ||
            elementSizeBytes(resource.element_type) == 0 ||
            !validRange(
                resource.dimension_offset,
                resource.rank,
                resource_dimension_count
            ) ||
            (resource.flags & ~(
                ResourceWorkspace |
                ResourceExternal |
                ResourceScratchpad |
                ResourceSpill)) != 0) {
            return false;
        }

        const bool uses_workspace =
            (resource.flags & ResourceWorkspace) != 0;
        const bool is_external =
            (resource.flags & ResourceExternal) != 0;
        const bool uses_scratchpad =
            (resource.flags & ResourceScratchpad) != 0;
        const uint32_t primary_storage_count =
            static_cast<uint32_t>(uses_workspace) +
            static_cast<uint32_t>(is_external) +
            static_cast<uint32_t>(uses_scratchpad);
        if (primary_storage_count != 1 ||
            ((resource.flags & ResourceSpill) != 0 &&
             !uses_workspace)) {
            return false;
        }
        if (uses_workspace &&
            (resource.workspace_offset > workspace_size ||
             resource.byte_size >
                 workspace_size - resource.workspace_offset)) {
            return false;
        }

        uint64_t element_count = 1;
        for (uint32_t dimension = 0; dimension < resource.rank; ++dimension) {
            const int64_t size =
                resource_dimensions[resource.dimension_offset + dimension];
            if (size < 0) {
                return false;
            }
            const uint64_t unsigned_size = static_cast<uint64_t>(size);
            if (unsigned_size != 0 &&
                element_count > UINT64_MAX / unsigned_size) {
                return false;
            }
            element_count *= unsigned_size;
        }
        const uint64_t element_size = elementSizeBytes(resource.element_type);
        if (element_count > UINT64_MAX / element_size ||
            resource.byte_size != element_count * element_size) {
            return false;
        }

        const bool is_route =
            resource.kind == ResourceKind::RouteInput ||
            resource.kind == ResourceKind::RouteOutput;
        if (!is_route && resource.route_id != 0) {
            return false;
        }
    }

    for (uint32_t index = 0; index < task_binding_count; ++index) {
        const TaskBinding& binding = task_bindings[index];
        if (binding.reserved != 0 ||
            binding.task_id != dispatch_tasks.tasks[index].id ||
            binding.input_count != dispatch_tasks.tasks[index].input_count ||
            binding.output_count != dispatch_tasks.tasks[index].output_count ||
            !validRange(
                binding.input_offset,
                binding.input_count,
                task_binding_data_count
            ) ||
            !validRange(
                binding.output_offset,
                binding.output_count,
                task_binding_data_count
            ) ||
            !validRange(
                binding.dependency_offset,
                binding.dependency_count,
                task_binding_data_count
            )) {
            return false;
        }

        for (uint32_t input = 0; input < binding.input_count; ++input) {
            if (task_binding_data[binding.input_offset + input] >=
                resource_count) {
                return false;
            }
        }
        for (uint32_t output = 0; output < binding.output_count; ++output) {
            if (task_binding_data[binding.output_offset + output] >=
                resource_count) {
                return false;
            }
        }
        for (uint32_t dependency = 0;
             dependency < binding.dependency_count;
             ++dependency) {
            if (dispatch_tasks.find(
                    task_binding_data[
                        binding.dependency_offset + dependency
                    ]
                ) == nullptr) {
                return false;
            }
        }
    }

    return true;
}

const Route* TileABI::findIncomingRoute(uint32_t route_id) const noexcept {
    return findRoute(incoming_routes, incoming_route_count, route_id);
}

const Route* TileABI::findOutgoingRoute(uint32_t route_id) const noexcept {
    return findRoute(outgoing_routes, outgoing_route_count, route_id);
}

const Resource* TileABI::findResource(uint32_t local_slot) const noexcept {
    if (local_slot >= resource_count ||
        resources == nullptr ||
        resources[local_slot].local_slot != local_slot) {
        return nullptr;
    }
    return &resources[local_slot];
}

const TaskBinding* TileABI::findTaskBinding(uint32_t task_id) const noexcept {
    for (uint32_t index = 0; index < task_binding_count; ++index) {
        if (task_bindings[index].task_id == task_id) {
            return &task_bindings[index];
        }
    }
    return nullptr;
}

}  // namespace golem::runtime
