#include "golem/runtime/tile_abi.h"

namespace golem::runtime {

namespace {

template <typename Entry>
bool validTable(const Entry *entries, uint32_t count) {
  return count == 0 || entries != nullptr;
}

const Route *findRoute(const Route *routes, uint32_t route_count,
                       uint32_t route_id) {
  for (uint32_t index = 0; index < route_count; ++index) {
    if (routes[index].id == route_id) {
      return &routes[index];
    }
  }
  return nullptr;
}

const TileABIIdIndex *findIdIndex(const TileABIIdIndex *entries,
                                  uint32_t count, uint32_t id) {
  uint32_t first = 0;
  uint32_t last = count;
  while (first < last) {
    const uint32_t middle = first + (last - first) / 2U;
    if (entries[middle].id < id)
      first = middle + 1U;
    else
      last = middle;
  }
  return first < count && entries[first].id == id ? &entries[first] : nullptr;
}

bool validRange(uint32_t offset, uint32_t count, uint32_t table_count) {
  return offset <= table_count && count <= table_count - offset;
}

bool validResourceKind(ResourceKind kind) {
  return kind >= ResourceKind::ModelInput && kind <= ResourceKind::RouteOutput;
}

bool validMemoryOwnerKind(MemoryOwnerKind kind) {
  return kind >= MemoryOwnerKind::ModelInput &&
         kind <= MemoryOwnerKind::LocalTemporary;
}

bool validContiguity(MemoryContiguity contiguity) {
  return contiguity >= MemoryContiguity::Contiguous &&
         contiguity <= MemoryContiguity::NonContiguous;
}

bool validMovementMode(MemoryMovementMode mode) {
  return mode >= MemoryMovementMode::LocalAlias &&
         mode <= MemoryMovementMode::Assembly;
}

bool validRouteTable(const Route *routes, uint32_t route_count,
                     uint32_t core_id, bool incoming) {
  if (!validTable(routes, route_count)) {
    return false;
  }

  for (uint32_t index = 0; index < route_count; ++index) {
    const Route &route = routes[index];
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

bool validModelIOTable(const ModelIO *entries, uint32_t count,
                       uint32_t core_id) {
  if (!validTable(entries, count)) {
    return false;
  }
  for (uint32_t index = 0; index < count; ++index) {
    if (entries[index].owner_core != core_id || entries[index].byte_size == 0) {
      return false;
    }
  }
  return true;
}

} // namespace

bool TileABI::valid() const noexcept {
  if (!validTable(boot_tasks, boot_task_count) || !dispatch_tasks.valid() ||
      !validRouteTable(incoming_routes, incoming_route_count, core_id, true) ||
      !validRouteTable(outgoing_routes, outgoing_route_count, core_id, false) ||
      !validModelIOTable(model_inputs, model_input_count, core_id) ||
      !validModelIOTable(model_outputs, model_output_count, core_id)) {
    return false;
  }

  for (uint32_t index = 0; index < boot_task_count; ++index) {
    const Task &task = boot_tasks[index];
    if (task.execute == nullptr || task.input_count != 0 ||
        task.output_count != 0) {
      return false;
    }
  }
  return !hasDeploymentPlan() || validDeploymentPlan();
}

bool TileABI::hasDeploymentPlan() const noexcept {
  return resources != nullptr || resource_count != 0 ||
         resource_dimensions != nullptr || resource_dimension_count != 0 ||
         workspace_size != 0 || task_bindings != nullptr ||
         task_binding_count != 0 || task_binding_data != nullptr ||
         task_binding_data_count != 0 || memory_owners != nullptr ||
         memory_owner_count != 0 || memory_views != nullptr ||
         memory_view_count != 0 || route_views != nullptr ||
         route_view_count != 0 || assemblies != nullptr ||
         assembly_count != 0 || segmented_movements != nullptr ||
         segmented_movement_count != 0 || memory_segments != nullptr ||
         memory_segment_count != 0 || task_id_index != nullptr ||
         task_id_index_count != 0 || static_tasks != nullptr ||
         static_task_count != 0;
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
    const Resource &resource = resources[slot];
    if (resource.local_slot != slot || !validResourceKind(resource.kind) ||
        resource.element_type == ElementType::Invalid ||
        elementSizeBytes(resource.element_type) == 0 ||
        !validRange(resource.dimension_offset, resource.rank,
                    resource_dimension_count) ||
        (resource.flags & ~(ResourceWorkspace | ResourceExternal |
                            ResourceScratchpad | ResourceSpill)) != 0) {
      return false;
    }

    const bool uses_workspace = (resource.flags & ResourceWorkspace) != 0;
    const bool is_external = (resource.flags & ResourceExternal) != 0;
    const bool uses_scratchpad = (resource.flags & ResourceScratchpad) != 0;
    const uint32_t primary_storage_count =
        static_cast<uint32_t>(uses_workspace) +
        static_cast<uint32_t>(is_external) +
        static_cast<uint32_t>(uses_scratchpad);
    if (primary_storage_count != 1 ||
        ((resource.flags & ResourceSpill) != 0 && !uses_workspace)) {
      return false;
    }
    if (uses_workspace &&
        (resource.workspace_offset > workspace_size ||
         resource.byte_size > workspace_size - resource.workspace_offset)) {
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
      if (unsigned_size != 0 && element_count > UINT64_MAX / unsigned_size) {
        return false;
      }
      element_count *= unsigned_size;
    }
    const uint64_t element_size = elementSizeBytes(resource.element_type);
    if (element_count > UINT64_MAX / element_size ||
        resource.byte_size != element_count * element_size) {
      return false;
    }

    const bool is_route = resource.kind == ResourceKind::RouteInput ||
                          resource.kind == ResourceKind::RouteOutput;
    if (!is_route && resource.route_id != 0) {
      return false;
    }
  }

  for (uint32_t index = 0; index < task_binding_count; ++index) {
    const TaskBinding &binding = task_bindings[index];
    if (binding.reserved != 0 ||
        binding.task_id != dispatch_tasks.tasks[index].id ||
        binding.input_count != dispatch_tasks.tasks[index].input_count ||
        binding.output_count != dispatch_tasks.tasks[index].output_count ||
        !validRange(binding.input_offset, binding.input_count,
                    task_binding_data_count) ||
        !validRange(binding.output_offset, binding.output_count,
                    task_binding_data_count) ||
        !validRange(binding.dependency_offset, binding.dependency_count,
                    task_binding_data_count)) {
      return false;
    }

    for (uint32_t input = 0; input < binding.input_count; ++input) {
      if (task_binding_data[binding.input_offset + input] >= resource_count) {
        return false;
      }
    }
    for (uint32_t output = 0; output < binding.output_count; ++output) {
      if (task_binding_data[binding.output_offset + output] >= resource_count) {
        return false;
      }
    }
    for (uint32_t dependency = 0; dependency < binding.dependency_count;
         ++dependency) {
      if (dispatch_tasks.find(
              task_binding_data[binding.dependency_offset + dependency]) ==
          nullptr) {
        return false;
      }
    }
  }

  const bool has_static_runtime = (abi_features & TileABIStaticRuntime) != 0;
  if (!has_static_runtime) {
    if (task_id_index_count != 0 || incoming_route_id_index_count != 0 ||
        outgoing_route_id_index_count != 0 || static_task_count != 0 ||
        static_resource_count != 0 || static_runtime_data_count != 0)
      return false;
  } else {
    if (!validTable(task_id_index, task_id_index_count) ||
        !validTable(incoming_route_id_index,
                    incoming_route_id_index_count) ||
        !validTable(outgoing_route_id_index,
                    outgoing_route_id_index_count) ||
        !validTable(static_tasks, static_task_count) ||
        !validTable(static_resources, static_resource_count) ||
        !validTable(static_runtime_data, static_runtime_data_count) ||
        task_id_index_count != task_binding_count ||
        incoming_route_id_index_count != incoming_route_count ||
        outgoing_route_id_index_count != outgoing_route_count ||
        static_task_count != task_binding_count ||
        static_resource_count != resource_count)
      return false;

    for (uint32_t index = 0; index < task_id_index_count; ++index) {
      const TileABIIdIndex &entry = task_id_index[index];
      if (entry.index >= task_binding_count ||
          task_bindings[entry.index].task_id != entry.id ||
          (index != 0 && task_id_index[index - 1].id >= entry.id))
        return false;
    }
    auto validRouteIndex = [](const TileABIIdIndex *entries, uint32_t count,
                              const Route *routes,
                              uint32_t route_count) -> bool {
      for (uint32_t index = 0; index < count; ++index) {
        if (entries[index].index >= route_count ||
            routes[entries[index].index].id != entries[index].id ||
            (index != 0 && entries[index - 1].id >= entries[index].id))
          return false;
      }
      return true;
    };
    if (!validRouteIndex(incoming_route_id_index,
                         incoming_route_id_index_count, incoming_routes,
                         incoming_route_count) ||
        !validRouteIndex(outgoing_route_id_index,
                         outgoing_route_id_index_count, outgoing_routes,
                         outgoing_route_count))
      return false;

    for (uint32_t task_index = 0; task_index < static_task_count;
         ++task_index) {
      const StaticTask &task = static_tasks[task_index];
      const TaskBinding &binding = task_bindings[task_index];
      const uint64_t expected = static_cast<uint64_t>(binding.input_count) +
                                binding.output_count +
                                binding.dependency_count;
      if (expected != task.initial_readiness ||
          !validRange(task.outgoing_route_offset, task.outgoing_route_count,
                      outgoing_route_count) ||
          !validRange(task.dependent_offset, task.dependent_count,
                      static_runtime_data_count))
        return false;
      for (uint32_t route = 0; route < task.outgoing_route_count; ++route) {
        if (outgoing_routes[task.outgoing_route_offset + route].source_task !=
            binding.task_id)
          return false;
      }
      for (uint32_t dependent = 0; dependent < task.dependent_count;
           ++dependent) {
        if (static_runtime_data[task.dependent_offset + dependent] >=
            task_binding_count)
          return false;
      }
    }
    for (uint32_t slot = 0; slot < static_resource_count; ++slot) {
      const StaticResource &resource = static_resources[slot];
      if (!validRange(resource.ready_consumer_offset,
                      resource.ready_consumer_count,
                      static_runtime_data_count) ||
          !validRange(resource.bound_consumer_offset,
                      resource.bound_consumer_count,
                      static_runtime_data_count))
        return false;
      for (uint32_t consumer = 0; consumer < resource.ready_consumer_count;
           ++consumer) {
        if (static_runtime_data[resource.ready_consumer_offset + consumer] >=
            task_binding_count)
          return false;
      }
      for (uint32_t consumer = 0; consumer < resource.bound_consumer_count;
           ++consumer) {
        if (static_runtime_data[resource.bound_consumer_offset + consumer] >=
            task_binding_count)
          return false;
      }

      uint64_t expected_ready = 0;
      uint64_t expected_bound = 0;
      for (uint32_t task_index = 0; task_index < task_binding_count;
           ++task_index) {
        const TaskBinding &binding = task_bindings[task_index];
        for (uint32_t input = 0; input < binding.input_count; ++input)
          expected_ready +=
              task_binding_data[binding.input_offset + input] == slot;
        for (uint32_t output = 0; output < binding.output_count; ++output)
          expected_bound +=
              task_binding_data[binding.output_offset + output] == slot;
      }
      if (expected_ready != resource.ready_consumer_count ||
          expected_bound != resource.bound_consumer_count)
        return false;
      for (uint32_t consumer = 0; consumer < resource.ready_consumer_count;
           ++consumer) {
        const uint32_t task_index =
            static_runtime_data[resource.ready_consumer_offset + consumer];
        const TaskBinding &binding = task_bindings[task_index];
        bool found = false;
        for (uint32_t input = 0; input < binding.input_count; ++input)
          found |= task_binding_data[binding.input_offset + input] == slot;
        if (!found)
          return false;
      }
      for (uint32_t consumer = 0; consumer < resource.bound_consumer_count;
           ++consumer) {
        const uint32_t task_index =
            static_runtime_data[resource.bound_consumer_offset + consumer];
        const TaskBinding &binding = task_bindings[task_index];
        bool found = false;
        for (uint32_t output = 0; output < binding.output_count; ++output)
          found |= task_binding_data[binding.output_offset + output] == slot;
        if (!found)
          return false;
      }
    }
    for (uint32_t source_index = 0; source_index < static_task_count;
         ++source_index) {
      const StaticTask &source = static_tasks[source_index];
      uint64_t expected_dependents = 0;
      for (uint32_t task_index = 0; task_index < task_binding_count;
           ++task_index) {
        const TaskBinding &binding = task_bindings[task_index];
        for (uint32_t dependency = 0; dependency < binding.dependency_count;
             ++dependency)
          expected_dependents +=
              task_binding_data[binding.dependency_offset + dependency] ==
              task_bindings[source_index].task_id;
      }
      if (expected_dependents != source.dependent_count)
        return false;
      for (uint32_t dependent = 0; dependent < source.dependent_count;
           ++dependent) {
        const uint32_t task_index =
            static_runtime_data[source.dependent_offset + dependent];
        const TaskBinding &binding = task_bindings[task_index];
        bool found = false;
        for (uint32_t dependency = 0; dependency < binding.dependency_count;
             ++dependency)
          found |=
              task_binding_data[binding.dependency_offset + dependency] ==
              task_bindings[source_index].task_id;
        if (!found)
          return false;
      }
    }
  }

  const bool has_memory_views = (abi_features & TileABIMemoryViews) != 0;
  const bool has_assembly_joins = (abi_features & TileABIAssemblyJoins) != 0;
  const bool has_segmented_movement =
      (abi_features & TileABISegmentedMovement) != 0;
  if (!has_memory_views) {
    return memory_owner_count == 0 && memory_view_count == 0 &&
           memory_view_geometry_count == 0 && route_view_count == 0 &&
           assembly_count == 0 && assembly_contribution_count == 0 &&
           segmented_movement_count == 0 && memory_segment_count == 0;
  }
  if (!validTable(memory_owners, memory_owner_count) ||
      !validTable(memory_views, memory_view_count) ||
      !validTable(memory_view_geometry, memory_view_geometry_count) ||
      !validTable(route_views, route_view_count) ||
      !validTable(assemblies, assembly_count) ||
      !validTable(assembly_contributions, assembly_contribution_count) ||
      !validTable(segmented_movements, segmented_movement_count) ||
      !validTable(memory_segments, memory_segment_count) ||
      route_view_count != incoming_route_count + outgoing_route_count ||
      (has_assembly_joins != (assembly_count != 0)) ||
      (has_segmented_movement != (segmented_movement_count != 0))) {
    return false;
  }

  for (uint32_t index = 0; index < memory_owner_count; ++index) {
    const MemoryOwner &owner = memory_owners[index];
    if (!validMemoryOwnerKind(owner.kind)) {
      return false;
    }
    const bool metadata_only = owner.kind == MemoryOwnerKind::Persistent ||
                               owner.kind == MemoryOwnerKind::LocalTemporary;
    if ((!metadata_only && owner.local_slot >= resource_count) ||
        (metadata_only && owner.local_slot != InvalidTileABIId &&
         owner.local_slot >= resource_count)) {
      return false;
    }
    for (uint32_t prior = 0; prior < index; ++prior) {
      if (memory_owners[prior].id == owner.id) {
        return false;
      }
    }
  }
  for (uint32_t index = 0; index < memory_view_count; ++index) {
    const MemoryView &view = memory_views[index];
    const MemoryOwner *owner = findMemoryOwner(view.owner_id);
    const uint64_t geometry_count = static_cast<uint64_t>(view.rank) * 3U;
    if (owner == nullptr || owner->local_slot != view.owner_slot ||
        !validContiguity(view.contiguity) || view.reserved0 != 0 ||
        view.reserved1 != 0 ||
        view.geometry_offset > memory_view_geometry_count ||
        geometry_count > memory_view_geometry_count - view.geometry_offset ||
        view.byte_offset > owner->byte_size ||
        view.byte_size > owner->byte_size - view.byte_offset) {
      return false;
    }
    for (uint32_t prior = 0; prior < index; ++prior) {
      if (memory_views[prior].id == view.id) {
        return false;
      }
    }
    if (view.rank == 0) {
      continue;
    }
    const int64_t *offsets = memory_view_geometry + view.geometry_offset;
    const int64_t *sizes = offsets + view.rank;
    const int64_t *strides = sizes + view.rank;
    for (uint32_t dimension = 0; dimension < view.rank; ++dimension) {
      if (offsets[dimension] < 0 || sizes[dimension] < 0 ||
          strides[dimension] <= 0) {
        return false;
      }
    }
  }

  for (uint32_t index = 0; index < route_view_count; ++index) {
    const RouteView &route_view = route_views[index];
    const bool incoming = (route_view.flags & RouteViewIncoming) != 0;
    const bool outgoing = (route_view.flags & RouteViewOutgoing) != 0;
    const Route *route = incoming ? findIncomingRoute(route_view.route_id)
                                  : findOutgoingRoute(route_view.route_id);
    const MemoryView *view = findMemoryView(route_view.view_id);
    if (incoming == outgoing || route == nullptr || view == nullptr ||
        route_view.local_resource_slot != route->local_slot ||
        route_view.owner_slot != view->owner_slot ||
        route_view.byte_offset != view->byte_offset ||
        route_view.byte_size != route->byte_size ||
        route_view.byte_size != view->byte_size ||
        !validMovementMode(route_view.movement_mode) ||
        (route_view.movement_mode != MemoryMovementMode::Contiguous &&
         route_view.movement_mode != MemoryMovementMode::Packed &&
         route_view.movement_mode != MemoryMovementMode::Segmented) ||
        ((route_view.movement_mode == MemoryMovementMode::Contiguous ||
          route_view.movement_mode == MemoryMovementMode::Packed) &&
         view->contiguity != MemoryContiguity::Contiguous) ||
        route_view.local_resource_slot >= resource_count ||
        route_view.owner_slot >= resource_count) {
      return false;
    }
    if ((route_view.flags & RouteViewAssemblyDestination) != 0) {
      if (!incoming || route_view.assembly_id == InvalidTileABIId ||
          findAssembly(route_view.assembly_id) == nullptr) {
        return false;
      }
    } else if (route_view.assembly_id != InvalidTileABIId) {
      return false;
    }
  }

  for (uint32_t index = 0; index < assembly_count; ++index) {
    const Assembly &assembly = assemblies[index];
    if (assembly.owner_slot >= resource_count || assembly.reserved != 0 ||
        !validRange(assembly.contribution_offset, assembly.contribution_count,
                    assembly_contribution_count)) {
      return false;
    }
    for (uint32_t contribution = 0; contribution < assembly.contribution_count;
         ++contribution) {
      const AssemblyContribution &entry =
          assembly_contributions[assembly.contribution_offset + contribution];
      const MemoryView *destination = findMemoryView(entry.destination_view_id);
      if (entry.assembly_id != assembly.id || destination == nullptr ||
          destination->owner_slot != assembly.owner_slot) {
        return false;
      }
      if (entry.route_id != InvalidTileABIId &&
          findIncomingRoute(entry.route_id) == nullptr) {
        return false;
      }
    }
  }

  uint32_t expected_segment_offset = 0;
  for (uint32_t index = 0; index < segmented_movement_count; ++index) {
    const SegmentedMovement &movement = segmented_movements[index];
    const bool incoming = (movement.flags & RouteViewIncoming) != 0;
    const bool outgoing = (movement.flags & RouteViewOutgoing) != 0;
    const RouteView *route_view = findRouteView(movement.route_id);
    const MemoryView *view =
        route_view == nullptr ? nullptr : findMemoryView(route_view->view_id);
    if (incoming == outgoing || movement.reserved != 0 ||
        movement.segment_count < 2 ||
        movement.segment_offset != expected_segment_offset ||
        !validRange(movement.segment_offset, movement.segment_count,
                    memory_segment_count) ||
        route_view == nullptr || view == nullptr ||
        route_view->movement_mode != MemoryMovementMode::Segmented ||
        route_view->completion_event_id != movement.completion_event_id ||
        route_view->assembly_id != movement.assembly_id ||
        route_view->flags != movement.flags ||
        route_view->byte_size != movement.byte_size ||
        (incoming && findIncomingRoute(movement.route_id) == nullptr) ||
        (outgoing && findOutgoingRoute(movement.route_id) == nullptr)) {
      return false;
    }
    uint64_t total_bytes = 0;
    for (uint32_t ordinal = 0; ordinal < movement.segment_count; ++ordinal) {
      const MemorySegment &segment =
          memory_segments[movement.segment_offset + ordinal];
      const uint64_t local_offset = incoming ? segment.destination_byte_offset
                                             : segment.source_byte_offset;
      if (segment.byte_size == 0 || segment.byte_size % sizeof(uint32_t) != 0 ||
          local_offset % sizeof(uint32_t) != 0 ||
          local_offset > UINT64_MAX - segment.byte_size ||
          local_offset + segment.byte_size >
              findMemoryOwner(view->owner_id)->byte_size ||
          total_bytes > UINT64_MAX - segment.byte_size) {
        return false;
      }
      total_bytes += segment.byte_size;
    }
    if (total_bytes != movement.byte_size)
      return false;
    expected_segment_offset += movement.segment_count;
    for (uint32_t prior = 0; prior < index; ++prior) {
      if (segmented_movements[prior].movement_id == movement.movement_id ||
          (segmented_movements[prior].route_id == movement.route_id &&
           segmented_movements[prior].flags == movement.flags))
        return false;
    }
  }
  if (expected_segment_offset != memory_segment_count)
    return false;

  return true;
}

const Route *TileABI::findIncomingRoute(uint32_t route_id) const noexcept {
  if ((abi_features & TileABIStaticRuntime) != 0) {
    const TileABIIdIndex *entry = findIdIndex(
        incoming_route_id_index, incoming_route_id_index_count, route_id);
    return entry == nullptr ? nullptr : &incoming_routes[entry->index];
  }
  return findRoute(incoming_routes, incoming_route_count, route_id);
}

const Route *TileABI::findOutgoingRoute(uint32_t route_id) const noexcept {
  if ((abi_features & TileABIStaticRuntime) != 0) {
    const TileABIIdIndex *entry = findIdIndex(
        outgoing_route_id_index, outgoing_route_id_index_count, route_id);
    return entry == nullptr ? nullptr : &outgoing_routes[entry->index];
  }
  return findRoute(outgoing_routes, outgoing_route_count, route_id);
}

const Resource *TileABI::findResource(uint32_t local_slot) const noexcept {
  if (local_slot >= resource_count || resources == nullptr ||
      resources[local_slot].local_slot != local_slot) {
    return nullptr;
  }
  return &resources[local_slot];
}

const TaskBinding *TileABI::findTaskBinding(uint32_t task_id) const noexcept {
  if ((abi_features & TileABIStaticRuntime) != 0) {
    const TileABIIdIndex *entry =
        findIdIndex(task_id_index, task_id_index_count, task_id);
    return entry == nullptr ? nullptr : &task_bindings[entry->index];
  }
  for (uint32_t index = 0; index < task_binding_count; ++index) {
    if (task_bindings[index].task_id == task_id) {
      return &task_bindings[index];
    }
  }
  return nullptr;
}

const MemoryOwner *TileABI::findMemoryOwner(uint32_t owner_id) const noexcept {
  if ((abi_features & TileABIStaticRuntime) != 0) {
    uint32_t first = 0;
    uint32_t last = memory_owner_count;
    while (first < last) {
      const uint32_t middle = first + (last - first) / 2U;
      if (memory_owners[middle].id < owner_id)
        first = middle + 1U;
      else
        last = middle;
    }
    return first < memory_owner_count && memory_owners[first].id == owner_id
               ? &memory_owners[first]
               : nullptr;
  }
  for (uint32_t index = 0; index < memory_owner_count; ++index) {
    if (memory_owners[index].id == owner_id) {
      return &memory_owners[index];
    }
  }
  return nullptr;
}

const MemoryView *TileABI::findMemoryView(uint32_t view_id) const noexcept {
  if ((abi_features & TileABIStaticRuntime) != 0) {
    uint32_t first = 0;
    uint32_t last = memory_view_count;
    while (first < last) {
      const uint32_t middle = first + (last - first) / 2U;
      if (memory_views[middle].id < view_id)
        first = middle + 1U;
      else
        last = middle;
    }
    return first < memory_view_count && memory_views[first].id == view_id
               ? &memory_views[first]
               : nullptr;
  }
  for (uint32_t index = 0; index < memory_view_count; ++index) {
    if (memory_views[index].id == view_id) {
      return &memory_views[index];
    }
  }
  return nullptr;
}

const RouteView *TileABI::findRouteView(uint32_t route_id) const noexcept {
  if ((abi_features & TileABIStaticRuntime) != 0) {
    uint32_t first = 0;
    uint32_t last = route_view_count;
    while (first < last) {
      const uint32_t middle = first + (last - first) / 2U;
      if (route_views[middle].route_id < route_id)
        first = middle + 1U;
      else
        last = middle;
    }
    return first < route_view_count && route_views[first].route_id == route_id
               ? &route_views[first]
               : nullptr;
  }
  for (uint32_t index = 0; index < route_view_count; ++index) {
    if (route_views[index].route_id == route_id) {
      return &route_views[index];
    }
  }
  return nullptr;
}

const Assembly *TileABI::findAssembly(uint32_t assembly_id) const noexcept {
  if ((abi_features & TileABIStaticRuntime) != 0) {
    uint32_t first = 0;
    uint32_t last = assembly_count;
    while (first < last) {
      const uint32_t middle = first + (last - first) / 2U;
      if (assemblies[middle].id < assembly_id)
        first = middle + 1U;
      else
        last = middle;
    }
    return first < assembly_count && assemblies[first].id == assembly_id
               ? &assemblies[first]
               : nullptr;
  }
  for (uint32_t index = 0; index < assembly_count; ++index) {
    if (assemblies[index].id == assembly_id) {
      return &assemblies[index];
    }
  }
  return nullptr;
}

const SegmentedMovement *
TileABI::findSegmentedMovement(uint32_t route_id,
                               uint32_t direction_flag) const noexcept {
  if ((abi_features & TileABIStaticRuntime) != 0) {
    uint32_t first = 0;
    uint32_t last = segmented_movement_count;
    while (first < last) {
      const uint32_t middle = first + (last - first) / 2U;
      if (segmented_movements[middle].route_id < route_id)
        first = middle + 1U;
      else
        last = middle;
    }
    for (uint32_t index = first;
         index < segmented_movement_count &&
         segmented_movements[index].route_id == route_id;
         ++index) {
      if ((segmented_movements[index].flags & direction_flag) != 0)
        return &segmented_movements[index];
    }
    return nullptr;
  }
  for (uint32_t index = 0; index < segmented_movement_count; ++index) {
    if (segmented_movements[index].route_id == route_id &&
        (segmented_movements[index].flags & direction_flag) != 0)
      return &segmented_movements[index];
  }
  return nullptr;
}

} // namespace golem::runtime
