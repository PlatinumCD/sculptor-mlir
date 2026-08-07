#include "sculptor-mlir/Dialect/Sculptor/Transforms/MaterializeTileRuntimeGraph.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeResourceUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::sculptor;

namespace deployment_attrs = mlir::sculptor::deployment_attrs;
namespace tile_runtime_attrs = mlir::sculptor::tile_runtime_attrs;

constexpr StringLiteral kKindAttr = "sculptor.deployment.kind";
constexpr StringLiteral kPhysicalTileIdAttr =
    "sculptor.deployment.physical_tile_id";
constexpr StringLiteral kGlobalRoutineIdAttr =
    "sculptor.deployment.global_routine_id";
constexpr StringLiteral kLocalRoutineIndexAttr =
    "sculptor.deployment.local_routine_index";
constexpr StringLiteral kRoutineKindAttr = "sculptor.deployment.routine_kind";
constexpr StringLiteral kInputResourceIdsAttr =
    "sculptor.deployment.input_resource_ids";
constexpr StringLiteral kOutputResourceIdsAttr =
    "sculptor.deployment.output_resource_ids";
constexpr StringLiteral kLocalBindingsAttr =
    "sculptor.deployment.local_bindings";

struct RoutineInfo {
  func::FuncOp function;
  int64_t globalId = -1;
  bool boot = false;
  SmallVector<int64_t> inputResourceIds;
  SmallVector<int64_t> outputResourceIds;
  SmallVector<int64_t> dependencyIds;
  SmallVector<Attribute> arrayBindings;
  TaskCreateOp task;
};

struct ModelIOInfo {
  TileRoutineModelIOAttr attr;
  bool input = false;
};

FailureOr<int64_t> requireNonNegativeI64(Operation *op, StringRef name) {
  auto value = op->getAttrOfType<IntegerAttr>(name);
  if (!value || value.getInt() < 0) {
    op->emitError("expected non-negative integer attribute '") << name << "'";
    return failure();
  }
  return value.getInt();
}

FailureOr<SmallVector<int64_t>> requireI64Array(Operation *op, StringRef name) {
  auto values = op->getAttrOfType<ArrayAttr>(name);
  if (!values) {
    op->emitError("expected array attribute '") << name << "'";
    return failure();
  }
  SmallVector<int64_t> result;
  result.reserve(values.size());
  for (Attribute value : values) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer || integer.getInt() < 0) {
      op->emitError("expected '")
          << name << "' to contain non-negative integers";
      return failure();
    }
    result.push_back(integer.getInt());
  }
  return result;
}

FailureOr<ArrayAttr> requireArray(Operation *op, StringRef name) {
  auto value = op->getAttrOfType<ArrayAttr>(name);
  if (!value) {
    op->emitError("expected array attribute '") << name << "'";
    return failure();
  }
  return value;
}

FailureOr<Type> getRoutineInputType(RoutineInfo &routine, int64_t port,
                                    Operation *anchor) {
  FunctionType type = routine.function.getFunctionType();
  if (port < 0 || port >= static_cast<int64_t>(type.getNumInputs())) {
    anchor->emitError("route or binding references invalid input port ")
        << port << " on routine " << routine.globalId;
    return failure();
  }
  return type.getInput(port);
}

FailureOr<Type> getRoutineOutputType(RoutineInfo &routine, int64_t port,
                                     Operation *anchor) {
  FunctionType type = routine.function.getFunctionType();
  if (port < 0 || port >= static_cast<int64_t>(type.getNumResults())) {
    anchor->emitError("route or binding references invalid output port ")
        << port << " on routine " << routine.globalId;
    return failure();
  }
  return type.getResult(port);
}

LogicalResult recordResourceType(std::map<int64_t, Type> &types,
                                 int64_t resourceId, Type type,
                                 Operation *anchor) {
  if (resourceId < 0)
    return anchor->emitError("expected non-negative global resource ID");
  auto [found, inserted] = types.emplace(resourceId, type);
  if (!inserted && found->second != type) {
    return anchor->emitError("global resource ")
           << resourceId << " has inconsistent boundary types";
  }
  return success();
}

FailureOr<SmallVector<RoutineInfo, 0>> collectRoutines(ModuleOp module) {
  SmallVector<RoutineInfo, 0> routines;
  DenseSet<int64_t> globalIds;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) {
      function.emitError("outlined tile routines must have definitions");
      return failure();
    }
    FailureOr<int64_t> globalId =
        requireNonNegativeI64(function, kGlobalRoutineIdAttr);
    auto kind = function->getAttrOfType<StringAttr>(kRoutineKindAttr);
    FailureOr<SmallVector<int64_t>> inputs =
        requireI64Array(function, kInputResourceIdsAttr);
    FailureOr<SmallVector<int64_t>> outputs =
        requireI64Array(function, kOutputResourceIdsAttr);
    if (failed(globalId) || failed(inputs) || failed(outputs) || !kind) {
      if (!kind)
        function.emitError("expected string attribute '")
            << kRoutineKindAttr << "'";
      return failure();
    }
    if (!globalIds.insert(*globalId).second) {
      function.emitError("duplicate global routine ID ") << *globalId;
      return failure();
    }
    bool boot = kind.getValue() == "boot";
    if (!boot && kind.getValue() != "compute") {
      function.emitError("expected routine kind to be 'boot' or 'compute'");
      return failure();
    }
    if (inputs->size() != function.getNumArguments() ||
        outputs->size() != function.getNumResults()) {
      function.emitError(
          "routine resource ID arrays must match its function signature");
      return failure();
    }
    routines.push_back(RoutineInfo{function, *globalId, boot,
                                   std::move(*inputs), std::move(*outputs)});
  }
  if (routines.empty()) {
    module.emitError("expected at least one outlined tile routine");
    return failure();
  }
  llvm::sort(routines, [](const RoutineInfo &lhs, const RoutineInfo &rhs) {
    return lhs.globalId < rhs.globalId;
  });
  return routines;
}

FailureOr<SmallVector<TileRoutineRouteAttr>>
collectRoutes(ModuleOp module, StringRef name, int64_t tileId, bool incoming) {
  FailureOr<ArrayAttr> values = requireArray(module, name);
  if (failed(values))
    return failure();
  SmallVector<TileRoutineRouteAttr> routes;
  DenseSet<int64_t> routeIds;
  for (Attribute value : *values) {
    auto route = dyn_cast<TileRoutineRouteAttr>(value);
    if (!route) {
      module.emitError("expected typed tile-routine routes in '")
          << name << "'";
      return failure();
    }
    int64_t owner = incoming ? route.getDestinationTile().getInt()
                             : route.getSourceTile().getInt();
    if (owner != tileId) {
      module.emitError(
          "route manifest contains an entry owned by another tile");
      return failure();
    }
    if (!routeIds.insert(route.getId().getInt()).second) {
      module.emitError("duplicate route ID ") << route.getId().getInt();
      return failure();
    }
    routes.push_back(route);
  }
  llvm::sort(routes, [](TileRoutineRouteAttr lhs, TileRoutineRouteAttr rhs) {
    return lhs.getId().getInt() < rhs.getId().getInt();
  });
  return routes;
}

FailureOr<SmallVector<TileRoutineBindingAttr>>
collectBindings(ModuleOp module) {
  FailureOr<ArrayAttr> values = requireArray(module, kLocalBindingsAttr);
  if (failed(values))
    return failure();
  SmallVector<TileRoutineBindingAttr> bindings;
  for (Attribute value : *values) {
    auto binding = dyn_cast<TileRoutineBindingAttr>(value);
    if (!binding) {
      module.emitError("expected typed tile-routine local bindings");
      return failure();
    }
    bindings.push_back(binding);
  }
  llvm::sort(bindings, [](TileRoutineBindingAttr lhs,
                          TileRoutineBindingAttr rhs) {
    return std::tuple<int64_t, int64_t, int64_t, int64_t>{
               lhs.getSourceRoutine().getInt(), lhs.getSourceOutput().getInt(),
               lhs.getDestinationRoutine().getInt(),
               lhs.getDestinationInput().getInt()} <
           std::tuple<int64_t, int64_t, int64_t, int64_t>{
               rhs.getSourceRoutine().getInt(), rhs.getSourceOutput().getInt(),
               rhs.getDestinationRoutine().getInt(),
               rhs.getDestinationInput().getInt()};
  });
  return bindings;
}

FailureOr<SmallVector<ModelIOInfo>>
collectModelIO(ModuleOp module, StringRef name, bool input, int64_t tileId) {
  FailureOr<ArrayAttr> values = requireArray(module, name);
  if (failed(values))
    return failure();
  SmallVector<ModelIOInfo> entries;
  std::set<std::tuple<int64_t, int64_t, int64_t, int64_t>> seen;
  for (Attribute value : *values) {
    auto entry = dyn_cast<TileRoutineModelIOAttr>(value);
    if (!entry || entry.getTile().getInt() != tileId) {
      module.emitError("expected selected-tile model I/O metadata in '")
          << name << "'";
      return failure();
    }
    auto key = std::make_tuple(
        entry.getIndex().getInt(), entry.getResourceId().getInt(),
        entry.getRoutine().getInt(), entry.getPort().getInt());
    if (seen.insert(key).second)
      entries.push_back(ModelIOInfo{entry, input});
  }
  llvm::sort(entries, [](const ModelIOInfo &lhs, const ModelIOInfo &rhs) {
    return std::tuple<int64_t, int64_t, int64_t, int64_t>{
               lhs.attr.getIndex().getInt(),
               lhs.attr.getResourceId().getInt(),
               lhs.attr.getRoutine().getInt(), lhs.attr.getPort().getInt()} <
           std::tuple<int64_t, int64_t, int64_t, int64_t>{
               rhs.attr.getIndex().getInt(),
               rhs.attr.getResourceId().getInt(),
               rhs.attr.getRoutine().getInt(), rhs.attr.getPort().getInt()};
  });
  return entries;
}

FailureOr<SmallVector<unsigned>>
buildTopologicalOrder(ArrayRef<RoutineInfo> routines, Operation *anchor) {
  DenseMap<int64_t, unsigned> indexById;
  for (auto indexed : llvm::enumerate(routines))
    indexById[indexed.value().globalId] = indexed.index();

  SmallVector<unsigned> indegree(routines.size(), 0);
  SmallVector<SmallVector<unsigned>> consumers(routines.size());
  for (auto indexed : llvm::enumerate(routines)) {
    DenseSet<int64_t> seen;
    for (int64_t dependencyId : indexed.value().dependencyIds) {
      auto dependency = indexById.find(dependencyId);
      if (dependency == indexById.end()) {
        anchor->emitError("routine dependency references unknown routine ")
            << dependencyId;
        return failure();
      }
      if (!seen.insert(dependencyId).second)
        continue;
      ++indegree[indexed.index()];
      consumers[dependency->second].push_back(indexed.index());
    }
  }

  SmallVector<unsigned> order;
  SmallVector<bool> emitted(routines.size(), false);
  while (order.size() != routines.size()) {
    std::optional<unsigned> selected;
    for (unsigned index = 0; index < routines.size(); ++index) {
      if (emitted[index] || indegree[index] != 0)
        continue;
      if (!selected ||
          std::tuple<bool, int64_t>{!routines[index].boot,
                                    routines[index].globalId} <
              std::tuple<bool, int64_t>{!routines[*selected].boot,
                                        routines[*selected].globalId})
        selected = index;
    }
    if (!selected) {
      anchor->emitError("outlined routine dependencies contain a cycle");
      return failure();
    }
    emitted[*selected] = true;
    order.push_back(*selected);
    for (unsigned consumer : consumers[*selected])
      --indegree[consumer];
  }
  return order;
}

DictionaryAttr buildArrayBinding(OpBuilder &builder, unsigned inputIndex,
                                 int64_t physicalArrayId,
                                 int64_t localArrayId) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr(tile_runtime_attrs::kArrayBindingInputIndexFieldName,
                           builder.getI64IntegerAttr(inputIndex)),
      builder.getNamedAttr(tile_runtime_attrs::kArrayBindingPhysicalIdFieldName,
                           builder.getI64IntegerAttr(physicalArrayId)),
      builder.getNamedAttr(tile_runtime_attrs::kArrayBindingLocalIdFieldName,
                           builder.getI64IntegerAttr(localArrayId)),
  });
}

template <typename OpT>
Value createResource(OpBuilder &builder, Location loc, Value graph, Type type,
                     int64_t globalResourceId) {
  auto resource = builder.create<OpT>(
      loc, TaskResourceType::get(builder.getContext(), type), graph);
  resource->setAttr(deployment_attrs::kGlobalResourceIdAttrName,
                    builder.getI64IntegerAttr(globalResourceId));
  return resource.getResult();
}

LogicalResult materializeRuntimeGraph(ModuleOp module) {
  auto kind = module->getAttrOfType<StringAttr>(kKindAttr);
  if (!kind || kind.getValue() != "tile_routine_core") {
    return module.emitError(
        "expected a standalone core from sculptor-extract-tile-module");
  }
  if (module.lookupSymbol("generate_task_graph")) {
    return module.emitError("tile runtime graph has already been materialized");
  }

  FailureOr<int64_t> tileId =
      requireNonNegativeI64(module, kPhysicalTileIdAttr);
  FailureOr<SmallVector<RoutineInfo, 0>> routines = collectRoutines(module);
  if (failed(tileId) || failed(routines))
    return failure();

  DenseMap<int64_t, unsigned> routineIndexById;
  for (auto indexed : llvm::enumerate(*routines))
    routineIndexById[indexed.value().globalId] = indexed.index();

  auto findRoutine = [&](int64_t id) -> FailureOr<RoutineInfo *> {
    auto found = routineIndexById.find(id);
    if (found == routineIndexById.end()) {
      module.emitError("deployment metadata references unknown routine ") << id;
      return failure();
    }
    return &(*routines)[found->second];
  };

  FailureOr<SmallVector<TileRoutineRouteAttr>> incoming = collectRoutes(
      module, deployment_attrs::kIncomingRoutesAttrName, *tileId, true);
  FailureOr<SmallVector<TileRoutineRouteAttr>> outgoing = collectRoutes(
      module, deployment_attrs::kOutgoingRoutesAttrName, *tileId, false);
  FailureOr<SmallVector<TileRoutineBindingAttr>> bindings =
      collectBindings(module);
  FailureOr<SmallVector<ModelIOInfo>> modelInputs = collectModelIO(
      module, deployment_attrs::kModelInputsAttrName, true, *tileId);
  FailureOr<SmallVector<ModelIOInfo>> modelOutputs = collectModelIO(
      module, deployment_attrs::kModelOutputsAttrName, false, *tileId);
  if (failed(incoming) || failed(outgoing) || failed(bindings) ||
      failed(modelInputs) || failed(modelOutputs))
    return failure();

  std::map<int64_t, Type> nonRouteTypes;
  DenseSet<int64_t> modelInputResourceIds;
  DenseSet<int64_t> modelOutputResourceIds;
  for (const ModelIOInfo &entry : *modelInputs) {
    FailureOr<RoutineInfo *> routine =
        findRoutine(entry.attr.getRoutine().getInt());
    if (failed(routine))
      return failure();
    FailureOr<Type> type =
        getRoutineInputType(**routine, entry.attr.getPort().getInt(), module);
    if (failed(type) ||
        failed(recordResourceType(
            nonRouteTypes, entry.attr.getResourceId().getInt(), *type, module)))
      return failure();
    modelInputResourceIds.insert(entry.attr.getResourceId().getInt());
  }
  for (const ModelIOInfo &entry : *modelOutputs) {
    FailureOr<RoutineInfo *> routine =
        findRoutine(entry.attr.getRoutine().getInt());
    if (failed(routine))
      return failure();
    FailureOr<Type> type =
        getRoutineOutputType(**routine, entry.attr.getPort().getInt(), module);
    if (failed(type) ||
        failed(recordResourceType(
            nonRouteTypes, entry.attr.getResourceId().getInt(), *type, module)))
      return failure();
    modelOutputResourceIds.insert(entry.attr.getResourceId().getInt());
  }
  for (int64_t resourceId : modelInputResourceIds) {
    if (modelOutputResourceIds.contains(resourceId)) {
      return module.emitError(
          "one global resource cannot be both model input and model output");
    }
  }

  for (TileRoutineBindingAttr binding : *bindings) {
    FailureOr<RoutineInfo *> source =
        findRoutine(binding.getSourceRoutine().getInt());
    FailureOr<RoutineInfo *> destination =
        findRoutine(binding.getDestinationRoutine().getInt());
    if (failed(source) || failed(destination))
      return failure();
    FailureOr<Type> sourceType = getRoutineOutputType(
        **source, binding.getSourceOutput().getInt(), module);
    FailureOr<Type> destinationType = getRoutineInputType(
        **destination, binding.getDestinationInput().getInt(), module);
    if (failed(sourceType) || failed(destinationType) ||
        *sourceType != *destinationType) {
      module.emitError("local routine binding has incompatible endpoint types");
      return failure();
    }
    if (failed(recordResourceType(nonRouteTypes,
                                  binding.getResourceId().getInt(), *sourceType,
                                  module)))
      return failure();
    (*destination)->dependencyIds.push_back((*source)->globalId);
  }

  std::map<int64_t, Type> incomingTypes;
  for (TileRoutineRouteAttr route : *incoming) {
    FailureOr<RoutineInfo *> destination =
        findRoutine(route.getDestinationRoutine().getInt());
    if (failed(destination))
      return failure();
    FailureOr<Type> type = getRoutineInputType(
        **destination, route.getDestinationInput().getInt(), module);
    if (failed(type))
      return failure();
    incomingTypes[route.getId().getInt()] = *type;
  }

  using OutgoingGroupKey = std::tuple<int64_t, int64_t, int64_t>;
  std::map<OutgoingGroupKey, SmallVector<TileRoutineRouteAttr>> outgoingGroups;
  std::map<OutgoingGroupKey, Type> outgoingGroupTypes;
  std::map<int64_t, OutgoingGroupKey> outgoingKeyByResourceId;
  for (TileRoutineRouteAttr route : *outgoing) {
    FailureOr<RoutineInfo *> source =
        findRoutine(route.getSourceRoutine().getInt());
    if (failed(source))
      return failure();
    FailureOr<Type> type = getRoutineOutputType(
        **source, route.getSourceOutput().getInt(), module);
    if (failed(type))
      return failure();

    OutgoingGroupKey key{route.getSourceRoutine().getInt(),
                         route.getSourceOutput().getInt(),
                         route.getResourceId().getInt()};
    auto [resourceKey, insertedResourceKey] =
        outgoingKeyByResourceId.emplace(route.getResourceId().getInt(), key);
    if (!insertedResourceKey && resourceKey->second != key) {
      return module.emitError("outgoing global resource ")
             << route.getResourceId().getInt()
             << " is produced by more than one routine result";
    }

    auto [groupType, insertedGroupType] =
        outgoingGroupTypes.emplace(key, *type);
    if (!insertedGroupType && groupType->second != *type) {
      return module.emitError(
          "coalesced outgoing routes have inconsistent tensor types");
    }

    auto &group = outgoingGroups[key];
    if (!group.empty()) {
      TileRoutineRouteAttr first = group.front();
      if (first.getByteSize() != route.getByteSize() ||
          first.getTensorId() != route.getTensorId() ||
          first.getSourceTile() != route.getSourceTile()) {
        return module.emitError(
            "coalesced outgoing routes have inconsistent payload metadata");
      }
    }
    group.push_back(route);
  }

  // Resolve every logical-array argument to the exact boot routine and lane.
  for (TileRoutineBindingAttr binding : *bindings) {
    FailureOr<RoutineInfo *> source =
        findRoutine(binding.getSourceRoutine().getInt());
    FailureOr<RoutineInfo *> destination =
        findRoutine(binding.getDestinationRoutine().getInt());
    if (failed(source) || failed(destination))
      return failure();
    int64_t inputPort = binding.getDestinationInput().getInt();
    FailureOr<Type> inputType =
        getRoutineInputType(**destination, inputPort, module);
    if (failed(inputType) || !isa<LogicalArrayType>(*inputType))
      continue;
    if (!(**source).boot) {
      return module.emitError(
          "logical-array binding must originate at a boot routine");
    }
    FailureOr<int64_t> localArrayId = requireNonNegativeI64(
        (**source).function, tile_runtime_attrs::kTaskLocalArrayIdAttrName);
    FailureOr<int64_t> physicalArrayId = requireNonNegativeI64(
        (**source).function, tile_runtime_attrs::kTaskPhysicalArrayIdAttrName);
    if (failed(localArrayId) || failed(physicalArrayId))
      return failure();
    OpBuilder attrBuilder(module.getContext());
    (**destination)
        .arrayBindings.push_back(buildArrayBinding(
            attrBuilder, inputPort, *physicalArrayId, *localArrayId));

    BlockArgument arrayArgument =
        (**destination).function.getArgument(inputPort);
    (**destination).function.walk([&](Operation *operation) {
      if (llvm::is_contained(operation->getOperands(), arrayArgument)) {
        operation->setAttr(tile_runtime_attrs::kTaskLocalArrayIdAttrName,
                           attrBuilder.getI64IntegerAttr(*localArrayId));
        operation->setAttr(tile_runtime_attrs::kTaskPhysicalArrayIdAttrName,
                           attrBuilder.getI64IntegerAttr(*physicalArrayId));
      }
    });
  }

  for (RoutineInfo &routine : *routines) {
    if (!routine.arrayBindings.empty()) {
      llvm::sort(routine.arrayBindings, [](Attribute lhs, Attribute rhs) {
        auto lhsIndex = cast<DictionaryAttr>(lhs).getAs<IntegerAttr>(
            tile_runtime_attrs::kArrayBindingInputIndexFieldName);
        auto rhsIndex = cast<DictionaryAttr>(rhs).getAs<IntegerAttr>(
            tile_runtime_attrs::kArrayBindingInputIndexFieldName);
        return lhsIndex.getInt() < rhsIndex.getInt();
      });
      routine.function->setAttr(
          tile_runtime_attrs::kTaskArrayBindingsAttrName,
          ArrayAttr::get(module.getContext(), routine.arrayBindings));
    }
  }

  FailureOr<SmallVector<unsigned>> routineOrder =
      buildTopologicalOrder(*routines, module);
  if (failed(routineOrder))
    return failure();

  OpBuilder builder(module.getContext());
  module->setAttr(tile_runtime_attrs::kTaskCoreIdAttrName,
                  builder.getI64IntegerAttr(*tileId));
  module->setAttr(kKindAttr, builder.getStringAttr("tile_runtime_core"));

  SmallVector<Attribute> incomingManifest;
  SmallVector<Attribute> outgoingManifest;
  auto convertRoute = [&](TileRoutineRouteAttr route) {
    return DeploymentRouteAttr::get(
        module.getContext(), route.getId(), route.getSourceTile(),
        route.getSourceRoutine(), route.getSourceOutput(),
        route.getDestinationTile(), route.getDestinationRoutine(),
        route.getDestinationInput(), route.getResourceId(),
        route.getByteSize());
  };
  for (TileRoutineRouteAttr route : *incoming)
    incomingManifest.push_back(convertRoute(route));
  for (TileRoutineRouteAttr route : *outgoing)
    outgoingManifest.push_back(convertRoute(route));
  module->setAttr(deployment_attrs::kIncomingRoutesAttrName,
                  builder.getArrayAttr(incomingManifest));
  module->setAttr(deployment_attrs::kOutgoingRoutesAttrName,
                  builder.getArrayAttr(outgoingManifest));

  auto buildModelManifest = [&](ArrayRef<ModelIOInfo> entries, bool input) {
    SmallVector<Attribute> manifest;
    std::set<std::pair<int64_t, int64_t>> emittedOwnership;
    for (const ModelIOInfo &entry : entries) {
      auto ownership = std::make_pair(entry.attr.getIndex().getInt(),
                                      entry.attr.getResourceId().getInt());
      if (!emittedOwnership.insert(ownership).second)
        continue;
      manifest.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr(input ? "input_index" : "output_index",
                               entry.attr.getIndex()),
          builder.getNamedAttr("owner_core",
                               builder.getI64IntegerAttr(*tileId)),
          builder.getNamedAttr("global_resource_id",
                               entry.attr.getResourceId()),
      }));
    }
    return builder.getArrayAttr(manifest);
  };
  module->setAttr(deployment_attrs::kModelInputsAttrName,
                  buildModelManifest(*modelInputs, true));
  module->setAttr(deployment_attrs::kModelOutputsAttrName,
                  buildModelManifest(*modelOutputs, false));
  module->removeAttr(kLocalBindingsAttr);

  builder.setInsertionPointToEnd(module.getBody());
  auto graphType = TaskGraphType::get(module.getContext());
  auto graphFunction = builder.create<func::FuncOp>(
      module.getLoc(), "generate_task_graph",
      builder.getFunctionType(TypeRange{}, TypeRange{graphType}));
  graphFunction.setPrivate();
  Block *entry = graphFunction.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value graph =
      builder.create<TaskGraphCreateOp>(module.getLoc(), graphType).getResult();

  std::map<int64_t, Value> nonRouteResources;
  for (const auto &[resourceId, type] : nonRouteTypes) {
    Value resource;
    if (modelInputResourceIds.contains(resourceId)) {
      resource = createResource<TaskGraphInputOp>(builder, module.getLoc(),
                                                  graph, type, resourceId);
    } else if (modelOutputResourceIds.contains(resourceId)) {
      resource = createResource<TaskGraphOutputOp>(builder, module.getLoc(),
                                                   graph, type, resourceId);
    } else {
      resource = createResource<TaskGraphIntermediateOp>(
          builder, module.getLoc(), graph, type, resourceId);
    }
    nonRouteResources.emplace(resourceId, resource);
  }

  std::map<int64_t, Value> incomingResources;
  for (TileRoutineRouteAttr route : *incoming) {
    Value resource = createResource<TaskGraphRouteInputOp>(
        builder, module.getLoc(), graph,
        incomingTypes.at(route.getId().getInt()),
        route.getResourceId().getInt());
    resource.getDefiningOp()->setAttr(deployment_attrs::kRouteIdAttrName,
                                      route.getId());
    incomingResources.emplace(route.getId().getInt(), resource);
  }
  std::map<int64_t, Value> outgoingResources;
  for (const auto &[key, routes] : outgoingGroups) {
    TileRoutineRouteAttr canonicalRoute = routes.front();
    Value resource = createResource<TaskGraphRouteOutputOp>(
        builder, module.getLoc(), graph, outgoingGroupTypes.at(key),
        canonicalRoute.getResourceId().getInt());
    resource.getDefiningOp()->setAttr(deployment_attrs::kRouteIdAttrName,
                                      canonicalRoute.getId());

    FailureOr<int64_t> computedByteSize = getTaskResourceByteSize(resource);
    if (failed(computedByteSize) ||
        *computedByteSize != canonicalRoute.getByteSize().getInt()) {
      return module.emitError(
          "outgoing route byte size does not match its static tensor type");
    }
    for (TileRoutineRouteAttr route : routes)
      outgoingResources.emplace(route.getId().getInt(), resource);
  }

  DenseMap<int64_t, Value> taskByRoutineId;
  for (auto ordered : llvm::enumerate(*routineOrder)) {
    RoutineInfo &routine = (*routines)[ordered.value()];
    SmallVector<Value> taskInputs;
    for (unsigned port = 0; port < routine.function.getNumArguments(); ++port) {
      llvm::SetVector<Value> candidates;
      for (TileRoutineRouteAttr route : *incoming) {
        if (route.getDestinationRoutine().getInt() == routine.globalId &&
            route.getDestinationInput().getInt() == port)
          candidates.insert(incomingResources.at(route.getId().getInt()));
      }
      for (TileRoutineBindingAttr binding : *bindings) {
        if (binding.getDestinationRoutine().getInt() == routine.globalId &&
            binding.getDestinationInput().getInt() == port)
          candidates.insert(
              nonRouteResources.at(binding.getResourceId().getInt()));
      }
      for (const ModelIOInfo &modelInput : *modelInputs) {
        if (modelInput.attr.getRoutine().getInt() == routine.globalId &&
            modelInput.attr.getPort().getInt() == port)
          candidates.insert(
              nonRouteResources.at(modelInput.attr.getResourceId().getInt()));
      }
      if (candidates.size() != 1) {
        routine.function.emitError("expected input port ")
            << port << " to have exactly one runtime resource binding";
        return failure();
      }
      taskInputs.push_back(candidates.front());
    }

    SmallVector<Value> taskOutputs;
    SmallVector<int64_t> resultIndices;
    for (unsigned port = 0; port < routine.function.getNumResults(); ++port) {
      llvm::SetVector<Value> resources;
      for (TileRoutineBindingAttr binding : *bindings) {
        if (binding.getSourceRoutine().getInt() == routine.globalId &&
            binding.getSourceOutput().getInt() == port)
          resources.insert(
              nonRouteResources.at(binding.getResourceId().getInt()));
      }
      for (TileRoutineRouteAttr route : *outgoing) {
        if (route.getSourceRoutine().getInt() == routine.globalId &&
            route.getSourceOutput().getInt() == port)
          resources.insert(outgoingResources.at(route.getId().getInt()));
      }
      for (const ModelIOInfo &modelOutput : *modelOutputs) {
        if (modelOutput.attr.getRoutine().getInt() == routine.globalId &&
            modelOutput.attr.getPort().getInt() == port)
          resources.insert(
              nonRouteResources.at(modelOutput.attr.getResourceId().getInt()));
      }
      if (resources.empty()) {
        routine.function.emitError("expected result port ")
            << port << " to have at least one runtime resource binding";
        return failure();
      }
      for (Value resource : resources) {
        taskOutputs.push_back(resource);
        resultIndices.push_back(port);
      }
    }

    SmallVector<Value> dependencies;
    DenseSet<int64_t> seenDependencies;
    for (int64_t dependencyId : routine.dependencyIds) {
      if (!seenDependencies.insert(dependencyId).second)
        continue;
      auto task = taskByRoutineId.find(dependencyId);
      if (task == taskByRoutineId.end()) {
        routine.function.emitError("dependency routine ")
            << dependencyId << " was not materialized before its consumer";
        return failure();
      }
      dependencies.push_back(task->second);
    }

    auto task = builder.create<TaskCreateOp>(
        module.getLoc(), TaskType::get(module.getContext()), graph,
        SymbolRefAttr::get(routine.function),
        builder.getStringAttr(
            routine.boot
                ? "analog"
                : (routine.arrayBindings.empty() ? "digital" : "analog")),
        builder.getStringAttr(routine.boot ? "sculptor.matrix_setup"
                                           : "sculptor.tile_routine"),
        builder.getStringAttr(routine.function.getSymName()),
        builder.getStringAttr("ra_tile"),
        builder.getI64IntegerAttr(routine.globalId), taskInputs, taskOutputs,
        dependencies);
    task->setAttr(deployment_attrs::kGlobalTaskIdAttrName,
                  builder.getI64IntegerAttr(routine.globalId));
    task->setAttr(tile_runtime_attrs::kTaskCoreIdAttrName,
                  builder.getI64IntegerAttr(*tileId));
    task->setAttr(tile_runtime_attrs::kTaskIndexAttrName,
                  builder.getI64IntegerAttr(ordered.index()));
    if (!resultIndices.empty())
      task->setAttr(tile_runtime_attrs::kTaskResultIndicesAttrName,
                    builder.getI64ArrayAttr(resultIndices));
    if (!routine.arrayBindings.empty())
      task->setAttr(tile_runtime_attrs::kTaskArrayBindingsAttrName,
                    builder.getArrayAttr(routine.arrayBindings));
    if (routine.boot) {
      for (StringRef attrName :
           {StringRef(tile_runtime_attrs::kTaskLocalArrayIdAttrName),
            StringRef(tile_runtime_attrs::kTaskPhysicalArrayIdAttrName)}) {
        Attribute value = routine.function->getAttr(attrName);
        if (!value) {
          routine.function.emitError("boot routine is missing '")
              << attrName << "'";
          return failure();
        }
        task->setAttr(attrName, value);
      }
    }
    routine.function->setAttr(deployment_attrs::kGlobalTaskIdAttrName,
                              builder.getI64IntegerAttr(routine.globalId));
    routine.function->setAttr(tile_runtime_attrs::kTaskCoreIdAttrName,
                              builder.getI64IntegerAttr(*tileId));
    routine.function->setAttr(tile_runtime_attrs::kTaskIndexAttrName,
                              builder.getI64IntegerAttr(ordered.index()));
    routine.task = task;
    taskByRoutineId[routine.globalId] = task.getResult();
  }

  builder.create<func::ReturnOp>(module.getLoc(), graph);
  if (failed(verify(module))) {
    module.emitError("materialized tile runtime graph failed verification");
    return failure();
  }
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {

void MaterializeTileRuntimeGraphPass::runOnOperation() {
  if (failed(materializeRuntimeGraph(getOperation())))
    signalPassFailure();
}

void registerMaterializeTileRuntimeGraphPass() {
  PassRegistration<MaterializeTileRuntimeGraphPass>();
}

} // namespace sculptor
} // namespace mlir
