#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractTileModule.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace {

using namespace mlir;
using namespace mlir::sculptor;

constexpr StringLiteral kKindAttr = "sculptor.deployment.kind";
constexpr StringLiteral kActiveTileIdsAttr =
    "sculptor.deployment.active_tile_ids";
constexpr StringLiteral kRoutesAttr = "sculptor.deployment.routes";
constexpr StringLiteral kLocalBindingsAttr =
    "sculptor.deployment.local_bindings";
constexpr StringLiteral kModelInputsAttr = "sculptor.deployment.model_inputs";
constexpr StringLiteral kModelOutputsAttr = "sculptor.deployment.model_outputs";
constexpr StringLiteral kIncomingRoutesAttr =
    "sculptor.deployment.incoming_routes";
constexpr StringLiteral kOutgoingRoutesAttr =
    "sculptor.deployment.outgoing_routes";
constexpr StringLiteral kPhysicalTileIdAttr =
    "sculptor.deployment.physical_tile_id";

FailureOr<ArrayAttr> requireArrayAttr(Operation *operation, StringRef name) {
  auto value = operation->getAttrOfType<ArrayAttr>(name);
  if (!value) {
    operation->emitError("expected array attribute '") << name << "'";
    return failure();
  }
  return value;
}

LogicalResult validateRoutes(ModuleOp module, ArrayAttr routes, int64_t tileId,
                             bool incoming) {
  for (Attribute value : routes) {
    auto route = dyn_cast<TileRoutineRouteAttr>(value);
    if (!route) {
      module.emitError("expected typed tile-routine route metadata");
      return failure();
    }
    int64_t owner = incoming ? route.getDestinationTile().getInt()
                             : route.getSourceTile().getInt();
    if (owner != tileId) {
      module.emitError(incoming ? "incoming route targets another tile"
                                : "outgoing route originates on another tile");
      return failure();
    }
  }
  return success();
}

LogicalResult validateLocalBindings(ModuleOp module, ArrayAttr bindings) {
  for (Attribute value : bindings) {
    if (!isa<TileRoutineBindingAttr>(value)) {
      module.emitError("expected typed tile-routine local binding metadata");
      return failure();
    }
  }
  return success();
}

FailureOr<ArrayAttr> filterModelIO(ModuleOp deployment, StringRef name,
                                   int64_t tileId) {
  FailureOr<ArrayAttr> values = requireArrayAttr(deployment, name);
  if (failed(values))
    return failure();
  SmallVector<Attribute> filtered;
  for (Attribute value : *values) {
    auto entry = dyn_cast<TileRoutineModelIOAttr>(value);
    if (!entry) {
      deployment.emitError("expected typed tile-routine model I/O metadata");
      return failure();
    }
    if (entry.getTile().getInt() == tileId)
      filtered.push_back(entry);
  }
  return ArrayAttr::get(deployment.getContext(), filtered);
}

LogicalResult extractTileModule(ModuleOp deployment, int64_t tileId) {
  if (tileId < 0) {
    deployment.emitError("expected tile-id to be a non-negative integer");
    return failure();
  }
  auto kind = deployment->getAttrOfType<StringAttr>(kKindAttr);
  if (!kind || kind.getValue() != "tile_routines") {
    deployment.emitError(
        "expected a deployment produced by sculptor-outline-tile-routines");
    return failure();
  }

  FailureOr<ArrayAttr> activeIds =
      requireArrayAttr(deployment, kActiveTileIdsAttr);
  if (failed(activeIds))
    return failure();
  bool active = false;
  for (Attribute value : *activeIds) {
    auto id = dyn_cast<IntegerAttr>(value);
    if (!id) {
      deployment.emitError("active tile IDs must be integers");
      return failure();
    }
    active |= id.getInt() == tileId;
  }
  if (!active) {
    deployment.emitError("requested tile ")
        << tileId << " is not active in the deployment";
    return failure();
  }

  SmallVector<ModuleOp> matches;
  for (ModuleOp nested : deployment.getOps<ModuleOp>()) {
    auto id = nested->getAttrOfType<IntegerAttr>(kPhysicalTileIdAttr);
    if (!id) {
      nested.emitError("outlined tile module has no physical tile ID");
      return failure();
    }
    if (id.getInt() == tileId)
      matches.push_back(nested);
  }
  if (matches.empty()) {
    deployment.emitError("active tile ") << tileId << " has no outlined module";
    return failure();
  }
  if (matches.size() != 1) {
    deployment.emitError("found multiple outlined modules for tile ") << tileId;
    return failure();
  }

  ModuleOp selected = matches.front();
  FailureOr<ArrayAttr> incoming =
      requireArrayAttr(selected, kIncomingRoutesAttr);
  FailureOr<ArrayAttr> outgoing =
      requireArrayAttr(selected, kOutgoingRoutesAttr);
  FailureOr<ArrayAttr> bindings =
      requireArrayAttr(selected, kLocalBindingsAttr);
  FailureOr<ArrayAttr> modelInputs =
      filterModelIO(deployment, kModelInputsAttr, tileId);
  FailureOr<ArrayAttr> modelOutputs =
      filterModelIO(deployment, kModelOutputsAttr, tileId);
  if (failed(incoming) || failed(outgoing) || failed(bindings) ||
      failed(modelInputs) || failed(modelOutputs) ||
      failed(validateRoutes(deployment, *incoming, tileId, true)) ||
      failed(validateRoutes(deployment, *outgoing, tileId, false)) ||
      failed(validateLocalBindings(deployment, *bindings)))
    return failure();

  // The selected tile owns the authoritative local memory plan.  Clear the
  // deployment-wide conservative summary first so attributes intentionally
  // absent from an exact tile plan (for example an interference default when
  // the tile has an explicit table) cannot leak into the extracted core.
  SmallVector<StringAttr> globalMemoryAttrs;
  for (NamedAttribute attribute : deployment->getAttrs())
    if (attribute.getName().getValue().starts_with("sculptor.memory."))
      globalMemoryAttrs.push_back(attribute.getName());
  for (StringAttr name : globalMemoryAttrs)
    deployment->removeAttr(name);

  for (NamedAttribute attribute : selected->getAttrs()) {
    if (attribute.getName().getValue() == SymbolTable::getSymbolAttrName())
      continue;
    deployment->setAttr(attribute.getName(), attribute.getValue());
  }
  Builder builder(deployment.getContext());
  deployment->setAttr(kKindAttr, builder.getStringAttr("tile_routine_core"));
  deployment->setAttr(kIncomingRoutesAttr, *incoming);
  deployment->setAttr(kOutgoingRoutesAttr, *outgoing);
  deployment->setAttr(kLocalBindingsAttr, *bindings);
  deployment->setAttr(kModelInputsAttr, *modelInputs);
  deployment->setAttr(kModelOutputsAttr, *modelOutputs);
  deployment->removeAttr(kActiveTileIdsAttr);
  deployment->removeAttr(kRoutesAttr);

  SmallVector<Operation *> erase;
  for (Operation &operation : *deployment.getBody()) {
    if (&operation != selected.getOperation())
      erase.push_back(&operation);
  }
  for (Operation *operation : llvm::reverse(erase))
    operation->erase();

  SmallVector<Operation *> contents;
  for (Operation &operation : *selected.getBody())
    contents.push_back(&operation);
  for (Operation *operation : contents)
    operation->moveBefore(selected);
  selected.erase();

  if (!deployment.getOps<ModuleOp>().empty()) {
    deployment.emitError("tile extraction left a nested module");
    return failure();
  }
  for (func::FuncOp function : deployment.getOps<func::FuncOp>()) {
    auto functionTile =
        function->getAttrOfType<IntegerAttr>(kPhysicalTileIdAttr);
    if (!functionTile || functionTile.getInt() != tileId) {
      function.emitError("extracted routine has an invalid physical tile ID");
      return failure();
    }
  }
  if (failed(verify(deployment))) {
    deployment.emitError("extracted tile module failed verification");
    return failure();
  }
  if (failed(tile_memory::verifyTileMemoryPlan(deployment)))
    return failure();
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {

FailureOr<OwningOpRef<ModuleOp>>
cloneExtractedTileModule(ModuleOp deployment, int64_t tileId) {
  if (tileId < 0) {
    deployment.emitError("expected tile-id to be a non-negative integer");
    return failure();
  }
  auto kind = deployment->getAttrOfType<StringAttr>(kKindAttr);
  if (!kind || kind.getValue() != "tile_routines") {
    deployment.emitError(
        "expected a deployment produced by sculptor-outline-tile-routines");
    return failure();
  }

  FailureOr<ArrayAttr> activeIds =
      requireArrayAttr(deployment, kActiveTileIdsAttr);
  if (failed(activeIds))
    return failure();
  bool active = llvm::any_of(*activeIds, [&](Attribute value) {
    auto id = dyn_cast<IntegerAttr>(value);
    return id && id.getInt() == tileId;
  });
  if (!active) {
    deployment.emitError("requested tile ")
        << tileId << " is not active in the deployment";
    return failure();
  }

  ModuleOp selected;
  for (ModuleOp nested : deployment.getOps<ModuleOp>()) {
    auto id = nested->getAttrOfType<IntegerAttr>(kPhysicalTileIdAttr);
    if (!id) {
      nested.emitError("outlined tile module has no physical tile ID");
      return failure();
    }
    if (id.getInt() != tileId)
      continue;
    if (selected) {
      deployment.emitError("found multiple outlined modules for tile ")
          << tileId;
      return failure();
    }
    selected = nested;
  }
  if (!selected) {
    deployment.emitError("active tile ") << tileId
                                          << " has no outlined module";
    return failure();
  }

  FailureOr<ArrayAttr> incoming =
      requireArrayAttr(selected, kIncomingRoutesAttr);
  FailureOr<ArrayAttr> outgoing =
      requireArrayAttr(selected, kOutgoingRoutesAttr);
  FailureOr<ArrayAttr> bindings =
      requireArrayAttr(selected, kLocalBindingsAttr);
  FailureOr<ArrayAttr> modelInputs =
      filterModelIO(deployment, kModelInputsAttr, tileId);
  FailureOr<ArrayAttr> modelOutputs =
      filterModelIO(deployment, kModelOutputsAttr, tileId);
  if (failed(incoming) || failed(outgoing) || failed(bindings) ||
      failed(modelInputs) || failed(modelOutputs) ||
      failed(validateRoutes(deployment, *incoming, tileId, true)) ||
      failed(validateRoutes(deployment, *outgoing, tileId, false)) ||
      failed(validateLocalBindings(deployment, *bindings)))
    return failure();

  OwningOpRef<ModuleOp> extracted =
      ModuleOp::create(deployment.getLoc(), deployment.getName());
  (*extracted)->setAttrs(deployment->getAttrDictionary());

  SmallVector<StringAttr> globalMemoryAttrs;
  for (NamedAttribute attribute : (*extracted)->getAttrs())
    if (attribute.getName().getValue().starts_with("sculptor.memory."))
      globalMemoryAttrs.push_back(attribute.getName());
  for (StringAttr name : globalMemoryAttrs)
    (*extracted)->removeAttr(name);

  for (NamedAttribute attribute : selected->getAttrs()) {
    if (attribute.getName().getValue() == SymbolTable::getSymbolAttrName())
      continue;
    (*extracted)->setAttr(attribute.getName(), attribute.getValue());
  }
  Builder builder(deployment.getContext());
  (*extracted)->setAttr(kKindAttr,
                        builder.getStringAttr("tile_routine_core"));
  (*extracted)->setAttr(kIncomingRoutesAttr, *incoming);
  (*extracted)->setAttr(kOutgoingRoutesAttr, *outgoing);
  (*extracted)->setAttr(kLocalBindingsAttr, *bindings);
  (*extracted)->setAttr(kModelInputsAttr, *modelInputs);
  (*extracted)->setAttr(kModelOutputsAttr, *modelOutputs);
  (*extracted)->removeAttr(kActiveTileIdsAttr);
  (*extracted)->removeAttr(kRoutesAttr);

  IRMapping mapping;
  OpBuilder opBuilder(deployment.getContext());
  opBuilder.setInsertionPointToEnd((*extracted).getBody());
  for (Operation &operation : *selected.getBody())
    opBuilder.clone(operation, mapping);

  for (func::FuncOp function : (*extracted).getOps<func::FuncOp>()) {
    auto functionTile =
        function->getAttrOfType<IntegerAttr>(kPhysicalTileIdAttr);
    if (!functionTile || functionTile.getInt() != tileId) {
      function.emitError("extracted routine has an invalid physical tile ID");
      return failure();
    }
  }
  if (failed(verify(*extracted))) {
    (*extracted).emitError("extracted tile module failed verification");
    return failure();
  }
  if (failed(tile_memory::verifyTileMemoryPlan(*extracted)))
    return failure();
  return extracted;
}

void ExtractTileModulePass::runOnOperation() {
  if (failed(extractTileModule(getOperation(), tileId)))
    signalPassFailure();
}

void registerExtractTileModulePass() {
  PassRegistration<ExtractTileModulePass>();
}

} // namespace sculptor
} // namespace mlir
