#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractCoreModule.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace {

namespace deployment_attrs = mlir::sculptor::deployment_attrs;
namespace runtime_attrs = mlir::sculptor::runtime_attrs;

mlir::FailureOr<mlir::ArrayAttr>
filterOwnershipManifest(mlir::ModuleOp module, llvm::StringRef attrName,
                        int64_t coreId) {
  mlir::Builder builder(module.getContext());
  auto manifest = module->getAttrOfType<mlir::ArrayAttr>(attrName);
  if (!manifest)
    return builder.getArrayAttr({});

  llvm::SmallVector<mlir::Attribute, 4> filtered;
  for (mlir::Attribute entry : manifest) {
    auto ownership = llvm::dyn_cast<mlir::DictionaryAttr>(entry);
    auto owner = ownership ? ownership.getAs<mlir::IntegerAttr>("owner_core")
                           : mlir::IntegerAttr();
    if (!owner) {
      module.emitError("expected '")
          << attrName << "' entries to contain an integer owner_core";
      return mlir::failure();
    }
    if (owner.getInt() == coreId)
      filtered.push_back(entry);
  }
  return builder.getArrayAttr(filtered);
}

mlir::FailureOr<mlir::ArrayAttr> getCoreRoutes(mlir::ModuleOp deployment,
                                               mlir::ModuleOp coreModule,
                                               llvm::StringRef attrName,
                                               int64_t coreId, bool incoming) {
  mlir::Builder builder(deployment.getContext());
  auto routes = coreModule->getAttrOfType<mlir::ArrayAttr>(attrName);
  if (!routes) {
    auto deploymentRoutes = deployment->getAttrOfType<mlir::ArrayAttr>(
        deployment_attrs::kRoutesAttrName);
    if (!deploymentRoutes)
      return builder.getArrayAttr({});

    llvm::SmallVector<mlir::Attribute, 4> filtered;
    for (mlir::Attribute entry : deploymentRoutes) {
      auto route = llvm::dyn_cast<mlir::sculptor::DeploymentRouteAttr>(entry);
      if (!route) {
        deployment.emitError("expected deployment route table to contain "
                             "#sculptor.deployment_route attributes");
        return mlir::failure();
      }
      int64_t routeCore = incoming ? route.getDestinationCore().getInt()
                                   : route.getSourceCore().getInt();
      if (routeCore == coreId)
        filtered.push_back(route);
    }
    routes = builder.getArrayAttr(filtered);
  }

  for (mlir::Attribute entry : routes) {
    auto route = llvm::dyn_cast<mlir::sculptor::DeploymentRouteAttr>(entry);
    if (!route) {
      coreModule.emitError("expected '")
          << attrName << "' to contain #sculptor.deployment_route attributes";
      return mlir::failure();
    }
    int64_t routeCore = incoming ? route.getDestinationCore().getInt()
                                 : route.getSourceCore().getInt();
    if (routeCore != coreId) {
      coreModule.emitError("route in '")
          << attrName << "' does not belong to core " << coreId;
      return mlir::failure();
    }
  }
  return routes;
}

mlir::LogicalResult extractCoreModule(mlir::ModuleOp deployment,
                                      int64_t coreId) {
  if (coreId < 0) {
    deployment.emitError("expected core-id to be a non-negative integer");
    return mlir::failure();
  }

  mlir::Attribute activeCoreIds =
      deployment->getAttr(deployment_attrs::kActiveCoreIdsAttrName);
  if (!activeCoreIds) {
    deployment.emitError("expected deployment attribute '")
        << deployment_attrs::kActiveCoreIdsAttrName << "'";
    return mlir::failure();
  }
  bool isActive = false;
  if (auto dense = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(activeCoreIds)) {
    isActive = llvm::is_contained(dense.asArrayRef(), coreId);
  } else if (auto array = llvm::dyn_cast<mlir::ArrayAttr>(activeCoreIds)) {
    for (mlir::Attribute entry : array) {
      auto integer = llvm::dyn_cast<mlir::IntegerAttr>(entry);
      if (!integer) {
        deployment.emitError("expected deployment attribute '")
            << deployment_attrs::kActiveCoreIdsAttrName
            << "' to contain integers";
        return mlir::failure();
      }
      isActive |= integer.getInt() == coreId;
    }
  } else {
    deployment.emitError("expected deployment attribute '")
        << deployment_attrs::kActiveCoreIdsAttrName
        << "' to be an integer array";
    return mlir::failure();
  }
  if (!isActive) {
    deployment.emitError("requested core ")
        << coreId << " is not an active core in the deployment";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::ModuleOp, 2> matches;
  for (mlir::ModuleOp nested : deployment.getOps<mlir::ModuleOp>()) {
    auto nestedCoreId = nested->getAttrOfType<mlir::IntegerAttr>(
        runtime_attrs::kTaskCoreIdAttrName);
    if (nestedCoreId && nestedCoreId.getInt() == coreId)
      matches.push_back(nested);
  }
  if (matches.empty()) {
    deployment.emitError("active core ")
        << coreId << " has no nested deployment module";
    return mlir::failure();
  }
  if (matches.size() != 1) {
    deployment.emitError("found multiple nested deployment modules for core ")
        << coreId;
    return mlir::failure();
  }

  mlir::ModuleOp selected = matches.front();
  auto incomingRoutes =
      getCoreRoutes(deployment, selected,
                    deployment_attrs::kIncomingRoutesAttrName, coreId, true);
  auto outgoingRoutes =
      getCoreRoutes(deployment, selected,
                    deployment_attrs::kOutgoingRoutesAttrName, coreId, false);
  auto modelInputs = filterOwnershipManifest(
      deployment, deployment_attrs::kModelInputsAttrName, coreId);
  auto modelOutputs = filterOwnershipManifest(
      deployment, deployment_attrs::kModelOutputsAttrName, coreId);
  if (failed(incomingRoutes) || failed(outgoingRoutes) || failed(modelInputs) ||
      failed(modelOutputs))
    return mlir::failure();

  for (mlir::NamedAttribute attr : selected->getAttrs()) {
    if (attr.getName().getValue() == mlir::SymbolTable::getSymbolAttrName())
      continue;
    deployment->setAttr(attr.getName(), attr.getValue());
  }
  mlir::Builder builder(deployment.getContext());
  deployment->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                      builder.getI64IntegerAttr(coreId));
  deployment->setAttr(deployment_attrs::kIncomingRoutesAttrName,
                      *incomingRoutes);
  deployment->setAttr(deployment_attrs::kOutgoingRoutesAttrName,
                      *outgoingRoutes);
  deployment->setAttr(deployment_attrs::kModelInputsAttrName, *modelInputs);
  deployment->setAttr(deployment_attrs::kModelOutputsAttrName, *modelOutputs);
  deployment->removeAttr(deployment_attrs::kActiveCoreIdsAttrName);
  deployment->removeAttr(deployment_attrs::kRoutesAttrName);

  llvm::SmallVector<mlir::Operation *, 16> erase;
  for (mlir::Operation &op : *deployment.getBody()) {
    if (&op != selected.getOperation())
      erase.push_back(&op);
  }
  for (mlir::Operation *op : llvm::reverse(erase))
    op->erase();

  llvm::SmallVector<mlir::Operation *, 16> selectedContents;
  for (mlir::Operation &op : *selected.getBody())
    selectedContents.push_back(&op);
  for (mlir::Operation *op : selectedContents)
    op->moveBefore(selected);
  selected.erase();

  if (failed(mlir::verify(deployment))) {
    deployment.emitError("extracted core module failed verification");
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace

namespace mlir {
namespace sculptor {

void ExtractCoreModulePass::runOnOperation() {
  if (failed(extractCoreModule(getOperation(), coreId)))
    signalPassFailure();
}

void registerExtractCoreModulePass() {
  PassRegistration<ExtractCoreModulePass>();
}

} // namespace sculptor
} // namespace mlir
