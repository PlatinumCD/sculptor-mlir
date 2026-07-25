#include "sculptor-mlir/Dialect/Sculptor/Conversion/EmitGolemTileABI.h"

#include "golem/GolemTileABI.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace sculptor {

namespace {

void removeConsumedSculptorAttributes(ModuleOp module) {
  module.walk([](Operation *op) {
    SmallVector<StringAttr> names;
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("sculptor."))
        names.push_back(attr.getName());
    for (StringAttr name : names)
      op->removeAttr(name);
  });
}

LogicalResult verifyPureLLVMModule(ModuleOp module) {
  Operation *badOperation = nullptr;
  module.walk([&](Operation *op) {
    if (!badOperation && op->getName().getDialectNamespace() == "sculptor")
      badOperation = op;
  });
  if (badOperation) {
    badOperation->emitError(
        "Sculptor operation remains after Golem tile ABI packaging");
    return failure();
  }

  bool hasSculptorAttrOrType = false;
  AttrTypeWalker walker;
  walker.addWalk([&](Type type) {
    if (type.getDialect().getNamespace() == "sculptor")
      hasSculptorAttrOrType = true;
  });
  walker.addWalk([&](Attribute attr) {
    if (attr.getDialect().getNamespace() == "sculptor")
      hasSculptorAttrOrType = true;
  });

  module.walk([&](Operation *op) {
    for (NamedAttribute attr : op->getAttrs())
      walker.walk(attr.getValue());
    for (Type type : op->getOperandTypes())
      walker.walk(type);
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          walker.walk(argument.getType());
  });
  if (hasSculptorAttrOrType) {
    module.emitError(
        "Sculptor type or typed attribute remains after Golem tile ABI "
        "packaging");
    return failure();
  }

  for (Operation &op : *module.getBody()) {
    if (op.getName().getDialectNamespace() != "llvm") {
      op.emitError("Golem tile ABI output must contain only top-level LLVM "
                   "dialect symbols");
      return failure();
    }
  }

  if (failed(verify(module))) {
    module.emitError("Golem tile ABI module failed verification");
    return failure();
  }
  return success();
}

} // namespace

void EmitGolemTileABIPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto model = golem_tile_abi::collectTileModel(module);
  if (failed(model) ||
      failed(golem_tile_abi::emitTaskAdapters(module, *model)) ||
      failed(golem_tile_abi::emitTileTables(module, *model))) {
    signalPassFailure();
    return;
  }

  model->taskGraphFunc.erase();
  removeConsumedSculptorAttributes(module);
  if (failed(verifyPureLLVMModule(module)))
    signalPassFailure();
}

void registerEmitGolemTileABIPass() {
  PassRegistration<EmitGolemTileABIPass>();
}

} // namespace sculptor
} // namespace mlir
