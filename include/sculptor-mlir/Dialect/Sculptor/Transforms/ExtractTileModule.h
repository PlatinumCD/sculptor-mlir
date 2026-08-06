#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXTRACTTILEMODULE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXTRACTTILEMODULE_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>

namespace mlir {
namespace sculptor {

struct ExtractTileModulePass
    : public PassWrapper<ExtractTileModulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractTileModulePass)

  Option<int64_t> tileId{*this, "tile-id",
                         llvm::cl::desc("Active physical tile to extract"),
                         llvm::cl::init(-1)};

  ExtractTileModulePass() = default;
  ExtractTileModulePass(const ExtractTileModulePass &pass) : PassWrapper(pass) {
    tileId = pass.tileId;
  }

  StringRef getArgument() const final { return "sculptor-extract-tile-module"; }
  StringRef getDescription() const final {
    return "Extract one outlined physical tile into a standalone module";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerExtractTileModulePass();

} // namespace sculptor
} // namespace mlir

#endif
