#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXTRACTCOREMODULE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXTRACTCOREMODULE_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>

namespace mlir {
namespace sculptor {

struct ExtractCoreModulePass
    : public PassWrapper<ExtractCoreModulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractCoreModulePass)

  Option<int64_t> coreId{*this, "core-id",
                         llvm::cl::desc("Active deployment core to extract"),
                         llvm::cl::init(-1)};

  ExtractCoreModulePass() = default;

  ExtractCoreModulePass(const ExtractCoreModulePass &pass)
      : PassWrapper(pass),
        coreId(*this, "core-id",
               llvm::cl::desc("Active deployment core to extract"),
               llvm::cl::init(-1)) {
    coreId = pass.coreId;
  }

  StringRef getArgument() const final { return "sculptor-extract-core-module"; }

  StringRef getDescription() const final {
    return "Extract one deployed core into a standalone top-level module";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerExtractCoreModulePass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXTRACTCOREMODULE_H
