#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_EMITGOLEMTILEABI_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_EMITGOLEMTILEABI_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct EmitGolemTileABIPass
    : public PassWrapper<EmitGolemTileABIPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitGolemTileABIPass)

  StringRef getArgument() const final { return "sculptor-emit-golem-tile-abi"; }

  StringRef getDescription() const final {
    return "Package one finalized deployment core for the Golem tile ABI";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<SculptorDialect, func::FuncDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

void registerEmitGolemTileABIPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_EMITGOLEMTILEABI_H
