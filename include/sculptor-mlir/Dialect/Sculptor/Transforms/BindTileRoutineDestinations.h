#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BINDTILEROUTINEDESTINATIONS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BINDTILEROUTINEDESTINATIONS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct BindTileRoutineDestinationsPass
    : public PassWrapper<BindTileRoutineDestinationsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindTileRoutineDestinationsPass)

  StringRef getArgument() const final {
    return "sculptor-bind-tile-routine-destinations";
  }
  StringRef getDescription() const final {
    return "Bind bufferized routine output arguments to planned owner views";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<SculptorDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override;
};

void registerBindTileRoutineDestinationsPass();

} // namespace mlir::sculptor

#endif
