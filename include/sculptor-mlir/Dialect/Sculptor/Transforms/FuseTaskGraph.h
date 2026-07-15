#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FUSETASKGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FUSETASKGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct FuseTaskGraphPass
    : public PassWrapper<FuseTaskGraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseTaskGraphPass)

  StringRef getArgument() const final { return "sculptor-fuse-task-graph"; }

  StringRef getDescription() const final {
    return "Fuse connected scheduled tasks within each logical island and core";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerFuseTaskGraphPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FUSETASKGRAPH_H
