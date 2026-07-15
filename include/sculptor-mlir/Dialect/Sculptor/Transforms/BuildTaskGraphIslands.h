#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BUILDTASKGRAPHISLANDS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BUILDTASKGRAPHISLANDS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct BuildTaskGraphIslandsPass
    : public PassWrapper<BuildTaskGraphIslandsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildTaskGraphIslandsPass)

  StringRef getArgument() const final {
    return "sculptor-build-task-graph-islands";
  }

  StringRef getDescription() const final {
    return "Build and attach logical placement islands to a task graph";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerBuildTaskGraphIslandsPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BUILDTASKGRAPHISLANDS_H
