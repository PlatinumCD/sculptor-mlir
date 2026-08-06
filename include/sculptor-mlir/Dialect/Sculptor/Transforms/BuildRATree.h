#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BUILDRATREE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BUILDRATREE_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct BuildRATreePass
    : public PassWrapper<BuildRATreePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildRATreePass)

  BuildRATreePass() = default;
  BuildRATreePass(const BuildRATreePass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "sculptor-build-ra-tree"; }

  StringRef getDescription() const final {
    return "Build a deterministic structural Resource Allocation Tree";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerBuildRATreePass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BUILDRATREE_H
