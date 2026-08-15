#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FUSEELEMENTWISEREGIONS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FUSEELEMENTWISEREGIONS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

// Fuses single-use, purely parallel linalg.generic chains before RA-tree
// construction. Fusion is confined to one semantic layer (or to wholly
// unassigned digital work), so it cannot erase a layer boundary.
struct FuseElementwiseRegionsPass
    : public PassWrapper<FuseElementwiseRegionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseElementwiseRegionsPass)

  StringRef getArgument() const final {
    return "sculptor-fuse-elementwise-regions";
  }
  StringRef getDescription() const final {
    return "Fuse semantic-safe single-use elementwise regions before RA "
           "planning";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<SculptorDialect, func::FuncDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

void registerFuseElementwiseRegionsPass();

} // namespace mlir::sculptor

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FUSEELEMENTWISEREGIONS_H
