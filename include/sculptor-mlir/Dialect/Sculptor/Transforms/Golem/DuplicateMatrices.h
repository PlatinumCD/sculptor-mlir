#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_GOLEM_DUPLICATEMATRICES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_GOLEM_DUPLICATEMATRICES_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

// Experimental transform that gives every expanded physical MVM stage an
// independent logical array setup. This exposes matrix replication to later
// RA-tree planning without assigning replicas to physical cores or arrays.
struct DuplicateMatricesPass
    : public PassWrapper<DuplicateMatricesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DuplicateMatricesPass)

  StringRef getArgument() const final { return "sculptor-duplicate-matrices"; }

  StringRef getDescription() const final {
    return "Duplicate expanded matrix setups so every physical MVM has an "
           "independent logical array";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, SculptorDialect>();
  }

  void runOnOperation() override;
};

void registerDuplicateMatricesPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_GOLEM_DUPLICATEMATRICES_H
