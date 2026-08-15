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

// Replicate expanded matrix setups before RA-tree planning. With no array
// capacity, every physical MVM receives an independent logical array (the
// original experimental behavior). With a finite capacity, replicas are
// allocated to high-fanout setups while preserving a hard upper bound on the
// persistent logical arrays that later realization must bind.
struct DuplicateMatricesPass
    : public PassWrapper<DuplicateMatricesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DuplicateMatricesPass)

  Option<int64_t> arrayCapacity{
      *this, "array-capacity",
      llvm::cl::desc("Maximum persistent logical arrays per function; zero "
                     "retains one replica per physical MVM"),
      llvm::cl::init(0)};

  Option<int64_t> minimumMVMsPerReplica{
      *this, "minimum-mvms-per-replica",
      llvm::cl::desc("Minimum physical-MVM consumers assigned to each added "
                     "replica"),
      llvm::cl::init(1)};

  Option<int64_t> maximumReplicasPerSetup{
      *this, "maximum-replicas-per-setup",
      llvm::cl::desc("Optional per-setup replica limit; zero is unlimited"),
      llvm::cl::init(0)};

  DuplicateMatricesPass() = default;
  DuplicateMatricesPass(const DuplicateMatricesPass &pass) : PassWrapper(pass) {
    arrayCapacity = pass.arrayCapacity;
    minimumMVMsPerReplica = pass.minimumMVMsPerReplica;
    maximumReplicasPerSetup = pass.maximumReplicasPerSetup;
  }

  StringRef getArgument() const final { return "sculptor-duplicate-matrices"; }

  StringRef getDescription() const final {
    return "Replicate expanded matrix setups within a physical-array budget";
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
