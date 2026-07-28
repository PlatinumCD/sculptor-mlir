#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_LOWERSCHEDULEDMVMTODIGITAL_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_LOWERSCHEDULEDMVMTODIGITAL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

// Replaces scheduled analog tile implementations with digital Linalg matmuls
// without changing task placement, task dependencies, or tensor communication.
struct LowerScheduledMVMToDigitalPass
    : public mlir::PassWrapper<LowerScheduledMVMToDigitalPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerScheduledMVMToDigitalPass)

  llvm::StringRef getArgument() const final {
    return "sculptor-lower-scheduled-mvm-to-digital";
  }

  llvm::StringRef getDescription() const final {
    return "Replace scheduled Golem MVM tile bodies with placement-preserving "
           "digital Linalg matmuls";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;
};

void registerLowerScheduledMVMToDigitalPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_LOWERSCHEDULEDMVMTODIGITAL_H
