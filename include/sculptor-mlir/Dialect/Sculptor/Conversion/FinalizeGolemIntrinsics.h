#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_FINALIZEGOLEMINTRINSICS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_FINALIZEGOLEMINTRINSICS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

// Final backend step from LLVM Golem shim calls to target Golem ISA intrinsics.
// This pass consumes shim calls emitted by LowerGolemToLLVMShims; it does not
// see sculptor.mvm or Sculptor Golem execution ops.
struct FinalizeGolemIntrinsicsPass
    : public mlir::PassWrapper<FinalizeGolemIntrinsicsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeGolemIntrinsicsPass)

  mlir::StringRef getArgument() const final {
    return "sculptor-finalize-golem-intrinsics";
  }

  mlir::StringRef getDescription() const final {
    return "Finalize LLVM Golem shim calls into Golem ISA intrinsics";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

void registerFinalizeGolemIntrinsicsPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_FINALIZEGOLEMINTRINSICS_H
