#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_LOWERGOLEMTOLLVMSHIMS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_LOWERGOLEMTOLLVMSHIMS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace sculptor {

// Lowers Sculptor Golem execution ops to LLVM-callable runtime/backend shims.
// This is after sculptor.mvm expansion and before final target ISA intrinsic
// selection.
struct LowerGolemToLLVMShimsPass
    : public mlir::PassWrapper<LowerGolemToLLVMShimsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGolemToLLVMShimsPass)

  // Returns the stable pipeline flag used by MLIR pass registration.
  mlir::StringRef getArgument() const final {
    return "sculptor-lower-golem-to-llvm-shims";
  }

  // Summarizes the lowering pass in MLIR pass-help output.
  mlir::StringRef getDescription() const final {
    return "Lower Sculptor Golem execution ops to LLVM runtime shims";
  }

  // Declares the dialects that conversion patterns and legality checks may
  // touch.
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect>();
  }

  // Applies the partial conversion and signals pass failure when illegal IR
  // remains.
  void runOnOperation() override;
};

// Makes the Sculptor Golem -> LLVM shim pass available to textual pipelines and
// pass managers.
void registerLowerGolemToLLVMShimsPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_LOWERGOLEMTOLLVMSHIMS_H
