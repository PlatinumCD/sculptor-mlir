#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_GOLEM_EXPANDMVMTOGOLEM_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_GOLEM_EXPANDMVMTOGOLEM_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>

namespace mlir {
namespace sculptor {

// Backend boundary from sculptor.mvm into Sculptor Golem execution IR. This pass
// creates logical array setup, vector load, execution, store, and recombination
// ops, but it does not lower those ops to LLVM shims or ISA intrinsics.
struct ExpandMVMToGolemPass
    : public mlir::PassWrapper<ExpandMVMToGolemPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandMVMToGolemPass)

  Option<int64_t> arrayRows{
      *this, "array-rows", llvm::cl::desc("Number of rows in the analog array"),
      llvm::cl::init(0)};

  Option<int64_t> arrayCols{
      *this, "array-cols",
      llvm::cl::desc("Number of columns in the analog array"),
      llvm::cl::init(0)};

  ExpandMVMToGolemPass() = default;

  ExpandMVMToGolemPass(const ExpandMVMToGolemPass &pass)
      : PassWrapper(pass),
        arrayRows(*this, "array-rows",
                  llvm::cl::desc("Number of rows in the analog array"),
                  llvm::cl::init(0)),
        arrayCols(*this, "array-cols",
                  llvm::cl::desc("Number of columns in the analog array"),
                  llvm::cl::init(0)) {
    arrayRows = pass.arrayRows;
    arrayCols = pass.arrayCols;
  }

  mlir::StringRef getArgument() const final {
    return "sculptor-expand-mvm-to-golem";
  }

  mlir::StringRef getDescription() const final {
    return "Expand sculptor.mvm ops into Sculptor Golem execution ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::sculptor::SculptorDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void registerExpandMVMToGolemPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_GOLEM_EXPANDMVMTOGOLEM_H
