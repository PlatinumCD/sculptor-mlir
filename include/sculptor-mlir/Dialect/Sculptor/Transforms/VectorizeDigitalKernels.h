#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZEDIGITALKERNELS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZEDIGITALKERNELS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct VectorizeDigitalKernelsPass
    : public PassWrapper<VectorizeDigitalKernelsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeDigitalKernelsPass)

  Option<int64_t> vectorBits{
      *this, "vector-bits",
      llvm::cl::desc("Vector width for Sculptor digital kernels"),
      llvm::cl::init(256)};

  VectorizeDigitalKernelsPass() = default;
  VectorizeDigitalKernelsPass(const VectorizeDigitalKernelsPass &pass)
      : PassWrapper(pass),
        vectorBits(*this, "vector-bits",
                   llvm::cl::desc("Vector width for Sculptor digital kernels"),
                   llvm::cl::init(256)) {
    vectorBits = pass.vectorBits;
  }

  StringRef getArgument() const final {
    return "sculptor-vectorize-digital-kernels";
  }
  StringRef getDescription() const final {
    return "Lower semantic digital kernels to fixed-width vector loops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void registerVectorizeDigitalKernelsPass();

} // namespace mlir::sculptor

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZEDIGITALKERNELS_H
