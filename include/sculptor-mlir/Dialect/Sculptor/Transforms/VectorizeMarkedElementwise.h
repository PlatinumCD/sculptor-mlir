#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZEMARKEDELEMENTWISE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZEMARKEDELEMENTWISE_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct VectorizeMarkedElementwisePass
    : public PassWrapper<VectorizeMarkedElementwisePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeMarkedElementwisePass)

  Option<bool> requireChange{
      *this, "require-change",
      llvm::cl::desc("Fail when no marked elementwise operation is lowered"),
      llvm::cl::init(false)};

  VectorizeMarkedElementwisePass() = default;

  VectorizeMarkedElementwisePass(const VectorizeMarkedElementwisePass &pass)
      : PassWrapper(pass),
        requireChange(
            *this, "require-change",
            llvm::cl::desc(
                "Fail when no marked elementwise operation is lowered"),
            llvm::cl::init(false)) {
    requireChange = pass.requireChange;
  }

  StringRef getArgument() const final {
    return "sculptor-vectorize-marked-elementwise";
  }

  StringRef getDescription() const final {
    return "Lower marked bufferized elementwise operations to vector loops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override;
};

void registerVectorizeMarkedElementwisePass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZEMARKEDELEMENTWISE_H
