#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FOLDATTENTIONLAYOUTS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FOLDATTENTIONLAYOUTS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

// Replaces attention transpose/reshape/batch-matmul chains with rank-four
// contractions whose affine indexing maps encode the required layouts.
struct FoldAttentionLayoutsPass
    : public PassWrapper<FoldAttentionLayoutsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAttentionLayoutsPass)

  StringRef getArgument() const final {
    return "sculptor-fold-attention-layouts";
  }
  StringRef getDescription() const final {
    return "Fold physical attention transposes into contraction indexing maps";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

void registerFoldAttentionLayoutsPass();

} // namespace mlir::sculptor

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FOLDATTENTIONLAYOUTS_H
