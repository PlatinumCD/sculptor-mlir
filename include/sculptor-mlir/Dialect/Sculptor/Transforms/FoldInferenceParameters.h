#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FOLDINFERENCEPARAMETERS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FOLDINFERENCEPARAMETERS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct FoldInferenceParametersPass
    : public PassWrapper<FoldInferenceParametersPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldInferenceParametersPass)

  StringRef getArgument() const final {
    return "sculptor-fold-inference-parameters";
  }
  StringRef getDescription() const final {
    return "Fold constant inference parameter graphs into weighted layers";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

void registerFoldInferenceParametersPass();

} // namespace mlir::sculptor

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FOLDINFERENCEPARAMETERS_H
