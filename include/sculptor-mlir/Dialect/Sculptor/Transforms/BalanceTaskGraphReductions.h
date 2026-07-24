#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BALANCETASKGRAPHREDUCTIONS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BALANCETASKGRAPHREDUCTIONS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct BalanceTaskGraphReductionsPass
    : public PassWrapper<BalanceTaskGraphReductionsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BalanceTaskGraphReductionsPass)

  Option<int64_t> reductionWidth{
      *this, "reduction-width",
      llvm::cl::desc("Number of parallel first-stage reduction lanes"),
      llvm::cl::init(2)};

  Option<bool> requireChange{
      *this, "require-change",
      llvm::cl::desc(
          "Fail when the task graph has no eligible marked reduction"),
      llvm::cl::init(false)};

  BalanceTaskGraphReductionsPass() = default;

  BalanceTaskGraphReductionsPass(const BalanceTaskGraphReductionsPass &pass)
      : PassWrapper(pass),
        reductionWidth(
            *this, "reduction-width",
            llvm::cl::desc("Number of parallel first-stage reduction lanes"),
            llvm::cl::init(2)),
        requireChange(
            *this, "require-change",
            llvm::cl::desc(
                "Fail when the task graph has no eligible marked reduction"),
            llvm::cl::init(false)) {
    reductionWidth = pass.reductionWidth;
    requireChange = pass.requireChange;
  }

  StringRef getArgument() const final {
    return "sculptor-balance-task-graph-reductions";
  }

  StringRef getDescription() const final {
    return "Rewrite marked associative task reductions into width-controlled "
           "parallel lanes";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void registerBalanceTaskGraphReductionsPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_BALANCETASKGRAPHREDUCTIONS_H
