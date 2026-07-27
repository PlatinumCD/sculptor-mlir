#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_OPTIMIZETASKGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_OPTIMIZETASKGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir {
namespace sculptor {

struct OptimizeTaskGraphPass
    : public PassWrapper<OptimizeTaskGraphPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeTaskGraphPass)

  Option<std::string> patterns{
      *this, "patterns",
      llvm::cl::desc("Comma-separated task-graph optimization patterns"),
      llvm::cl::init("streaming-convolution")};

  Option<bool> requireChange{
      *this, "require-change",
      llvm::cl::desc("Fail when no selected optimization pattern applies"),
      llvm::cl::init(false)};

  OptimizeTaskGraphPass() = default;

  OptimizeTaskGraphPass(const OptimizeTaskGraphPass &pass)
      : PassWrapper(pass),
        patterns(
            *this, "patterns",
            llvm::cl::desc("Comma-separated task-graph optimization patterns"),
            llvm::cl::init("streaming-convolution")),
        requireChange(*this, "require-change",
                      llvm::cl::desc(
                          "Fail when no selected optimization pattern applies"),
                      llvm::cl::init(false)) {
    patterns = pass.patterns;
    requireChange = pass.requireChange;
  }

  StringRef getArgument() const final { return "sculptor-optimize-task-graph"; }

  StringRef getDescription() const final {
    return "Apply placement-preserving optimizations to a scheduled task graph";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void registerOptimizeTaskGraphPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_OPTIMIZETASKGRAPH_H
