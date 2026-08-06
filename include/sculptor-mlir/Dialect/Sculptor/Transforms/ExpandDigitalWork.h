#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPANDDIGITALWORK_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPANDDIGITALWORK_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>

namespace mlir {
namespace sculptor {

struct ExpandDigitalWorkPass
    : public PassWrapper<ExpandDigitalWorkPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandDigitalWorkPass)

  Option<int64_t> parallelWorkers{
      *this, "parallel-workers",
      llvm::cl::desc("Target number of independent digital work units"),
      llvm::cl::init(4)};

  Option<bool> requireChange{
      *this, "require-change",
      llvm::cl::desc("Fail when no digital operation can be expanded"),
      llvm::cl::init(false)};

  ExpandDigitalWorkPass() = default;
  ExpandDigitalWorkPass(const ExpandDigitalWorkPass &pass);

  StringRef getArgument() const final {
    return "sculptor-expand-digital-work";
  }

  StringRef getDescription() const final {
    return "Expose balanced parallel-only digital work units through "
           "TilingInterface";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override;
};

void registerExpandDigitalWorkPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPANDDIGITALWORK_H
