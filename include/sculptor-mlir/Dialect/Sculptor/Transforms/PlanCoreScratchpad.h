#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANCORESCRATCHPAD_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANCORESCRATCHPAD_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct PlanCoreScratchpadPass
    : public PassWrapper<PlanCoreScratchpadPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanCoreScratchpadPass)

  Option<std::string> localMemory{
      *this, "local-memory",
      llvm::cl::desc("Local storage mode: workspace or scratchpad"),
      llvm::cl::init("scratchpad")};
  Option<int64_t> bytes{*this, "bytes",
                        llvm::cl::desc("Scratchpad capacity in bytes"),
                        llvm::cl::init(0)};
  Option<int64_t> alignment{
      *this, "alignment",
      llvm::cl::desc("Scratchpad allocation alignment in bytes"),
      llvm::cl::init(64)};
  Option<bool> doubleBufferBoundaries{
      *this, "double-buffer-boundaries",
      llvm::cl::desc("Reserve two buffers for region boundary resources"),
      llvm::cl::init(false)};

  PlanCoreScratchpadPass() = default;
  PlanCoreScratchpadPass(const PlanCoreScratchpadPass &pass)
      : PassWrapper(pass),
        localMemory(
            *this, "local-memory",
            llvm::cl::desc("Local storage mode: workspace or scratchpad"),
            llvm::cl::init("scratchpad")),
        bytes(*this, "bytes", llvm::cl::desc("Scratchpad capacity in bytes"),
              llvm::cl::init(0)),
        alignment(*this, "alignment",
                  llvm::cl::desc("Scratchpad allocation alignment in bytes"),
                  llvm::cl::init(64)),
        doubleBufferBoundaries(
            *this, "double-buffer-boundaries",
            llvm::cl::desc("Reserve two buffers for region boundary resources"),
            llvm::cl::init(false)) {
    localMemory = pass.localMemory;
    bytes = pass.bytes;
    alignment = pass.alignment;
    doubleBufferBoundaries = pass.doubleBufferBoundaries;
  }

  StringRef getArgument() const final {
    return "sculptor-plan-core-scratchpad";
  }
  StringRef getDescription() const final {
    return "Plan one resident producer-consumer region in an extracted core";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};

void registerPlanCoreScratchpadPass();

} // namespace mlir::sculptor

#endif
