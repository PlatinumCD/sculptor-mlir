#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANTILESCRATCHPAD_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANTILESCRATCHPAD_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct PlanTileScratchpadPass
    : public PassWrapper<PlanTileScratchpadPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanTileScratchpadPass)

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

  PlanTileScratchpadPass() = default;
  PlanTileScratchpadPass(const PlanTileScratchpadPass &pass)
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
    return "sculptor-plan-tile-scratchpad";
  }
  StringRef getDescription() const final {
    return "Select and allocate scratchpad regions across an extracted core";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};

void registerPlanTileScratchpadPass();

} // namespace mlir::sculptor

#endif
