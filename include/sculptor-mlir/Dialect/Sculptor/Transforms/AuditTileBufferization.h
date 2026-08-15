#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_AUDITTILEBUFFERIZATION_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_AUDITTILEBUFFERIZATION_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct AuditTileBufferizationPass
    : public PassWrapper<AuditTileBufferizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AuditTileBufferizationPass)

  Option<bool> strict{
      *this, "strict",
      llvm::cl::desc("Reject escaping allocations, missing deallocations, and "
                     "unplanned full-tensor copies"),
      llvm::cl::init(false)};
  Option<bool> printReport{
      *this, "print",
      llvm::cl::desc("Print the structured audit to the diagnostic stream"),
      llvm::cl::init(false)};

  AuditTileBufferizationPass() = default;
  AuditTileBufferizationPass(const AuditTileBufferizationPass &pass)
      : PassWrapper(pass),
        strict(*this, "strict",
               llvm::cl::desc("Reject escaping allocations, missing "
                              "deallocations, and unplanned full-tensor "
                              "copies"),
               llvm::cl::init(false)),
        printReport(
            *this, "print",
            llvm::cl::desc("Print the structured audit to the diagnostic "
                           "stream"),
            llvm::cl::init(false)) {
    strict = pass.strict;
    printReport = pass.printReport;
  }

  StringRef getArgument() const final {
    return "sculptor-audit-tile-bufferization";
  }
  StringRef getDescription() const final {
    return "Audit bufferized tile routines against the tile memory plan";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<SculptorDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override;
};

void registerAuditTileBufferizationPass();

} // namespace mlir::sculptor

#endif
