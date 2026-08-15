#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_REPORTTILEMEMORY_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_REPORTTILEMEMORY_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct ReportTileMemoryPass
    : public PassWrapper<ReportTileMemoryPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportTileMemoryPass)

  Option<std::string> stage{
      *this, "stage",
      llvm::cl::desc("Report stage label; 'auto' infers it from the IR"),
      llvm::cl::init("auto")};
  Option<bool> printReport{
      *this, "print",
      llvm::cl::desc("Print the structured report to the diagnostic stream"),
      llvm::cl::init(false)};

  ReportTileMemoryPass() = default;
  ReportTileMemoryPass(const ReportTileMemoryPass &pass)
      : PassWrapper(pass),
        stage(
            *this, "stage",
            llvm::cl::desc("Report stage label; 'auto' infers it from the IR"),
            llvm::cl::init("auto")),
        printReport(*this, "print",
                    llvm::cl::desc(
                        "Print the structured report to the diagnostic stream"),
                    llvm::cl::init(false)) {
    stage = pass.stage;
    printReport = pass.printReport;
  }

  StringRef getArgument() const final { return "sculptor-report-tile-memory"; }
  StringRef getDescription() const final {
    return "Report tile-local resources, allocations, copies, and peak bytes";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};

void registerReportTileMemoryPass();

} // namespace mlir::sculptor

#endif
