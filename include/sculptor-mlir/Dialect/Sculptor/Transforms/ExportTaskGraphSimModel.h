#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHSIMMODEL_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHSIMMODEL_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir {
namespace sculptor {

// Exports a scheduled analog task graph as a structured model for external
// placement and runtime simulation.
struct ExportTaskGraphSimModelPass
    : public mlir::PassWrapper<ExportTaskGraphSimModelPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportTaskGraphSimModelPass)

  Option<std::string> output{
      *this, "output",
      llvm::cl::desc("Path where the task graph simulation model will be "
                     "written"),
      llvm::cl::init("")};

  ExportTaskGraphSimModelPass() = default;

  ExportTaskGraphSimModelPass(const ExportTaskGraphSimModelPass &pass)
      : PassWrapper(pass),
        output(*this, "output",
               llvm::cl::desc("Path where the task graph simulation model "
                              "will be written"),
               llvm::cl::init("")) {
    output = pass.output;
  }

  mlir::StringRef getArgument() const final {
    return "sculptor-export-task-graph-sim-model";
  }

  mlir::StringRef getDescription() const final {
    return "Export scheduled analog task graph as a structured simulation model";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerExportTaskGraphSimModelPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHSIMMODEL_H
