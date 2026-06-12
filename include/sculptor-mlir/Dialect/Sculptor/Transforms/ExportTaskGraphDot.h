#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHDOT_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHDOT_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir {
namespace sculptor {

// Exports assembled analog task graph dependencies to a Graphviz DOT file.
struct ExportTaskGraphDotPass
    : public mlir::PassWrapper<ExportTaskGraphDotPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportTaskGraphDotPass)

  Option<std::string> output{
      *this, "output",
      llvm::cl::desc("Path where the task graph DOT file will be written"),
      llvm::cl::init("")};

  ExportTaskGraphDotPass() = default;

  ExportTaskGraphDotPass(const ExportTaskGraphDotPass &pass)
      : PassWrapper(pass),
        output(*this, "output",
               llvm::cl::desc(
                   "Path where the task graph DOT file will be written"),
               llvm::cl::init("")) {
    output = pass.output;
  }

  mlir::StringRef getArgument() const final {
    return "sculptor-export-task-graph-dot";
  }

  mlir::StringRef getDescription() const final {
    return "Export assembled analog task graph dependencies to DOT";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerExportTaskGraphDotPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHDOT_H
