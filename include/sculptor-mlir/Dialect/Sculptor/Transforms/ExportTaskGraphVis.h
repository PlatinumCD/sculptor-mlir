#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHVIS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHVIS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir {
namespace sculptor {

// Exports assembled analog task graph dependencies to visualization formats.
struct ExportTaskGraphVisPass
    : public mlir::PassWrapper<ExportTaskGraphVisPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportTaskGraphVisPass)

  Option<std::string> output{
      *this, "output",
      llvm::cl::desc("Path where the task graph visualization file will be "
                     "written"),
      llvm::cl::init("")};

  Option<std::string> format{
      *this, "format",
      llvm::cl::desc("Task graph visualization format: dot or graphml"),
      llvm::cl::init("dot")};

  ExportTaskGraphVisPass() = default;

  ExportTaskGraphVisPass(const ExportTaskGraphVisPass &pass)
      : PassWrapper(pass),
        output(*this, "output",
               llvm::cl::desc("Path where the task graph visualization file "
                              "will be written"),
               llvm::cl::init("")),
        format(*this, "format",
               llvm::cl::desc("Task graph visualization format: dot or "
                              "graphml"),
               llvm::cl::init("dot")) {
    output = pass.output;
    format = pass.format;
  }

  mlir::StringRef getArgument() const final {
    return "sculptor-export-task-graph-vis";
  }

  mlir::StringRef getDescription() const final {
    return "Export assembled analog task graph dependencies to DOT or GraphML";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerExportTaskGraphVisPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHVIS_H
