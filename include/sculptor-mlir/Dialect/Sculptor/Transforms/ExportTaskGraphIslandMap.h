#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHISLANDMAP_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHISLANDMAP_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir {
namespace sculptor {

struct ExportTaskGraphIslandMapPass
    : public PassWrapper<ExportTaskGraphIslandMapPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportTaskGraphIslandMapPass)

  Option<std::string> output{
      *this, "output",
      llvm::cl::desc("Path where the task-index to island-id map will be "
                     "written"),
      llvm::cl::init("")};

  ExportTaskGraphIslandMapPass() = default;

  ExportTaskGraphIslandMapPass(const ExportTaskGraphIslandMapPass &pass)
      : PassWrapper(pass),
        output(*this, "output",
               llvm::cl::desc("Path where the task-index to island-id map "
                              "will be written"),
               llvm::cl::init("")) {
    output = pass.output;
  }

  StringRef getArgument() const final {
    return "sculptor-export-task-graph-island-map";
  }

  StringRef getDescription() const final {
    return "Export task graph DAG order and logical island membership";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerExportTaskGraphIslandMapPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPORTTASKGRAPHISLANDMAP_H
