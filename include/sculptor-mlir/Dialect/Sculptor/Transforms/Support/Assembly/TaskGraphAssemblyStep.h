#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_ASSEMBLY_TASKGRAPHASSEMBLYSTEP_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_ASSEMBLY_TASKGRAPHASSEMBLYSTEP_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <vector>

namespace mlir {
namespace sculptor {

class TaskGraphAssemblyStep {
public:
  virtual ~TaskGraphAssemblyStep() = default;

  virtual StringRef getName() const = 0;

  virtual LogicalResult assemble(ModuleOp module, func::FuncOp forward) const = 0;
};

using TaskGraphAssemblySteps =
    std::vector<std::unique_ptr<TaskGraphAssemblyStep>>;

void registerTaskGraphGeneratorAssembler(TaskGraphAssemblySteps &steps);
void registerTaskGraphResourceAssembler(TaskGraphAssemblySteps &steps);
void registerTaskGraphTaskAssembler(TaskGraphAssemblySteps &steps);
void registerTaskGraphExecutionPlanAssembler(TaskGraphAssemblySteps &steps);

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_ASSEMBLY_TASKGRAPHASSEMBLYSTEP_H
