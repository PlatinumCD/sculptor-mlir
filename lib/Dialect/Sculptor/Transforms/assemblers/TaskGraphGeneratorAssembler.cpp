#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyStep.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>
#include <string>

namespace {

std::string makeUniqueTaskGraphFunctionName(mlir::ModuleOp module) {
  if (!module.lookupSymbol("generate_task_graph"))
    return "generate_task_graph";

  unsigned functionIndex = 0;
  std::string functionName =
      "generate_task_graph_" + std::to_string(functionIndex);
  while (module.lookupSymbol(functionName)) {
    ++functionIndex;
    functionName = "generate_task_graph_" + std::to_string(functionIndex);
  }

  return functionName;
}

class TaskGraphGeneratorAssembler final
    : public mlir::sculptor::TaskGraphAssemblyStep {
public:
  mlir::StringRef getName() const final { return "TaskGraphGenerator"; }

  mlir::LogicalResult assemble(mlir::ModuleOp module,
                               mlir::func::FuncOp forward) const final {
    std::string functionName = makeUniqueTaskGraphFunctionName(module);
    mlir::IRRewriter rewriter(forward.getContext());
    auto taskGraphType = mlir::sculptor::TaskGraphType::get(rewriter.getContext());
    auto functionType =
        rewriter.getFunctionType(llvm::ArrayRef<mlir::Type>{}, taskGraphType);

    rewriter.setInsertionPointToEnd(module.getBody());
    auto generateFunc = rewriter.create<mlir::func::FuncOp>(
        forward.getLoc(), functionName, functionType);
    generateFunc.setPrivate();

    mlir::Block *entryBlock = generateFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);
    rewriter.create<mlir::sculptor::TaskGraphCreateOp>(forward.getLoc(),
                                                     taskGraphType);

    mlir::sculptor::assembler_utils::setGeneratedTaskGraphFunc(forward,
                                                             generateFunc);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerTaskGraphGeneratorAssembler(TaskGraphAssemblySteps &steps) {
  steps.push_back(std::make_unique<TaskGraphGeneratorAssembler>());
}

} // namespace sculptor
} // namespace mlir
