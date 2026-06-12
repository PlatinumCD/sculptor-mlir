#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyStep.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <memory>

namespace {

bool isSupportedTaskResourceShapedType(mlir::ShapedType shapedType) {
  return shapedType.hasStaticShape() && shapedType.getRank() > 0 &&
         shapedType.getRank() <= 5 &&
         llvm::all_of(shapedType.getShape(),
                      [](int64_t dim) { return dim > 0; }) &&
         llvm::isa<mlir::Float32Type>(shapedType.getElementType());
}

llvm::StringRef getSupportedTaskResourceTypeDescription() {
  return "float scalar, !sculptor.logical.array, or static ranked "
         "tensor/memref<f32> with rank between 1 and 5";
}

llvm::FailureOr<mlir::sculptor::TaskResourceType>
buildTaskResourceType(mlir::Type valueType) {
  if (llvm::isa<mlir::sculptor::LogicalArrayType>(valueType)) {
    return mlir::sculptor::TaskResourceType::get(valueType.getContext(),
                                               valueType);
  }

  if (llvm::isa<mlir::FloatType>(valueType)) {
    return mlir::sculptor::TaskResourceType::get(valueType.getContext(),
                                               valueType);
  }

  if (auto rankedTensorType = llvm::dyn_cast<mlir::RankedTensorType>(valueType))
    if (isSupportedTaskResourceShapedType(rankedTensorType))
      return mlir::sculptor::TaskResourceType::get(valueType.getContext(),
                                                 valueType);

  if (auto memRefType = llvm::dyn_cast<mlir::MemRefType>(valueType))
    if (isSupportedTaskResourceShapedType(memRefType))
      return mlir::sculptor::TaskResourceType::get(valueType.getContext(),
                                                 valueType);

  return llvm::failure();
}

void collectOrderedWeightDependencies(
    mlir::ModuleOp module, mlir::Block &forwardBlock,
    llvm::SmallVectorImpl<mlir::StringAttr> &persistentSymbols) {
  persistentSymbols.clear();
  llvm::DenseSet<mlir::StringAttr> seenSymbols;

  for (mlir::Operation &op : forwardBlock) {
    auto call = llvm::dyn_cast<mlir::func::CallOp>(&op);
    if (!call)
      continue;

    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      continue;

    auto calleeFunc =
        module.lookupSymbol<mlir::func::FuncOp>(calleeAttr.getValue());
    if (!calleeFunc)
      continue;

    auto weightDependencies =
        calleeFunc->getAttrOfType<mlir::ArrayAttr>("weight_dependencies");
    if (!weightDependencies)
      continue;

    for (mlir::Attribute attr : weightDependencies) {
      auto dependencyName = llvm::dyn_cast<mlir::StringAttr>(attr);
      if (!dependencyName || !seenSymbols.insert(dependencyName).second)
        continue;

      persistentSymbols.push_back(dependencyName);
    }
  }
}

class TaskGraphResourceAssembler final
    : public mlir::sculptor::TaskGraphAssemblyStep {
public:
  mlir::StringRef getName() const final { return "TaskGraphResource"; }

  mlir::LogicalResult assemble(mlir::ModuleOp module,
                               mlir::func::FuncOp forward) const final {
    auto taskGraphFunc =
        mlir::sculptor::assembler_utils::lookupGeneratedTaskGraphFunc(module,
                                                                    forward);
    if (!taskGraphFunc) {
      forward.emitError("expected task graph scaffold to create a generator "
                        "function");
      return mlir::failure();
    }

    auto graph =
        mlir::sculptor::assembler_utils::matchTaskGraphCreateOp(taskGraphFunc);
    if (!graph) {
      taskGraphFunc.emitError("expected generated task graph function to begin "
                              "with sculptor.task_graph.create");
      return mlir::failure();
    }

    if (!forward.getBody().hasOneBlock()) {
      forward.emitError("expected forward to have a single block");
      return mlir::failure();
    }

    mlir::Block &forwardBlock = forward.getBody().front();
    auto returnOp =
        llvm::dyn_cast<mlir::func::ReturnOp>(forwardBlock.getTerminator());
    if (!returnOp) {
      forward.emitError("expected forward to terminate with func.return");
      return mlir::failure();
    }

    mlir::IRRewriter rewriter(forward.getContext());
    rewriter.setInsertionPointToEnd(&taskGraphFunc.getBody().front());

    for (mlir::BlockArgument argument : forwardBlock.getArguments()) {
      auto resourceType = buildTaskResourceType(argument.getType());
      if (failed(resourceType)) {
        forward.emitError() << "expected forward arguments to be "
                            << getSupportedTaskResourceTypeDescription();
        return mlir::failure();
      }

      auto inputResource = rewriter.create<mlir::sculptor::TaskGraphInputOp>(
          forward.getLoc(), *resourceType, graph.getResult());
      inputResource->setAttr(
          mlir::sculptor::assembler_utils::kForwardInputIndexAttrName,
          rewriter.getI64IntegerAttr(argument.getArgNumber()));
    }

    llvm::DenseSet<mlir::Value> returnedValues;
    for (auto indexedReturnValue : llvm::enumerate(returnOp.getOperands())) {
      mlir::Value returnValue = indexedReturnValue.value();
      auto resourceType = buildTaskResourceType(returnValue.getType());
      if (failed(resourceType)) {
        forward.emitError() << "expected forward results to be "
                            << getSupportedTaskResourceTypeDescription();
        return mlir::failure();
      }

      auto outputResource = rewriter.create<mlir::sculptor::TaskGraphOutputOp>(
          forward.getLoc(), *resourceType, graph.getResult());
      outputResource->setAttr(
          mlir::sculptor::assembler_utils::kForwardOutputIndexAttrName,
          rewriter.getI64IntegerAttr(indexedReturnValue.index()));
      returnedValues.insert(returnValue);
    }

    llvm::SmallVector<mlir::func::CallOp> orderedCalls;
    for (mlir::Operation &op : forwardBlock) {
      if (auto call = llvm::dyn_cast<mlir::func::CallOp>(&op))
        orderedCalls.push_back(call);
    }

    for (auto indexedCall : llvm::enumerate(orderedCalls)) {
      mlir::func::CallOp call = indexedCall.value();
      for (auto indexedResult : llvm::enumerate(call.getResults())) {
        mlir::Value result = indexedResult.value();
        if (returnedValues.contains(result) || result.use_empty())
          continue;

        auto resourceType = buildTaskResourceType(result.getType());
        if (failed(resourceType)) {
          call.emitError() << "expected task result to be "
                           << getSupportedTaskResourceTypeDescription();
          return mlir::failure();
        }

        auto temporaryResource =
            rewriter.create<mlir::sculptor::TaskGraphTemporaryOp>(
                call.getLoc(), *resourceType, graph.getResult());
        temporaryResource->setAttr(
            mlir::sculptor::assembler_utils::kForwardCallIndexAttrName,
            rewriter.getI64IntegerAttr(indexedCall.index()));
        temporaryResource->setAttr(
            mlir::sculptor::assembler_utils::kForwardResultIndexAttrName,
            rewriter.getI64IntegerAttr(indexedResult.index()));
      }
    }

    llvm::SmallVector<mlir::StringAttr> persistentSymbols;
    collectOrderedWeightDependencies(module, forwardBlock, persistentSymbols);

    auto persistentHandleType =
        mlir::sculptor::RuntimeHandleType::get(forward.getContext());
    auto persistentResourceType = mlir::sculptor::TaskResourceType::get(
        forward.getContext(), persistentHandleType);
    for (mlir::StringAttr persistentSymbol : persistentSymbols) {
      auto persistentResource =
          rewriter.create<mlir::sculptor::TaskGraphPersistentOp>(
              forward.getLoc(), persistentResourceType, graph.getResult());
      persistentResource->setAttr(
          mlir::sculptor::assembler_utils::kPersistentSymbolAttrName,
          persistentSymbol);
    }

    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerTaskGraphResourceAssembler(TaskGraphAssemblySteps &steps) {
  steps.push_back(std::make_unique<TaskGraphResourceAssembler>());
}

} // namespace sculptor
} // namespace mlir
