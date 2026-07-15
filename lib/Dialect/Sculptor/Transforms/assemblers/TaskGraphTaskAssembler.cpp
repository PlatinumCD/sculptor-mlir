#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyStep.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphAssemblyUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskMetadata.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_metadata = mlir::sculptor::task_metadata;

mlir::FailureOr<task_metadata::TaskFunctionMetadata>
collectTaskMetadata(mlir::ModuleOp module, mlir::func::CallOp call) {
  auto calleeAttr = call.getCalleeAttr();
  if (!calleeAttr) {
    call.emitError("expected direct func.call with callee symbol");
    return mlir::failure();
  }

  auto calleeFunc =
      module.lookupSymbol<mlir::func::FuncOp>(calleeAttr.getValue());
  if (!calleeFunc) {
    call.emitError("could not resolve task callee '")
        << calleeAttr.getValue() << "'";
    return mlir::failure();
  }

  return task_metadata::getTaskFunctionMetadata(call, calleeFunc);
}

void collectSourceCallsFromValue(
    mlir::Value value, mlir::Block *forwardBlock,
    llvm::SetVector<mlir::func::CallOp> &sourceCalls,
    llvm::SmallDenseSet<mlir::Value, 16> &visitedValues) {
  if (!visitedValues.insert(value).second)
    return;

  auto blockArgument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (blockArgument)
    return;

  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp || definingOp->getBlock() != forwardBlock)
    return;

  if (auto sourceCall = llvm::dyn_cast<mlir::func::CallOp>(definingOp)) {
    sourceCalls.insert(sourceCall);
    return;
  }

  for (mlir::Value operand : definingOp->getOperands())
    collectSourceCallsFromValue(operand, forwardBlock, sourceCalls,
                                visitedValues);
}

llvm::FailureOr<llvm::SmallVector<llvm::StringRef>>
collectWeightDependencyNames(mlir::func::CallOp call, mlir::ModuleOp module) {
  llvm::SmallVector<llvm::StringRef> dependencyNames;

  auto calleeAttr = call.getCalleeAttr();
  if (!calleeAttr)
    return dependencyNames;

  auto calleeFunc =
      module.lookupSymbol<mlir::func::FuncOp>(calleeAttr.getValue());
  if (!calleeFunc)
    return dependencyNames;

  auto weightDependencies =
      calleeFunc->getAttrOfType<mlir::ArrayAttr>("weight_dependencies");
  if (!weightDependencies)
    return dependencyNames;

  dependencyNames.reserve(weightDependencies.size());
  for (mlir::Attribute attr : weightDependencies) {
    auto dependencyName = llvm::dyn_cast<mlir::StringAttr>(attr);
    if (!dependencyName) {
      call.emitError("expected weight_dependencies to be an array of "
                     "string attrs");
      return mlir::failure();
    }

    dependencyNames.push_back(dependencyName.getValue());
  }

  return dependencyNames;
}

llvm::FailureOr<llvm::SmallVector<mlir::func::CallOp>>
lookupWeightDependencyCalls(
    mlir::func::CallOp call, mlir::ModuleOp module,
    const llvm::StringMap<mlir::func::CallOp> &callByCallee) {
  auto dependencyNames = collectWeightDependencyNames(call, module);
  if (failed(dependencyNames))
    return mlir::failure();

  llvm::SmallVector<mlir::func::CallOp> dependencyCalls;
  dependencyCalls.reserve(dependencyNames->size());
  for (llvm::StringRef dependencyName : *dependencyNames) {
    auto dependencyCallIt = callByCallee.find(dependencyName);
    if (dependencyCallIt == callByCallee.end()) {
      call.emitError("could not find forward call for weight dependency '")
          << dependencyName << "'";
      return mlir::failure();
    }

    dependencyCalls.push_back(dependencyCallIt->second);
  }

  return dependencyCalls;
}

llvm::FailureOr<llvm::SmallVector<mlir::Value>>
lookupWeightDependencyPersistentInputs(
    mlir::func::CallOp call, mlir::ModuleOp module,
    const llvm::StringMap<mlir::Value> &persistentResourceBySymbol) {
  auto dependencyNames = collectWeightDependencyNames(call, module);
  if (failed(dependencyNames))
    return mlir::failure();

  llvm::SmallVector<mlir::Value> inputResources;
  inputResources.reserve(dependencyNames->size());
  for (llvm::StringRef dependencyName : *dependencyNames) {
    auto persistentIt = persistentResourceBySymbol.find(dependencyName);
    if (persistentIt == persistentResourceBySymbol.end()) {
      call.emitError("could not find persistent resource for weight "
                     "dependency '")
          << dependencyName << "'";
      return mlir::failure();
    }

    inputResources.push_back(persistentIt->second);
  }

  return inputResources;
}

llvm::FailureOr<llvm::SmallVector<mlir::Value>> collectTaskDependencies(
    mlir::func::CallOp call, mlir::ModuleOp module, mlir::Block *forwardBlock,
    llvm::DenseMap<mlir::Operation *, mlir::Value> &taskByCall,
    llvm::DenseMap<mlir::Operation *, unsigned> &callOrder,
    llvm::StringMap<mlir::func::CallOp> &callByCallee) {
  llvm::SetVector<mlir::func::CallOp> sourceCalls;
  llvm::SmallDenseSet<mlir::Value, 16> visitedValues;

  for (mlir::Value operand : call.getOperands())
    collectSourceCallsFromValue(operand, forwardBlock, sourceCalls,
                                visitedValues);

  auto weightDependencyCalls =
      lookupWeightDependencyCalls(call, module, callByCallee);
  if (failed(weightDependencyCalls))
    return mlir::failure();
  for (mlir::func::CallOp dependencyCall : *weightDependencyCalls)
    sourceCalls.insert(dependencyCall);

  llvm::SmallVector<mlir::func::CallOp> orderedSourceCalls(sourceCalls.begin(),
                                                           sourceCalls.end());
  std::sort(orderedSourceCalls.begin(), orderedSourceCalls.end(),
            [&](mlir::func::CallOp lhs, mlir::func::CallOp rhs) {
              return callOrder.lookup(lhs.getOperation()) <
                     callOrder.lookup(rhs.getOperation());
            });

  llvm::SmallVector<mlir::Value> dependencies;
  for (mlir::func::CallOp sourceCall : orderedSourceCalls) {
    auto taskIt = taskByCall.find(sourceCall.getOperation());
    if (taskIt == taskByCall.end()) {
      call.emitError("expected task dependency to be created before use");
      return mlir::failure();
    }

    dependencies.push_back(taskIt->second);
  }

  return dependencies;
}

mlir::FailureOr<unsigned> getIndexAttrValue(mlir::Operation *op,
                                            llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr) {
    op->emitError("expected assembly bookkeeping attr '") << attrName << "'";
    return mlir::failure();
  }

  int64_t value = attr.getInt();
  if (value < 0) {
    op->emitError("expected non-negative assembly bookkeeping attr '")
        << attrName << "'";
    return mlir::failure();
  }

  return static_cast<unsigned>(value);
}

mlir::FailureOr<llvm::StringMap<mlir::Value>>
collectPersistentResourceBySymbol(mlir::func::FuncOp taskGraphFunc) {
  llvm::StringMap<mlir::Value> resourceBySymbol;

  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    auto persistentResource =
        llvm::dyn_cast<mlir::sculptor::TaskGraphPersistentOp>(&op);
    if (!persistentResource)
      continue;

    auto persistentSymbol = op.getAttrOfType<mlir::StringAttr>(
        mlir::sculptor::assembler_utils::kPersistentSymbolAttrName);
    if (!persistentSymbol || persistentSymbol.getValue().empty()) {
      op.emitError("expected persistent resource to carry assembly bookkeeping "
                   "symbol attr");
      return mlir::failure();
    }

    auto insertIt = resourceBySymbol.try_emplace(
        persistentSymbol.getValue(), persistentResource.getResult());
    if (!insertIt.second) {
      op.emitError("expected persistent resource symbols to be unique");
      return mlir::failure();
    }
  }

  return resourceBySymbol;
}

void collectOrderedCalls(
    mlir::Block &forwardBlock,
    llvm::SmallVectorImpl<mlir::func::CallOp> &orderedCalls,
    llvm::DenseMap<mlir::Operation *, unsigned> &callOrder,
    llvm::StringMap<mlir::func::CallOp> &callByCallee) {
  orderedCalls.clear();
  callOrder.clear();
  callByCallee.clear();

  for (mlir::Operation &op : forwardBlock) {
    auto call = llvm::dyn_cast<mlir::func::CallOp>(&op);
    if (!call)
      continue;

    callOrder[call.getOperation()] = orderedCalls.size();
    if (auto calleeAttr = call.getCalleeAttr())
      callByCallee.try_emplace(calleeAttr.getValue(), call);
    orderedCalls.push_back(call);
  }
}

mlir::LogicalResult collectResourceByValue(
    mlir::func::FuncOp forward, mlir::func::ReturnOp returnOp,
    llvm::ArrayRef<mlir::func::CallOp> orderedCalls,
    mlir::func::FuncOp taskGraphFunc,
    llvm::DenseMap<mlir::Value, mlir::Value> &resourceByValue) {
  resourceByValue.clear();
  mlir::Block &forwardBlock = forward.getBody().front();

  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    if (auto inputResource =
            llvm::dyn_cast<mlir::sculptor::TaskGraphInputOp>(&op)) {
      auto inputIndex = getIndexAttrValue(
          &op, mlir::sculptor::assembler_utils::kForwardInputIndexAttrName);
      if (failed(inputIndex))
        return mlir::failure();
      if (*inputIndex >= forwardBlock.getNumArguments()) {
        op.emitError("forward input index out of range");
        return mlir::failure();
      }

      resourceByValue.try_emplace(forwardBlock.getArgument(*inputIndex),
                                  inputResource.getResult());
      continue;
    }

    if (auto outputResource =
            llvm::dyn_cast<mlir::sculptor::TaskGraphOutputOp>(&op)) {
      auto outputIndex = getIndexAttrValue(
          &op, mlir::sculptor::assembler_utils::kForwardOutputIndexAttrName);
      if (failed(outputIndex))
        return mlir::failure();
      if (*outputIndex >= returnOp.getNumOperands()) {
        op.emitError("forward output index out of range");
        return mlir::failure();
      }

      resourceByValue.try_emplace(returnOp.getOperand(*outputIndex),
                                  outputResource.getResult());
      continue;
    }

    if (auto intermediateResource =
            llvm::dyn_cast<mlir::sculptor::TaskGraphIntermediateOp>(&op)) {
      auto callIndex = getIndexAttrValue(
          &op, mlir::sculptor::assembler_utils::kForwardCallIndexAttrName);
      if (failed(callIndex))
        return mlir::failure();

      auto resultIndex = getIndexAttrValue(
          &op, mlir::sculptor::assembler_utils::kForwardResultIndexAttrName);
      if (failed(resultIndex))
        return mlir::failure();

      if (*callIndex >= orderedCalls.size()) {
        op.emitError("forward call index out of range");
        return mlir::failure();
      }

      mlir::func::CallOp call = orderedCalls[*callIndex];
      if (*resultIndex >= call.getNumResults()) {
        op.emitError("forward result index out of range");
        return mlir::failure();
      }

      resourceByValue[call.getResult(*resultIndex)] =
          intermediateResource.getResult();
    }
  }

  return mlir::success();
}

class TaskGraphTaskAssembler final
    : public mlir::sculptor::TaskGraphAssemblyStep {
public:
  mlir::StringRef getName() const final { return "TaskGraphTask"; }

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

    llvm::SmallVector<mlir::func::CallOp> orderedCalls;
    llvm::DenseMap<mlir::Operation *, unsigned> callOrder;
    llvm::StringMap<mlir::func::CallOp> callByCallee;
    collectOrderedCalls(forwardBlock, orderedCalls, callOrder, callByCallee);

    llvm::DenseMap<mlir::Value, mlir::Value> resourceByValue;
    if (failed(collectResourceByValue(forward, returnOp, orderedCalls,
                                      taskGraphFunc, resourceByValue)))
      return mlir::failure();

    auto persistentResourceBySymbol =
        collectPersistentResourceBySymbol(taskGraphFunc);
    if (failed(persistentResourceBySymbol))
      return mlir::failure();

    mlir::IRRewriter rewriter(forward.getContext());
    rewriter.setInsertionPointToEnd(&taskGraphFunc.getBody().front());

    llvm::DenseMap<mlir::Operation *, mlir::Value> taskByCall;
    for (mlir::func::CallOp call : orderedCalls) {
      auto calleeAttr = call.getCalleeAttr();
      if (!calleeAttr) {
        call.emitError("expected direct func.call with callee symbol");
        return mlir::failure();
      }

      auto taskMetadata = collectTaskMetadata(module, call);
      if (failed(taskMetadata))
        return mlir::failure();

      if (auto calleeFunc =
              module.lookupSymbol<mlir::func::FuncOp>(calleeAttr.getValue())) {
        calleeFunc->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
      }

      llvm::FailureOr<llvm::SmallVector<mlir::Value>> dependencies =
          collectTaskDependencies(call, module, &forwardBlock, taskByCall,
                                  callOrder, callByCallee);
      if (failed(dependencies))
        return mlir::failure();

      llvm::SmallVector<mlir::Value> inputResources;
      inputResources.reserve(call.getNumOperands());
      for (mlir::Value operand : call.getOperands()) {
        auto resourceIt = resourceByValue.find(operand);
        if (resourceIt == resourceByValue.end()) {
          call.emitError("expected operand resource to be created before use");
          return mlir::failure();
        }

        inputResources.push_back(resourceIt->second);
      }

      auto weightDependencyInputs = lookupWeightDependencyPersistentInputs(
          call, module, *persistentResourceBySymbol);
      if (failed(weightDependencyInputs))
        return mlir::failure();
      inputResources.append(weightDependencyInputs->begin(),
                            weightDependencyInputs->end());

      llvm::SmallVector<mlir::Value> outputResources;
      llvm::SmallVector<int64_t> outputResultIndices;
      outputResources.reserve(call.getNumResults());
      outputResultIndices.reserve(call.getNumResults());
      for (auto indexedResult : llvm::enumerate(call.getResults())) {
        auto resourceIt = resourceByValue.find(indexedResult.value());
        if (resourceIt == resourceByValue.end())
          continue;

        outputResources.push_back(resourceIt->second);
        outputResultIndices.push_back(
            static_cast<int64_t>(indexedResult.index()));
      }

      auto task = rewriter.create<mlir::sculptor::TaskCreateOp>(
          call.getLoc(), mlir::sculptor::TaskType::get(rewriter.getContext()),
          graph.getResult(), calleeAttr, taskMetadata->domain,
          taskMetadata->taskKind, taskMetadata->taskName,
          taskMetadata->sourceLayer, taskMetadata->sourceTaskOrdinal,
          inputResources, outputResources, *dependencies);
      if (!outputResultIndices.empty())
        task->setAttr(runtime_attrs::kTaskResultIndicesAttrName,
                      rewriter.getI64ArrayAttr(outputResultIndices));
      taskByCall[call.getOperation()] = task.getResult();
    }

    rewriter.create<mlir::func::ReturnOp>(forward.getLoc(), graph.getResult());
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerTaskGraphTaskAssembler(TaskGraphAssemblySteps &steps) {
  steps.push_back(std::make_unique<TaskGraphTaskAssembler>());
}

} // namespace sculptor
} // namespace mlir
