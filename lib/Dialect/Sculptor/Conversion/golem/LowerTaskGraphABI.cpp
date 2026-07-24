#include "sculptor-mlir/Dialect/Sculptor/Conversion/golem/LowerTaskGraphABI.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphExecutionPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;

struct ArrayBinding {
  int64_t physicalArrayId;
  int64_t localArrayId;

  bool operator==(const ArrayBinding &other) const {
    return physicalArrayId == other.physicalArrayId &&
           localArrayId == other.localArrayId;
  }
};

bool returnsTaskGraph(mlir::func::FuncOp func) {
  mlir::FunctionType type = func.getFunctionType();
  return type.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(type.getResult(0));
}

bool isLogicalArrayResource(mlir::Value resource) {
  auto resourceType =
      mlir::dyn_cast<mlir::sculptor::TaskResourceType>(resource.getType());
  return resourceType && mlir::isa<mlir::sculptor::LogicalArrayType>(
                             resourceType.getValueType());
}

mlir::FailureOr<mlir::func::FuncOp>
lookupTaskCallee(mlir::ModuleOp module, mlir::sculptor::TaskCreateOp taskOp) {
  auto callee = module.lookupSymbol<mlir::func::FuncOp>(
      taskOp.getCalleeAttr().getValue());
  if (!callee)
    return taskOp.emitOpError("expected task callee '")
           << taskOp.getCalleeAttr().getValue() << "'";
  return callee;
}

mlir::FailureOr<int64_t>
getTaskOrCalleeI64Attr(mlir::ModuleOp module,
                       mlir::sculptor::TaskCreateOp taskOp,
                       llvm::StringRef attrName) {
  mlir::FailureOr<mlir::func::FuncOp> callee = lookupTaskCallee(module, taskOp);
  if (mlir::failed(callee))
    return mlir::failure();

  auto taskAttr = taskOp->getAttrOfType<mlir::IntegerAttr>(attrName);
  auto calleeAttr = (*callee)->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (taskAttr && calleeAttr && taskAttr.getInt() != calleeAttr.getInt()) {
    return taskOp.emitOpError("expected task and callee '")
           << (*callee).getSymName() << "' to agree on '" << attrName << "'";
  }
  if (taskAttr) {
    if (!calleeAttr)
      (*callee)->setAttr(attrName, taskAttr);
    return taskAttr.getInt();
  }
  if (calleeAttr) {
    taskOp->setAttr(attrName, calleeAttr);
    return calleeAttr.getInt();
  }

  return taskOp.emitOpError(
             "expected scheduled array task or callee to carry '")
         << attrName << "'";
}

mlir::FailureOr<ArrayBinding>
getArrayBinding(mlir::ModuleOp module, mlir::sculptor::TaskCreateOp taskOp) {
  mlir::FailureOr<int64_t> physicalArrayId = getTaskOrCalleeI64Attr(
      module, taskOp, runtime_attrs::kTaskPhysicalArrayIdAttrName);
  mlir::FailureOr<int64_t> localArrayId = getTaskOrCalleeI64Attr(
      module, taskOp, runtime_attrs::kTaskLocalArrayIdAttrName);
  if (mlir::failed(physicalArrayId) || mlir::failed(localArrayId))
    return mlir::failure();
  return ArrayBinding{*physicalArrayId, *localArrayId};
}

bool containsDependency(mlir::ValueRange dependencies, mlir::Value candidate) {
  for (mlir::Value dependency : dependencies)
    if (dependency == candidate)
      return true;
  return false;
}

mlir::FailureOr<unsigned>
getConvertedResultIndex(mlir::sculptor::TaskCreateOp taskOp,
                        mlir::func::FuncOp callee, unsigned oldResultIndex,
                        mlir::TypeConverter &typeConverter) {
  mlir::FunctionType calleeType = callee.getFunctionType();
  if (oldResultIndex >= calleeType.getNumResults()) {
    return taskOp.emitOpError("expected result index ")
           << oldResultIndex << " to be within callee '" << callee.getSymName()
           << "' result range";
  }

  unsigned convertedResultIndex = 0;
  for (unsigned index = 0; index < oldResultIndex; ++index) {
    llvm::SmallVector<mlir::Type, 2> convertedTypes;
    if (mlir::failed(typeConverter.convertType(calleeType.getResult(index),
                                               convertedTypes))) {
      return taskOp.emitOpError("failed to convert result ")
             << index << " of callee '" << callee.getSymName() << "'";
    }
    convertedResultIndex += convertedTypes.size();
  }

  llvm::SmallVector<mlir::Type, 2> convertedTypes;
  if (mlir::failed(typeConverter.convertType(
          calleeType.getResult(oldResultIndex), convertedTypes))) {
    return taskOp.emitOpError("failed to convert result ")
           << oldResultIndex << " of callee '" << callee.getSymName() << "'";
  }
  if (convertedTypes.size() != 1) {
    return taskOp.emitOpError("expected retained output result ")
           << oldResultIndex << " of callee '" << callee.getSymName()
           << "' to convert to exactly one runtime value";
  }

  return convertedResultIndex;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
buildConvertedResultIndices(mlir::ModuleOp module,
                            mlir::sculptor::TaskCreateOp taskOp,
                            llvm::ArrayRef<unsigned> retainedOutputIndices,
                            mlir::TypeConverter &typeConverter) {
  mlir::FailureOr<mlir::func::FuncOp> callee = lookupTaskCallee(module, taskOp);
  if (mlir::failed(callee))
    return mlir::failure();

  auto resultIndicesAttr = taskOp->getAttrOfType<mlir::ArrayAttr>(
      runtime_attrs::kTaskResultIndicesAttrName);
  if (resultIndicesAttr &&
      resultIndicesAttr.size() != taskOp.getOutputs().size()) {
    return taskOp.emitOpError("expected runtime attr '")
           << runtime_attrs::kTaskResultIndicesAttrName
           << "' to match the number of task outputs";
  }

  llvm::SmallVector<int64_t, 4> convertedResultIndices;
  convertedResultIndices.reserve(retainedOutputIndices.size());
  for (unsigned outputIndex : retainedOutputIndices) {
    unsigned oldResultIndex = outputIndex;
    if (resultIndicesAttr) {
      auto indexAttr =
          mlir::dyn_cast<mlir::IntegerAttr>(resultIndicesAttr[outputIndex]);
      if (!indexAttr || indexAttr.getInt() < 0) {
        return taskOp.emitOpError("expected runtime attr '")
               << runtime_attrs::kTaskResultIndicesAttrName
               << "' to contain non-negative integer result indices";
      }
      oldResultIndex = static_cast<unsigned>(indexAttr.getInt());
    }

    mlir::FailureOr<unsigned> convertedIndex =
        getConvertedResultIndex(taskOp, *callee, oldResultIndex, typeConverter);
    if (mlir::failed(convertedIndex))
      return mlir::failure();
    convertedResultIndices.push_back(*convertedIndex);
  }

  return convertedResultIndices;
}

mlir::LogicalResult
lowerTaskGraphFunctionABI(mlir::ModuleOp module,
                          mlir::func::FuncOp taskGraphFunc,
                          mlir::TypeConverter &typeConverter) {
  bool hadExecutionPlan =
      taskGraphFunc->hasAttr(runtime_attrs::kTaskGraphResourceCountAttrName);

  llvm::DenseMap<mlir::Value, mlir::sculptor::TaskCreateOp> producerByResource;
  for (mlir::sculptor::TaskCreateOp taskOp :
       taskGraphFunc.getOps<mlir::sculptor::TaskCreateOp>()) {
    for (mlir::Value output : taskOp.getOutputs()) {
      auto inserted = producerByResource.try_emplace(output, taskOp);
      if (!inserted.second) {
        return taskOp.emitOpError(
            "expected each task resource to have at most one producer");
      }
    }
  }

  for (mlir::sculptor::TaskCreateOp taskOp :
       taskGraphFunc.getOps<mlir::sculptor::TaskCreateOp>()) {
    llvm::SmallVector<mlir::Value, 8> retainedInputs;
    llvm::SmallVector<mlir::Value, 8> retainedOutputs;
    llvm::SmallVector<mlir::Value, 8> dependencies(taskOp.getDependencies());
    llvm::SmallVector<unsigned, 4> retainedOutputIndices;
    std::optional<mlir::Value> logicalInputResource;

    for (mlir::Value input : taskOp.getInputs()) {
      if (!isLogicalArrayResource(input)) {
        retainedInputs.push_back(input);
        continue;
      }

      if (logicalInputResource && *logicalInputResource != input) {
        return taskOp.emitOpError(
            "cannot lower a task that consumes multiple logical arrays with "
            "one local array binding");
      }
      logicalInputResource = input;

      auto producerIt = producerByResource.find(input);
      if (producerIt == producerByResource.end()) {
        return taskOp.emitOpError(
            "expected logical array input to have a matrix-setup task "
            "producer");
      }

      mlir::FailureOr<ArrayBinding> producerBinding =
          getArrayBinding(module, producerIt->second);
      mlir::FailureOr<ArrayBinding> consumerBinding =
          getArrayBinding(module, taskOp);
      if (mlir::failed(producerBinding) || mlir::failed(consumerBinding))
        return mlir::failure();
      if (!(*producerBinding == *consumerBinding)) {
        return taskOp.emitOpError(
            "expected logical array producer and consumer to share the same "
            "physical and local array binding");
      }

      mlir::Value setupDependency = producerIt->second.getResult();
      if (!containsDependency(dependencies, setupDependency))
        dependencies.push_back(setupDependency);
    }

    for (auto indexedOutput : llvm::enumerate(taskOp.getOutputs())) {
      if (isLogicalArrayResource(indexedOutput.value())) {
        if (mlir::failed(getArrayBinding(module, taskOp)))
          return mlir::failure();
        continue;
      }
      retainedOutputs.push_back(indexedOutput.value());
      retainedOutputIndices.push_back(indexedOutput.index());
    }

    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> convertedResultIndices =
        buildConvertedResultIndices(module, taskOp, retainedOutputIndices,
                                    typeConverter);
    if (mlir::failed(convertedResultIndices))
      return mlir::failure();

    taskOp.getInputsMutable().assign(retainedInputs);
    taskOp.getOutputsMutable().assign(retainedOutputs);
    taskOp.getDependenciesMutable().assign(dependencies);

    if (convertedResultIndices->empty()) {
      taskOp->removeAttr(runtime_attrs::kTaskResultIndicesAttrName);
    } else {
      mlir::Builder builder(taskOp.getContext());
      taskOp->setAttr(runtime_attrs::kTaskResultIndicesAttrName,
                      builder.getI64ArrayAttr(*convertedResultIndices));
    }
  }

  llvm::SmallVector<mlir::Operation *, 8> deadLogicalResources;
  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    auto intermediate =
        mlir::dyn_cast<mlir::sculptor::TaskGraphIntermediateOp>(&op);
    if (!intermediate || !isLogicalArrayResource(intermediate.getResult()))
      continue;
    if (!intermediate.getResult().use_empty()) {
      return intermediate.emitOpError(
          "expected logical array resource to have no uses after task ABI "
          "lowering");
    }
    deadLogicalResources.push_back(&op);
  }
  for (mlir::Operation *resourceOp : deadLogicalResources)
    resourceOp->erase();

  if (hadExecutionPlan &&
      mlir::failed(
          mlir::sculptor::rebuildTaskGraphExecutionPlan(taskGraphFunc))) {
    return taskGraphFunc.emitError(
        "failed to rebuild task graph resources after logical array lowering");
  }

  return mlir::success();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace golem {

LogicalResult lowerTaskGraphABI(ModuleOp module, TypeConverter &typeConverter) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    if (failed(lowerTaskGraphFunctionABI(module, func, typeConverter)))
      return failure();
  }
  return success();
}

} // namespace golem
} // namespace sculptor
} // namespace mlir
