#include "sculptor-mlir/Dialect/Sculptor/Conversion/golem/LowerTaskGraphABI.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphExecutionPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
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

struct OrderedArrayBinding {
  unsigned inputIndex;
  ArrayBinding binding;
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

mlir::FailureOr<std::optional<mlir::ArrayAttr>>
getOrderedArrayBindingsAttr(mlir::ModuleOp module,
                            mlir::sculptor::TaskCreateOp taskOp) {
  mlir::FailureOr<mlir::func::FuncOp> callee = lookupTaskCallee(module, taskOp);
  if (mlir::failed(callee))
    return mlir::failure();

  auto taskBindings = taskOp->getAttrOfType<mlir::ArrayAttr>(
      runtime_attrs::kTaskArrayBindingsAttrName);
  auto calleeBindings = (*callee)->getAttrOfType<mlir::ArrayAttr>(
      runtime_attrs::kTaskArrayBindingsAttrName);
  if (taskBindings && calleeBindings && taskBindings != calleeBindings) {
    return taskOp.emitOpError("expected task and callee '")
           << (*callee).getSymName() << "' to agree on '"
           << runtime_attrs::kTaskArrayBindingsAttrName << "'";
  }
  if (taskBindings) {
    if (!calleeBindings)
      (*callee)->setAttr(runtime_attrs::kTaskArrayBindingsAttrName,
                         taskBindings);
    return std::optional<mlir::ArrayAttr>(taskBindings);
  }
  if (calleeBindings) {
    taskOp->setAttr(runtime_attrs::kTaskArrayBindingsAttrName, calleeBindings);
    return std::optional<mlir::ArrayAttr>(calleeBindings);
  }
  return std::optional<mlir::ArrayAttr>();
}

mlir::FailureOr<llvm::SmallVector<OrderedArrayBinding, 4>>
parseOrderedArrayBindings(mlir::ModuleOp module,
                          mlir::sculptor::TaskCreateOp taskOp) {
  auto bindingsAttr = getOrderedArrayBindingsAttr(module, taskOp);
  if (mlir::failed(bindingsAttr))
    return mlir::failure();
  if (!*bindingsAttr)
    return llvm::SmallVector<OrderedArrayBinding, 4>{};

  llvm::SmallSet<unsigned, 4> seenInputIndices;
  llvm::SmallVector<OrderedArrayBinding, 4> bindings;
  bindings.reserve((**bindingsAttr).size());
  for (mlir::Attribute attribute : **bindingsAttr) {
    auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
    if (!dictionary) {
      return taskOp.emitOpError("expected '")
             << runtime_attrs::kTaskArrayBindingsAttrName
             << "' to contain dictionary attributes";
    }
    auto inputIndexAttr = dictionary.getAs<mlir::IntegerAttr>(
        runtime_attrs::kArrayBindingInputIndexFieldName);
    auto physicalArrayAttr = dictionary.getAs<mlir::IntegerAttr>(
        runtime_attrs::kArrayBindingPhysicalIdFieldName);
    auto localArrayAttr = dictionary.getAs<mlir::IntegerAttr>(
        runtime_attrs::kArrayBindingLocalIdFieldName);
    if (!inputIndexAttr || !physicalArrayAttr || !localArrayAttr ||
        inputIndexAttr.getInt() < 0 || physicalArrayAttr.getInt() < 0 ||
        localArrayAttr.getInt() < 0 ||
        static_cast<uint64_t>(inputIndexAttr.getInt()) >
            std::numeric_limits<unsigned>::max()) {
      return taskOp.emitOpError("expected valid non-negative fields in '")
             << runtime_attrs::kTaskArrayBindingsAttrName << "'";
    }

    unsigned inputIndex = static_cast<unsigned>(inputIndexAttr.getInt());
    if (inputIndex >= taskOp.getInputs().size() ||
        !isLogicalArrayResource(taskOp.getInputs()[inputIndex])) {
      return taskOp.emitOpError("array binding input index ")
             << inputIndex << " does not identify a logical-array input";
    }
    if (!seenInputIndices.insert(inputIndex).second) {
      return taskOp.emitOpError("duplicate array binding for input index ")
             << inputIndex;
    }
    bindings.push_back(
        OrderedArrayBinding{inputIndex, ArrayBinding{physicalArrayAttr.getInt(),
                                                     localArrayAttr.getInt()}});
  }

  llvm::sort(bindings, [](const OrderedArrayBinding &lhs,
                          const OrderedArrayBinding &rhs) {
    return lhs.inputIndex < rhs.inputIndex;
  });
  return bindings;
}

const OrderedArrayBinding *
findOrderedArrayBinding(llvm::ArrayRef<OrderedArrayBinding> bindings,
                        unsigned inputIndex) {
  for (const OrderedArrayBinding &binding : bindings)
    if (binding.inputIndex == inputIndex)
      return &binding;
  return nullptr;
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
    unsigned logicalInputCount =
        llvm::count_if(taskOp.getInputs(), [](mlir::Value input) {
          return isLogicalArrayResource(input);
        });
    auto orderedBindings = parseOrderedArrayBindings(module, taskOp);
    if (mlir::failed(orderedBindings))
      return mlir::failure();
    if (!orderedBindings->empty() &&
        orderedBindings->size() != logicalInputCount) {
      return taskOp.emitOpError("expected '")
             << runtime_attrs::kTaskArrayBindingsAttrName
             << "' to contain one entry per logical-array input";
    }
    if (logicalInputCount > 1 && orderedBindings->empty()) {
      return taskOp.emitOpError(
          "expected ordered array bindings for a task consuming multiple "
          "logical arrays");
    }

    for (auto indexedInput : llvm::enumerate(taskOp.getInputs())) {
      mlir::Value input = indexedInput.value();
      if (!isLogicalArrayResource(input)) {
        retainedInputs.push_back(input);
        continue;
      }

      auto producerIt = producerByResource.find(input);
      if (producerIt == producerByResource.end()) {
        return taskOp.emitOpError(
            "expected logical array input to have a matrix-setup task "
            "producer");
      }

      mlir::FailureOr<ArrayBinding> producerBinding =
          getArrayBinding(module, producerIt->second);
      if (mlir::failed(producerBinding))
        return mlir::failure();

      ArrayBinding consumerBinding;
      if (orderedBindings->empty()) {
        auto scalarBinding = getArrayBinding(module, taskOp);
        if (mlir::failed(scalarBinding))
          return mlir::failure();
        consumerBinding = *scalarBinding;
      } else {
        const OrderedArrayBinding *orderedBinding =
            findOrderedArrayBinding(*orderedBindings, indexedInput.index());
        if (!orderedBinding) {
          return taskOp.emitOpError(
                     "missing ordered array binding for logical input ")
                 << indexedInput.index();
        }
        consumerBinding = orderedBinding->binding;
      }
      if (!(*producerBinding == consumerBinding)) {
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
    if (!orderedBindings->empty()) {
      taskOp->removeAttr(runtime_attrs::kTaskArrayBindingsAttrName);
      auto callee = lookupTaskCallee(module, taskOp);
      if (mlir::failed(callee))
        return mlir::failure();
      (*callee)->removeAttr(runtime_attrs::kTaskArrayBindingsAttrName);
    }

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
