#include "sculptor-mlir/Dialect/Sculptor/Transforms/ScheduleTaskGraph.h"

// ScheduleTaskGraph makes the target hardware budget explicit in IR and runs
// the selected task scheduler to assign task indices, cores, and arrays.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Assembly/TaskGraphExecutionPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScorer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace {

namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_schedulers = mlir::sculptor::task_schedulers;

bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

mlir::FailureOr<task_schedulers::HardwareBudget>
buildHardwareBudget(mlir::ModuleOp module, int64_t numCores,
                    int64_t arraysPerCore, llvm::StringRef topology,
                    int64_t meshRows, int64_t meshCols, int64_t randomSeed) {
  if (numCores <= 0) {
    module.emitError("expected Sculptor scheduling budget to have at least one "
                     "core");
    return mlir::failure();
  }

  if (arraysPerCore <= 0) {
    module.emitError("expected Sculptor scheduling budget to have at least one "
                     "array per core");
    return mlir::failure();
  }

  if (randomSeed < 0) {
    module.emitError("expected Sculptor random scheduling seed to be "
                     "non-negative");
    return mlir::failure();
  }

  if (topology != "mesh") {
    module.emitError("unknown Sculptor scheduling topology '")
        << topology << "'";
    return mlir::failure();
  }

  if (meshRows <= 0) {
    module.emitError("expected mesh topology to have at least one row");
    return mlir::failure();
  }

  if (meshCols < 0) {
    module.emitError("expected mesh topology column count to be non-negative");
    return mlir::failure();
  }

  if (meshCols == 0) {
    if (numCores % meshRows != 0) {
      module.emitError("expected mesh topology rows to evenly divide the "
                       "number of cores when mesh-cols is inferred");
      return mlir::failure();
    }
    meshCols = numCores / meshRows;
  }

  if (meshCols <= 0) {
    module.emitError("expected mesh topology to have at least one column");
    return mlir::failure();
  }

  if (meshRows > std::numeric_limits<int64_t>::max() / meshCols) {
    module.emitError("Sculptor scheduling mesh topology overflows core count");
    return mlir::failure();
  }

  if (meshRows * meshCols != numCores) {
    module.emitError("expected mesh topology dimensions to match the number "
                     "of cores");
    return mlir::failure();
  }

  if (numCores > std::numeric_limits<int64_t>::max() / arraysPerCore) {
    module.emitError("Sculptor scheduling budget overflows total array count");
    return mlir::failure();
  }

  task_schedulers::HardwareBudget budget;
  budget.numCores = numCores;
  budget.arraysPerCore = arraysPerCore;
  budget.topology = topology.str();
  budget.meshRows = meshRows;
  budget.meshCols = meshCols;
  budget.numAnalogArrays = numCores * arraysPerCore;
  budget.randomSeed = randomSeed;
  budget.analogArrays.reserve(static_cast<size_t>(budget.numAnalogArrays));

  for (int64_t analogArray = 0; analogArray < budget.numAnalogArrays;
       ++analogArray)
    budget.analogArrays.push_back(analogArray);

  return budget;
}

mlir::ArrayAttr buildI64ArrayAttr(mlir::Builder &builder,
                                  llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

void attachBudgetAttrs(mlir::Operation *op, mlir::Builder &builder,
                       const task_schedulers::HardwareBudget &budget) {
  op->setAttr(schedule_attrs::kNumCoresAttrName,
              builder.getI64IntegerAttr(budget.numCores));
  op->setAttr(schedule_attrs::kArraysPerCoreAttrName,
              builder.getI64IntegerAttr(budget.arraysPerCore));
  op->setAttr(schedule_attrs::kTopologyAttrName,
              builder.getStringAttr(budget.topology));
  op->setAttr(schedule_attrs::kMeshRowsAttrName,
              builder.getI64IntegerAttr(budget.meshRows));
  op->setAttr(schedule_attrs::kMeshColsAttrName,
              builder.getI64IntegerAttr(budget.meshCols));
  op->setAttr(schedule_attrs::kNumAnalogArraysAttrName,
              builder.getI64IntegerAttr(budget.numAnalogArrays));
  op->setAttr(schedule_attrs::kAnalogArraysAttrName,
              buildI64ArrayAttr(builder, budget.analogArrays));
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_attrs = mlir::sculptor::task_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;

static bool isLogicalArrayResource(Value value) {
  auto resourceType = dyn_cast<sculptor::TaskResourceType>(value.getType());
  return resourceType &&
         isa<sculptor::LogicalArrayType>(resourceType.getValueType());
}

static FailureOr<func::FuncOp> lookupTaskCallee(ModuleOp module,
                                                sculptor::TaskCreateOp taskOp,
                                                StringRef placementKind) {
  auto callee =
      module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
  if (!callee) {
    return taskOp.emitError("expected task callee '")
           << taskOp.getCalleeAttr().getValue()
           << "' to resolve to a function for " << placementKind
           << " placement";
  }

  return callee;
}

static std::string buildUniqueSymbolName(ModuleOp module, StringRef baseName) {
  if (!module.lookupSymbol(baseName))
    return baseName.str();

  unsigned index = 0;
  std::string candidate = (baseName + "_" + Twine(index)).str();
  while (module.lookupSymbol(candidate)) {
    ++index;
    candidate = (baseName + "_" + Twine(index)).str();
  }
  return candidate;
}

static FailureOr<func::FuncOp>
cloneParentTaskCalleeForFusion(ModuleOp module,
                               sculptor::TaskCreateOp parentTask,
                               func::FuncOp parentFunc) {
  std::string cloneBaseName = (parentFunc.getSymName() + "_fusion").str();
  std::string cloneName = buildUniqueSymbolName(module, cloneBaseName);

  func::FuncOp clonedFunc = parentFunc.clone();
  clonedFunc.setSymName(cloneName);

  Builder builder(module.getContext());
  OpBuilder opBuilder(module.getContext());
  opBuilder.setInsertionPointAfter(parentFunc);
  opBuilder.insert(clonedFunc);

  parentTask.setCalleeAttr(
      FlatSymbolRefAttr::get(builder.getContext(), cloneName));

  if (SymbolTable::symbolKnownUseEmpty(parentFunc, module))
    parentFunc.erase();

  return clonedFunc;
}

static void attachCorePlacementAttrs(Operation *op, Builder &builder,
                                     int64_t coreId) {
  op->setAttr(runtime_attrs::kTaskCoreIdAttrName,
              builder.getI64IntegerAttr(coreId));
}

static void
attachAnalogArrayPlacementAttrs(Operation *op, Builder &builder,
                                const PhysicalArrayPlacement &placement) {
  attachCorePlacementAttrs(op, builder, placement.coreId);
  op->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
              builder.getI64IntegerAttr(placement.physicalArrayId));
  op->setAttr(runtime_attrs::kTaskLocalArrayIdAttrName,
              builder.getI64IntegerAttr(placement.localArrayId));
}

static bool hasDependency(sculptor::TaskCreateOp taskOp, Value dependency) {
  for (Value candidate : taskOp.getDependencies()) {
    if (candidate == dependency)
      return true;
  }
  return false;
}

static bool isMatrixSetupMVMPair(sculptor::TaskCreateOp parentTask,
                                 sculptor::TaskCreateOp childTask) {
  return parentTask.getTaskKind() == task_graph_names::kMatrixSetupTaskKind &&
         childTask.getTaskKind() == task_graph_names::kMVMTaskKind;
}

static FailureOr<func::ReturnOp> getSingleBlockReturn(func::FuncOp func,
                                                      Operation *diagnosticOp) {
  if (!func || func.isDeclaration() || !func.getBody().hasOneBlock()) {
    diagnosticOp->emitError("expected task callee to have one block for task "
                            "fusion");
    return failure();
  }

  auto returnOp =
      dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
  if (!returnOp) {
    diagnosticOp->emitError(
        "expected task callee to terminate with func.return "
        "for task fusion");
    return failure();
  }

  return returnOp;
}

static Value lookupParentTaskValue(sculptor::TaskCreateOp parentTask,
                                   func::FuncOp parentFunc,
                                   func::ReturnOp parentReturn,
                                   Value resource) {
  Block &entryBlock = parentFunc.getBody().front();
  for (auto indexedInput : llvm::enumerate(parentTask.getInputs())) {
    if (indexedInput.value() == resource)
      return entryBlock.getArgument(indexedInput.index());
  }

  for (auto indexedOutput : llvm::enumerate(parentTask.getOutputs())) {
    if (indexedOutput.value() == resource)
      return parentReturn.getOperand(indexedOutput.index());
  }

  return {};
}

static Value appendParentTaskInput(sculptor::TaskCreateOp parentTask,
                                   func::FuncOp parentFunc, Value resource,
                                   Type argumentType, Location loc,
                                   Builder &builder) {
  SmallVector<Value> inputs(parentTask.getInputs().begin(),
                            parentTask.getInputs().end());
  inputs.push_back(resource);
  parentTask.getInputsMutable().assign(inputs);

  Block &entryBlock = parentFunc.getBody().front();
  entryBlock.addArgument(argumentType, loc);

  auto oldType = parentFunc.getFunctionType();
  SmallVector<Type> inputTypes(oldType.getInputs().begin(),
                               oldType.getInputs().end());
  inputTypes.push_back(argumentType);
  parentFunc.setType(builder.getFunctionType(inputTypes, oldType.getResults()));
  return entryBlock.getArgument(entryBlock.getNumArguments() - 1);
}

static LogicalResult mapChildInputsToParentValues(
    sculptor::TaskCreateOp parentTask, func::FuncOp parentFunc,
    func::ReturnOp parentReturn, sculptor::TaskCreateOp childTask,
    func::FuncOp childFunc, IRMapping &mapping,
    SmallVectorImpl<unsigned> &appendedChildInputIndices, Builder &builder) {
  Block &childEntryBlock = childFunc.getBody().front();
  if (childEntryBlock.getNumArguments() != childTask.getInputs().size()) {
    childTask.emitError("expected child task input count to match child callee "
                        "argument count for task fusion");
    return failure();
  }

  for (auto indexedInput : llvm::enumerate(childTask.getInputs())) {
    Value mappedValue = lookupParentTaskValue(
        parentTask, parentFunc, parentReturn, indexedInput.value());
    if (!mappedValue) {
      mappedValue = appendParentTaskInput(
          parentTask, parentFunc, indexedInput.value(),
          childEntryBlock.getArgument(indexedInput.index()).getType(),
          childEntryBlock.getArgument(indexedInput.index()).getLoc(), builder);
      appendedChildInputIndices.push_back(
          static_cast<unsigned>(indexedInput.index()));
    }
    mapping.map(childEntryBlock.getArgument(indexedInput.index()), mappedValue);
  }

  return success();
}

static LogicalResult
collectMappedReturnValues(func::FuncOp childFunc, func::ReturnOp childReturn,
                          const IRMapping &mapping,
                          SmallVectorImpl<Value> &mappedReturns) {
  for (Value returnValue : childReturn.getOperands()) {
    Value mapped = mapping.lookupOrNull(returnValue);
    if (!mapped) {
      childFunc.emitError("expected child return operand to be mapped by task "
                          "fusion");
      return failure();
    }
    mappedReturns.push_back(mapped);
  }
  return success();
}

static LogicalResult appendSelectedValues(ArrayRef<Value> values,
                                          ArrayRef<unsigned> indices,
                                          SmallVectorImpl<Value> &selected,
                                          Operation *diagnosticOp,
                                          StringRef valueName) {
  for (unsigned index : indices) {
    if (index >= values.size()) {
      diagnosticOp->emitError("expected ")
          << valueName << " index to be in range for task fusion";
      return failure();
    }
    selected.push_back(values[index]);
  }
  return success();
}

static LogicalResult cloneChildBodyIntoParent(
    func::FuncOp parentFunc, func::FuncOp childFunc, func::ReturnOp childReturn,
    IRMapping &mapping, ArrayRef<unsigned> keptParentReturnIndices,
    ArrayRef<unsigned> keptChildReturnIndices, OpBuilder &builder) {
  Block &childBlock = childFunc.getBody().front();
  builder.setInsertionPoint(parentFunc.getBody().front().getTerminator());
  for (Operation &op : childBlock.without_terminator()) {
    for (Value operand : op.getOperands()) {
      if (!mapping.contains(operand)) {
        childFunc.emitError("expected child operation operands to be mapped by "
                            "task fusion");
        return failure();
      }
    }
    builder.clone(op, mapping);
  }

  SmallVector<Value> mappedChildReturns;
  if (failed(collectMappedReturnValues(childFunc, childReturn, mapping,
                                       mappedChildReturns)))
    return failure();

  auto parentReturn =
      cast<func::ReturnOp>(parentFunc.getBody().front().getTerminator());
  SmallVector<Value> parentReturns(parentReturn.getOperands().begin(),
                                   parentReturn.getOperands().end());
  SmallVector<Value> fusedReturns;
  fusedReturns.reserve(keptParentReturnIndices.size() +
                       keptChildReturnIndices.size());
  if (failed(appendSelectedValues(parentReturns, keptParentReturnIndices,
                                  fusedReturns, parentFunc.getOperation(),
                                  "parent return")))
    return failure();
  if (failed(appendSelectedValues(mappedChildReturns, keptChildReturnIndices,
                                  fusedReturns, childFunc.getOperation(),
                                  "child return")))
    return failure();
  parentReturn->setOperands(fusedReturns);

  SmallVector<Type> resultTypes;
  resultTypes.reserve(fusedReturns.size());
  for (Value value : fusedReturns)
    resultTypes.push_back(value.getType());

  auto oldType = parentFunc.getFunctionType();
  parentFunc.setType(builder.getFunctionType(oldType.getInputs(), resultTypes));
  return success();
}

static LogicalResult appendSelectedI64ArrayAttr(
    Operation *parentOp, Operation *childOp, StringRef attrName,
    ArrayRef<unsigned> childElementIndices, Builder &builder) {
  if (childElementIndices.empty())
    return success();

  auto parentAttr = parentOp->getAttrOfType<ArrayAttr>(attrName);
  auto childAttr = childOp->getAttrOfType<ArrayAttr>(attrName);
  if (!parentAttr || !childAttr) {
    parentOp->emitError("expected both fused tasks to carry '")
        << attrName << "' when appending external fused inputs";
    return failure();
  }

  SmallVector<int64_t> values;
  values.reserve(parentAttr.size() + childElementIndices.size());
  for (Attribute attr : parentAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr) {
      parentOp->emitError("expected array attr '")
          << attrName << "' to contain integer values";
      return failure();
    }
    values.push_back(intAttr.getInt());
  }

  for (unsigned index : childElementIndices) {
    if (index >= childAttr.size()) {
      childOp->emitError("expected array attr '")
          << attrName << "' to have one entry per child input";
      return failure();
    }
    auto intAttr = dyn_cast<IntegerAttr>(childAttr[index]);
    if (!intAttr) {
      childOp->emitError("expected array attr '")
          << attrName << "' to contain integer values";
      return failure();
    }
    values.push_back(intAttr.getInt());
  }

  parentOp->setAttr(attrName, builder.getI64ArrayAttr(values));
  return success();
}

static LogicalResult appendSelectedI64ArrayElements(
    Operation *op, ArrayAttr attr, StringRef attrName,
    ArrayRef<unsigned> elementIndices, SmallVectorImpl<int64_t> &values) {
  for (unsigned index : elementIndices) {
    if (index >= attr.size()) {
      op->emitError("expected array attr '")
          << attrName << "' to have one entry per selected element";
      return failure();
    }
    auto intAttr = dyn_cast<IntegerAttr>(attr[index]);
    if (!intAttr) {
      op->emitError("expected array attr '")
          << attrName << "' to contain integer values";
      return failure();
    }
    values.push_back(intAttr.getInt());
  }
  return success();
}

static LogicalResult setSelectedOutputSlotsAttr(
    sculptor::TaskCreateOp parentTask, sculptor::TaskCreateOp childTask,
    ArrayRef<unsigned> keptParentOutputIndices,
    ArrayRef<unsigned> keptChildOutputIndices, Builder &builder) {
  Operation *parentOp = parentTask.getOperation();
  Operation *childOp = childTask.getOperation();
  StringRef attrName = runtime_attrs::kTaskOutputSlotsAttrName;
  auto parentAttr = parentOp->getAttrOfType<ArrayAttr>(attrName);
  auto childAttr = childOp->getAttrOfType<ArrayAttr>(attrName);
  if (!parentAttr && !childAttr)
    return success();
  if (!parentAttr || !childAttr) {
    parentOp->emitError("expected both fused tasks to carry '")
        << attrName << "' or neither to carry it";
    return failure();
  }

  SmallVector<int64_t> values;
  values.reserve(keptParentOutputIndices.size() +
                 keptChildOutputIndices.size());
  if (failed(appendSelectedI64ArrayElements(parentOp, parentAttr, attrName,
                                            keptParentOutputIndices, values)))
    return failure();
  if (failed(appendSelectedI64ArrayElements(childOp, childAttr, attrName,
                                            keptChildOutputIndices, values)))
    return failure();

  parentOp->setAttr(attrName, builder.getI64ArrayAttr(values));
  return success();
}

static LogicalResult addI64Attr(Operation *parentOp, Operation *childOp,
                                StringRef attrName, Builder &builder) {
  auto parentAttr = parentOp->getAttrOfType<IntegerAttr>(attrName);
  auto childAttr = childOp->getAttrOfType<IntegerAttr>(attrName);
  if (!parentAttr && !childAttr)
    return success();
  int64_t value = 0;
  if (parentAttr)
    value += parentAttr.getInt();
  if (childAttr)
    value += childAttr.getInt();
  parentOp->setAttr(attrName, builder.getI64IntegerAttr(value));
  return success();
}

static void setSequentialTaskResultIndicesAttr(sculptor::TaskCreateOp taskOp,
                                               Builder &builder) {
  SmallVector<int64_t> values;
  values.reserve(taskOp.getOutputs().size());
  for (unsigned index = 0, count = taskOp.getOutputs().size(); index < count;
       ++index)
    values.push_back(static_cast<int64_t>(index));
  taskOp->setAttr(runtime_attrs::kTaskResultIndicesAttrName,
                  builder.getI64ArrayAttr(values));
}

static void setSelectedTaskOutputs(sculptor::TaskCreateOp parentTask,
                                   sculptor::TaskCreateOp childTask,
                                   ArrayRef<unsigned> keptParentOutputIndices,
                                   ArrayRef<unsigned> keptChildOutputIndices) {
  SmallVector<Value> fusedOutputs;
  fusedOutputs.reserve(keptParentOutputIndices.size() +
                       keptChildOutputIndices.size());
  for (unsigned index : keptParentOutputIndices)
    fusedOutputs.push_back(parentTask.getOutputs()[index]);
  for (unsigned index : keptChildOutputIndices)
    fusedOutputs.push_back(childTask.getOutputs()[index]);
  parentTask.getOutputsMutable().assign(fusedOutputs);
}

static bool taskConsumesResource(sculptor::TaskCreateOp taskOp,
                                 Value resource) {
  for (Value input : taskOp.getInputs()) {
    if (input == resource)
      return true;
  }
  return false;
}

static bool isExternallyNeededOutput(Value resource,
                                     sculptor::TaskCreateOp parentTask,
                                     sculptor::TaskCreateOp childTask) {
  if (resource.getDefiningOp<sculptor::TaskGraphOutputOp>())
    return true;

  Block *block = parentTask->getBlock();
  for (Operation &op : *block) {
    auto taskOp = dyn_cast<sculptor::TaskCreateOp>(&op);
    if (!taskOp || taskOp == parentTask || taskOp == childTask)
      continue;
    if (taskConsumesResource(taskOp, resource))
      return true;
  }
  return false;
}

static void collectExternallyNeededOutputIndices(
    sculptor::TaskCreateOp taskOp, sculptor::TaskCreateOp parentTask,
    sculptor::TaskCreateOp childTask, SmallVectorImpl<unsigned> &indices) {
  for (auto indexedOutput : llvm::enumerate(taskOp.getOutputs())) {
    if (isExternallyNeededOutput(indexedOutput.value(), parentTask, childTask))
      indices.push_back(static_cast<unsigned>(indexedOutput.index()));
  }
}

static void
replaceChildDependenciesWithParent(sculptor::TaskCreateOp parentTask,
                                   sculptor::TaskCreateOp childTask) {
  Value parentResult = parentTask.getResult();
  Value childResult = childTask.getResult();
  Block *block = parentTask->getBlock();
  for (Operation &op : *block) {
    auto taskOp = dyn_cast<sculptor::TaskCreateOp>(&op);
    if (!taskOp || taskOp == childTask)
      continue;

    bool changed = false;
    bool hasParent = hasDependency(taskOp, parentResult);
    SmallVector<Value> dependencies;
    dependencies.reserve(taskOp.getDependencies().size());
    for (Value dependency : taskOp.getDependencies()) {
      if (dependency != childResult) {
        dependencies.push_back(dependency);
        continue;
      }

      changed = true;
      if (!hasParent) {
        dependencies.push_back(parentResult);
        hasParent = true;
      }
    }

    if (changed)
      taskOp.getDependenciesMutable().assign(dependencies);
  }
}

static LogicalResult
appendChildDependenciesToParent(sculptor::TaskCreateOp parentTask,
                                sculptor::TaskCreateOp childTask) {
  Value parentResult = parentTask.getResult();
  SmallVector<Value> dependencies(parentTask.getDependencies().begin(),
                                  parentTask.getDependencies().end());
  llvm::SmallPtrSet<Value, 8> seen(dependencies.begin(), dependencies.end());

  for (Value childDependency : childTask.getDependencies()) {
    if (childDependency == parentResult || seen.contains(childDependency))
      continue;

    auto dependencyTask =
        childDependency.getDefiningOp<sculptor::TaskCreateOp>();
    if (!dependencyTask) {
      childTask.emitError("expected child task dependency to be produced by a "
                          "task for task fusion");
      return failure();
    }

    if (!dependencyTask->isBeforeInBlock(parentTask.getOperation())) {
      childTask.emitError("expected external child task dependency to appear "
                          "before fused parent task");
      return failure();
    }

    dependencies.push_back(childDependency);
    seen.insert(childDependency);
  }

  parentTask.getDependenciesMutable().assign(dependencies);
  return success();
}

static LogicalResult
eraseUnusedTaskGraphTemporaryResources(func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected scheduled task graph function to have "
                            "one block");
    return failure();
  }

  SmallVector<Operation *> unusedResources;
  for (Operation &op : taskGraphFunc.getBody().front()) {
    auto temporaryOp = dyn_cast<sculptor::TaskGraphTemporaryOp>(&op);
    if (temporaryOp && temporaryOp.getResult().use_empty())
      unusedResources.push_back(&op);
  }

  for (Operation *op : unusedResources)
    op->erase();

  return success();
}

static bool isTaskGraphFunction(func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         isa<sculptor::TaskGraphType>(functionType.getResult(0));
}

static bool isGeneratedTaskCallee(func::FuncOp func) {
  if (func->hasAttr(task_attrs::kTaskKindAttrName))
    return true;

  return func.getSymName().starts_with("task_");
}

static bool callsGeneratedTaskCallee(ModuleOp module, func::FuncOp func) {
  bool found = false;
  func.walk([&](func::CallOp callOp) {
    if (found)
      return;

    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (callee && isGeneratedTaskCallee(callee))
      found = true;
  });
  return found;
}

static void eraseUnusedTaskCallees(ModuleOp module) {
  llvm::StringSet<> liveTaskCallees;
  module.walk([&](sculptor::TaskCreateOp taskOp) {
    liveTaskCallees.insert(taskOp.getCalleeAttr().getValue());
  });

  bool hasTaskGraph = false;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (isTaskGraphFunction(func)) {
      hasTaskGraph = true;
      break;
    }
  }

  SmallVector<func::FuncOp> staleEntryPoints;
  llvm::SmallPtrSet<Operation *, 4> staleEntryPointOps;
  if (hasTaskGraph) {
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.getName() != "forward" ||
          !callsGeneratedTaskCallee(module, func))
        continue;

      staleEntryPoints.push_back(func);
      staleEntryPointOps.insert(func.getOperation());
    }
  }

  llvm::StringSet<> calledTaskCallees;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (staleEntryPointOps.contains(func.getOperation()))
      continue;

    func.walk([&](func::CallOp callOp) {
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (callee && isGeneratedTaskCallee(callee))
        calledTaskCallees.insert(callee.getSymName());
    });
  }

  for (func::FuncOp func : staleEntryPoints)
    func.erase();

  SmallVector<func::FuncOp> deadTaskCallees;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!func.isPrivate() || !isGeneratedTaskCallee(func))
      continue;

    if (liveTaskCallees.contains(func.getSymName()) ||
        calledTaskCallees.contains(func.getSymName()))
      continue;

    deadTaskCallees.push_back(func);
  }

  for (func::FuncOp func : deadTaskCallees)
    func.erase();
}

FailureOr<TaskGraphDAG> parseTaskGraphDAG(func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected scheduled task graph function to have "
                            "one block");
    return failure();
  }

  TaskGraphDAG dag;
  Block &block = taskGraphFunc.getBody().front();

  for (Operation &op : block) {
    for (Value result : op.getResults()) {
      if (isLogicalArrayResource(result))
        dag.logicalArrayResources.push_back(result);
    }

    auto taskOp = dyn_cast<sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    TaskGraphNode node;
    node.op = taskOp;
    node.index = dag.nodes.size();
    dag.nodeIndexByTaskResult.try_emplace(taskOp.getResult(), node.index);
    dag.nodes.push_back(std::move(node));
  }

  for (TaskGraphNode &node : dag.nodes) {
    for (Value dependency : node.op.getDependencies()) {
      auto predecessorIt = dag.nodeIndexByTaskResult.find(dependency);
      if (predecessorIt == dag.nodeIndexByTaskResult.end()) {
        node.op.emitError("expected task dependency to reference an "
                          "sculptor.task.create result in the same task graph");
        return failure();
      }

      unsigned predecessorIndex = predecessorIt->second;
      if (predecessorIndex >= node.index) {
        node.op.emitError("expected task dependency to reference an earlier "
                          "task in the task graph");
        return failure();
      }

      node.predecessors.push_back(predecessorIndex);
      dag.nodes[predecessorIndex].successors.push_back(node.index);
      ++dag.dependencyCount;
    }
  }

  return dag;
}

FailureOr<PhysicalArrayPlacement>
resolvePhysicalArrayPlacement(Operation *diagnosticOp,
                              const HardwareBudget &budget,
                              int64_t physicalArrayId) {
  int64_t coreId = physicalArrayId / budget.arraysPerCore;
  int64_t localArrayId = physicalArrayId % budget.arraysPerCore;
  if (physicalArrayId < 0 || coreId < 0 || coreId >= budget.numCores ||
      localArrayId < 0 || localArrayId >= budget.arraysPerCore) {
    if (diagnosticOp)
      diagnosticOp->emitError("assigned analog array is outside the hardware "
                              "budget");
    return failure();
  }

  return PhysicalArrayPlacement{physicalArrayId, coreId, localArrayId};
}

LogicalResult attachTaskCorePlacement(ModuleOp module,
                                      sculptor::TaskCreateOp taskOp,
                                      const HardwareBudget &budget,
                                      int64_t coreId) {
  if (coreId < 0 || coreId >= budget.numCores) {
    taskOp.emitError("assigned core is outside the hardware budget");
    return failure();
  }

  auto callee = lookupTaskCallee(module, taskOp, "core");
  if (failed(callee))
    return failure();

  Builder builder(module.getContext());
  attachCorePlacementAttrs(taskOp.getOperation(), builder, coreId);
  attachCorePlacementAttrs(callee->getOperation(), builder, coreId);
  return success();
}

LogicalResult attachTaskAnalogArrayPlacement(ModuleOp module,
                                             sculptor::TaskCreateOp taskOp,
                                             const HardwareBudget &budget,
                                             int64_t physicalArrayId) {
  auto placement = resolvePhysicalArrayPlacement(taskOp.getOperation(), budget,
                                                 physicalArrayId);
  if (failed(placement))
    return failure();

  auto callee = lookupTaskCallee(module, taskOp, "analog array");
  if (failed(callee))
    return failure();

  Builder builder(module.getContext());
  attachAnalogArrayPlacementAttrs(taskOp.getOperation(), builder, *placement);
  attachAnalogArrayPlacementAttrs(callee->getOperation(), builder, *placement);
  return success();
}

static std::optional<int64_t> getOptionalI64Attr(Operation *op,
                                                 StringRef attrName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

static FailureOr<int64_t> getRequiredI64Attr(Operation *op,
                                             StringRef attrName) {
  auto attr = op->getAttrOfType<IntegerAttr>(attrName);
  if (!attr) {
    op->emitError("expected scheduled task graph attr '") << attrName << "'";
    return failure();
  }

  return attr.getInt();
}

static bool isAnalogTask(sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() == "analog";
}

static bool shouldInferDigitalOpsFromCallee(sculptor::TaskCreateOp taskOp) {
  return !isAnalogTask(taskOp) ||
         taskOp.getTaskKind() == task_graph_names::kConvTileMVMTaskKind;
}

static LogicalResult collectResourceProducers(
    const TaskGraphDAG &dag,
    llvm::DenseMap<Value, const TaskGraphNode *> &producerByResource) {
  for (const TaskGraphNode &node : dag.nodes) {
    sculptor::TaskCreateOp taskOp = node.op;
    for (Value output : taskOp.getOutputs()) {
      if (!producerByResource.try_emplace(output, &node).second) {
        taskOp.emitError("expected task graph resource to have one producer");
        return failure();
      }
    }
  }

  return success();
}

static LogicalResult normalizeTaskPlacement(ModuleOp module,
                                            sculptor::TaskCreateOp taskOp,
                                            const HardwareBudget &budget) {
  std::optional<int64_t> coreId =
      getOptionalI64Attr(taskOp, runtime_attrs::kTaskCoreIdAttrName);
  std::optional<int64_t> physicalArrayId =
      getOptionalI64Attr(taskOp, runtime_attrs::kTaskPhysicalArrayIdAttrName);
  std::optional<int64_t> localArrayId =
      getOptionalI64Attr(taskOp, runtime_attrs::kTaskLocalArrayIdAttrName);

  if (physicalArrayId) {
    auto placement = resolvePhysicalArrayPlacement(taskOp.getOperation(),
                                                   budget, *physicalArrayId);
    if (failed(placement))
      return failure();

    if (coreId && *coreId != placement->coreId) {
      taskOp.emitError("expected scheduled task core to match assigned analog "
                       "array core");
      return failure();
    }
    if (localArrayId && *localArrayId != placement->localArrayId) {
      taskOp.emitError("expected scheduled task local array to match assigned "
                       "analog array");
      return failure();
    }

    return attachTaskAnalogArrayPlacement(module, taskOp, budget,
                                          *physicalArrayId);
  }

  if (isAnalogTask(taskOp)) {
    taskOp.emitError("expected scheduled analog task to have physical analog "
                     "array placement");
    return failure();
  }

  auto requiredCoreId =
      getRequiredI64Attr(taskOp, runtime_attrs::kTaskCoreIdAttrName);
  if (failed(requiredCoreId))
    return failure();
  return attachTaskCorePlacement(module, taskOp, budget, *requiredCoreId);
}

static int64_t getStaticElementCount(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.hasStaticShape())
    return 0;
  return shapedType.getNumElements();
}

static int64_t getStaticElementCount(Operation *op) {
  for (Type resultType : op->getResultTypes()) {
    int64_t elementCount = getStaticElementCount(resultType);
    if (elementCount > 0)
      return elementCount;
  }

  for (Value operand : llvm::reverse(op->getOperands())) {
    int64_t elementCount = getStaticElementCount(operand.getType());
    if (elementCount > 0)
      return elementCount;
  }

  return 0;
}

static bool isScalarDigitalOp(Operation *op) {
  StringRef dialectNamespace = op->getName().getDialectNamespace();
  return dialectNamespace == "arith" || dialectNamespace == "math";
}

static int64_t countScalarDigitalOps(Operation *linalgOp) {
  int64_t scalarOps = 0;
  for (Region &region : linalgOp->getRegions()) {
    region.walk([&](Operation *nestedOp) {
      if (nestedOp == linalgOp || nestedOp->hasTrait<OpTrait::IsTerminator>())
        return;
      if (isScalarDigitalOp(nestedOp))
        ++scalarOps;
    });
  }
  return scalarOps;
}

static bool isSingleScalarOpLinalg(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName == "linalg.add" || opName == "linalg.sub" ||
         opName == "linalg.mul" || opName == "linalg.div" ||
         opName == "linalg.max" || opName == "linalg.min";
}

static int64_t inferDigitalOpsFromCallee(func::FuncOp callee) {
  if (!callee || callee.isDeclaration() || !callee.getBody().hasOneBlock())
    return 0;

  int64_t digitalOps = 0;
  for (Operation &op : callee.getBody().front().without_terminator()) {
    StringRef dialectNamespace = op.getName().getDialectNamespace();
    if (dialectNamespace != "linalg")
      continue;

    int64_t elementCount = getStaticElementCount(&op);
    if (elementCount <= 0)
      continue;

    if (op.getName().getStringRef() == "linalg.generic") {
      digitalOps += elementCount * countScalarDigitalOps(&op);
      continue;
    }

    if (isSingleScalarOpLinalg(&op))
      digitalOps += elementCount;
  }

  return digitalOps;
}

static FailureOr<int64_t>
getOrAttachTaskDigitalOps(ModuleOp module, sculptor::TaskCreateOp taskOp,
                          Builder &builder) {
  if (auto digitalOpsAttr = taskOp->getAttrOfType<IntegerAttr>(
          runtime_attrs::kTaskDigitalOpsAttrName))
    return digitalOpsAttr.getInt();

  int64_t digitalOps = 0;
  if (shouldInferDigitalOpsFromCallee(taskOp)) {
    auto callee =
        module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
    if (!callee) {
      return taskOp.emitError("expected task callee '")
             << taskOp.getCalleeAttr().getValue()
             << "' to resolve to a function for digital op accounting";
    }
    digitalOps = inferDigitalOpsFromCallee(callee);
  }

  taskOp->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                  builder.getI64IntegerAttr(digitalOps));
  return digitalOps;
}

static FailureOr<int64_t> computeTotalDigitalOps(ModuleOp module,
                                                 const TaskGraphDAG &dag,
                                                 Builder &builder) {
  int64_t totalDigitalOps = 0;
  for (const TaskGraphNode &node : dag.nodes) {
    auto digitalOps = getOrAttachTaskDigitalOps(module, node.op, builder);
    if (failed(digitalOps))
      return failure();
    totalDigitalOps += *digitalOps;
  }
  return totalDigitalOps;
}

static LogicalResult attachLogicalArrayScheduleMetadata(
    func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag,
    const llvm::DenseMap<Value, const TaskGraphNode *> &producerByResource,
    Builder &builder) {
  SmallVector<int64_t, 8> logicalArrayToAnalogArray;
  logicalArrayToAnalogArray.reserve(dag.logicalArrayResources.size());

  for (auto indexedResource : llvm::enumerate(dag.logicalArrayResources)) {
    Value resource = indexedResource.value();
    auto producerIt = producerByResource.find(resource);
    if (producerIt == producerByResource.end()) {
      if (Operation *resourceOp = resource.getDefiningOp()) {
        resourceOp->emitError("expected logical array resource to be produced "
                              "by a scheduled task");
      } else {
        taskGraphFunc.emitError(
            "expected logical array resource to be produced "
            "by a scheduled task");
      }
      return failure();
    }

    auto physicalArrayId = getRequiredI64Attr(
        producerIt->second->op, runtime_attrs::kTaskPhysicalArrayIdAttrName);
    if (failed(physicalArrayId))
      return failure();

    if (failed(resolvePhysicalArrayPlacement(producerIt->second->op, budget,
                                             *physicalArrayId)))
      return failure();

    if (Operation *resourceOp = resource.getDefiningOp())
      resourceOp->setAttr(schedule_attrs::kLogicalArrayIndexAttrName,
                          builder.getI64IntegerAttr(
                              static_cast<int64_t>(indexedResource.index())));

    logicalArrayToAnalogArray.push_back(*physicalArrayId);
  }

  taskGraphFunc->setAttr(schedule_attrs::kNumLogicalArraysAttrName,
                         builder.getI64IntegerAttr(static_cast<int64_t>(
                             dag.logicalArrayResources.size())));
  taskGraphFunc->setAttr(schedule_attrs::kLogicalArrayToAnalogArrayAttrName,
                         builder.getI64ArrayAttr(logicalArrayToAnalogArray));
  return success();
}

static void attachGraphScoreMetadata(func::FuncOp taskGraphFunc,
                                     const TaskGraphScore &score,
                                     Builder &builder) {
  taskGraphFunc->setAttr(schedule_attrs::kGraphScoreAttrName,
                         builder.getI64IntegerAttr(score.score));
  taskGraphFunc->setAttr(schedule_attrs::kBoundaryPenaltyAttrName,
                         builder.getI64IntegerAttr(score.boundaryPenalty));
  taskGraphFunc->setAttr(schedule_attrs::kCoreTransferBytesAttrName,
                         builder.getI64ArrayAttr(score.coreTransferBytes));
  taskGraphFunc->setAttr(
      schedule_attrs::kInterCoreTransferBytesAttrName,
      builder.getI64IntegerAttr(score.interCoreTransferBytes));
  taskGraphFunc->setAttr(schedule_attrs::kCoreTransferCostAttrName,
                         builder.getI64ArrayAttr(score.coreTransferCost));
  taskGraphFunc->setAttr(schedule_attrs::kTotalTransferCostAttrName,
                         builder.getI64IntegerAttr(score.totalTransferCost));
}

static FailureOr<TaskGraphScore> scoreTaskGraph(ModuleOp module,
                                                func::FuncOp taskGraphFunc,
                                                const HardwareBudget &budget,
                                                const TaskGraphDAG &dag) {
  MeshTaskGraphScorer scorer;
  return scorer.score(module, taskGraphFunc, budget, dag);
}

static LogicalResult
attachTransferScheduleMetadata(ModuleOp module, func::FuncOp taskGraphFunc,
                               const HardwareBudget &budget,
                               const TaskGraphDAG &dag, Builder &builder) {
  FailureOr<TaskGraphScore> score =
      scoreTaskGraph(module, taskGraphFunc, budget, dag);
  if (failed(score))
    return failure();

  attachGraphScoreMetadata(taskGraphFunc, *score, builder);
  return success();
}

LogicalResult finalizeTaskGraphScheduleMetadata(ModuleOp module,
                                                func::FuncOp taskGraphFunc,
                                                const HardwareBudget &budget,
                                                const TaskGraphDAG &dag) {
  Builder builder(module.getContext());
  llvm::DenseMap<Value, const TaskGraphNode *> producerByResource;
  if (failed(collectResourceProducers(dag, producerByResource)))
    return failure();

  for (const TaskGraphNode &node : dag.nodes) {
    if (failed(normalizeTaskPlacement(module, node.op, budget)))
      return failure();
  }

  auto totalDigitalOps = computeTotalDigitalOps(module, dag, builder);
  if (failed(totalDigitalOps))
    return failure();

  if (failed(attachLogicalArrayScheduleMetadata(taskGraphFunc, budget, dag,
                                                producerByResource, builder)))
    return failure();

  if (failed(attachTransferScheduleMetadata(module, taskGraphFunc, budget, dag,
                                            builder)))
    return failure();

  taskGraphFunc->setAttr(
      schedule_attrs::kTaskCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag.nodes.size())));
  taskGraphFunc->setAttr(
      schedule_attrs::kDependencyCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag.dependencyCount)));
  taskGraphFunc->setAttr(schedule_attrs::kTotalDigitalOpsAttrName,
                         builder.getI64IntegerAttr(*totalDigitalOps));
  return success();
}

FailureOr<sculptor::TaskCreateOp> fuseTasks(ModuleOp module,
                                            sculptor::TaskCreateOp parentTask,
                                            sculptor::TaskCreateOp childTask) {
  if (!parentTask || !childTask)
    return failure();

  if (parentTask == childTask) {
    parentTask.emitError("cannot fuse a task with itself");
    return failure();
  }

  if (parentTask->getBlock() != childTask->getBlock()) {
    childTask.emitError("expected fused tasks to be in the same task graph "
                        "block");
    return failure();
  }

  if (parentTask.getGraph() != childTask.getGraph()) {
    childTask.emitError(
        "expected fused tasks to belong to the same task graph");
    return failure();
  }

  if (!hasDependency(childTask, parentTask.getResult())) {
    childTask.emitError("expected child task to directly depend on parent task "
                        "for task fusion");
    return failure();
  }

  if (isMatrixSetupMVMPair(parentTask, childTask)) {
    childTask.emitError("refusing to fuse matrix setup and MVM tasks");
    return failure();
  }

  auto parentFunc = lookupTaskCallee(module, parentTask, "task fusion");
  auto childFunc = lookupTaskCallee(module, childTask, "task fusion");
  if (failed(parentFunc) || failed(childFunc))
    return failure();

  parentFunc = cloneParentTaskCalleeForFusion(module, parentTask, *parentFunc);
  if (failed(parentFunc))
    return failure();

  auto parentReturn = getSingleBlockReturn(*parentFunc, parentTask);
  auto childReturn = getSingleBlockReturn(*childFunc, childTask);
  if (failed(parentReturn) || failed(childReturn))
    return failure();

  if (parentFunc->getBody().front().getNumArguments() !=
      parentTask.getInputs().size()) {
    parentTask.emitError("expected parent task input count to match parent "
                         "callee argument count for task fusion");
    return failure();
  }

  if (parentReturn->getNumOperands() != parentTask.getOutputs().size() ||
      childReturn->getNumOperands() != childTask.getOutputs().size()) {
    childTask.emitError("expected task output counts to match callee return "
                        "counts for task fusion");
    return failure();
  }

  SmallVector<unsigned> keptParentOutputIndices;
  SmallVector<unsigned> keptChildOutputIndices;
  collectExternallyNeededOutputIndices(parentTask, parentTask, childTask,
                                       keptParentOutputIndices);
  collectExternallyNeededOutputIndices(childTask, parentTask, childTask,
                                       keptChildOutputIndices);

  IRMapping mapping;
  SmallVector<unsigned> appendedChildInputIndices;
  OpBuilder builder(module.getContext());
  if (failed(mapChildInputsToParentValues(
          parentTask, *parentFunc, *parentReturn, childTask, *childFunc,
          mapping, appendedChildInputIndices, builder)))
    return failure();

  if (failed(appendChildDependenciesToParent(parentTask, childTask)))
    return failure();

  if (failed(cloneChildBodyIntoParent(*parentFunc, *childFunc, *childReturn,
                                      mapping, keptParentOutputIndices,
                                      keptChildOutputIndices, builder)))
    return failure();

  if (failed(appendSelectedI64ArrayAttr(parentTask.getOperation(),
                                        childTask.getOperation(),
                                        runtime_attrs::kTaskInputSlotsAttrName,
                                        appendedChildInputIndices, builder)))
    return failure();

  if (failed(setSelectedOutputSlotsAttr(parentTask, childTask,
                                        keptParentOutputIndices,
                                        keptChildOutputIndices, builder)) ||
      failed(addI64Attr(parentTask.getOperation(), childTask.getOperation(),
                        runtime_attrs::kTaskDigitalOpsAttrName, builder)))
    return failure();

  setSelectedTaskOutputs(parentTask, childTask, keptParentOutputIndices,
                         keptChildOutputIndices);
  setSequentialTaskResultIndicesAttr(parentTask, builder);
  replaceChildDependenciesWithParent(parentTask, childTask);
  if (!childTask.getResult().use_empty()) {
    childTask.emitError("expected child task result to have no remaining users "
                        "after task fusion");
    return failure();
  }

  childTask.erase();
  if (SymbolTable::symbolKnownUseEmpty(*childFunc, module))
    childFunc->erase();

  return parentTask;
}

LogicalResult
registerTaskGraphScheduler(TaskGraphSchedulerRegistry &registry,
                           std::unique_ptr<TaskGraphScheduler> scheduler) {
  if (!scheduler)
    return failure();

  StringRef name = scheduler->getName();
  if (name.empty() || registry.contains(name))
    return failure();

  registry.try_emplace(name, std::move(scheduler));
  return success();
}

const TaskGraphScheduler *
lookupTaskGraphScheduler(const TaskGraphSchedulerRegistry &registry,
                         StringRef name) {
  auto it = registry.find(name);
  if (it == registry.end())
    return nullptr;
  return it->second.get();
}

} // namespace task_schedulers

void ScheduleTaskGraphPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto budget = buildHardwareBudget(module, cores, arraysPerCore, topology,
                                    meshRows, meshCols, randomSeed);
  if (failed(budget)) {
    signalPassFailure();
    return;
  }

  task_schedulers::TaskGraphSchedulerRegistry registry;
  task_schedulers::registerRandomTaskScheduler(registry);
  task_schedulers::registerSnakeTaskScheduler(registry);
  task_schedulers::registerGreedyHeavyEdgeTaskScheduler(registry);
  task_schedulers::registerManhattanCutTaskScheduler(registry);
  task_schedulers::registerBoundaryAwareCutTaskScheduler(registry);
  task_schedulers::registerBoundaryAwareCutOptimizedTaskScheduler(registry);
  const task_schedulers::TaskGraphScheduler *selectedScheduler =
      task_schedulers::lookupTaskGraphScheduler(registry, schedule);
  if (!selectedScheduler) {
    if (schedule.empty()) {
      module.emitError("expected task graph schedule name");
    } else {
      module.emitError("unknown task graph schedule '") << schedule << "'";
    }
    signalPassFailure();
    return;
  }

  mlir::Builder builder(module.getContext());
  attachBudgetAttrs(module.getOperation(), builder, *budget);

  bool foundTaskGraph = false;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;

    attachBudgetAttrs(func.getOperation(), builder, *budget);
    auto dag = task_schedulers::parseTaskGraphDAG(func);
    if (failed(dag)) {
      signalPassFailure();
      return;
    }

    if (failed(selectedScheduler->schedule(module, func, *budget, *dag))) {
      func.emitError("failed to apply task graph schedule '")
          << selectedScheduler->getName() << "'";
      signalPassFailure();
      return;
    }

    if (failed(task_schedulers::fuseTaskGraphRoutines(module, func, *budget,
                                                      *dag))) {
      func.emitError("failed to fuse task graph routines after schedule '")
          << selectedScheduler->getName() << "'";
      signalPassFailure();
      return;
    }

    task_schedulers::eraseUnusedTaskCallees(module);

    if (failed(task_schedulers::eraseUnusedTaskGraphTemporaryResources(func)) ||
        failed(mlir::sculptor::rebuildTaskGraphExecutionPlan(func))) {
      func.emitError("failed to compact task graph resources after schedule '")
          << selectedScheduler->getName() << "'";
      signalPassFailure();
      return;
    }

    auto scheduledDag = task_schedulers::parseTaskGraphDAG(func);
    if (failed(scheduledDag)) {
      signalPassFailure();
      return;
    }

    if (failed(task_schedulers::finalizeTaskGraphScheduleMetadata(
            module, func, *budget, *scheduledDag))) {
      func.emitError("failed to finalize task graph schedule metadata");
      signalPassFailure();
      return;
    }

    foundTaskGraph = true;
  }

  if (!foundTaskGraph) {
    module.emitError("expected at least one task graph function returning "
                     "!sculptor.task_graph");
    signalPassFailure();
  }
}

void registerScheduleTaskGraphPass() {
  PassRegistration<ScheduleTaskGraphPass>();
}

} // namespace sculptor
} // namespace mlir
