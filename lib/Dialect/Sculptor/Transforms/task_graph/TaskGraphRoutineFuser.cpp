#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphRoutineFuser.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTimingAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_attrs = mlir::sculptor::task_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace timing_attrs = mlir::sculptor::timing_attrs;

static constexpr llvm::StringLiteral kFusedMixedTaskDomain("digital");
static constexpr llvm::StringLiteral kFusedMixedTaskKind("mixed.fused");

using TaskFusionPatternFn = LogicalResult (*)(ModuleOp module,
                                              func::FuncOp taskGraphFunc,
                                              const TaskGraphDAG &dag,
                                              bool &changed);

struct TaskFusionPattern {
  llvm::StringRef name;
  TaskFusionPatternFn apply;
};

struct IslandFusibleComponent {
  llvm::SmallVector<const TaskGraphNode *, 8> nodes;
  int64_t coreId = 0;
  int64_t islandId = 0;
  std::string fusedTaskName;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> localArrayId;
};

struct ComponentBoundary {
  llvm::SmallVector<Value, 8> inputs;
  llvm::SmallVector<Value, 8> outputs;
  llvm::SmallVector<Value, 8> dependencies;
  Operation *latestDependency = nullptr;
  Operation *earliestExternalUser = nullptr;
};

static bool hasDependency(mlir::sculptor::TaskCreateOp taskOp,
                          Value dependency) {
  for (Value candidate : taskOp.getDependencies()) {
    if (candidate == dependency)
      return true;
  }
  return false;
}

static std::optional<int64_t> getOptionalI64Attr(Operation *op,
                                                 StringRef attrName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

static std::optional<int64_t>
getOptionalScheduledCore(mlir::sculptor::TaskCreateOp taskOp) {
  return getOptionalI64Attr(taskOp.getOperation(),
                            runtime_attrs::kTaskCoreIdAttrName);
}

static std::string sanitizeSymbolComponent(StringRef value) {
  std::string result;
  result.reserve(value.size());

  bool previousWasUnderscore = false;
  for (char c : value) {
    unsigned char uc = static_cast<unsigned char>(c);
    bool symbolSafe = std::isalnum(uc) || c == '_';
    char next = symbolSafe ? c : '_';

    if (next == '_' && previousWasUnderscore)
      continue;

    result.push_back(next);
    previousWasUnderscore = next == '_';
  }

  while (!result.empty() && result.back() == '_')
    result.pop_back();
  while (!result.empty() && result.front() == '_')
    result.erase(result.begin());

  if (result.empty())
    return "task";

  return result;
}

static llvm::StringSet<> collectTaskNames(const TaskGraphDAG &dag) {
  llvm::StringSet<> usedTaskNames;
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    usedTaskNames.insert(taskOp.getTaskName());
  }
  return usedTaskNames;
}

static std::string buildUniqueFusedTaskName(StringRef baseName,
                                            llvm::StringSet<> &usedTaskNames) {
  std::string sanitizedBase = sanitizeSymbolComponent(baseName);
  if (usedTaskNames.insert(sanitizedBase).second)
    return sanitizedBase;

  unsigned index = 0;
  std::string candidate = sanitizedBase + "_" + std::to_string(index);
  while (!usedTaskNames.insert(candidate).second) {
    ++index;
    candidate = sanitizedBase + "_" + std::to_string(index);
  }
  return candidate;
}

static FailureOr<func::FuncOp>
lookupTaskCallee(ModuleOp module, mlir::sculptor::TaskCreateOp taskOp) {
  auto callee =
      module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
  if (!callee) {
    return taskOp.emitError("expected task callee '")
           << taskOp.getCalleeAttr().getValue()
           << "' to resolve to a function for task fusion";
  }
  return callee;
}

static bool containsValue(ArrayRef<Value> values, Value value) {
  return llvm::is_contained(values, value);
}

static void appendUniqueValue(SmallVectorImpl<Value> &values, Value value) {
  if (!containsValue(values, value))
    values.push_back(value);
}

static bool isMatrixSetupTask(mlir::sculptor::TaskCreateOp taskOp) {
  return taskOp.getTaskKind() == task_graph_names::kMatrixSetupTaskKind;
}

static bool isComponentFusibleTask(mlir::sculptor::TaskCreateOp taskOp) {
  // Matrix setup produces logical-array resources used by schedule metadata.
  // Keep it explicit and let fused MVM/digital components depend on it.
  return !isMatrixSetupTask(taskOp) && getOptionalScheduledCore(taskOp) &&
         getOptionalI64Attr(taskOp, schedule_attrs::kIslandIdAttrName);
}

static std::optional<int64_t>
getOptionalPhysicalArray(mlir::sculptor::TaskCreateOp taskOp) {
  return getOptionalI64Attr(taskOp.getOperation(),
                            runtime_attrs::kTaskPhysicalArrayIdAttrName);
}

static std::optional<int64_t> getSingleComponentPhysicalArray(
    ArrayRef<const TaskGraphNode *> componentNodes) {
  std::optional<int64_t> physicalArrayId;
  for (const TaskGraphNode *node : componentNodes) {
    std::optional<int64_t> taskPhysicalArray =
        getOptionalPhysicalArray(node->op);
    if (!taskPhysicalArray)
      continue;
    if (physicalArrayId && *physicalArrayId != *taskPhysicalArray)
      return std::nullopt;
    physicalArrayId = *taskPhysicalArray;
  }
  return physicalArrayId;
}

static std::optional<int64_t>
getSingleComponentLocalArray(ArrayRef<const TaskGraphNode *> componentNodes) {
  std::optional<int64_t> localArrayId;
  for (const TaskGraphNode *node : componentNodes) {
    std::optional<int64_t> taskLocalArray = getOptionalI64Attr(
        node->op.operator->(), runtime_attrs::kTaskLocalArrayIdAttrName);
    if (!taskLocalArray)
      continue;
    if (localArrayId && *localArrayId != *taskLocalArray)
      return std::nullopt;
    localArrayId = *taskLocalArray;
  }
  return localArrayId;
}

static std::string
buildFusedIslandComponentTaskName(ArrayRef<const TaskGraphNode *> nodes,
                                  llvm::StringSet<> &usedTaskNames) {
  if (nodes.empty())
    return buildUniqueFusedTaskName("same_core_component", usedTaskNames);

  mlir::sculptor::TaskCreateOp firstTask = nodes.front()->op;
  mlir::sculptor::TaskCreateOp lastTask = nodes.back()->op;
  std::string baseName = firstTask.getSourceLayer().str();
  baseName += "_same_core_component_core_";
  std::optional<int64_t> coreId = getOptionalScheduledCore(firstTask);
  baseName += std::to_string(coreId.value_or(0));
  baseName += "_";
  baseName += std::to_string(firstTask.getSourceTaskOrdinal());
  baseName += "_";
  baseName += std::to_string(lastTask.getSourceTaskOrdinal());
  return buildUniqueFusedTaskName(baseName, usedTaskNames);
}

static bool isBeforeInSameBlock(Operation *lhs, Operation *rhs) {
  return lhs && rhs && lhs->getBlock() == rhs->getBlock() &&
         lhs->isBeforeInBlock(rhs);
}

static Operation *getLaterOperation(Operation *current, Operation *candidate) {
  if (!current)
    return candidate;
  if (!candidate)
    return current;
  return current->isBeforeInBlock(candidate) ? candidate : current;
}

static Operation *getEarlierOperation(Operation *current,
                                      Operation *candidate) {
  if (!current)
    return candidate;
  if (!candidate)
    return current;
  return candidate->isBeforeInBlock(current) ? candidate : current;
}

static void collectTaskConsumers(
    const TaskGraphDAG &dag,
    llvm::DenseMap<Value, llvm::SmallVector<const TaskGraphNode *, 4>>
        &consumersByTaskResult) {
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    for (Value dependency : taskOp.getDependencies()) {
      auto producer = dependency.getDefiningOp<mlir::sculptor::TaskCreateOp>();
      if (!producer)
        continue;
      consumersByTaskResult[dependency].push_back(&node);
    }
  }
}

static void collectResourceConsumers(
    const TaskGraphDAG &dag,
    llvm::DenseMap<Value, llvm::SmallVector<const TaskGraphNode *, 4>>
        &consumersByResource) {
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    for (Value input : taskOp.getInputs())
      consumersByResource[input].push_back(&node);
  }
}

static bool allComponentNodesHaveCore(ArrayRef<const TaskGraphNode *> nodes,
                                      int64_t coreId) {
  for (const TaskGraphNode *node : nodes) {
    std::optional<int64_t> taskCore = getOptionalScheduledCore(node->op);
    if (!taskCore || *taskCore != coreId)
      return false;
  }
  return true;
}

static llvm::SmallVector<IslandFusibleComponent, 16>
collectIslandFusibleComponents(const TaskGraphDAG &dag) {
  llvm::SmallVector<IslandFusibleComponent, 16> components;
  llvm::SmallVector<bool, 64> eligible(dag.nodes.size(), false);
  llvm::SmallVector<bool, 64> visited(dag.nodes.size(), false);
  llvm::StringSet<> usedTaskNames = collectTaskNames(dag);

  for (const TaskGraphNode &node : dag.nodes)
    eligible[node.index] = isComponentFusibleTask(node.op);

  for (const TaskGraphNode &seed : dag.nodes) {
    if (!eligible[seed.index] || visited[seed.index])
      continue;

    std::optional<int64_t> coreId = getOptionalScheduledCore(seed.op);
    std::optional<int64_t> islandId = getOptionalI64Attr(
        seed.op.operator->(), schedule_attrs::kIslandIdAttrName);
    if (!coreId || !islandId)
      continue;

    llvm::SmallVector<unsigned, 16> stack;
    llvm::SmallVector<const TaskGraphNode *, 8> nodes;
    stack.push_back(seed.index);
    visited[seed.index] = true;

    while (!stack.empty()) {
      unsigned index = stack.pop_back_val();
      const TaskGraphNode &node = dag.nodes[index];
      nodes.push_back(&node);

      auto maybePushNeighbor = [&](unsigned neighborIndex) {
        if (!eligible[neighborIndex] || visited[neighborIndex])
          return;
        const TaskGraphNode &neighbor = dag.nodes[neighborIndex];
        std::optional<int64_t> neighborCore =
            getOptionalScheduledCore(neighbor.op);
        if (!neighborCore || *neighborCore != *coreId)
          return;
        std::optional<int64_t> neighborIslandId = getOptionalI64Attr(
            neighbor.op.operator->(), schedule_attrs::kIslandIdAttrName);
        if (!neighborIslandId || *neighborIslandId != *islandId)
          return;
        visited[neighborIndex] = true;
        stack.push_back(neighborIndex);
      };

      for (unsigned predecessor : node.predecessors)
        maybePushNeighbor(predecessor);
      for (unsigned successor : node.successors)
        maybePushNeighbor(successor);
    }

    if (nodes.size() <= 1)
      continue;

    llvm::sort(nodes, [](const TaskGraphNode *lhs, const TaskGraphNode *rhs) {
      return lhs->index < rhs->index;
    });

    if (!allComponentNodesHaveCore(nodes, *coreId))
      continue;

    IslandFusibleComponent component;
    component.nodes = std::move(nodes);
    component.coreId = *coreId;
    component.islandId = *islandId;
    component.physicalArrayId =
        getSingleComponentPhysicalArray(component.nodes);
    component.localArrayId = getSingleComponentLocalArray(component.nodes);
    component.fusedTaskName =
        buildFusedIslandComponentTaskName(component.nodes, usedTaskNames);
    components.push_back(std::move(component));
  }

  return components;
}

static bool isNodeInComponent(const llvm::SmallPtrSetImpl<Operation *> &ops,
                              const TaskGraphNode *node) {
  if (!node)
    return false;
  mlir::sculptor::TaskCreateOp taskOp = node->op;
  return ops.contains(taskOp.getOperation());
}

static bool isComponentOutputResource(
    Value resource, const llvm::SmallPtrSetImpl<Operation *> &componentOps,
    const llvm::DenseMap<Value, llvm::SmallVector<const TaskGraphNode *, 4>>
        &consumersByResource) {
  if (resource.getDefiningOp<mlir::sculptor::TaskGraphOutputOp>())
    return true;

  auto consumerIt = consumersByResource.find(resource);
  if (consumerIt == consumersByResource.end())
    return false;

  for (const TaskGraphNode *consumer : consumerIt->second)
    if (!isNodeInComponent(componentOps, consumer))
      return true;
  return false;
}

static FailureOr<ComponentBoundary> computeComponentBoundary(
    const IslandFusibleComponent &component, const TaskGraphDAG &dag,
    const llvm::DenseMap<Value, const TaskGraphNode *> &producerByResource,
    const llvm::DenseMap<Value, llvm::SmallVector<const TaskGraphNode *, 4>>
        &consumersByResource,
    const llvm::DenseMap<Value, llvm::SmallVector<const TaskGraphNode *, 4>>
        &consumersByTaskResult) {
  ComponentBoundary boundary;
  llvm::SmallPtrSet<Operation *, 16> componentOps;
  for (const TaskGraphNode *node : component.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node->op;
    componentOps.insert(taskOp.getOperation());
  }

  for (const TaskGraphNode *node : component.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node->op;
    for (Value input : taskOp.getInputs()) {
      auto producerIt = producerByResource.find(input);
      if (producerIt != producerByResource.end() &&
          isNodeInComponent(componentOps, producerIt->second))
        continue;
      appendUniqueValue(boundary.inputs, input);
    }

    for (Value dependency : taskOp.getDependencies()) {
      auto dependencyTask =
          dependency.getDefiningOp<mlir::sculptor::TaskCreateOp>();
      if (!dependencyTask)
        return taskOp.emitError("expected component task dependency to be "
                                "produced by a task");
      if (componentOps.contains(dependencyTask.getOperation()))
        continue;
      appendUniqueValue(boundary.dependencies, dependency);
      boundary.latestDependency = getLaterOperation(
          boundary.latestDependency, dependencyTask.getOperation());
    }

    for (Value output : taskOp.getOutputs()) {
      if (isComponentOutputResource(output, componentOps, consumersByResource))
        appendUniqueValue(boundary.outputs, output);
    }

    auto consumersIt = consumersByTaskResult.find(taskOp.getResult());
    if (consumersIt == consumersByTaskResult.end())
      continue;
    for (const TaskGraphNode *consumer : consumersIt->second) {
      if (isNodeInComponent(componentOps, consumer))
        continue;
      mlir::sculptor::TaskCreateOp consumerTask = consumer->op;
      boundary.earliestExternalUser = getEarlierOperation(
          boundary.earliestExternalUser, consumerTask.getOperation());
    }
  }

  return boundary;
}

static bool
hasLegalComponentInsertionWindow(const ComponentBoundary &boundary) {
  return !boundary.latestDependency || !boundary.earliestExternalUser ||
         isBeforeInSameBlock(boundary.latestDependency,
                             boundary.earliestExternalUser);
}

static FailureOr<Type> getTaskResourceValueType(Value resource) {
  auto resourceType =
      dyn_cast<mlir::sculptor::TaskResourceType>(resource.getType());
  if (!resourceType) {
    if (Operation *op = resource.getDefiningOp())
      return op->emitError("expected fused task boundary value to be a task "
                           "resource");
    return failure();
  }
  return resourceType.getValueType();
}

static LogicalResult collectComponentMappedReturnValues(
    func::FuncOp childFunc, func::ReturnOp childReturn,
    const IRMapping &mapping, SmallVectorImpl<Value> &mappedReturns) {
  for (Value returnValue : childReturn.getOperands()) {
    Value mapped = mapping.lookupOrNull(returnValue);
    if (!mapped) {
      childFunc.emitError("expected component return operand to be mapped");
      return failure();
    }
    mappedReturns.push_back(mapped);
  }
  return success();
}

static FailureOr<func::FuncOp>
buildIslandComponentCallee(ModuleOp module,
                           const IslandFusibleComponent &component,
                           const ComponentBoundary &boundary) {
  MLIRContext *context = module.getContext();
  Builder builder(context);

  SmallVector<Type, 8> inputTypes;
  inputTypes.reserve(boundary.inputs.size());
  for (Value input : boundary.inputs) {
    FailureOr<Type> valueType = getTaskResourceValueType(input);
    if (failed(valueType))
      return failure();
    inputTypes.push_back(*valueType);
  }

  SmallVector<Type, 8> resultTypes;
  resultTypes.reserve(boundary.outputs.size());
  for (Value output : boundary.outputs) {
    FailureOr<Type> valueType = getTaskResourceValueType(output);
    if (failed(valueType))
      return failure();
    resultTypes.push_back(*valueType);
  }

  std::string functionName = "task_";
  functionName += sanitizeSymbolComponent(component.fusedTaskName);
  functionName += "_";
  mlir::sculptor::TaskCreateOp firstTask = component.nodes.front()->op;
  functionName += std::to_string(firstTask.getSourceTaskOrdinal());
  unsigned suffix = 0;
  std::string candidate = functionName;
  while (module.lookupSymbol<func::FuncOp>(candidate)) {
    candidate = functionName + "_" + std::to_string(suffix++);
  }
  functionName = candidate;

  auto functionType = builder.getFunctionType(inputTypes, resultTypes);
  func::FuncOp fusedFunc =
      func::FuncOp::create(firstTask.getLoc(), functionName, functionType);
  fusedFunc.setPrivate();

  auto domain = builder.getStringAttr(kFusedMixedTaskDomain);
  auto kind = builder.getStringAttr(kFusedMixedTaskKind);
  auto name = builder.getStringAttr(component.fusedTaskName);
  fusedFunc->setAttr(task_attrs::kTaskDomainAttrName, domain);
  fusedFunc->setAttr(task_attrs::kTaskKindAttrName, kind);
  fusedFunc->setAttr(task_attrs::kTaskNameAttrName, name);
  fusedFunc->setAttr(task_attrs::kSourceLayerAttrName,
                     builder.getStringAttr(firstTask.getSourceLayer()));
  fusedFunc->setAttr(
      task_attrs::kSourceTaskOrdinalAttrName,
      builder.getI64IntegerAttr(firstTask.getSourceTaskOrdinal()));

  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  moduleBuilder.insert(fusedFunc);

  Block *entryBlock = fusedFunc.addEntryBlock();
  OpBuilder bodyBuilder(entryBlock, entryBlock->begin());

  llvm::DenseMap<Value, Value> resourceValueByResource;
  for (auto indexedInput : llvm::enumerate(boundary.inputs))
    resourceValueByResource.try_emplace(
        indexedInput.value(), entryBlock->getArgument(indexedInput.index()));

  for (const TaskGraphNode *node : component.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node->op;
    auto callee = lookupTaskCallee(module, taskOp);
    if (failed(callee))
      return failure();

    func::FuncOp calleeFunc = *callee;
    auto returnOp = dyn_cast_or_null<func::ReturnOp>(
        calleeFunc.getBody().front().getTerminator());
    if (!returnOp) {
      taskOp.emitError("expected component task callee to terminate with "
                       "func.return");
      return failure();
    }

    Block &calleeBlock = calleeFunc.getBody().front();
    if (calleeBlock.getNumArguments() != taskOp.getInputs().size() ||
        returnOp.getNumOperands() != taskOp.getOutputs().size()) {
      taskOp.emitError("expected component task callee signature to match "
                       "task inputs and outputs");
      return failure();
    }

    IRMapping mapping;
    for (auto indexedInput : llvm::enumerate(taskOp.getInputs())) {
      auto mappedInput = resourceValueByResource.find(indexedInput.value());
      if (mappedInput == resourceValueByResource.end()) {
        taskOp.emitError("expected component task input to be available in "
                         "fused component");
        return failure();
      }
      mapping.map(calleeBlock.getArgument(indexedInput.index()),
                  mappedInput->second);
    }

    bodyBuilder.setInsertionPointToEnd(entryBlock);
    for (Operation &op : calleeBlock.without_terminator()) {
      for (Value operand : op.getOperands()) {
        if (!mapping.contains(operand)) {
          taskOp.emitError("expected component task operation operands to be "
                           "mapped in fused component");
          return failure();
        }
      }
      bodyBuilder.clone(op, mapping);
    }

    SmallVector<Value, 4> mappedReturns;
    if (failed(collectComponentMappedReturnValues(calleeFunc, returnOp, mapping,
                                                  mappedReturns)))
      return failure();

    for (auto indexedOutput : llvm::enumerate(taskOp.getOutputs()))
      resourceValueByResource[indexedOutput.value()] =
          mappedReturns[indexedOutput.index()];
  }

  SmallVector<Value, 8> fusedReturns;
  fusedReturns.reserve(boundary.outputs.size());
  for (Value output : boundary.outputs) {
    auto mappedOutput = resourceValueByResource.find(output);
    if (mappedOutput == resourceValueByResource.end()) {
      firstTask.emitError(
          "expected fused component output resource to be produced");
      return failure();
    }
    fusedReturns.push_back(mappedOutput->second);
  }
  bodyBuilder.setInsertionPointToEnd(entryBlock);
  bodyBuilder.create<func::ReturnOp>(fusedFunc.getLoc(), fusedReturns);

  return fusedFunc;
}

static void replaceComponentDependenciesWithFusedTask(
    mlir::sculptor::TaskCreateOp fusedTask,
    const llvm::SmallPtrSetImpl<Operation *> &componentOps,
    func::FuncOp taskGraphFunc) {
  Value fusedResult = fusedTask.getResult();
  for (Operation &op : taskGraphFunc.getBody().front()) {
    auto taskOp = dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp || componentOps.contains(taskOp.getOperation()) ||
        taskOp == fusedTask)
      continue;

    bool changed = false;
    bool hasFusedDependency = hasDependency(taskOp, fusedResult);
    SmallVector<Value, 8> dependencies;
    dependencies.reserve(taskOp.getDependencies().size());
    for (Value dependency : taskOp.getDependencies()) {
      auto dependencyTask =
          dependency.getDefiningOp<mlir::sculptor::TaskCreateOp>();
      if (!dependencyTask ||
          !componentOps.contains(dependencyTask.getOperation())) {
        dependencies.push_back(dependency);
        continue;
      }

      changed = true;
      if (!hasFusedDependency) {
        dependencies.push_back(fusedResult);
        hasFusedDependency = true;
      }
    }

    if (changed)
      taskOp.getDependenciesMutable().assign(dependencies);
  }
}

static FailureOr<mlir::sculptor::TaskCreateOp>
fuseIslandComponent(ModuleOp module, func::FuncOp taskGraphFunc,
                    const IslandFusibleComponent &component,
                    const ComponentBoundary &boundary) {
  FailureOr<func::FuncOp> fusedFunc =
      buildIslandComponentCallee(module, component, boundary);
  if (failed(fusedFunc))
    return failure();

  OpBuilder builder(module.getContext());
  if (boundary.latestDependency)
    builder.setInsertionPointAfter(boundary.latestDependency);
  else
    builder.setInsertionPoint(component.nodes.front()->op);

  mlir::sculptor::TaskCreateOp firstTask = component.nodes.front()->op;
  auto fusedTask = builder.create<mlir::sculptor::TaskCreateOp>(
      firstTask.getLoc(), firstTask.getResult().getType(), firstTask.getGraph(),
      FlatSymbolRefAttr::get(builder.getContext(), fusedFunc->getSymName()),
      builder.getStringAttr(kFusedMixedTaskDomain),
      builder.getStringAttr(kFusedMixedTaskKind),
      builder.getStringAttr(component.fusedTaskName),
      builder.getStringAttr(firstTask.getSourceLayer()),
      builder.getI64IntegerAttr(firstTask.getSourceTaskOrdinal()),
      boundary.inputs, boundary.outputs, boundary.dependencies);

  fusedTask->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                     builder.getI64IntegerAttr(component.coreId));
  (*fusedFunc)
      ->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                builder.getI64IntegerAttr(component.coreId));
  fusedTask->setAttr(schedule_attrs::kIslandIdAttrName,
                     builder.getI64IntegerAttr(component.islandId));
  if (component.physicalArrayId) {
    fusedTask->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
                       builder.getI64IntegerAttr(*component.physicalArrayId));
    (*fusedFunc)
        ->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
                  builder.getI64IntegerAttr(*component.physicalArrayId));
  }
  if (component.localArrayId) {
    fusedTask->setAttr(runtime_attrs::kTaskLocalArrayIdAttrName,
                       builder.getI64IntegerAttr(*component.localArrayId));
    (*fusedFunc)
        ->setAttr(runtime_attrs::kTaskLocalArrayIdAttrName,
                  builder.getI64IntegerAttr(*component.localArrayId));
  }

  int64_t digitalOps = 0;
  for (const TaskGraphNode *node : component.nodes) {
    if (auto taskDigitalOps = node->op->getAttrOfType<IntegerAttr>(
            runtime_attrs::kTaskDigitalOpsAttrName))
      digitalOps += taskDigitalOps.getInt();
  }
  fusedTask->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                     builder.getI64IntegerAttr(digitalOps));

  auto sumTimingAttr = [&](llvm::StringRef attrName) -> std::optional<double> {
    double total = 0.0;
    for (const TaskGraphNode *node : component.nodes) {
      auto value = node->op->getAttrOfType<FloatAttr>(attrName);
      if (!value)
        return std::nullopt;
      total += value.getValueAsDouble();
    }
    return total;
  };
  for (llvm::StringRef attrName : {
           timing_attrs::kAnalogLoadLatencyNsAttrName,
           timing_attrs::kAnalogExecuteLatencyNsAttrName,
           timing_attrs::kAnalogStoreLatencyNsAttrName,
           timing_attrs::kIntrinsicLatencyNsAttrName,
       }) {
    if (std::optional<double> total = sumTimingAttr(attrName))
      fusedTask->setAttr(attrName, builder.getF64FloatAttr(*total));
  }

  llvm::SmallPtrSet<Operation *, 16> componentOps;
  for (const TaskGraphNode *node : component.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node->op;
    componentOps.insert(taskOp.getOperation());
  }
  replaceComponentDependenciesWithFusedTask(fusedTask, componentOps,
                                            taskGraphFunc);

  for (const TaskGraphNode *node : llvm::reverse(component.nodes)) {
    mlir::sculptor::TaskCreateOp taskOp = node->op;
    if (!taskOp.getResult().use_empty()) {
      taskOp.emitError("expected fused component task result to have no "
                       "remaining users");
      return failure();
    }
    taskOp.erase();
  }

  return fusedTask;
}

static LogicalResult fuseIslandComponents(ModuleOp module,
                                          func::FuncOp taskGraphFunc,
                                          const TaskGraphDAG &dag,
                                          bool &changed) {
  changed = false;

  llvm::SmallVector<IslandFusibleComponent, 16> components =
      collectIslandFusibleComponents(dag);
  if (components.empty())
    return success();

  llvm::DenseMap<Value, const TaskGraphNode *> producerByResource;
  llvm::DenseMap<Value, llvm::SmallVector<const TaskGraphNode *, 4>>
      consumersByResource;
  llvm::DenseMap<Value, llvm::SmallVector<const TaskGraphNode *, 4>>
      consumersByTaskResult;
  if (failed(collectResourceProducers(dag, producerByResource)))
    return failure();
  collectResourceConsumers(dag, consumersByResource);
  collectTaskConsumers(dag, consumersByTaskResult);

  for (const IslandFusibleComponent &component : components) {
    FailureOr<ComponentBoundary> boundary =
        computeComponentBoundary(component, dag, producerByResource,
                                 consumersByResource, consumersByTaskResult);
    if (failed(boundary))
      return failure();
    if (!hasLegalComponentInsertionWindow(*boundary))
      continue;

    if (failed(
            fuseIslandComponent(module, taskGraphFunc, component, *boundary)))
      return failure();

    changed = true;
    return success();
  }

  return success();
}

static llvm::ArrayRef<TaskFusionPattern> getDefaultTaskFusionPatterns() {
  static const TaskFusionPattern patterns[] = {
      {"same-island-core-component", fuseIslandComponents},
  };
  return patterns;
}

static LogicalResult applyPatternToFixedPoint(ModuleOp module,
                                              func::FuncOp taskGraphFunc,
                                              const TaskFusionPattern &pattern,
                                              TaskGraphDAG &dag) {
  while (true) {
    bool changed = false;
    if (failed(pattern.apply(module, taskGraphFunc, dag, changed))) {
      taskGraphFunc.emitError("failed task fusion pattern '")
          << pattern.name << "'";
      return failure();
    }

    if (!changed)
      return success();

    auto nextDag = task_graph::parseTaskGraphDAG(taskGraphFunc);
    if (failed(nextDag))
      return failure();

    dag = std::move(*nextDag);
  }
}

} // namespace

LogicalResult fuseTaskGraphRoutines(ModuleOp module, func::FuncOp taskGraphFunc,
                                    const TaskGraphDAG &dag) {
  TaskGraphDAG workingDag = dag;
  for (const TaskFusionPattern &pattern : getDefaultTaskFusionPatterns()) {
    if (failed(applyPatternToFixedPoint(module, taskGraphFunc, pattern,
                                        workingDag)))
      return failure();
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
