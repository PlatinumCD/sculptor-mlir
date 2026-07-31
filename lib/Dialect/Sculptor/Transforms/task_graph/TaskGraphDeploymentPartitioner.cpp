#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDeploymentPartitioner.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResourceUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::sculptor;

namespace deployment_attrs = mlir::sculptor::deployment_attrs;
namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;

enum class ResourceKind {
  Input,
  Output,
  Intermediate,
  Persistent,
};

struct TaskRecord {
  TaskCreateOp op;
  unsigned globalId = 0;
  int64_t coreId = 0;
};

struct ResourceConsumer {
  unsigned taskIndex = 0;
  unsigned inputIndex = 0;
};

struct ResourceRecord {
  Operation *op = nullptr;
  Value value;
  ResourceKind kind = ResourceKind::Intermediate;
  unsigned globalId = 0;
  std::optional<unsigned> producerTask;
  unsigned producerOutput = 0;
  SmallVector<ResourceConsumer, 4> consumers;
};

struct RouteRecord {
  unsigned id = 0;
  unsigned resourceIndex = 0;
  unsigned sourceTask = 0;
  unsigned sourceOutput = 0;
  int64_t sourceCore = 0;
  unsigned destinationTask = 0;
  unsigned destinationInput = 0;
  int64_t destinationCore = 0;
  int64_t byteSize = 0;
};

struct GraphAnalysis {
  func::FuncOp graphFunc;
  TaskGraphCreateOp graphCreate;
  SmallVector<TaskRecord, 32> tasks;
  SmallVector<ResourceRecord, 32> resources;
  DenseMap<Operation *, unsigned> taskIndex;
  DenseMap<Value, unsigned> resourceIndex;
  SmallVector<RouteRecord, 32> routes;
  SmallVector<SmallVector<std::optional<unsigned>, 8>, 32> inputRoute;
  SmallVector<SmallVector<SmallVector<unsigned, 2>, 4>, 32> outputRoutes;
  SmallVector<llvm::DenseSet<int64_t>, 32> localResourceCores;
  SmallVector<int64_t, 8> activeCores;
  SmallVector<SmallVector<int64_t, 4>, 4> inputOwners;
  SmallVector<int64_t, 4> outputOwners;
};

struct CoreBuildState {
  ModuleOp module;
  func::FuncOp graphFunc;
  Value graph;
  DenseMap<unsigned, Value> localResources;
  DenseMap<unsigned, Value> routeInputs;
  DenseMap<unsigned, Value> routeOutputs;
  DenseMap<unsigned, Value> tasks;
};

bool returnsTaskGraph(func::FuncOp func) {
  FunctionType type = func.getFunctionType();
  return type.getNumResults() == 1 && isa<TaskGraphType>(type.getResult(0));
}

std::optional<ResourceKind> classifyResource(Operation *op) {
  if (isa<TaskGraphInputOp>(op))
    return ResourceKind::Input;
  if (isa<TaskGraphOutputOp>(op))
    return ResourceKind::Output;
  if (isa<TaskGraphIntermediateOp>(op))
    return ResourceKind::Intermediate;
  if (isa<TaskGraphPersistentOp>(op))
    return ResourceKind::Persistent;
  return std::nullopt;
}

bool isFinalizedLayoutAttr(StringRef name) {
  return name == runtime_attrs::kTaskGraphResourceCountAttrName ||
         name == runtime_attrs::kTaskGraphInputSlotsAttrName ||
         name == runtime_attrs::kTaskGraphOutputSlotsAttrName ||
         name == runtime_attrs::kTaskGraphRouteInputSlotsAttrName ||
         name == runtime_attrs::kTaskGraphRouteOutputSlotsAttrName ||
         name == runtime_attrs::kTaskGraphTempOffsetsAttrName ||
         name == runtime_attrs::kTaskGraphTempBaseSlotAttrName ||
         name == runtime_attrs::kTaskGraphTempCountAttrName ||
         name == runtime_attrs::kTaskGraphWorkspaceSizeAttrName ||
         name == runtime_attrs::kResourceSlotAttrName ||
         name == runtime_attrs::kResourceTempIndexAttrName ||
         name == runtime_attrs::kResourceTempOffsetAttrName ||
         name == runtime_attrs::kTaskIndexAttrName ||
         name == runtime_attrs::kTaskInputSlotsAttrName ||
         name == runtime_attrs::kTaskOutputSlotsAttrName;
}

LogicalResult rejectFinalizedLayout(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    for (NamedAttribute attr : op->getAttrs()) {
      if (!isFinalizedLayoutAttr(attr.getName().getValue()))
        continue;
      op->emitError()
          << "sculptor-partition-task-graph-by-core must run before "
             "--sculptor-finalize-task-graph-resources; found stale runtime "
             "layout attribute '"
          << attr.getName().getValue() << "'";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult findGlobalTaskGraph(ModuleOp module, GraphAnalysis &analysis) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    if (analysis.graphFunc) {
      func.emitError("expected exactly one global task graph function");
      return failure();
    }
    analysis.graphFunc = func;
  }

  if (!analysis.graphFunc) {
    module.emitError("expected one global task graph function returning "
                     "!sculptor.task_graph");
    return failure();
  }
  if (!analysis.graphFunc.getBody().hasOneBlock()) {
    analysis.graphFunc.emitError(
        "expected the global task graph function to contain one block");
    return failure();
  }

  for (TaskGraphCreateOp create :
       analysis.graphFunc.getBody().front().getOps<TaskGraphCreateOp>()) {
    if (analysis.graphCreate) {
      create.emitError("expected exactly one sculptor.task_graph.create");
      return failure();
    }
    analysis.graphCreate = create;
  }
  if (!analysis.graphCreate) {
    analysis.graphFunc.emitError(
        "expected one sculptor.task_graph.create operation");
    return failure();
  }
  return success();
}

LogicalResult collectTasks(GraphAnalysis &analysis, int64_t numCores) {
  Block &block = analysis.graphFunc.getBody().front();
  for (TaskCreateOp task : block.getOps<TaskCreateOp>()) {
    auto coreAttr =
        task->getAttrOfType<IntegerAttr>(runtime_attrs::kTaskCoreIdAttrName);
    if (!coreAttr) {
      task.emitError("expected scheduled task to have integer attribute '")
          << runtime_attrs::kTaskCoreIdAttrName << "'";
      return failure();
    }

    int64_t coreId = coreAttr.getInt();
    if (coreId < 0 || coreId >= numCores) {
      task.emitError("scheduled core ID ")
          << coreId << " is outside hardware core range [0, " << numCores
          << ")";
      return failure();
    }

    unsigned taskIndex = analysis.tasks.size();
    analysis.taskIndex[task.getOperation()] = taskIndex;
    analysis.tasks.push_back(TaskRecord{task, taskIndex, coreId});
  }
  if (analysis.tasks.empty()) {
    analysis.graphFunc.emitError(
        "expected the global task graph to contain at least one task");
    return failure();
  }

  std::set<int64_t> cores;
  for (const TaskRecord &task : analysis.tasks)
    cores.insert(task.coreId);
  analysis.activeCores.assign(cores.begin(), cores.end());
  return success();
}

LogicalResult collectResources(GraphAnalysis &analysis) {
  Block &block = analysis.graphFunc.getBody().front();
  for (Operation &op : block) {
    std::optional<ResourceKind> kind = classifyResource(&op);
    if (!kind)
      continue;

    Value value = op.getResult(0);
    auto resourceType = dyn_cast<TaskResourceType>(value.getType());
    if (!resourceType) {
      op.emitError("expected a task resource result");
      return failure();
    }
    if (isa<LogicalArrayType>(resourceType.getValueType())) {
      op.emitError(
          "expected logical-array ABI lowering before per-core partitioning");
      return failure();
    }

    unsigned resourceIndex = analysis.resources.size();
    analysis.resourceIndex[value] = resourceIndex;
    analysis.resources.push_back(
        ResourceRecord{&op, value, *kind, resourceIndex});
  }
  return success();
}

FailureOr<unsigned> lookupResourceIndex(GraphAnalysis &analysis, Value value,
                                        Operation *user) {
  auto it = analysis.resourceIndex.find(value);
  if (it != analysis.resourceIndex.end())
    return it->second;
  user->emitError("references a task resource not owned by the global graph");
  return failure();
}

LogicalResult connectResources(GraphAnalysis &analysis) {
  for (TaskRecord task : analysis.tasks) {
    for (auto indexedOutput : llvm::enumerate(task.op.getOutputs())) {
      FailureOr<unsigned> resourceIndex =
          lookupResourceIndex(analysis, indexedOutput.value(), task.op);
      if (failed(resourceIndex))
        return failure();
      ResourceRecord &resource = analysis.resources[*resourceIndex];
      if (resource.producerTask) {
        task.op.emitError("task resource has more than one producer");
        return failure();
      }
      resource.producerTask = task.globalId;
      resource.producerOutput = indexedOutput.index();
    }

    for (auto indexedInput : llvm::enumerate(task.op.getInputs())) {
      FailureOr<unsigned> resourceIndex =
          lookupResourceIndex(analysis, indexedInput.value(), task.op);
      if (failed(resourceIndex))
        return failure();
      analysis.resources[*resourceIndex].consumers.push_back(ResourceConsumer{
          task.globalId, static_cast<unsigned>(indexedInput.index())});
    }
  }
  return success();
}

LogicalResult deriveOwnershipAndRoutes(GraphAnalysis &analysis) {
  analysis.localResourceCores.resize(analysis.resources.size());
  analysis.inputRoute.resize(analysis.tasks.size());
  analysis.outputRoutes.resize(analysis.tasks.size());
  for (TaskRecord task : analysis.tasks) {
    analysis.inputRoute[task.globalId].resize(task.op.getInputs().size());
    analysis.outputRoutes[task.globalId].resize(task.op.getOutputs().size());
  }

  for (ResourceRecord &resource : analysis.resources) {
    switch (resource.kind) {
    case ResourceKind::Input: {
      if (resource.producerTask) {
        resource.op->emitError("model input resource cannot have a producer");
        return failure();
      }
      std::set<int64_t> consumerCores;
      for (const ResourceConsumer &consumer : resource.consumers)
        consumerCores.insert(analysis.tasks[consumer.taskIndex].coreId);
      if (consumerCores.empty())
        consumerCores.insert(analysis.activeCores.front());
      SmallVector<int64_t, 4> owners(consumerCores.begin(),
                                     consumerCores.end());
      for (int64_t owner : owners)
        analysis.localResourceCores[resource.globalId].insert(owner);
      analysis.inputOwners.push_back(std::move(owners));
      break;
    }
    case ResourceKind::Output: {
      if (!resource.producerTask) {
        resource.op->emitError("model output resource requires one producer");
        return failure();
      }
      int64_t owner = analysis.tasks[*resource.producerTask].coreId;
      analysis.localResourceCores[resource.globalId].insert(owner);
      analysis.outputOwners.push_back(owner);
      break;
    }
    case ResourceKind::Persistent: {
      if (resource.producerTask) {
        resource.op->emitError(
            "persistent resource cannot be produced by a task");
        return failure();
      }
      for (const ResourceConsumer &consumer : resource.consumers) {
        analysis.localResourceCores[resource.globalId].insert(
            analysis.tasks[consumer.taskIndex].coreId);
      }
      if (resource.consumers.empty()) {
        analysis.localResourceCores[resource.globalId].insert(
            analysis.activeCores.front());
      }
      break;
    }
    case ResourceKind::Intermediate: {
      if (!resource.producerTask) {
        resource.op->emitError(
            "intermediate resource requires one task producer");
        return failure();
      }
      unsigned producerTask = *resource.producerTask;
      int64_t producerCore = analysis.tasks[producerTask].coreId;
      bool needsLocalResource = resource.consumers.empty();
      for (const ResourceConsumer &consumer : resource.consumers) {
        if (analysis.tasks[consumer.taskIndex].coreId == producerCore)
          needsLocalResource = true;
      }
      if (needsLocalResource)
        analysis.localResourceCores[resource.globalId].insert(producerCore);
      break;
    }
    }

    if (!resource.producerTask)
      continue;
    unsigned producerTask = *resource.producerTask;
    int64_t producerCore = analysis.tasks[producerTask].coreId;
    for (const ResourceConsumer &consumer : resource.consumers) {
      int64_t consumerCore = analysis.tasks[consumer.taskIndex].coreId;
      if (consumerCore == producerCore)
        continue;

      auto resourceType =
          cast<TaskResourceType>(resource.value.getType()).getValueType();
      if (!isa<RankedTensorType>(resourceType)) {
        resource.op->emitError(
            "cannot route a non-tensor resource between cores");
        return failure();
      }
      FailureOr<int64_t> byteSize = getTaskResourceByteSize(resource.value);
      if (failed(byteSize)) {
        resource.op->emitError(
            "cannot determine static byte size for cross-core resource");
        return failure();
      }

      analysis.routes.push_back(RouteRecord{
          0,
          resource.globalId,
          producerTask,
          resource.producerOutput,
          producerCore,
          consumer.taskIndex,
          consumer.inputIndex,
          consumerCore,
          *byteSize,
      });
    }
  }

  llvm::sort(
      analysis.routes, [](const RouteRecord &lhs, const RouteRecord &rhs) {
        return std::tie(lhs.sourceTask, lhs.sourceOutput, lhs.destinationCore,
                        lhs.destinationTask, lhs.destinationInput) <
               std::tie(rhs.sourceTask, rhs.sourceOutput, rhs.destinationCore,
                        rhs.destinationTask, rhs.destinationInput);
      });
  for (auto indexedRoute : llvm::enumerate(analysis.routes)) {
    RouteRecord &route = indexedRoute.value();
    route.id = indexedRoute.index();
    analysis.inputRoute[route.destinationTask][route.destinationInput] =
        route.id;
    analysis.outputRoutes[route.sourceTask][route.sourceOutput].push_back(
        route.id);
  }

  return success();
}

bool hasRouteBetween(const GraphAnalysis &analysis, unsigned sourceTask,
                     unsigned destinationTask) {
  return llvm::any_of(analysis.routes, [&](const RouteRecord &route) {
    return route.sourceTask == sourceTask &&
           route.destinationTask == destinationTask;
  });
}

LogicalResult validateDependencies(GraphAnalysis &analysis) {
  for (TaskRecord task : analysis.tasks) {
    for (Value dependency : task.op.getDependencies()) {
      auto dependencyTask = dependency.getDefiningOp<TaskCreateOp>();
      auto dependencyIt =
          dependencyTask
              ? analysis.taskIndex.find(dependencyTask.getOperation())
              : analysis.taskIndex.end();
      if (!dependencyTask || dependencyIt == analysis.taskIndex.end()) {
        task.op.emitError("has a dependency outside the global task graph");
        return failure();
      }

      unsigned sourceTask = dependencyIt->second;
      if (analysis.tasks[sourceTask].coreId == task.coreId)
        continue;
      if (hasRouteBetween(analysis, sourceTask, task.globalId))
        continue;

      task.op.emitError()
          << "cannot partition cross-core control-only dependency from global "
             "task "
          << sourceTask << " on core " << analysis.tasks[sourceTask].coreId
          << " to global task " << task.globalId << " on core " << task.coreId
          << ": no routed tensor edge can satisfy the dependency";
      return failure();
    }
  }
  return success();
}

FailureOr<int64_t> getHardwareCoreCount(ModuleOp module,
                                        func::FuncOp graphFunc) {
  auto attr =
      module->getAttrOfType<IntegerAttr>(schedule_attrs::kNumCoresAttrName);
  if (!attr) {
    attr = graphFunc->getAttrOfType<IntegerAttr>(
        schedule_attrs::kNumCoresAttrName);
  }
  if (!attr || attr.getInt() <= 0) {
    module.emitError("expected positive integer hardware attribute '")
        << schedule_attrs::kNumCoresAttrName << "'";
    return failure();
  }
  return attr.getInt();
}

LogicalResult analyzeGraph(ModuleOp module, GraphAnalysis &analysis) {
  if (failed(rejectFinalizedLayout(module)) ||
      failed(findGlobalTaskGraph(module, analysis)))
    return failure();

  FailureOr<int64_t> numCores =
      getHardwareCoreCount(module, analysis.graphFunc);
  if (failed(numCores) || failed(collectTasks(analysis, *numCores)) ||
      failed(collectResources(analysis)) ||
      failed(connectResources(analysis)) ||
      failed(deriveOwnershipAndRoutes(analysis)) ||
      failed(validateDependencies(analysis)))
    return failure();
  return success();
}

void copyNonLayoutAttrs(Operation *source, Operation *target,
                        ArrayRef<StringRef> skipped = {}) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (isFinalizedLayoutAttr(name) ||
        name == deployment_attrs::kGlobalTaskIdAttrName ||
        name == deployment_attrs::kGlobalResourceIdAttrName ||
        name == deployment_attrs::kRouteIdAttrName ||
        llvm::is_contained(skipped, name))
      continue;
    target->setAttr(attr.getName(), attr.getValue());
  }
}

Operation *createLocalResource(OpBuilder &builder, Location loc, Value graph,
                               const ResourceRecord &resource) {
  OperationState state(loc, resource.op->getName());
  state.addOperands(graph);
  state.addTypes(resource.value.getType());
  Operation *created = builder.create(state);
  copyNonLayoutAttrs(resource.op, created);
  created->setAttr(
      deployment_attrs::kGlobalResourceIdAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(resource.globalId)));
  return created;
}

Value createRouteResource(OpBuilder &builder, Location loc, Value graph,
                          const ResourceRecord &resource,
                          const RouteRecord &route, bool incoming) {
  OperationState state(loc, incoming
                                ? TaskGraphRouteInputOp::getOperationName()
                                : TaskGraphRouteOutputOp::getOperationName());
  state.addOperands(graph);
  state.addTypes(resource.value.getType());
  state.addAttribute(deployment_attrs::kRouteIdAttrName,
                     builder.getI64IntegerAttr(route.id));
  state.addAttribute(deployment_attrs::kGlobalResourceIdAttrName,
                     builder.getI64IntegerAttr(resource.globalId));
  return builder.create(state)->getResult(0);
}

FailureOr<SmallVector<unsigned, 4>>
getOriginalResultIndices(TaskCreateOp task) {
  SmallVector<unsigned, 4> resultIndices;
  auto attr =
      task->getAttrOfType<ArrayAttr>(runtime_attrs::kTaskResultIndicesAttrName);
  if (!attr) {
    for (unsigned index = 0; index < task.getOutputs().size(); ++index)
      resultIndices.push_back(index);
    return resultIndices;
  }

  resultIndices.reserve(attr.size());
  for (Attribute value : attr) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer || integer.getInt() < 0) {
      task.emitError("expected runtime result indices to contain "
                     "non-negative integers");
      return failure();
    }
    resultIndices.push_back(static_cast<unsigned>(integer.getInt()));
  }
  return resultIndices;
}

LogicalResult cloneTaskIntoCore(OpBuilder &builder, GraphAnalysis &analysis,
                                CoreBuildState &state, TaskRecord task) {
  SmallVector<Value, 8> inputs;
  for (auto indexedInput : llvm::enumerate(task.op.getInputs())) {
    std::optional<unsigned> routeId =
        analysis.inputRoute[task.globalId][indexedInput.index()];
    if (routeId) {
      auto routeIt = state.routeInputs.find(*routeId);
      if (routeIt == state.routeInputs.end()) {
        task.op.emitError("missing local route input while partitioning");
        return failure();
      }
      inputs.push_back(routeIt->second);
      continue;
    }

    auto resourceIt = analysis.resourceIndex.find(indexedInput.value());
    auto localIt = resourceIt == analysis.resourceIndex.end()
                       ? state.localResources.end()
                       : state.localResources.find(resourceIt->second);
    if (localIt == state.localResources.end()) {
      task.op.emitError("missing local task input while partitioning");
      return failure();
    }
    inputs.push_back(localIt->second);
  }

  SmallVector<Value, 8> outputs;
  SmallVector<int64_t, 8> resultIndices;
  FailureOr<SmallVector<unsigned, 4>> originalResultIndices =
      getOriginalResultIndices(task.op);
  if (failed(originalResultIndices))
    return failure();
  if (originalResultIndices->size() != task.op.getOutputs().size()) {
    task.op.emitError("expected runtime result index metadata to match task "
                      "output count");
    return failure();
  }

  for (auto indexedOutput : llvm::enumerate(task.op.getOutputs())) {
    auto resourceIt = analysis.resourceIndex.find(indexedOutput.value());
    if (resourceIt == analysis.resourceIndex.end()) {
      task.op.emitError("missing task output resource while partitioning");
      return failure();
    }
    auto localIt = state.localResources.find(resourceIt->second);
    if (localIt != state.localResources.end()) {
      outputs.push_back(localIt->second);
      resultIndices.push_back((*originalResultIndices)[indexedOutput.index()]);
    }
    for (unsigned routeId :
         analysis.outputRoutes[task.globalId][indexedOutput.index()]) {
      auto routeIt = state.routeOutputs.find(routeId);
      if (routeIt == state.routeOutputs.end()) {
        task.op.emitError("missing local route output while partitioning");
        return failure();
      }
      outputs.push_back(routeIt->second);
      resultIndices.push_back((*originalResultIndices)[indexedOutput.index()]);
    }
  }

  SmallVector<Value, 8> dependencies;
  for (Value dependency : task.op.getDependencies()) {
    TaskCreateOp dependencyTask = dependency.getDefiningOp<TaskCreateOp>();
    unsigned dependencyIndex =
        analysis.taskIndex.lookup(dependencyTask.getOperation());
    if (analysis.tasks[dependencyIndex].coreId != task.coreId)
      continue;
    auto localIt = state.tasks.find(dependencyIndex);
    if (localIt == state.tasks.end()) {
      task.op.emitError(
          "expected local dependencies to precede their consumers");
      return failure();
    }
    dependencies.push_back(localIt->second);
  }

  auto cloned = builder.create<TaskCreateOp>(
      task.op.getLoc(), task.op.getResult().getType(), state.graph,
      task.op.getCalleeAttr(), task.op.getDomainAttr(),
      task.op.getTaskKindAttr(), task.op.getTaskNameAttr(),
      task.op.getSourceLayerAttr(), task.op.getSourceTaskOrdinalAttr(), inputs,
      outputs, dependencies);

  static const StringRef skippedAttrs[] = {
      "callee",
      "domain",
      "taskKind",
      "taskName",
      "sourceLayer",
      "sourceTaskOrdinal",
      "operandSegmentSizes",
      runtime_attrs::kTaskResultIndicesAttrName,
  };
  copyNonLayoutAttrs(task.op, cloned, skippedAttrs);
  cloned->setAttr(deployment_attrs::kGlobalTaskIdAttrName,
                  builder.getI64IntegerAttr(task.globalId));
  if (!resultIndices.empty()) {
    cloned->setAttr(runtime_attrs::kTaskResultIndicesAttrName,
                    builder.getI64ArrayAttr(resultIndices));
  }
  state.tasks[task.globalId] = cloned.getResult();
  return success();
}

void copyGlobalGraphAttrs(func::FuncOp source, ModuleOp target) {
  for (NamedAttribute attr : source->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name.starts_with("sculptor.schedule.") ||
        name.starts_with("sculptor.timing."))
      target->setAttr(attr.getName(), attr.getValue());
  }
}

void copyCoreGraphAttrs(func::FuncOp source, func::FuncOp target) {
  static constexpr StringLiteral coreGraphAttrs[] = {
      schedule_attrs::kNumCoresAttrName,
      schedule_attrs::kArraysPerCoreAttrName,
      schedule_attrs::kTopologyAttrName,
      schedule_attrs::kMeshRowsAttrName,
      schedule_attrs::kMeshColsAttrName,
      schedule_attrs::kNumAnalogArraysAttrName,
      schedule_attrs::kAnalogArraysAttrName,
      schedule_attrs::kPlacementCostModeAttrName,
      schedule_attrs::kSearchCompletionTimeProxyAttrName,
      schedule_attrs::kSearchCommunicationProxyAttrName,
      schedule_attrs::kSearchResourceLoadProxyAttrName,
      "sculptor.timing.model",
  };
  for (StringRef name : coreGraphAttrs) {
    if (Attribute attr = source->getAttr(name))
      target->setAttr(name, attr);
  }
}

DeploymentRouteAttr buildRouteAttr(Builder &builder, const RouteRecord &route,
                                   const ResourceRecord &resource) {
  return DeploymentRouteAttr::get(
      builder.getContext(), builder.getI64IntegerAttr(route.id),
      builder.getI64IntegerAttr(route.sourceCore),
      builder.getI64IntegerAttr(route.sourceTask),
      builder.getI64IntegerAttr(route.sourceOutput),
      builder.getI64IntegerAttr(route.destinationCore),
      builder.getI64IntegerAttr(route.destinationTask),
      builder.getI64IntegerAttr(route.destinationInput),
      builder.getI64IntegerAttr(resource.globalId),
      builder.getI64IntegerAttr(route.byteSize));
}

ArrayAttr buildRouteArray(Builder &builder, const GraphAnalysis &analysis,
                          function_ref<bool(const RouteRecord &)> include) {
  SmallVector<Attribute, 8> attrs;
  for (const RouteRecord &route : analysis.routes) {
    if (include(route)) {
      attrs.push_back(buildRouteAttr(builder, route,
                                     analysis.resources[route.resourceIndex]));
    }
  }
  return builder.getArrayAttr(attrs);
}

DictionaryAttr buildOwnershipAttr(Builder &builder, StringRef ordinalName,
                                  unsigned ordinal, int64_t ownerCore,
                                  unsigned resourceId) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr(ordinalName, builder.getI64IntegerAttr(ordinal)),
      builder.getNamedAttr("owner_core", builder.getI64IntegerAttr(ownerCore)),
      builder.getNamedAttr("global_resource_id",
                           builder.getI64IntegerAttr(resourceId)),
  });
}

void attachDeploymentManifest(ModuleOp module, const GraphAnalysis &analysis) {
  Builder builder(module.getContext());
  copyGlobalGraphAttrs(analysis.graphFunc, module);
  if (!module->hasAttr(schedule_attrs::kNumCoresAttrName)) {
    module->setAttr(
        schedule_attrs::kNumCoresAttrName,
        analysis.graphFunc->getAttr(schedule_attrs::kNumCoresAttrName));
  }
  module->setAttr(deployment_attrs::kActiveCoreIdsAttrName,
                  builder.getI64ArrayAttr(analysis.activeCores));
  module->setAttr(deployment_attrs::kRoutesAttrName,
                  buildRouteArray(builder, analysis,
                                  [](const RouteRecord &) { return true; }));

  SmallVector<Attribute, 4> modelInputs;
  SmallVector<Attribute, 4> modelOutputs;
  unsigned inputOrdinal = 0;
  unsigned outputOrdinal = 0;
  for (const ResourceRecord &resource : analysis.resources) {
    if (resource.kind == ResourceKind::Input) {
      for (int64_t owner : analysis.inputOwners[inputOrdinal]) {
        modelInputs.push_back(buildOwnershipAttr(
            builder, "input_index", inputOrdinal, owner, resource.globalId));
      }
      ++inputOrdinal;
    } else if (resource.kind == ResourceKind::Output) {
      modelOutputs.push_back(buildOwnershipAttr(
          builder, "output_index", outputOrdinal,
          analysis.outputOwners[outputOrdinal], resource.globalId));
      ++outputOrdinal;
    }
  }
  module->setAttr(deployment_attrs::kModelInputsAttrName,
                  builder.getArrayAttr(modelInputs));
  module->setAttr(deployment_attrs::kModelOutputsAttrName,
                  builder.getArrayAttr(modelOutputs));
}

FailureOr<CoreBuildState>
createCoreModule(ModuleOp outer, GraphAnalysis &analysis, int64_t coreId) {
  OpBuilder outerBuilder(outer.getContext());
  outerBuilder.setInsertionPointToEnd(outer.getBody());
  std::string moduleName = ("core_" + std::to_string(coreId));
  ModuleOp coreModule =
      outerBuilder.create<ModuleOp>(analysis.graphFunc.getLoc(), moduleName);
  coreModule->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                      outerBuilder.getI64IntegerAttr(coreId));
  coreModule->setAttr(
      deployment_attrs::kIncomingRoutesAttrName,
      buildRouteArray(outerBuilder, analysis, [coreId](const RouteRecord &r) {
        return r.destinationCore == coreId;
      }));
  coreModule->setAttr(
      deployment_attrs::kOutgoingRoutesAttrName,
      buildRouteArray(outerBuilder, analysis, [coreId](const RouteRecord &r) {
        return r.sourceCore == coreId;
      }));

  OpBuilder builder(coreModule.getContext());
  builder.setInsertionPointToEnd(coreModule.getBody());
  auto graphType = TaskGraphType::get(builder.getContext());
  auto functionType = builder.getFunctionType({}, graphType);
  func::FuncOp localGraph = builder.create<func::FuncOp>(
      analysis.graphFunc.getLoc(), analysis.graphFunc.getSymName(),
      functionType);
  localGraph.setPrivate();
  copyCoreGraphAttrs(analysis.graphFunc, localGraph);

  Block *entry = localGraph.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto graphCreate = builder.create<TaskGraphCreateOp>(
      analysis.graphCreate.getLoc(), graphType);
  copyNonLayoutAttrs(analysis.graphCreate, graphCreate);

  CoreBuildState state;
  state.module = coreModule;
  state.graphFunc = localGraph;
  state.graph = graphCreate.getResult();

  auto createLocalResourcesOfKind = [&](ResourceKind kind) {
    for (const ResourceRecord &resource : analysis.resources) {
      if (resource.kind != kind ||
          !analysis.localResourceCores[resource.globalId].contains(coreId))
        continue;
      Operation *created = createLocalResource(builder, resource.op->getLoc(),
                                               state.graph, resource);
      state.localResources[resource.globalId] = created->getResult(0);
    }
  };

  createLocalResourcesOfKind(ResourceKind::Input);
  for (const RouteRecord &route : analysis.routes) {
    if (route.destinationCore != coreId)
      continue;
    state.routeInputs[route.id] = createRouteResource(
        builder, analysis.resources[route.resourceIndex].op->getLoc(),
        state.graph, analysis.resources[route.resourceIndex], route, true);
  }
  createLocalResourcesOfKind(ResourceKind::Persistent);
  createLocalResourcesOfKind(ResourceKind::Intermediate);
  for (const RouteRecord &route : analysis.routes) {
    if (route.sourceCore != coreId)
      continue;
    state.routeOutputs[route.id] = createRouteResource(
        builder, analysis.resources[route.resourceIndex].op->getLoc(),
        state.graph, analysis.resources[route.resourceIndex], route, false);
  }
  createLocalResourcesOfKind(ResourceKind::Output);

  int64_t localDependencyCount = 0;
  for (TaskRecord task : analysis.tasks) {
    if (task.coreId != coreId)
      continue;
    for (Value dependency : task.op.getDependencies()) {
      auto dependencyTask = dependency.getDefiningOp<TaskCreateOp>();
      unsigned dependencyIndex =
          analysis.taskIndex.lookup(dependencyTask.getOperation());
      if (analysis.tasks[dependencyIndex].coreId == coreId)
        ++localDependencyCount;
    }
    if (failed(cloneTaskIntoCore(builder, analysis, state, task)))
      return failure();
  }

  localGraph->setAttr(
      schedule_attrs::kTaskCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(state.tasks.size())));
  localGraph->setAttr(schedule_attrs::kDependencyCountAttrName,
                      builder.getI64IntegerAttr(localDependencyCount));
  builder.create<func::ReturnOp>(analysis.graphFunc.getLoc(), state.graph);
  return state;
}

Operation *getTopLevelSymbol(Operation *symbol, ModuleOp module) {
  Operation *top = symbol;
  while (top && top->getParentOp() != module)
    top = top->getParentOp();
  return top;
}

FailureOr<llvm::SetVector<Operation *>>
collectSymbolClosure(ModuleOp module, const GraphAnalysis &analysis,
                     int64_t coreId) {
  SymbolTable symbolTable(module);
  llvm::SetVector<Operation *> closure;
  SmallVector<Operation *, 16> worklist;
  auto addResolvedSymbol = [&](Operation *user,
                               SymbolRefAttr symbolRef) -> LogicalResult {
    Operation *referenced =
        SymbolTable::lookupNearestSymbolFrom(user, symbolRef);
    if (!referenced) {
      user->emitError("cannot resolve referenced symbol '") << symbolRef << "'";
      return failure();
    }
    Operation *top = getTopLevelSymbol(referenced, module);
    if (!top) {
      user->emitError(
          "referenced symbol does not belong to the deployment module");
      return failure();
    }
    if (closure.insert(top))
      worklist.push_back(top);
    return success();
  };

  for (TaskRecord task : analysis.tasks) {
    if (task.coreId != coreId)
      continue;
    Operation *callee = symbolTable.lookup(task.op.getCallee());
    if (!callee) {
      task.op.emitError("cannot resolve task callee '")
          << task.op.getCallee() << "'";
      return failure();
    }
    Operation *top = getTopLevelSymbol(callee, module);
    if (!top) {
      task.op.emitError("task callee does not resolve to a top-level symbol");
      return failure();
    }
    if (closure.insert(top))
      worklist.push_back(top);
  }

  for (const ResourceRecord &resource : analysis.resources) {
    if (!analysis.localResourceCores[resource.globalId].contains(coreId))
      continue;
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(resource.op);
    if (!uses) {
      resource.op->emitError(
          "cannot determine symbol closure for local task resource");
      return failure();
    }
    for (const SymbolTable::SymbolUse &use : *uses) {
      if (failed(addResolvedSymbol(use.getUser(), use.getSymbolRef())))
        return failure();
    }
  }

  while (!worklist.empty()) {
    Operation *symbol = worklist.pop_back_val();
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(symbol);
    if (!uses) {
      symbol->emitError(
          "cannot determine transitive symbol closure for per-core module");
      return failure();
    }
    for (const SymbolTable::SymbolUse &use : *uses) {
      if (failed(addResolvedSymbol(use.getUser(), use.getSymbolRef())))
        return failure();
    }
  }
  return closure;
}

void cloneSymbolClosure(ModuleOp source, ModuleOp destination,
                        const llvm::SetVector<Operation *> &closure) {
  OpBuilder builder(destination.getContext());
  builder.setInsertionPoint(destination.getBody()->getTerminator());
  for (Operation &op : *source.getBody()) {
    if (closure.contains(&op))
      builder.clone(op);
  }
}

LogicalResult verifyCoreIsolation(ModuleOp coreModule) {
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(coreModule);
  if (!uses) {
    coreModule.emitError(
        "cannot verify symbol isolation for deployed core module");
    return failure();
  }
  for (const SymbolTable::SymbolUse &use : *uses) {
    Operation *symbol =
        SymbolTable::lookupNearestSymbolFrom(use.getUser(), use.getSymbolRef());
    if (!symbol || !coreModule->isAncestor(symbol)) {
      use.getUser()->emitError("cross-module symbol reference '")
          << use.getSymbolRef() << "' remains after per-core partitioning";
      return failure();
    }
  }
  return success();
}

bool sourceSymbolsAreDeploymentOnly(
    ModuleOp module, func::FuncOp graphFunc,
    ArrayRef<llvm::SetVector<Operation *>> closures) {
  llvm::DenseSet<Operation *> symbols;
  for (const llvm::SetVector<Operation *> &closure : closures)
    symbols.insert(closure.begin(), closure.end());

  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(module);
  if (!uses)
    return false;
  for (const SymbolTable::SymbolUse &use : *uses) {
    Operation *referenced =
        SymbolTable::lookupNearestSymbolFrom(use.getUser(), use.getSymbolRef());
    Operation *topReferenced =
        referenced ? getTopLevelSymbol(referenced, module) : nullptr;
    if (!topReferenced || !symbols.contains(topReferenced))
      continue;

    Operation *topUser = use.getUser();
    while (topUser && topUser->getParentOp() != module)
      topUser = topUser->getParentOp();
    if (topUser != graphFunc.getOperation() && !symbols.contains(topUser))
      return false;
  }
  return true;
}

void eraseOriginalGraphAndSymbols(
    ModuleOp module, func::FuncOp graphFunc,
    ArrayRef<llvm::SetVector<Operation *>> closures, bool eraseSourceSymbols) {
  llvm::DenseSet<Operation *> symbols;
  for (const llvm::SetVector<Operation *> &closure : closures)
    symbols.insert(closure.begin(), closure.end());

  graphFunc.erase();
  if (!eraseSourceSymbols)
    return;

  SmallVector<Operation *, 32> orderedSymbols;
  for (Operation &op : *module.getBody()) {
    if (symbols.contains(&op))
      orderedSymbols.push_back(&op);
  }
  for (Operation *symbol : llvm::reverse(orderedSymbols))
    symbol->erase();
}

LogicalResult partitionGraph(ModuleOp module, GraphAnalysis &analysis) {
  SmallVector<llvm::SetVector<Operation *>, 8> closures;
  closures.reserve(analysis.activeCores.size());
  for (int64_t coreId : analysis.activeCores) {
    FailureOr<llvm::SetVector<Operation *>> closure =
        collectSymbolClosure(module, analysis, coreId);
    if (failed(closure))
      return failure();
    closures.push_back(std::move(*closure));
  }
  bool eraseSourceSymbols =
      sourceSymbolsAreDeploymentOnly(module, analysis.graphFunc, closures);

  attachDeploymentManifest(module, analysis);
  SmallVector<ModuleOp, 8> coreModules;
  for (auto indexedCore : llvm::enumerate(analysis.activeCores)) {
    int64_t coreId = indexedCore.value();
    FailureOr<CoreBuildState> state =
        createCoreModule(module, analysis, coreId);
    if (failed(state))
      return failure();
    cloneSymbolClosure(module, state->module, closures[indexedCore.index()]);
    coreModules.push_back(state->module);
  }

  for (ModuleOp coreModule : coreModules) {
    if (failed(verifyCoreIsolation(coreModule)) ||
        failed(verify(coreModule.getOperation()))) {
      coreModule.emitError("failed to construct an isolated core module");
      return failure();
    }
  }

  eraseOriginalGraphAndSymbols(module, analysis.graphFunc, closures,
                               eraseSourceSymbols);
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_graph {

LogicalResult partitionTaskGraphByCore(ModuleOp module) {
  GraphAnalysis analysis;
  if (failed(analyzeGraph(module, analysis)) ||
      failed(partitionGraph(module, analysis)))
    return failure();
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
