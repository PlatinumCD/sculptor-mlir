#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphReductionBalancer.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTimingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

struct ReductionLeaf {
  Value resource;
  llvm::SmallVector<Value, 4> dependencies;
};

static FailureOr<RankedTensorType> getReductionTensorType(TaskCreateOp taskOp) {
  if (taskOp.getOutputs().size() != 1)
    return failure();
  auto resourceType =
      dyn_cast<TaskResourceType>(taskOp.getOutputs().front().getType());
  if (!resourceType)
    return failure();
  auto tensorType = dyn_cast<RankedTensorType>(resourceType.getValueType());
  if (!tensorType)
    return failure();
  return tensorType;
}

static LogicalResult rejectStalePlacementMetadata(func::FuncOp taskGraphFunc) {
  if (taskGraphFunc->hasAttr(schedule_attrs::kNumCoresAttrName) ||
      taskGraphFunc->hasAttr(timing_attrs::kTimingModelAttrName)) {
    return taskGraphFunc.emitError(
        "task reduction balancing must run before timing and scheduling");
  }

  for (TaskCreateOp taskOp : taskGraphFunc.getOps<TaskCreateOp>()) {
    if (taskOp->hasAttr(schedule_attrs::kIslandIdAttrName) ||
        taskOp->hasAttr(runtime_attrs::kTaskCoreIdAttrName) ||
        taskOp->hasAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName)) {
      return taskOp.emitError(
          "task reduction balancing must run before island construction and "
          "placement");
    }
  }
  return success();
}

static std::string getReductionKindName(TaskReductionKind kind) {
  return stringifyTaskReductionKind(kind).str();
}

static std::string getReductionHelperBaseName(TaskReductionKind kind,
                                              RankedTensorType tensorType,
                                              unsigned arity, int64_t treeId,
                                              int64_t level,
                                              std::optional<int64_t> lane) {
  std::string typeText;
  llvm::raw_string_ostream stream(typeText);
  tensorType.print(stream);
  stream.flush();
  uint64_t typeHash =
      static_cast<uint64_t>(static_cast<size_t>(llvm::hash_value(typeText)));
  return "__sculptor_reduce_" + getReductionKindName(kind) + "_" +
         std::to_string(arity) + "_tree" + std::to_string(treeId) + "_level" +
         std::to_string(level) +
         (lane ? "_lane" + std::to_string(*lane) : "_root") + "_" +
         llvm::utohexstr(typeHash, /*LowerCase=*/true);
}

static Value buildReductionElement(TaskReductionKind kind, OpBuilder &builder,
                                   Location loc, Value lhs, Value rhs) {
  switch (kind) {
  case TaskReductionKind::Add:
    return builder.create<arith::AddFOp>(loc, lhs, rhs);
  case TaskReductionKind::Max:
    return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
  case TaskReductionKind::Min:
    return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
  }
  llvm_unreachable("unknown task reduction kind");
}

class ReductionCalleeCache {
public:
  explicit ReductionCalleeCache(ModuleOp module) : module(module) {}

  FailureOr<func::FuncOp> getOrCreate(Location loc, TaskReductionAttr reduction,
                                      RankedTensorType tensorType,
                                      unsigned arity, int64_t treeId,
                                      int64_t level,
                                      std::optional<int64_t> lane) {
    if (arity < 2)
      return failure();

    std::string cacheKey = getReductionKindName(reduction.getKind()) + ":" +
                           std::to_string(arity) + ":" +
                           std::to_string(treeId) + ":" +
                           std::to_string(level) + ":" +
                           (lane ? std::to_string(*lane) : "root") + ":";
    llvm::raw_string_ostream keyStream(cacheKey);
    tensorType.print(keyStream);
    keyStream.flush();
    auto cached = helperByKey.find(cacheKey);
    if (cached != helperByKey.end())
      return cached->second;

    llvm::SmallVector<Type, 8> inputTypes(arity, tensorType);
    FunctionType functionType =
        FunctionType::get(module.getContext(), inputTypes, tensorType);
    std::string baseName = getReductionHelperBaseName(
        reduction.getKind(), tensorType, arity, treeId, level, lane);
    std::string functionName = baseName;
    unsigned suffix = 0;
    while (func::FuncOp existing =
               module.lookupSymbol<func::FuncOp>(functionName)) {
      auto existingReduction = existing->getAttrOfType<TaskReductionAttr>(
          task_graph_attrs::kTaskReductionHelperAttrName);
      auto existingLane = existing->getAttrOfType<IntegerAttr>(
          task_graph_attrs::kTaskReductionLaneAttrName);
      auto existingTree = existing->getAttrOfType<IntegerAttr>(
          task_graph_attrs::kTaskReductionTreeIdAttrName);
      auto existingLevel = existing->getAttrOfType<IntegerAttr>(
          task_graph_attrs::kTaskReductionLevelAttrName);
      if (existing.getFunctionType() == functionType &&
          existingReduction == reduction && existingTree &&
          existingTree.getInt() == treeId && existingLevel &&
          existingLevel.getInt() == level &&
          ((!lane && !existingLane) ||
           (lane && existingLane && existingLane.getInt() == *lane))) {
        helperByKey.try_emplace(cacheKey, existing);
        return existing;
      }
      functionName = baseName + "_" + std::to_string(++suffix);
    }

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    auto helper = builder.create<func::FuncOp>(loc, functionName, functionType);
    helper.setPrivate();
    helper->setAttr(task_graph_attrs::kTaskReductionHelperAttrName, reduction);
    helper->setAttr(task_graph_attrs::kTaskReductionTreeIdAttrName,
                    builder.getI64IntegerAttr(treeId));
    helper->setAttr(task_graph_attrs::kTaskReductionLevelAttrName,
                    builder.getI64IntegerAttr(level));
    if (lane) {
      helper->setAttr(task_graph_attrs::kTaskReductionLaneAttrName,
                      builder.getI64IntegerAttr(*lane));
    }

    Block *entry = helper.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    Value empty = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                  tensorType.getElementType());
    AffineMap identity = builder.getMultiDimIdentityMap(tensorType.getRank());
    llvm::SmallVector<utils::IteratorType, 4> iterators(
        tensorType.getRank(), utils::IteratorType::parallel);
    llvm::SmallVector<Value, 8> inputs(entry->getArguments().begin(),
                                       entry->getArguments().end());
    llvm::SmallVector<AffineMap, 8> indexingMaps(arity + 1, identity);
    Value result =
        builder
            .create<linalg::GenericOp>(
                loc, tensorType, inputs, ValueRange{empty}, indexingMaps,
                iterators,
                [kind = reduction.getKind(), arity](OpBuilder &nestedBuilder,
                                                    Location nestedLoc,
                                                    ValueRange args) {
                  Value combined = args.front();
                  for (unsigned input = 1; input < arity; ++input) {
                    combined = buildReductionElement(
                        kind, nestedBuilder, nestedLoc, combined, args[input]);
                  }
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, combined);
                })
            .getResult(0);
    builder.create<func::ReturnOp>(loc, result);

    helperByKey.try_emplace(cacheKey, helper);
    return helper;
  }

private:
  ModuleOp module;
  llvm::StringMap<func::FuncOp> helperByKey;
};

static llvm::DenseMap<Value, TaskCreateOp>
collectProducerTasks(func::FuncOp taskGraphFunc) {
  llvm::DenseMap<Value, TaskCreateOp> producerByResource;
  for (TaskCreateOp taskOp : taskGraphFunc.getOps<TaskCreateOp>()) {
    for (Value output : taskOp.getOutputs())
      producerByResource.try_emplace(output, taskOp);
  }
  return producerByResource;
}

static llvm::SmallVector<Value, 4>
deduplicateDependencies(llvm::ArrayRef<Value> dependencies) {
  llvm::DenseSet<Value> seen;
  llvm::SmallVector<Value, 4> result;
  result.reserve(dependencies.size());
  for (Value dependency : dependencies) {
    if (seen.insert(dependency).second)
      result.push_back(dependency);
  }
  return result;
}

static llvm::StringMap<int64_t>
collectNextSourceOrdinals(func::FuncOp taskGraphFunc) {
  llvm::StringMap<int64_t> nextOrdinalByLayer;
  for (TaskCreateOp taskOp : taskGraphFunc.getOps<TaskCreateOp>()) {
    int64_t &next = nextOrdinalByLayer[taskOp.getSourceLayer()];
    next =
        std::max(next, static_cast<int64_t>(taskOp.getSourceTaskOrdinal()) + 1);
  }
  return nextOrdinalByLayer;
}

static int64_t collectNextReductionTreeId(func::FuncOp taskGraphFunc) {
  int64_t nextTreeId = 0;
  for (TaskCreateOp taskOp : taskGraphFunc.getOps<TaskCreateOp>()) {
    auto treeId = taskOp->getAttrOfType<IntegerAttr>(
        task_graph_attrs::kTaskReductionTreeIdAttrName);
    if (treeId)
      nextTreeId = std::max(nextTreeId, treeId.getInt() + 1);
  }
  return nextTreeId;
}

static void attachReductionTreeMetadata(TaskCreateOp taskOp, OpBuilder &builder,
                                        int64_t treeId, int64_t level,
                                        std::optional<int64_t> lane,
                                        int64_t width) {
  taskOp->setAttr(task_graph_attrs::kTaskReductionTreeIdAttrName,
                  builder.getI64IntegerAttr(treeId));
  taskOp->setAttr(task_graph_attrs::kTaskReductionLevelAttrName,
                  builder.getI64IntegerAttr(level));
  if (lane) {
    taskOp->setAttr(task_graph_attrs::kTaskReductionLaneAttrName,
                    builder.getI64IntegerAttr(*lane));
  }
  taskOp->setAttr(task_graph_attrs::kTaskReductionWidthAttrName,
                  builder.getI64IntegerAttr(width));
}

static FailureOr<TaskCreateOp> createReductionTask(
    ModuleOp module, TaskCreateOp originalTask,
    ReductionCalleeCache &calleeCache, TaskReductionAttr reduction,
    RankedTensorType tensorType, llvm::ArrayRef<ReductionLeaf> inputs,
    Value outputResource, llvm::StringRef suffix, int64_t treeId, int64_t level,
    std::optional<int64_t> lane, int64_t width,
    llvm::StringMap<int64_t> &nextOrdinalByLayer, OpBuilder &builder) {
  auto callee =
      calleeCache.getOrCreate(originalTask.getLoc(), reduction, tensorType,
                              inputs.size(), treeId, level, lane);
  if (failed(callee))
    return failure();

  llvm::SmallVector<Value, 8> inputResources;
  llvm::SmallVector<Value, 8> dependencies;
  inputResources.reserve(inputs.size());
  for (const ReductionLeaf &input : inputs) {
    inputResources.push_back(input.resource);
    dependencies.append(input.dependencies.begin(), input.dependencies.end());
  }
  dependencies = deduplicateDependencies(dependencies);

  std::string taskName = originalTask.getTaskName().str() + suffix.str();
  int64_t sourceOrdinal = nextOrdinalByLayer[originalTask.getSourceLayer()]++;
  auto task = builder.create<TaskCreateOp>(
      originalTask.getLoc(), originalTask.getResult().getType(),
      originalTask.getGraph(),
      FlatSymbolRefAttr::get(builder.getContext(), callee->getSymName()),
      builder.getStringAttr(task_graph_names::kDigitalDomain),
      builder.getStringAttr(task_graph_names::kReductionTaskKind),
      builder.getStringAttr(taskName),
      builder.getStringAttr(originalTask.getSourceLayer()),
      builder.getI64IntegerAttr(sourceOrdinal), inputResources,
      ValueRange{outputResource}, dependencies);
  task->setAttr(task_graph_attrs::kTaskReductionAttrName, reduction);
  attachReductionTreeMetadata(task, builder, treeId, level, lane, width);
  task->setAttr(runtime_attrs::kTaskResultIndicesAttrName,
                builder.getI64ArrayAttr({0}));
  return task;
}

static LogicalResult rewriteReductionTask(
    ModuleOp module, func::FuncOp taskGraphFunc, TaskCreateOp reductionTask,
    ReductionCalleeCache &calleeCache, int64_t reductionWidth, int64_t treeId,
    llvm::StringMap<int64_t> &nextOrdinalByLayer,
    llvm::SmallVectorImpl<func::FuncOp> &obsoleteCallees) {
  auto reduction = reductionTask->getAttrOfType<TaskReductionAttr>(
      task_graph_attrs::kTaskReductionAttrName);
  if (!reduction)
    return reductionTask.emitError("expected typed task reduction metadata");

  auto tensorType = getReductionTensorType(reductionTask);
  if (failed(tensorType))
    return reductionTask.emitError("expected a valid reduction tensor type");

  llvm::DenseMap<Value, TaskCreateOp> producerByResource =
      collectProducerTasks(taskGraphFunc);
  llvm::DenseSet<Value> inputProducerTokens;
  for (Value input : reductionTask.getInputs()) {
    auto producer = producerByResource.find(input);
    if (producer != producerByResource.end()) {
      if (!producer->second->isBeforeInBlock(reductionTask)) {
        return reductionTask.emitError(
            "expected every reduction input producer to precede the "
            "reduction task");
      }
      inputProducerTokens.insert(producer->second.getResult());
    }
  }

  llvm::SmallVector<Value, 4> controlDependencies;
  for (Value dependency : reductionTask.getDependencies()) {
    if (!inputProducerTokens.contains(dependency))
      controlDependencies.push_back(dependency);
  }

  llvm::SmallVector<ReductionLeaf, 8> level;
  level.reserve(reductionTask.getInputs().size());
  for (Value input : reductionTask.getInputs()) {
    ReductionLeaf leaf;
    leaf.resource = input;
    leaf.dependencies.append(controlDependencies.begin(),
                             controlDependencies.end());
    auto producer = producerByResource.find(input);
    if (producer != producerByResource.end())
      leaf.dependencies.push_back(producer->second.getResult());
    leaf.dependencies = deduplicateDependencies(leaf.dependencies);
    level.push_back(std::move(leaf));
  }

  func::FuncOp originalCallee =
      module.lookupSymbol<func::FuncOp>(reductionTask.getCallee());
  if (originalCallee)
    obsoleteCallees.push_back(originalCallee);

  OpBuilder builder(reductionTask);
  OpBuilder resourceBuilder(reductionTask.getContext());
  auto taskOps = taskGraphFunc.getOps<TaskCreateOp>();
  auto firstTask = taskOps.begin();
  if (firstTask == taskOps.end())
    return reductionTask.emitError(
        "expected the reduction task in the task graph");
  resourceBuilder.setInsertionPoint(*firstTask);
  unsigned width = static_cast<unsigned>(reductionWidth);
  unsigned fanIn = static_cast<unsigned>(level.size());
  unsigned baseLaneSize = fanIn / width;
  unsigned extraLanes = fanIn % width;
  unsigned inputOffset = 0;
  llvm::SmallVector<ReductionLeaf, 8> laneOutputs;
  laneOutputs.reserve(width);
  for (unsigned lane = 0; lane < width; ++lane) {
    unsigned laneSize = baseLaneSize + (lane < extraLanes ? 1 : 0);
    llvm::ArrayRef<ReductionLeaf> laneInputs(level.data() + inputOffset,
                                             laneSize);
    inputOffset += laneSize;
    if (laneInputs.size() == 1) {
      laneOutputs.push_back(laneInputs.front());
      continue;
    }

    Value outputResource =
        resourceBuilder
            .create<TaskGraphIntermediateOp>(
                reductionTask.getLoc(),
                TaskResourceType::get(resourceBuilder.getContext(),
                                      *tensorType),
                reductionTask.getGraph())
            .getResult();
    std::string suffix = ".level0.lane" + std::to_string(lane);
    auto laneTask = createReductionTask(
        module, reductionTask, calleeCache, reduction, *tensorType, laneInputs,
        outputResource, suffix, treeId, /*level=*/0, lane, reductionWidth,
        nextOrdinalByLayer, builder);
    if (failed(laneTask))
      return failure();

    ReductionLeaf laneOutput;
    laneOutput.resource = outputResource;
    laneOutput.dependencies.push_back(laneTask->getResult());
    laneOutputs.push_back(std::move(laneOutput));
  }

  auto rootTask = createReductionTask(
      module, reductionTask, calleeCache, reduction, *tensorType, laneOutputs,
      reductionTask.getOutputs().front(), ".level1.root", treeId,
      /*level=*/1, /*lane=*/std::nullopt, reductionWidth, nextOrdinalByLayer,
      builder);

  if (failed(rootTask))
    return reductionTask.emitError("failed to create a reduction tree root");
  reductionTask.getResult().replaceAllUsesWith(rootTask->getResult());
  reductionTask.erase();
  return success();
}

static LogicalResult verifyRewrittenTaskGraph(func::FuncOp taskGraphFunc) {
  auto dag = parseTaskGraphDAG(taskGraphFunc);
  if (failed(dag))
    return failure();
  if (failed(buildTaskExecutionGraph(taskGraphFunc, *dag)))
    return failure();
  llvm::DenseMap<Value, unsigned> producerByResource;
  return collectResourceProducers(*dag, producerByResource);
}

} // namespace

LogicalResult balanceTaskGraphReductions(ModuleOp module,
                                         func::FuncOp taskGraphFunc,
                                         int64_t reductionWidth) {
  if (failed(rejectStalePlacementMetadata(taskGraphFunc)))
    return failure();

  llvm::SmallVector<TaskCreateOp, 4> candidates;
  for (TaskCreateOp taskOp : taskGraphFunc.getOps<TaskCreateOp>()) {
    if (!taskOp->hasAttr(task_graph_attrs::kTaskReductionAttrName) ||
        taskOp->hasAttr(task_graph_attrs::kTaskReductionTreeIdAttrName) ||
        static_cast<int64_t>(taskOp.getInputs().size()) <= reductionWidth)
      continue;
    candidates.push_back(taskOp);
  }
  if (candidates.empty())
    return success();

  ReductionCalleeCache calleeCache(module);
  llvm::StringMap<int64_t> nextOrdinalByLayer =
      collectNextSourceOrdinals(taskGraphFunc);
  int64_t nextTreeId = collectNextReductionTreeId(taskGraphFunc);
  llvm::SmallVector<func::FuncOp, 4> obsoleteCallees;
  for (TaskCreateOp candidate : candidates) {
    if (failed(rewriteReductionTask(module, taskGraphFunc, candidate,
                                    calleeCache, reductionWidth, nextTreeId++,
                                    nextOrdinalByLayer, obsoleteCallees)))
      return failure();
  }

  if (failed(verifyRewrittenTaskGraph(taskGraphFunc)))
    return taskGraphFunc.emitError(
        "balanced reduction rewrite produced an invalid task graph");

  llvm::SmallPtrSet<Operation *, 4> erased;
  for (func::FuncOp callee : obsoleteCallees) {
    if (!callee || !erased.insert(callee.getOperation()).second)
      continue;
    if (SymbolTable::symbolKnownUseEmpty(callee, module))
      callee.erase();
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
