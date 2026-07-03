#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleMetadata.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScorer.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;

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
    auto placement =
        resolvePhysicalArrayPlacement(taskOp.getOperation(), budget,
                                      *physicalArrayId);
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
  taskGraphFunc->setAttr(
      schedule_attrs::kTransferCostPerInterCoreByteAttrName,
      builder.getFloatAttr(builder.getF64Type(),
                           score.transferCostPerInterCoreByte));
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

} // namespace

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

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
