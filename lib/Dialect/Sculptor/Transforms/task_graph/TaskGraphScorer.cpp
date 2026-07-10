#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphScorer.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/MeshGeometry.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphResources.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_schedulers = mlir::sculptor::task_schedulers;

constexpr int64_t kMaxDenseCoreTransferEntries = 1'000'000;

static mlir::FailureOr<int64_t> getRequiredI64Attr(mlir::Operation *op,
                                                   llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr) {
    op->emitError("expected scheduled task graph attr '") << attrName << "'";
    return mlir::failure();
  }

  return attr.getInt();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

FailureOr<TaskGraphScore>
MeshTaskGraphScorer::score(ModuleOp module, func::FuncOp taskGraphFunc,
                           const HardwareBudget &budget,
                           const TaskGraphDAG &dag) const {
  (void)module;

  if (budget.topology != "mesh") {
    taskGraphFunc.emitError("expected mesh topology for task graph scorer");
    return failure();
  }

  if (budget.meshCols <= 0 || budget.meshRows <= 0 ||
      budget.meshRows * budget.meshCols != budget.numCores) {
    taskGraphFunc.emitError("expected mesh scorer dimensions to match the "
                            "scheduled core count");
    return failure();
  }

  llvm::DenseMap<Value, const TaskGraphNode *> producerByResource;
  if (failed(collectResourceProducers(dag, producerByResource)))
    return failure();

  TaskGraphScore result;
  int64_t coreTransferEntryCount = budget.numCores * budget.numCores;
  bool recordDenseCoreTransfers =
      coreTransferEntryCount <= kMaxDenseCoreTransferEntries;
  if (recordDenseCoreTransfers) {
    result.coreTransferBytes.assign(static_cast<size_t>(coreTransferEntryCount),
                                    0);
    result.coreTransferCost.assign(static_cast<size_t>(coreTransferEntryCount),
                                   0);
  }

  for (const TaskGraphNode &consumer : dag.nodes) {
    sculptor::TaskCreateOp consumerOp = consumer.op;
    auto destinationCore =
        getRequiredI64Attr(consumerOp, runtime_attrs::kTaskCoreIdAttrName);
    if (failed(destinationCore))
      return failure();

    for (Value input : consumerOp.getInputs()) {
      auto producerIt = producerByResource.find(input);
      if (producerIt == producerByResource.end())
        continue;

      auto sourceCore = getRequiredI64Attr(producerIt->second->op,
                                           runtime_attrs::kTaskCoreIdAttrName);
      if (failed(sourceCore))
        return failure();

      if (*sourceCore == *destinationCore)
        continue;

      int64_t byteSize = getResourceByteSize(input);
      int64_t meshDistance =
          getMeshDistance(*sourceCore, *destinationCore, budget);
      int64_t transferCost = byteSize * meshDistance;

      if (recordDenseCoreTransfers) {
        size_t transferIndex = static_cast<size_t>(
            *sourceCore * budget.numCores + *destinationCore);
        result.coreTransferBytes[transferIndex] += byteSize;
        result.coreTransferCost[transferIndex] += transferCost;
      }
      result.interCoreTransferBytes += byteSize;
      result.totalTransferCost += transferCost;
      result.score += transferCost;
    }
  }

  if (!dag.nodes.empty()) {
    sculptor::TaskCreateOp firstTask = dag.nodes.front().op;
    sculptor::TaskCreateOp lastTask = dag.nodes.back().op;
    auto firstCore =
        getRequiredI64Attr(firstTask, runtime_attrs::kTaskCoreIdAttrName);
    auto lastCore =
        getRequiredI64Attr(lastTask, runtime_attrs::kTaskCoreIdAttrName);
    if (failed(firstCore) || failed(lastCore))
      return failure();

    unsigned firstBoundaryMask = getMeshBoundaryMask(*firstCore, budget);
    unsigned lastBoundaryMask = getMeshBoundaryMask(*lastCore, budget);
    if ((firstBoundaryMask & lastBoundaryMask) == 0) {
      result.boundaryPenalty = getBoundaryPenalty(result.totalTransferCost);
      result.score += result.boundaryPenalty;
    }
  }

  if (result.interCoreTransferBytes > 0) {
    result.transferCostPerInterCoreByte =
        static_cast<double>(result.totalTransferCost) /
        static_cast<double>(result.interCoreTransferBytes);
  }

  return result;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
