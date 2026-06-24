#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScorer.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_schedulers = mlir::sculptor::task_schedulers;

static mlir::FailureOr<int64_t>
getRequiredI64Attr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr) {
    op->emitError("expected scheduled task graph attr '") << attrName << "'";
    return mlir::failure();
  }

  return attr.getInt();
}

static int64_t getResourceByteSize(mlir::Value resource) {
  mlir::Operation *resourceOp = resource.getDefiningOp();
  if (!resourceOp)
    return 0;

  auto byteSizeAttr = resourceOp->getAttrOfType<mlir::IntegerAttr>(
      runtime_attrs::kResourceByteSizeAttrName);
  if (!byteSizeAttr)
    return 0;
  return byteSizeAttr.getInt();
}

static int64_t getMeshDistance(
    int64_t sourceCore, int64_t destinationCore,
    const task_schedulers::HardwareBudget &budget) {
  int64_t sourceRow = sourceCore / budget.meshCols;
  int64_t sourceCol = sourceCore % budget.meshCols;
  int64_t destinationRow = destinationCore / budget.meshCols;
  int64_t destinationCol = destinationCore % budget.meshCols;
  int64_t rowDistance = sourceRow > destinationRow ? sourceRow - destinationRow
                                                   : destinationRow - sourceRow;
  int64_t colDistance = sourceCol > destinationCol ? sourceCol - destinationCol
                                                   : destinationCol - sourceCol;
  return rowDistance + colDistance;
}

static unsigned
getMeshBoundaryMask(int64_t coreId,
                    const task_schedulers::HardwareBudget &budget) {
  constexpr unsigned kTop = 1u << 0;
  constexpr unsigned kBottom = 1u << 1;
  constexpr unsigned kLeft = 1u << 2;
  constexpr unsigned kRight = 1u << 3;

  int64_t row = coreId / budget.meshCols;
  int64_t col = coreId % budget.meshCols;
  unsigned mask = 0;
  if (row == 0)
    mask |= kTop;
  if (row == budget.meshRows - 1)
    mask |= kBottom;
  if (col == 0)
    mask |= kLeft;
  if (col == budget.meshCols - 1)
    mask |= kRight;
  return mask;
}

static int64_t getBoundaryPenalty(int64_t totalTransferCost) {
  if (totalTransferCost <= 0)
    return 0;
  return (totalTransferCost + 4) / 5;
}

static mlir::LogicalResult collectResourceProducers(
    const task_schedulers::TaskGraphDAG &dag,
    llvm::DenseMap<mlir::Value, const task_schedulers::TaskGraphNode *>
        &producerByResource) {
  for (const task_schedulers::TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    for (mlir::Value output : taskOp.getOutputs()) {
      if (!producerByResource.try_emplace(output, &node).second) {
        taskOp.emitError("expected task graph resource to have one producer");
        return mlir::failure();
      }
    }
  }

  return mlir::success();
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
  result.coreTransferBytes.assign(
      static_cast<size_t>(coreTransferEntryCount), 0);
  result.coreTransferCost.assign(static_cast<size_t>(coreTransferEntryCount),
                                 0);

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
      size_t transferIndex =
          static_cast<size_t>(*sourceCore * budget.numCores + *destinationCore);

      result.coreTransferBytes[transferIndex] += byteSize;
      result.coreTransferCost[transferIndex] += transferCost;
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
      result.boundaryPenalty =
          getBoundaryPenalty(result.totalTransferCost);
      result.score += result.boundaryPenalty;
    }
  }

  return result;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
