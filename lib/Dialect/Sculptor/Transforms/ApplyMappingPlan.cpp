#include "sculptor-mlir/Dialect/Sculptor/Transforms/ApplyMappingPlan.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

bool matchesWorkUnit(tensor::ExtractSliceOp slice,
                     const MappingWorkUnit &workUnit) {
  if (slice.getType().getRank() !=
          static_cast<int64_t>(workUnit.resultSizes.size()) ||
      slice.getStaticOffsets() != ArrayRef<int64_t>(workUnit.resultOffsets) ||
      slice.getStaticSizes() != ArrayRef<int64_t>(workUnit.resultSizes))
    return false;
  return llvm::all_of(slice.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; });
}

FailureOr<tensor::ExtractSliceOp>
findConsumerSlice(Operation *operation, const MappingWorkUnit &workUnit) {
  if (workUnit.resultNumber < 0 ||
      workUnit.resultNumber >= static_cast<int64_t>(operation->getNumResults()))
    return failure();

  tensor::ExtractSliceOp match;
  for (OpOperand &use : operation->getResult(workUnit.resultNumber).getUses()) {
    auto slice = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
    if (!slice || !matchesWorkUnit(slice, workUnit))
      continue;
    if (match) {
      operation->emitError("mapping work unit ")
          << workUnit.id << " matches multiple consumer slices";
      return failure();
    }
    match = slice;
  }
  if (!match) {
    operation->emitError("mapping work unit ")
        << workUnit.id << " has no matching tensor.extract_slice consumer";
    return failure();
  }
  return match;
}

LogicalResult applyWorkUnits(func::FuncOp function, const ComputeGraph &graph,
                             const ResourceAllocationTree &tree) {
  DenseMap<int64_t, int64_t> leafByWorkUnit;
  for (const StructuralRATreeNode &node : tree.nodes) {
    if (node.kind == RATreeNodeKind::Leaf && node.workUnitId >= 0)
      leafByWorkUnit[node.workUnitId] = node.id;
  }

  DenseMap<int64_t, SmallVector<const MappingWorkUnit *>> workUnitsByOperation;
  for (const MappingWorkUnit &workUnit : tree.workUnits)
    workUnitsByOperation[workUnit.operationId].push_back(&workUnit);

  IRRewriter rewriter(function.getContext());
  int64_t appliedWorkUnits = 0;
  for (auto &[operationId, workUnits] : workUnitsByOperation) {
    if (operationId < 0 ||
        operationId >= static_cast<int64_t>(graph.operations.size())) {
      function.emitError("mapping plan references unknown operation ")
          << operationId;
      return failure();
    }
    Operation *operation = graph.operations[operationId].operation;
    auto tiling = dyn_cast<TilingInterface>(operation);
    if (!tiling) {
      operation->emitError(
          "selected mapping work units require TilingInterface");
      return failure();
    }
    SmallVector<Operation *> deadOperandCandidates;
    for (Value operand : operation->getOperands()) {
      if (auto empty = operand.getDefiningOp<tensor::EmptyOp>())
        deadOperandCandidates.push_back(empty);
    }

    llvm::sort(workUnits,
               [](const MappingWorkUnit *lhs, const MappingWorkUnit *rhs) {
                 return lhs->id < rhs->id;
               });
    for (const MappingWorkUnit *workUnit : workUnits) {
      FailureOr<tensor::ExtractSliceOp> consumerSlice =
          findConsumerSlice(operation, *workUnit);
      if (failed(consumerSlice))
        return failure();

      SmallVector<OpFoldResult> offsets;
      SmallVector<OpFoldResult> sizes;
      offsets.reserve(workUnit->resultOffsets.size());
      sizes.reserve(workUnit->resultSizes.size());
      for (int64_t offset : workUnit->resultOffsets)
        offsets.push_back(rewriter.getIndexAttr(offset));
      for (int64_t size : workUnit->resultSizes)
        sizes.push_back(rewriter.getIndexAttr(size));

      rewriter.setInsertionPoint(*consumerSlice);
      FailureOr<TilingResult> tiled = tiling.generateResultTileValue(
          rewriter, workUnit->resultNumber, offsets, sizes);
      if (failed(tiled) || tiled->tiledValues.size() != 1) {
        operation->emitError("failed to materialize mapping work unit ")
            << workUnit->id;
        return failure();
      }

      SmallVector<Operation *> generatedSlices = tiled->generatedSlices;
      for (Operation *generated : generatedSlices) {
        auto slice = dyn_cast<tensor::ExtractSliceOp>(generated);
        if (!slice || !slice.getSource().getDefiningOp<tensor::EmptyOp>())
          continue;
        auto tileType = cast<RankedTensorType>(slice.getType());
        rewriter.setInsertionPoint(slice);
        Value empty = rewriter.create<tensor::EmptyOp>(
            slice.getLoc(), tileType.getShape(), tileType.getElementType());
        rewriter.replaceOp(slice, empty);
      }

      int64_t leafId = leafByWorkUnit.lookup(workUnit->id);
      for (Operation *tiledOperation : tiled->tiledOps) {
        tiledOperation->setAttr(kMappingWorkUnitIdAttrName,
                                rewriter.getI64IntegerAttr(workUnit->id));
        tiledOperation->setAttr(kRALeafIdAttrName,
                                rewriter.getI64IntegerAttr(leafId));
      }
      rewriter.replaceOp(*consumerSlice, tiled->tiledValues.front());
      ++appliedWorkUnits;
    }

    if (!operation->use_empty()) {
      operation->emitError(
          "consumer-aligned tiling left unsupported uses of the full result");
      return failure();
    }
    rewriter.eraseOp(operation);
    for (Operation *candidate : deadOperandCandidates) {
      if (candidate->use_empty())
        rewriter.eraseOp(candidate);
    }
  }

  auto plan = function->getAttrOfType<MappingPlanAttr>(kMappingPlanAttrName);
  if (plan)
    function->setAttr("sculptor.mapping.applied_strategies", plan.getPlanner());
  function->setAttr("sculptor.mapping.applied_work_unit_count",
                    rewriter.getI64IntegerAttr(appliedWorkUnits));
  function->removeAttr(kRATreeAttrName);
  function->removeAttr(kMappingPlanAttrName);
  function->removeAttr(kLogicalTileGraphAttrName);
  function->removeAttr(kLogicalTilePlacementAttrName);
  function->removeAttr(kLogicalTileAnnealingTraceAttrName);
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {

void ApplyMappingPlanPass::runOnOperation() {
  bool foundPlan = false;
  for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
    auto treeAttr =
        function->getAttrOfType<RATreeAttr>(mapping::kRATreeAttrName);
    if (!treeAttr)
      continue;
    foundPlan = true;

    FailureOr<mapping::ComputeGraph> graph =
        mapping::buildComputeGraph(function);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    FailureOr<mapping::ResourceAllocationTree> tree =
        mapping::deserializeResourceAllocationTree(treeAttr, *graph, function);
    if (failed(tree) || failed(applyWorkUnits(function, *graph, *tree))) {
      signalPassFailure();
      return;
    }
  }

  if (!foundPlan) {
    getOperation().emitError(
        "expected at least one function with a selected mapping plan");
    signalPassFailure();
  }
}

void registerApplyMappingPlanPass() {
  PassRegistration<ApplyMappingPlanPass>();
}

} // namespace sculptor
} // namespace mlir
