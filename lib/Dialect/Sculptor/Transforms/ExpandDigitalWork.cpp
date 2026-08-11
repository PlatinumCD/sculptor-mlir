#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExpandDigitalWork.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ReductionTree.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ShardDataflow.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"

#include <algorithm>
#include <optional>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

void chooseWorkerFactors(ArrayRef<ComputeIterationDimension> domain,
                         int64_t parallelWorkers, size_t dimension,
                         int64_t currentProduct,
                         SmallVectorImpl<int64_t> &current,
                         SmallVectorImpl<int64_t> &best, int64_t &bestProduct) {
  if (dimension == domain.size()) {
    if (currentProduct > bestProduct ||
        (currentProduct == bestProduct &&
         std::lexicographical_compare(best.begin(), best.end(), current.begin(),
                                      current.end()))) {
      bestProduct = currentProduct;
      best.assign(current.begin(), current.end());
    }
    return;
  }

  const ComputeIterationDimension &loop = domain[dimension];
  int64_t maximumFactor = 1;
  if (loop.kind == ComputeIteratorKind::Parallel &&
      !ShapedType::isDynamic(loop.staticExtent) && loop.staticExtent > 1) {
    maximumFactor =
        std::min(loop.staticExtent, parallelWorkers / currentProduct);
  }

  for (int64_t factor = maximumFactor; factor >= 1; --factor) {
    current[dimension] = factor;
    chooseWorkerFactors(domain, parallelWorkers, dimension + 1,
                        currentProduct * factor, current, best, bestProduct);
  }
}

std::optional<int64_t> getStaticValue(OpFoldResult value) {
  return getConstantIntValue(value);
}

void dissolveDigitalMappingStages(func::FuncOp function) {
  function.walk([](Operation *operation) {
    auto stageKind = operation->getAttrOfType<StringAttr>(kStageKindAttrName);
    if (!stageKind || stageKind.getValue() != kDigitalStageKind)
      return;

    operation->removeAttr(kStageIdAttrName);
    operation->removeAttr(kStageKindAttrName);
    operation->removeAttr(kStageNameAttrName);
  });
}

FailureOr<SmallVector<MappingWorkUnitAttr>>
buildWorkUnits(const ComputeOperation &operation, int64_t parallelWorkers,
               int64_t &nextWorkUnitId, OpBuilder &builder) {
  SmallVector<MappingWorkUnitAttr> result;
  if (operation.kind != ComputeOperationKind::Structured ||
      operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
          LogicalLaneKind::Digital ||
      operation.mvmWaveId || operation.operation->getNumResults() != 1)
    return result;

  auto resultType =
      dyn_cast<RankedTensorType>(operation.operation->getResult(0).getType());
  auto tiling = dyn_cast<TilingInterface>(operation.operation);
  if (!resultType || !resultType.hasStaticShape() || !tiling ||
      operation.iterationDomain.empty() ||
      llvm::any_of(operation.iterationDomain,
                   [](const ComputeIterationDimension &dimension) {
                     return ShapedType::isDynamic(dimension.staticExtent) ||
                            dimension.staticExtent <= 0;
                   }))
    return result;

  SmallVector<int64_t> current(operation.iterationDomain.size(), 1);
  SmallVector<int64_t> factors(operation.iterationDomain.size(), 1);
  int64_t workerCount = 1;
  chooseWorkerFactors(operation.iterationDomain, parallelWorkers,
                      /*dimension=*/0, /*currentProduct=*/1, current, factors,
                      workerCount);
  if (workerCount < 2)
    return result;

  SmallVector<int64_t> coordinates(operation.iterationDomain.size(), 0);
  for (int64_t worker = 0; worker < workerCount; ++worker) {
    int64_t linear = worker;
    for (int64_t dimension = static_cast<int64_t>(factors.size()) - 1;
         dimension >= 0; --dimension) {
      coordinates[dimension] = linear % factors[dimension];
      linear /= factors[dimension];
    }

    SmallVector<int64_t> iterationOffsets;
    SmallVector<int64_t> iterationSizes;
    SmallVector<OpFoldResult> foldedOffsets;
    SmallVector<OpFoldResult> foldedSizes;
    iterationOffsets.reserve(factors.size());
    iterationSizes.reserve(factors.size());
    for (auto [dimension, factor] : llvm::enumerate(factors)) {
      int64_t extent = operation.iterationDomain[dimension].staticExtent;
      int64_t baseSize = extent / factor;
      int64_t remainder = extent % factor;
      int64_t coordinate = coordinates[dimension];
      int64_t size = baseSize + (coordinate < remainder ? 1 : 0);
      int64_t offset = coordinate * baseSize + std::min(coordinate, remainder);
      iterationOffsets.push_back(offset);
      iterationSizes.push_back(size);
      foldedOffsets.push_back(builder.getIndexAttr(offset));
      foldedSizes.push_back(builder.getIndexAttr(size));
    }

    SmallVector<OpFoldResult> foldedResultOffsets;
    SmallVector<OpFoldResult> foldedResultSizes;
    if (failed(tiling.getResultTilePosition(
            builder, /*resultNumber=*/0, foldedOffsets, foldedSizes,
            foldedResultOffsets, foldedResultSizes)))
      return SmallVector<MappingWorkUnitAttr>{};

    SmallVector<int64_t> resultOffsets;
    SmallVector<int64_t> resultSizes;
    for (OpFoldResult value : foldedResultOffsets) {
      std::optional<int64_t> constant = getStaticValue(value);
      if (!constant)
        return SmallVector<MappingWorkUnitAttr>{};
      resultOffsets.push_back(*constant);
    }
    for (OpFoldResult value : foldedResultSizes) {
      std::optional<int64_t> constant = getStaticValue(value);
      if (!constant || *constant <= 0)
        return SmallVector<MappingWorkUnitAttr>{};
      resultSizes.push_back(*constant);
    }
    if (resultOffsets.size() != static_cast<size_t>(resultType.getRank()) ||
        resultSizes.size() != static_cast<size_t>(resultType.getRank()))
      return SmallVector<MappingWorkUnitAttr>{};

    result.push_back(MappingWorkUnitAttr::get(
        builder.getContext(), builder.getI64IntegerAttr(nextWorkUnitId++),
        builder.getI64IntegerAttr(operation.id), builder.getI64IntegerAttr(0),
        builder.getI64ArrayAttr(resultOffsets),
        builder.getI64ArrayAttr(resultSizes),
        builder.getI64ArrayAttr(iterationOffsets),
        builder.getI64ArrayAttr(iterationSizes), builder.getI64IntegerAttr(-1),
        builder.getI64IntegerAttr(-1), builder.getI64IntegerAttr(-1)));
  }
  return result;
}

} // namespace

namespace mlir {
namespace sculptor {

ExpandDigitalWorkPass::ExpandDigitalWorkPass(const ExpandDigitalWorkPass &pass)
    : PassWrapper(pass),
      parallelWorkers(
          *this, "parallel-workers",
          llvm::cl::desc("Target number of independent digital work units"),
          llvm::cl::init(4)),
      requireChange(
          *this, "require-change",
          llvm::cl::desc("Fail when no digital operation can be expanded"),
          llvm::cl::init(false)),
      dataflow(*this, "dataflow",
               llvm::cl::desc("Digital dataflow mode: bulk or sharded"),
               llvm::cl::init("bulk")),
      shardPropagationDepth(
          *this, "shard-propagation-depth",
          llvm::cl::desc("Maximum shard propagation depth; zero is unbounded"),
          llvm::cl::init(0)),
      requireCompleteShardChain(
          *this, "require-complete-shard-chain",
          llvm::cl::desc(
              "Fail when an eligible shard chain reaches a boundary"),
          llvm::cl::init(false)),
      reductionTree(
          *this, "reduction-tree",
          llvm::cl::desc("Associative reduction policy: none or balanced"),
          llvm::cl::init("none")),
      reductionFanIn(
          *this, "reduction-fan-in",
          llvm::cl::desc("Maximum reduction fan-in; only two is supported"),
          llvm::cl::init(2)),
      reductionMinimumWidth(
          *this, "reduction-min-width",
          llvm::cl::desc("Minimum number of leaves for reduction balancing"),
          llvm::cl::init(3)) {
  parallelWorkers = pass.parallelWorkers;
  requireChange = pass.requireChange;
  dataflow = pass.dataflow;
  shardPropagationDepth = pass.shardPropagationDepth;
  requireCompleteShardChain = pass.requireCompleteShardChain;
  reductionTree = pass.reductionTree;
  reductionFanIn = pass.reductionFanIn;
  reductionMinimumWidth = pass.reductionMinimumWidth;
}

void ExpandDigitalWorkPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (parallelWorkers < 1) {
    module.emitError("parallel-workers must be positive");
    signalPassFailure();
    return;
  }
  FailureOr<mapping::DigitalDataflowMode> parsedDataflow =
      mapping::parseDigitalDataflowMode(dataflow, module);
  FailureOr<mapping::ReductionTreePolicy> parsedReductionTree =
      mapping::parseReductionTreePolicy(reductionTree, module);
  if (failed(parsedDataflow) || failed(parsedReductionTree) ||
      shardPropagationDepth < 0) {
    if (succeeded(parsedDataflow))
      module.emitError("shard-propagation-depth must be nonnegative");
    signalPassFailure();
    return;
  }

  int64_t expandedOperations = 0;
  int64_t expandedWorkUnits = 0;
  int64_t reductionTreeCount = 0;
  int64_t reductionNodeCount = 0;
  int64_t maximumReductionFanIn = 0;
  int64_t shardGroupCount = 0;
  int64_t shardEdgeCount = 0;
  int64_t assemblyBoundaryCount = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    OpBuilder builder(&getContext());

    function.walk([](Operation *operation) {
      operation->removeAttr(mapping::kExpandedDigitalWorkAttrName);
    });
    dissolveDigitalMappingStages(function);
    function->removeAttr("sculptor.mapping.digital_parallel_workers");
    function->removeAttr("sculptor.mapping.expanded_digital_operation_count");
    function->removeAttr("sculptor.mapping.expanded_digital_work_unit_count");
    function->removeAttr(mapping::kShardWorkUnitEdgesAttrName);

    FailureOr<int64_t> functionReductionTrees = mapping::buildReductionTrees(
        function, *parsedReductionTree, reductionFanIn, reductionMinimumWidth);
    if (failed(functionReductionTrees)) {
      signalPassFailure();
      return;
    }
    function->setAttr(
        mapping::kReductionTreePolicyAttrName,
        StringAttr::get(&getContext(), mapping::stringifyReductionTreePolicy(
                                           *parsedReductionTree)));
    function->setAttr("sculptor.mapping.reduction_tree_count",
                      builder.getI64IntegerAttr(*functionReductionTrees));
    reductionTreeCount += *functionReductionTrees;
    reductionNodeCount += function
                              ->getAttrOfType<IntegerAttr>(
                                  "sculptor.mapping.reduction_node_count")
                              .getInt();
    maximumReductionFanIn =
        std::max(maximumReductionFanIn,
                 function
                     ->getAttrOfType<IntegerAttr>(
                         "sculptor.mapping.maximum_reduction_fan_in")
                     .getInt());

    FailureOr<mapping::ComputeGraph> graph =
        mapping::buildComputeGraph(function);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }

    int64_t nextWorkUnitId = 0;
    int64_t functionOperations = 0;
    int64_t functionWorkUnits = 0;
    for (const mapping::ComputeOperation &operation : graph->operations) {
      FailureOr<SmallVector<MappingWorkUnitAttr>> workUnits =
          buildWorkUnits(operation, parallelWorkers, nextWorkUnitId, builder);
      if (failed(workUnits)) {
        signalPassFailure();
        return;
      }
      if (workUnits->empty())
        continue;
      SmallVector<Attribute> attributes(workUnits->begin(), workUnits->end());
      operation.operation->setAttr(mapping::kExpandedDigitalWorkAttrName,
                                   builder.getArrayAttr(attributes));
      ++functionOperations;
      functionWorkUnits += static_cast<int64_t>(workUnits->size());
    }

    if (functionOperations > 0) {
      function->setAttr("sculptor.mapping.digital_parallel_workers",
                        builder.getI64IntegerAttr(parallelWorkers));
      function->setAttr("sculptor.mapping.expanded_digital_operation_count",
                        builder.getI64IntegerAttr(functionOperations));
      function->setAttr("sculptor.mapping.expanded_digital_work_unit_count",
                        builder.getI64IntegerAttr(functionWorkUnits));
    }
    if (failed(mapping::planShardDataflow(function, *graph, *parsedDataflow,
                                          shardPropagationDepth,
                                          requireCompleteShardChain))) {
      signalPassFailure();
      return;
    }
    function->setAttr(
        mapping::kShardDataflowModeAttrName,
        StringAttr::get(&getContext(), mapping::stringifyDigitalDataflowMode(
                                           *parsedDataflow)));
    shardGroupCount +=
        function->getAttrOfType<IntegerAttr>(mapping::kShardGroupCountAttrName)
            .getInt();
    shardEdgeCount +=
        function->getAttrOfType<IntegerAttr>(mapping::kShardEdgeCountAttrName)
            .getInt();
    assemblyBoundaryCount += function
                                 ->getAttrOfType<IntegerAttr>(
                                     mapping::kAssemblyBoundaryCountAttrName)
                                 .getInt();
    expandedOperations += functionOperations;
    expandedWorkUnits += functionWorkUnits;
  }

  if (requireChange && expandedOperations == 0) {
    module.emitError("no digital operation has legal parallel work units");
    signalPassFailure();
    return;
  }
  module->setAttr("sculptor.mapping.expanded_digital_operation_count",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   expandedOperations));
  module->setAttr(
      "sculptor.mapping.expanded_digital_work_unit_count",
      IntegerAttr::get(IntegerType::get(&getContext(), 64), expandedWorkUnits));
  module->setAttr("sculptor.mapping.reduction_tree_count",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   reductionTreeCount));
  module->setAttr(
      mapping::kReductionTreePolicyAttrName,
      StringAttr::get(&getContext(), mapping::stringifyReductionTreePolicy(
                                         *parsedReductionTree)));
  module->setAttr("sculptor.mapping.reduction_node_count",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   reductionNodeCount));
  module->setAttr("sculptor.mapping.maximum_reduction_fan_in",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   maximumReductionFanIn));
  module->setAttr(
      mapping::kShardDataflowModeAttrName,
      StringAttr::get(&getContext(),
                      mapping::stringifyDigitalDataflowMode(*parsedDataflow)));
  module->setAttr(
      mapping::kShardGroupCountAttrName,
      IntegerAttr::get(IntegerType::get(&getContext(), 64), shardGroupCount));
  module->setAttr(
      mapping::kShardEdgeCountAttrName,
      IntegerAttr::get(IntegerType::get(&getContext(), 64), shardEdgeCount));
  module->setAttr(mapping::kAssemblyBoundaryCountAttrName,
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   assemblyBoundaryCount));
}

void registerExpandDigitalWorkPass() {
  PassRegistration<ExpandDigitalWorkPass>();
}

} // namespace sculptor
} // namespace mlir
