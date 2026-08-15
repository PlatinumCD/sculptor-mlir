#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExpandDigitalWork.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ReductionTree.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ShardDataflow.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

constexpr StringLiteral kDigitalTilingPolicyAttrName =
    "sculptor.mapping.digital_tiling_policy";

enum class DigitalTilingPolicy { DimensionFirst, CommunicationAware };

struct WorkUnitSpec {
  SmallVector<int64_t> resultOffsets;
  SmallVector<int64_t> resultSizes;
  SmallVector<int64_t> iterationOffsets;
  SmallVector<int64_t> iterationSizes;
};

struct WorkUnitCandidate {
  SmallVector<int64_t> factors;
  SmallVector<WorkUnitSpec> units;
};

struct CandidateCommunicationEdge {
  int64_t producerId = -1;
  int64_t consumerId = -1;
  unsigned operandNumber = 0;
  int64_t byteSize = 1;
  AffineMap consumerOperandMap;
};

FailureOr<DigitalTilingPolicy> parseDigitalTilingPolicy(StringRef value,
                                                        Operation *anchor) {
  if (value == "dimension-first")
    return DigitalTilingPolicy::DimensionFirst;
  if (value == "communication-aware")
    return DigitalTilingPolicy::CommunicationAware;
  anchor->emitError("unknown digital tiling policy '") << value << "'";
  return failure();
}

StringRef stringifyDigitalTilingPolicy(DigitalTilingPolicy policy) {
  switch (policy) {
  case DigitalTilingPolicy::DimensionFirst:
    return "dimension-first";
  case DigitalTilingPolicy::CommunicationAware:
    return "communication-aware";
  }
  llvm_unreachable("unknown digital tiling policy");
}

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

void enumerateExactWorkerFactors(
    ArrayRef<ComputeIterationDimension> domain, int64_t remainingWorkers,
    size_t dimension, SmallVectorImpl<int64_t> &current,
    SmallVectorImpl<SmallVector<int64_t>> &results) {
  if (dimension == domain.size()) {
    if (remainingWorkers == 1)
      results.emplace_back(current.begin(), current.end());
    return;
  }

  const ComputeIterationDimension &loop = domain[dimension];
  int64_t maximumFactor = 1;
  if (loop.kind == ComputeIteratorKind::Parallel &&
      !ShapedType::isDynamic(loop.staticExtent) && loop.staticExtent > 1)
    maximumFactor = std::min(loop.staticExtent, remainingWorkers);

  for (int64_t factor = maximumFactor; factor >= 1; --factor) {
    if (remainingWorkers % factor != 0)
      continue;
    current[dimension] = factor;
    enumerateExactWorkerFactors(domain, remainingWorkers / factor,
                                dimension + 1, current, results);
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

bool isExpandableDigitalOperation(const ComputeOperation &operation) {
  if (operation.kind != ComputeOperationKind::Structured ||
      operation.requiredLane.value_or(LogicalLaneKind::Digital) !=
          LogicalLaneKind::Digital ||
      operation.mvmWaveId || operation.operation->getNumResults() != 1)
    return false;

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
    return false;
  return true;
}

std::optional<WorkUnitCandidate>
buildWorkUnitCandidate(const ComputeOperation &operation,
                       ArrayRef<int64_t> factors, OpBuilder &builder) {
  auto resultType =
      cast<RankedTensorType>(operation.operation->getResult(0).getType());
  auto tiling = cast<TilingInterface>(operation.operation);
  int64_t workerCount = 1;
  for (int64_t factor : factors) {
    std::optional<int64_t> next = llvm::checkedMul(workerCount, factor);
    if (!next)
      return std::nullopt;
    workerCount = *next;
  }
  if (workerCount < 2)
    return std::nullopt;

  WorkUnitCandidate candidate;
  candidate.factors.assign(factors.begin(), factors.end());

  SmallVector<int64_t> coordinates(factors.size(), 0);
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
      return std::nullopt;

    WorkUnitSpec unit;
    unit.iterationOffsets = std::move(iterationOffsets);
    unit.iterationSizes = std::move(iterationSizes);
    for (OpFoldResult value : foldedResultOffsets) {
      std::optional<int64_t> constant = getStaticValue(value);
      if (!constant)
        return std::nullopt;
      unit.resultOffsets.push_back(*constant);
    }
    for (OpFoldResult value : foldedResultSizes) {
      std::optional<int64_t> constant = getStaticValue(value);
      if (!constant || *constant <= 0)
        return std::nullopt;
      unit.resultSizes.push_back(*constant);
    }
    if (unit.resultOffsets.size() !=
            static_cast<size_t>(resultType.getRank()) ||
        unit.resultSizes.size() != static_cast<size_t>(resultType.getRank()))
      return std::nullopt;
    candidate.units.push_back(std::move(unit));
  }
  return candidate;
}

SmallVector<WorkUnitCandidate, 0>
buildWorkUnitCandidates(const ComputeOperation &operation,
                        int64_t parallelWorkers,
                        int64_t minimumWorkItemsPerUnit,
                        DigitalTilingPolicy policy,
                        OpBuilder &builder) {
  SmallVector<WorkUnitCandidate, 0> result;
  if (!isExpandableDigitalOperation(operation))
    return result;

  int64_t totalWorkItems = 1;
  for (const ComputeIterationDimension &dimension :
       operation.iterationDomain) {
    std::optional<int64_t> product =
        llvm::checkedMul(totalWorkItems, dimension.staticExtent);
    if (!product) {
      totalWorkItems = std::numeric_limits<int64_t>::max();
      break;
    }
    totalWorkItems = *product;
  }
  int64_t usefulWorkers = std::max<int64_t>(
      1, totalWorkItems / minimumWorkItemsPerUnit);
  parallelWorkers = std::min(parallelWorkers, usefulWorkers);
  if (parallelWorkers < 2)
    return result;

  SmallVector<SmallVector<int64_t>> factorVectors;
  SmallVector<int64_t> current(operation.iterationDomain.size(), 1);
  if (policy == DigitalTilingPolicy::DimensionFirst) {
    SmallVector<int64_t> factors(operation.iterationDomain.size(), 1);
    int64_t workerCount = 1;
    chooseWorkerFactors(operation.iterationDomain, parallelWorkers,
                        /*dimension=*/0, /*currentProduct=*/1, current, factors,
                        workerCount);
    if (workerCount >= 2)
      factorVectors.push_back(std::move(factors));
  } else {
    enumerateExactWorkerFactors(operation.iterationDomain, parallelWorkers,
                                /*dimension=*/0, current, factorVectors);
  }

  for (const SmallVector<int64_t> &factors : factorVectors) {
    std::optional<WorkUnitCandidate> candidate =
        buildWorkUnitCandidate(operation, factors, builder);
    if (candidate)
      result.push_back(std::move(*candidate));
  }
  return result;
}

std::optional<int64_t> getStaticTensorByteSize(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return std::nullopt;
  unsigned bitWidth = tensor.getElementTypeBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;
  std::optional<int64_t> bytes = llvm::checkedMul(
      tensor.getNumElements(), static_cast<int64_t>(bitWidth / 8));
  return bytes;
}

std::optional<uint64_t>
getCandidateShardPenalty(const WorkUnitCandidate &producer,
                         const WorkUnitCandidate &consumer,
                         AffineMap consumerOperandMap,
                         int64_t fullTensorBytes) {
  uint64_t sourceElements = 0;
  for (const WorkUnitSpec &source : producer.units) {
    uint64_t elements = 1;
    for (int64_t size : source.resultSizes) {
      if (size <= 0 || elements > std::numeric_limits<uint64_t>::max() /
                                      static_cast<uint64_t>(size))
        return std::nullopt;
      elements *= static_cast<uint64_t>(size);
    }
    if (sourceElements > std::numeric_limits<uint64_t>::max() - elements)
      return std::nullopt;
    sourceElements += elements;
  }
  if (sourceElements == 0 || fullTensorBytes < 0 ||
      static_cast<uint64_t>(fullTensorBytes) % sourceElements != 0)
    return std::nullopt;
  uint64_t elementBytes =
      static_cast<uint64_t>(fullTensorBytes) / sourceElements;

  uint64_t routedElements = 0;
  uint64_t intersections = 0;
  for (const WorkUnitSpec &target : consumer.units) {
    std::optional<StaticTileRegion> demand =
        mapIterationTileThroughIndexingMap(consumerOperandMap,
                                           target.iterationOffsets,
                                           target.iterationSizes);
    if (!demand)
      return std::nullopt;
    uint64_t demandedElements = 1;
    for (int64_t size : demand->sizes) {
      if (size <= 0 || demandedElements >
                           std::numeric_limits<uint64_t>::max() /
                               static_cast<uint64_t>(size))
        return std::nullopt;
      demandedElements *= static_cast<uint64_t>(size);
    }
    uint64_t coveredElements = 0;
    for (const WorkUnitSpec &source : producer.units) {
      uint64_t overlapElements = 1;
      bool overlaps = true;
      for (auto [sourceOffset, sourceSize, demandOffset, demandSize] :
           llvm::zip_equal(source.resultOffsets, source.resultSizes,
                           demand->offsets, demand->sizes)) {
        int64_t begin = std::max(sourceOffset, demandOffset);
        int64_t end =
            std::min(sourceOffset + sourceSize, demandOffset + demandSize);
        if (begin >= end) {
          overlaps = false;
          break;
        }
        uint64_t extent = static_cast<uint64_t>(end - begin);
        if (overlapElements >
            std::numeric_limits<uint64_t>::max() / extent)
          return std::nullopt;
        overlapElements *= extent;
      }
      if (!overlaps)
        continue;
      if (coveredElements >
          std::numeric_limits<uint64_t>::max() - overlapElements)
        return std::nullopt;
      coveredElements += overlapElements;
      ++intersections;
    }
    if (coveredElements != demandedElements ||
        routedElements >
            std::numeric_limits<uint64_t>::max() - demandedElements)
      return std::nullopt;
    routedElements += demandedElements;
  }

  if (routedElements > std::numeric_limits<uint64_t>::max() / elementBytes)
    return std::nullopt;
  uint64_t routedBytes = routedElements * elementBytes;
  // Preserve byte volume as the primary objective and use the exact number of
  // routed shard fragments as a deterministic message-overhead tiebreaker.
  constexpr uint64_t kFragmentScale = 1024;
  if (routedBytes >
      (std::numeric_limits<uint64_t>::max() - intersections) /
          kFragmentScale)
    return std::nullopt;
  return routedBytes * kFragmentScale + intersections;
}

SmallVector<CandidateCommunicationEdge> buildCandidateCommunicationEdges(
    const ComputeGraph &graph,
    const DenseMap<int64_t, SmallVector<WorkUnitCandidate, 0>> &candidates) {
  DenseMap<Operation *, int64_t> operationIds;
  for (const ComputeOperation &operation : graph.operations) {
    operationIds[operation.operation] = operation.id;
    for (Operation *member : operation.members)
      operationIds[member] = operation.id;
  }

  SmallVector<CandidateCommunicationEdge> edges;
  for (int64_t producerId : graph.topologicalOrder) {
    auto producerCandidates = candidates.find(producerId);
    if (producerCandidates == candidates.end())
      continue;
    const ComputeOperation &producer = graph.operations[producerId];
    if (producer.operation->getNumResults() != 1)
      continue;
    std::optional<int64_t> byteSize =
        getStaticTensorByteSize(producer.operation->getResult(0).getType());
    if (!byteSize)
      continue;
    for (OpOperand &use : producer.operation->getResult(0).getUses()) {
      auto consumerId = operationIds.find(use.getOwner());
      if (consumerId == operationIds.end() ||
          !candidates.contains(consumerId->second))
        continue;
      auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
      if (!consumer || consumer->getNumResults() != 1)
        continue;
      edges.push_back({producerId, consumerId->second, use.getOperandNumber(),
                       *byteSize, consumer.getMatchingIndexingMap(&use)});
    }
  }
  return edges;
}

uint64_t getCommunicationPenalty(
    ArrayRef<CandidateCommunicationEdge> edges,
    const DenseMap<int64_t, SmallVector<WorkUnitCandidate, 0>> &candidates,
    const DenseMap<int64_t, size_t> &selection) {
  uint64_t penalty = 0;
  for (const CandidateCommunicationEdge &edge : edges) {
    const WorkUnitCandidate &producer =
        candidates.find(edge.producerId)
            ->second[selection.lookup(edge.producerId)];
    const WorkUnitCandidate &consumer =
        candidates.find(edge.consumerId)
            ->second[selection.lookup(edge.consumerId)];
    std::optional<uint64_t> exactPenalty = getCandidateShardPenalty(
        producer, consumer, edge.consumerOperandMap, edge.byteSize);
    uint64_t edgePenalty = exactPenalty.value_or(
        std::numeric_limits<uint64_t>::max());
    if (std::numeric_limits<uint64_t>::max() - penalty < edgePenalty)
      return std::numeric_limits<uint64_t>::max();
    penalty += edgePenalty;
  }
  return penalty;
}

DenseMap<int64_t, size_t> selectCommunicationAwareCandidates(
    const ComputeGraph &graph,
    const DenseMap<int64_t, SmallVector<WorkUnitCandidate, 0>> &candidates) {
  DenseMap<int64_t, size_t> selection;
  for (const auto &entry : candidates)
    selection[entry.first] = 0;
  SmallVector<CandidateCommunicationEdge> edges =
      buildCandidateCommunicationEdges(graph, candidates);

  SmallVector<int64_t> forwardOrder;
  for (int64_t operationId : graph.topologicalOrder)
    if (candidates.contains(operationId))
      forwardOrder.push_back(operationId);
  SmallVector<int64_t> reverseOrder(forwardOrder.rbegin(), forwardOrder.rend());

  auto refine = [&](ArrayRef<int64_t> order) {
    bool changed = false;
    for (int64_t operationId : order) {
      size_t original = selection.lookup(operationId);
      size_t best = original;
      uint64_t bestPenalty =
          getCommunicationPenalty(edges, candidates, selection);
      for (size_t candidate = 0;
           candidate < candidates.find(operationId)->second.size();
           ++candidate) {
        selection[operationId] = candidate;
        uint64_t candidatePenalty =
            getCommunicationPenalty(edges, candidates, selection);
        if (candidatePenalty < bestPenalty) {
          bestPenalty = candidatePenalty;
          best = candidate;
        }
      }
      selection[operationId] = best;
      changed |= best != original;
    }
    return changed;
  };

  for (size_t iteration = 0;
       iteration < std::max<size_t>(1, candidates.size() * 2); ++iteration) {
    bool changed = refine(forwardOrder);
    changed |= refine(reverseOrder);
    if (!changed)
      break;
  }
  return selection;
}

SmallVector<MappingWorkUnitAttr>
materializeWorkUnits(const ComputeOperation &operation,
                     const WorkUnitCandidate &candidate,
                     int64_t &nextWorkUnitId, OpBuilder &builder) {
  SmallVector<MappingWorkUnitAttr> result;
  result.reserve(candidate.units.size());
  for (const WorkUnitSpec &unit : candidate.units) {
    result.push_back(MappingWorkUnitAttr::get(
        builder.getContext(), builder.getI64IntegerAttr(nextWorkUnitId++),
        builder.getI64IntegerAttr(operation.id), builder.getI64IntegerAttr(0),
        builder.getI64ArrayAttr(unit.resultOffsets),
        builder.getI64ArrayAttr(unit.resultSizes),
        builder.getI64ArrayAttr(unit.iterationOffsets),
        builder.getI64ArrayAttr(unit.iterationSizes),
        builder.getI64IntegerAttr(-1), builder.getI64IntegerAttr(-1),
        builder.getI64IntegerAttr(-1)));
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
      minimumWorkItemsPerUnit(
          *this, "minimum-work-items-per-unit",
          llvm::cl::desc(
              "Minimum static iteration work assigned to each digital work "
              "unit; reduces workers for small operations"),
          llvm::cl::init(1)),
      requireChange(
          *this, "require-change",
          llvm::cl::desc("Fail when no digital operation can be expanded"),
          llvm::cl::init(false)),
      dataflow(*this, "dataflow",
               llvm::cl::desc("Digital dataflow mode: bulk or sharded"),
               llvm::cl::init("bulk")),
      tilingPolicy(
          *this, "tiling-policy",
          llvm::cl::desc(
              "Digital tiling policy: dimension-first or communication-aware"),
          llvm::cl::init("dimension-first")),
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
  minimumWorkItemsPerUnit = pass.minimumWorkItemsPerUnit;
  requireChange = pass.requireChange;
  dataflow = pass.dataflow;
  tilingPolicy = pass.tilingPolicy;
  shardPropagationDepth = pass.shardPropagationDepth;
  requireCompleteShardChain = pass.requireCompleteShardChain;
  reductionTree = pass.reductionTree;
  reductionFanIn = pass.reductionFanIn;
  reductionMinimumWidth = pass.reductionMinimumWidth;
}

void ExpandDigitalWorkPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (parallelWorkers < 1 || minimumWorkItemsPerUnit < 1) {
    module.emitError(
        "parallel-workers and minimum-work-items-per-unit must be positive");
    signalPassFailure();
    return;
  }
  FailureOr<mapping::DigitalDataflowMode> parsedDataflow =
      mapping::parseDigitalDataflowMode(dataflow, module);
  FailureOr<DigitalTilingPolicy> parsedTilingPolicy =
      parseDigitalTilingPolicy(tilingPolicy, module);
  FailureOr<mapping::ReductionTreePolicy> parsedReductionTree =
      mapping::parseReductionTreePolicy(reductionTree, module);
  if (failed(parsedDataflow) || failed(parsedTilingPolicy) ||
      failed(parsedReductionTree) || shardPropagationDepth < 0) {
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
    function->removeAttr(
        "sculptor.mapping.digital_minimum_work_items_per_unit");
    function->removeAttr("sculptor.mapping.expanded_digital_operation_count");
    function->removeAttr("sculptor.mapping.expanded_digital_work_unit_count");
    function->removeAttr(kDigitalTilingPolicyAttrName);
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

    DenseMap<int64_t, SmallVector<WorkUnitCandidate, 0>> candidates;
    for (const mapping::ComputeOperation &operation : graph->operations) {
      SmallVector<WorkUnitCandidate, 0> operationCandidates =
          buildWorkUnitCandidates(operation, parallelWorkers,
                                  minimumWorkItemsPerUnit,
                                  *parsedTilingPolicy, builder);
      if (!operationCandidates.empty())
        candidates[operation.id] = std::move(operationCandidates);
    }
    DenseMap<int64_t, size_t> selection;
    if (*parsedTilingPolicy == DigitalTilingPolicy::CommunicationAware)
      selection = selectCommunicationAwareCandidates(*graph, candidates);
    else
      for (const auto &entry : candidates)
        selection[entry.first] = 0;

    int64_t nextWorkUnitId = 0;
    int64_t functionOperations = 0;
    int64_t functionWorkUnits = 0;
    for (int64_t operationId : graph->topologicalOrder) {
      auto found = candidates.find(operationId);
      if (found == candidates.end())
        continue;
      const mapping::ComputeOperation &operation =
          graph->operations[operationId];
      SmallVector<MappingWorkUnitAttr> workUnits = materializeWorkUnits(
          operation, found->second[selection.lookup(operationId)],
          nextWorkUnitId, builder);
      SmallVector<Attribute> attributes(workUnits.begin(), workUnits.end());
      operation.operation->setAttr(mapping::kExpandedDigitalWorkAttrName,
                                   builder.getArrayAttr(attributes));
      ++functionOperations;
      functionWorkUnits += static_cast<int64_t>(workUnits.size());
    }

    if (functionOperations > 0) {
      function->setAttr("sculptor.mapping.digital_parallel_workers",
                        builder.getI64IntegerAttr(parallelWorkers));
      function->setAttr(
          "sculptor.mapping.digital_minimum_work_items_per_unit",
          builder.getI64IntegerAttr(minimumWorkItemsPerUnit));
      function->setAttr("sculptor.mapping.expanded_digital_operation_count",
                        builder.getI64IntegerAttr(functionOperations));
      function->setAttr("sculptor.mapping.expanded_digital_work_unit_count",
                        builder.getI64IntegerAttr(functionWorkUnits));
    }
    function->setAttr(kDigitalTilingPolicyAttrName,
                      builder.getStringAttr(
                          stringifyDigitalTilingPolicy(*parsedTilingPolicy)));
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
      "sculptor.mapping.digital_minimum_work_items_per_unit",
      IntegerAttr::get(IntegerType::get(&getContext(), 64),
                       minimumWorkItemsPerUnit));
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
  module->setAttr(kDigitalTilingPolicyAttrName,
                  StringAttr::get(&getContext(), stringifyDigitalTilingPolicy(
                                                     *parsedTilingPolicy)));
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
