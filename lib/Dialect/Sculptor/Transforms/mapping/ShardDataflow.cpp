#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ShardDataflow.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

MappingWorkUnitAttr withShardIdentity(MappingWorkUnitAttr attr, int64_t groupId,
                                      int64_t index, int64_t count) {
  return MappingWorkUnitAttr::get(
      attr.getContext(), attr.getId(), attr.getOperationId(),
      attr.getResultNumber(), attr.getResultOffsets(), attr.getResultSizes(),
      attr.getIterationOffsets(), attr.getIterationSizes(),
      IntegerAttr::get(IntegerType::get(attr.getContext(), 64), groupId),
      IntegerAttr::get(IntegerType::get(attr.getContext(), 64), index),
      IntegerAttr::get(IntegerType::get(attr.getContext(), 64), count));
}

FailureOr<int64_t> getRegionByteSize(ArrayRef<int64_t> sizes,
                                     RankedTensorType type,
                                     Operation *anchor) {
  unsigned bitWidth = type.getElementTypeBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0) {
    anchor->emitError("shard dataflow requires byte-addressable element types");
    return failure();
  }
  int64_t bytes = bitWidth / 8;
  for (int64_t size : sizes) {
    if (size <= 0) {
      anchor->emitError("shard dataflow requires positive static tile sizes");
      return failure();
    }
    std::optional<int64_t> next = llvm::checkedMul(bytes, size);
    if (!next) {
      anchor->emitError("shard byte size overflows int64");
      return failure();
    }
    bytes = *next;
  }
  return bytes;
}

std::optional<StaticTileRegion>
intersectTileRegions(ArrayRef<int64_t> lhsOffsets,
                     ArrayRef<int64_t> lhsSizes,
                     ArrayRef<int64_t> rhsOffsets,
                     ArrayRef<int64_t> rhsSizes) {
  if (lhsOffsets.size() != lhsSizes.size() ||
      rhsOffsets.size() != rhsSizes.size() ||
      lhsOffsets.size() != rhsOffsets.size())
    return std::nullopt;

  StaticTileRegion intersection;
  intersection.offsets.reserve(lhsOffsets.size());
  intersection.sizes.reserve(lhsSizes.size());
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhsOffsets, lhsSizes, rhsOffsets, rhsSizes)) {
    if (lhsSize <= 0 || rhsSize <= 0)
      return std::nullopt;
    int64_t begin = std::max(lhsOffset, rhsOffset);
    int64_t end = std::min(lhsOffset + lhsSize, rhsOffset + rhsSize);
    if (begin >= end)
      return std::nullopt;
    intersection.offsets.push_back(begin);
    intersection.sizes.push_back(end - begin);
  }
  return intersection;
}

bool regionsOverlap(ArrayRef<int64_t> lhsOffsets, ArrayRef<int64_t> lhsSizes,
                    ArrayRef<int64_t> rhsOffsets,
                    ArrayRef<int64_t> rhsSizes) {
  return intersectTileRegions(lhsOffsets, lhsSizes, rhsOffsets, rhsSizes)
      .has_value();
}

FailureOr<int64_t> getRegionElementCount(ArrayRef<int64_t> sizes,
                                         Operation *anchor) {
  int64_t elements = 1;
  for (int64_t size : sizes) {
    if (size <= 0) {
      anchor->emitError("shard dataflow requires positive static tile sizes");
      return failure();
    }
    std::optional<int64_t> next = llvm::checkedMul(elements, size);
    if (!next) {
      anchor->emitError("shard element count overflows int64");
      return failure();
    }
    elements = *next;
  }
  return elements;
}

bool sameTileRegion(ArrayRef<int64_t> lhsOffsets, ArrayRef<int64_t> lhsSizes,
                    ArrayRef<int64_t> rhsOffsets, ArrayRef<int64_t> rhsSizes) {
  return lhsOffsets == rhsOffsets && lhsSizes == rhsSizes;
}

SmallVector<int64_t> getIntegerArray(ArrayAttr values) {
  SmallVector<int64_t> result;
  result.reserve(values.size());
  for (Attribute value : values)
    result.push_back(cast<IntegerAttr>(value).getInt());
  return result;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

std::optional<StaticTileRegion>
mapIterationTileThroughIndexingMap(AffineMap indexingMap,
                                   ArrayRef<int64_t> iterationOffsets,
                                   ArrayRef<int64_t> iterationSizes) {
  if (indexingMap.getNumSymbols() != 0 ||
      indexingMap.getNumDims() != iterationOffsets.size() ||
      iterationOffsets.size() != iterationSizes.size())
    return std::nullopt;

  StaticTileRegion region;
  region.offsets.reserve(indexingMap.getNumResults());
  region.sizes.reserve(indexingMap.getNumResults());
  for (AffineExpr expression : indexingMap.getResults()) {
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      unsigned position = dimension.getPosition();
      if (position >= iterationOffsets.size() || iterationSizes[position] <= 0)
        return std::nullopt;
      region.offsets.push_back(iterationOffsets[position]);
      region.sizes.push_back(iterationSizes[position]);
      continue;
    }
    if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
      region.offsets.push_back(constant.getValue());
      region.sizes.push_back(1);
      continue;
    }
    return std::nullopt;
  }
  return region;
}

FailureOr<DigitalDataflowMode> parseDigitalDataflowMode(StringRef value,
                                                        Operation *anchor) {
  if (value == "bulk")
    return DigitalDataflowMode::Bulk;
  if (value == "sharded")
    return DigitalDataflowMode::Sharded;
  anchor->emitError("unknown digital dataflow mode '") << value << "'";
  return failure();
}

StringRef stringifyDigitalDataflowMode(DigitalDataflowMode mode) {
  switch (mode) {
  case DigitalDataflowMode::Bulk:
    return "bulk";
  case DigitalDataflowMode::Sharded:
    return "sharded";
  }
  llvm_unreachable("unknown digital dataflow mode");
}

LogicalResult planShardDataflow(func::FuncOp function,
                                const ComputeGraph &graph,
                                DigitalDataflowMode mode,
                                int64_t propagationDepth,
                                bool requireCompleteChain) {
  function->removeAttr(kShardWorkUnitEdgesAttrName);
  function->removeAttr(kShardGroupCountAttrName);
  function->removeAttr(kShardEdgeCountAttrName);
  function->removeAttr(kAssemblyBoundaryCountAttrName);
  OpBuilder builder(function.getContext());
  if (mode == DigitalDataflowMode::Bulk) {
    function->setAttr(kShardGroupCountAttrName, builder.getI64IntegerAttr(0));
    function->setAttr(kShardEdgeCountAttrName, builder.getI64IntegerAttr(0));
    function->setAttr(kAssemblyBoundaryCountAttrName,
                      builder.getI64IntegerAttr(0));
    return success();
  }

  DenseMap<int64_t, SmallVector<MappingWorkUnitAttr>> unitsByOperation;
  DenseMap<Operation *, int64_t> operationIds;
  for (const ComputeOperation &operation : graph.operations) {
    operationIds[operation.operation] = operation.id;
    for (Operation *member : operation.members)
      operationIds[member] = operation.id;
    auto expanded = operation.operation->getAttrOfType<ArrayAttr>(
        kExpandedDigitalWorkAttrName);
    if (!expanded)
      continue;
    for (Attribute value : expanded) {
      auto unit = dyn_cast<MappingWorkUnitAttr>(value);
      if (!unit)
        return operation.operation->emitError(
            "expanded digital work contains an untyped work unit");
      unitsByOperation[operation.id].push_back(unit);
    }
  }

  int64_t nextShardGroup = 0;
  DenseMap<int64_t, int64_t> propagationLevel;
  for (int64_t operationId : graph.topologicalOrder) {
    auto found = unitsByOperation.find(operationId);
    if (found == unitsByOperation.end())
      continue;
    SmallVector<Attribute> rewritten;
    rewritten.reserve(found->second.size());
    for (auto [index, unit] : llvm::enumerate(found->second)) {
      MappingWorkUnitAttr updated =
          withShardIdentity(unit, nextShardGroup, index, found->second.size());
      found->second[index] = updated;
      rewritten.push_back(updated);
    }
    graph.operations[operationId].operation->setAttr(
        kExpandedDigitalWorkAttrName, builder.getArrayAttr(rewritten));
    propagationLevel[operationId] = 0;
    ++nextShardGroup;
  }

  DenseMap<int64_t, llvm::DenseSet<int64_t>> producersByConsumer;
  for (int64_t producerId : graph.topologicalOrder) {
    if (!unitsByOperation.contains(producerId))
      continue;
    const ComputeOperation &producer = graph.operations[producerId];
    if (producer.operation->getNumResults() != 1)
      continue;
    for (OpOperand &use : producer.operation->getResult(0).getUses()) {
      auto consumerId = operationIds.find(use.getOwner());
      if (consumerId != operationIds.end() &&
          unitsByOperation.contains(consumerId->second))
        producersByConsumer[consumerId->second].insert(producerId);
    }
  }

  SmallVector<Attribute> exactEdges;
  std::set<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>> seen;
  std::set<std::pair<int64_t, int64_t>> assemblyBoundaries;
  for (int64_t producerId : graph.topologicalOrder) {
    auto producerUnits = unitsByOperation.find(producerId);
    if (producerUnits == unitsByOperation.end())
      continue;
    const ComputeOperation &producer = graph.operations[producerId];
    if (producer.operation->getNumResults() != 1)
      continue;
    auto producerType =
        dyn_cast<RankedTensorType>(producer.operation->getResult(0).getType());
    if (!producerType || !producerType.hasStaticShape())
      continue;

    for (OpOperand &use : producer.operation->getResult(0).getUses()) {
      auto consumerIdIt = operationIds.find(use.getOwner());
      if (consumerIdIt == operationIds.end()) {
        assemblyBoundaries.insert({producerId, -1});
        if (requireCompleteChain) {
          use.getOwner()->emitError(
              "complete shard chain reaches an unexpanded graph boundary");
          return failure();
        }
        continue;
      }
      int64_t consumerId = consumerIdIt->second;
      auto consumerUnits = unitsByOperation.find(consumerId);
      if (consumerUnits == unitsByOperation.end()) {
        assemblyBoundaries.insert({producerId, consumerId});
        if (requireCompleteChain) {
          use.getOwner()->emitError(
              "complete shard chain reaches an unexpanded consumer");
          return failure();
        }
        continue;
      }
      if (propagationDepth > 0 &&
          propagationLevel.lookup(producerId) >= propagationDepth) {
        assemblyBoundaries.insert({producerId, consumerId});
        if (requireCompleteChain) {
          use.getOwner()->emitError(
              "complete shard chain exceeds shard-propagation-depth");
          return failure();
        }
        continue;
      }
      auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
      if (!consumer || consumer->getNumResults() != 1) {
        assemblyBoundaries.insert({producerId, consumerId});
        if (requireCompleteChain) {
          use.getOwner()->emitError(
              "complete shard chain reaches an unsupported boundary");
          return failure();
        }
        continue;
      }
      AffineMap operandMap = consumer.getMatchingIndexingMap(&use);
      struct WorkUnitIntersection {
        MappingWorkUnitAttr source;
        MappingWorkUnitAttr target;
        size_t targetIndex;
        StaticTileRegion region;
      };
      SmallVector<WorkUnitIntersection> intersections;
      bool completeCoverage = true;
      for (auto [targetIndex, target] :
           llvm::enumerate(consumerUnits->second)) {
        std::optional<StaticTileRegion> inputRegion =
            mapIterationTileThroughIndexingMap(
                operandMap, getIntegerArray(target.getIterationOffsets()),
                getIntegerArray(target.getIterationSizes()));
        if (!inputRegion) {
          completeCoverage = false;
          break;
        }

        FailureOr<int64_t> demandedElements =
            getRegionElementCount(inputRegion->sizes, use.getOwner());
        if (failed(demandedElements))
          return failure();
        int64_t coveredElements = 0;
        SmallVector<StaticTileRegion> targetIntersections;
        for (MappingWorkUnitAttr source : producerUnits->second) {
          std::optional<StaticTileRegion> intersection = intersectTileRegions(
              getIntegerArray(source.getResultOffsets()),
              getIntegerArray(source.getResultSizes()), inputRegion->offsets,
              inputRegion->sizes);
          if (!intersection)
            continue;
          if (llvm::any_of(targetIntersections,
                           [&](const StaticTileRegion &existing) {
                             return regionsOverlap(
                                 existing.offsets, existing.sizes,
                                 intersection->offsets, intersection->sizes);
                           })) {
            use.getOwner()->emitError(
                "producer work units overlap within one consumer demand");
            return failure();
          }
          FailureOr<int64_t> intersectionElements =
              getRegionElementCount(intersection->sizes, use.getOwner());
          if (failed(intersectionElements))
            return failure();
          std::optional<int64_t> next =
              llvm::checkedAdd(coveredElements, *intersectionElements);
          if (!next) {
            use.getOwner()->emitError(
                "consumer shard coverage overflows int64");
            return failure();
          }
          coveredElements = *next;
          targetIntersections.push_back(*intersection);
          intersections.push_back({source, target, targetIndex,
                                   std::move(*intersection)});
        }
        if (coveredElements != *demandedElements) {
          completeCoverage = false;
          break;
        }
      }
      if (!completeCoverage || intersections.empty()) {
        assemblyBoundaries.insert({producerId, consumerId});
        if (requireCompleteChain) {
          use.getOwner()->emitError(
              "complete shard chain has incomplete work-unit coverage");
          return failure();
        }
        continue;
      }

      // Preserve the original one-to-one shard identity propagation. For an
      // intersection graph, the target work units have their own stable shard
      // group because one target can depend on multiple source shards or
      // multiple producer groups.
      bool oneToOne = producersByConsumer.lookup(consumerId).size() == 1 &&
                      intersections.size() == producerUnits->second.size() &&
                      intersections.size() == consumerUnits->second.size();
      SmallVector<int64_t> sourceShardByTarget(consumerUnits->second.size(),
                                               -1);
      SmallVector<bool> seenSources(producerUnits->second.size(), false);
      if (oneToOne) {
        for (const WorkUnitIntersection &intersection : intersections) {
          SmallVector<int64_t> sourceOffsets =
              getIntegerArray(intersection.source.getResultOffsets());
          SmallVector<int64_t> sourceSizes =
              getIntegerArray(intersection.source.getResultSizes());
          if (!sameTileRegion(sourceOffsets, sourceSizes,
                              intersection.region.offsets,
                              intersection.region.sizes) ||
              sourceShardByTarget[intersection.targetIndex] >= 0) {
            oneToOne = false;
            break;
          }
          auto sourceIt = llvm::find_if(
              producerUnits->second, [&](MappingWorkUnitAttr candidate) {
                return candidate.getId() == intersection.source.getId();
              });
          if (sourceIt == producerUnits->second.end()) {
            oneToOne = false;
            break;
          }
          size_t sourceIndex =
              static_cast<size_t>(sourceIt - producerUnits->second.begin());
          if (seenSources[sourceIndex]) {
            oneToOne = false;
            break;
          }
          seenSources[sourceIndex] = true;
          sourceShardByTarget[intersection.targetIndex] =
              intersection.source.getShardIndex().getInt();
        }
      }
      if (oneToOne) {
        int64_t shardGroup =
            intersections.front().source.getShardGroupId().getInt();
        SmallVector<Attribute> rewrittenConsumerUnits;
        rewrittenConsumerUnits.reserve(consumerUnits->second.size());
        for (auto [targetIndex, unit] :
             llvm::enumerate(consumerUnits->second)) {
          MappingWorkUnitAttr updated = withShardIdentity(
              unit, shardGroup, sourceShardByTarget[targetIndex],
              consumerUnits->second.size());
          consumerUnits->second[targetIndex] = updated;
          rewrittenConsumerUnits.push_back(updated);
        }
        graph.operations[consumerId].operation->setAttr(
            kExpandedDigitalWorkAttrName,
            builder.getArrayAttr(rewrittenConsumerUnits));
      }
      propagationLevel[consumerId] =
          std::max(propagationLevel.lookup(consumerId),
                   propagationLevel.lookup(producerId) + 1);

      int64_t tensorId = -1;
      for (const ComputeTensor &tensor : graph.tensors) {
        if (tensor.value == use.get() &&
            llvm::is_contained(tensor.producerOperations, producerId) &&
            llvm::is_contained(tensor.consumerOperations, consumerId)) {
          tensorId = tensor.id;
          break;
        }
      }
      if (tensorId < 0)
        return use.getOwner()->emitError(
            "shard edge cannot resolve its compute tensor");

      for (const WorkUnitIntersection &intersection : intersections) {
        MappingWorkUnitAttr source = intersection.source;
        MappingWorkUnitAttr target =
            consumerUnits->second[intersection.targetIndex];
        auto identity =
            std::make_tuple(producerId, source.getId().getInt(), consumerId,
                            target.getId().getInt(), use.getOperandNumber());
        if (!seen.insert(identity).second)
          continue;
        FailureOr<int64_t> bytes = getRegionByteSize(
            intersection.region.sizes, producerType, use.getOwner());
        if (failed(bytes))
          return failure();
        exactEdges.push_back(MappingWorkUnitEdgeAttr::get(
            function.getContext(), builder.getI64IntegerAttr(producerId),
            source.getId(), builder.getI64IntegerAttr(consumerId),
            target.getId(), builder.getI64IntegerAttr(tensorId),
            builder.getI64IntegerAttr(0),
            builder.getI64IntegerAttr(use.getOperandNumber()),
            builder.getI64IntegerAttr(*bytes)));
      }
    }
  }

  function->setAttr(kShardWorkUnitEdgesAttrName,
                    builder.getArrayAttr(exactEdges));
  std::set<int64_t> shardGroups;
  for (const auto &entry : unitsByOperation) {
    for (MappingWorkUnitAttr unit : entry.second)
      shardGroups.insert(unit.getShardGroupId().getInt());
  }
  function->setAttr(kShardGroupCountAttrName,
                    builder.getI64IntegerAttr(shardGroups.size()));
  function->setAttr(kShardEdgeCountAttrName,
                    builder.getI64IntegerAttr(exactEdges.size()));
  function->setAttr(kAssemblyBoundaryCountAttrName,
                    builder.getI64IntegerAttr(assemblyBoundaries.size()));
  return success();
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
