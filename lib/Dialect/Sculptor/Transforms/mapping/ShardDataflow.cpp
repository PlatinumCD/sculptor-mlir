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

FailureOr<int64_t> getShardByteSize(MappingWorkUnitAttr attr,
                                    RankedTensorType type, Operation *anchor) {
  unsigned bitWidth = type.getElementTypeBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0) {
    anchor->emitError("shard dataflow requires byte-addressable element types");
    return failure();
  }
  int64_t bytes = bitWidth / 8;
  for (Attribute value : attr.getResultSizes()) {
    auto size = dyn_cast<IntegerAttr>(value);
    if (!size || size.getInt() <= 0) {
      anchor->emitError("shard dataflow requires positive static tile sizes");
      return failure();
    }
    std::optional<int64_t> next = llvm::checkedMul(bytes, size.getInt());
    if (!next) {
      anchor->emitError("shard byte size overflows int64");
      return failure();
    }
    bytes = *next;
  }
  return bytes;
}

bool sameResultTile(MappingWorkUnitAttr lhs, MappingWorkUnitAttr rhs) {
  return lhs.getResultOffsets() == rhs.getResultOffsets() &&
         lhs.getResultSizes() == rhs.getResultSizes();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

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
      if (producersByConsumer.lookup(consumerId).size() > 1) {
        assemblyBoundaries.insert({producerId, consumerId});
        if (requireCompleteChain) {
          use.getOwner()->emitError("complete shard chain reaches incompatible "
                                    "producer shard groups");
          return failure();
        }
        continue;
      }

      auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
      if (!consumer || consumer->getNumResults() != 1 ||
          consumer->getResult(0).getType() != producerType ||
          llvm::any_of(cast<TilingInterface>(consumer.getOperation())
                           .getLoopIteratorTypes(),
                       [](utils::IteratorType iterator) {
                         return iterator != utils::IteratorType::parallel;
                       })) {
        assemblyBoundaries.insert({producerId, consumerId});
        if (requireCompleteChain) {
          use.getOwner()->emitError(
              "complete shard chain reaches a non-elementwise boundary");
          return failure();
        }
        continue;
      }
      AffineMap operandMap = consumer.getMatchingIndexingMap(&use);
      AffineMap resultMap =
          consumer.getIndexingMapMatchingResult(consumer->getResult(0));
      if (operandMap != resultMap) {
        assemblyBoundaries.insert({producerId, consumerId});
        if (requireCompleteChain) {
          use.getOwner()->emitError(
              "complete shard chain reaches an incompatible indexing map");
          return failure();
        }
        continue;
      }

      SmallVector<std::pair<MappingWorkUnitAttr, MappingWorkUnitAttr>> pairs;
      for (MappingWorkUnitAttr source : producerUnits->second) {
        auto target = llvm::find_if(consumerUnits->second,
                                    [&](MappingWorkUnitAttr candidate) {
                                      return sameResultTile(source, candidate);
                                    });
        if (target == consumerUnits->second.end()) {
          pairs.clear();
          break;
        }
        pairs.push_back({source, *target});
      }
      if (pairs.size() != producerUnits->second.size() ||
          pairs.size() != consumerUnits->second.size()) {
        assemblyBoundaries.insert({producerId, consumerId});
        if (requireCompleteChain) {
          use.getOwner()->emitError(
              "complete shard chain has incompatible work-unit coverage");
          return failure();
        }
        continue;
      }

      int64_t shardGroup = pairs.front().first.getShardGroupId().getInt();
      SmallVector<Attribute> rewrittenConsumerUnits;
      rewrittenConsumerUnits.reserve(consumerUnits->second.size());
      for (auto [index, unit] : llvm::enumerate(consumerUnits->second)) {
        MappingWorkUnitAttr updated = withShardIdentity(
            unit, shardGroup, index, consumerUnits->second.size());
        consumerUnits->second[index] = updated;
        rewrittenConsumerUnits.push_back(updated);
      }
      graph.operations[consumerId].operation->setAttr(
          kExpandedDigitalWorkAttrName,
          builder.getArrayAttr(rewrittenConsumerUnits));
      propagationLevel[consumerId] = propagationLevel.lookup(producerId) + 1;

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

      for (const auto &pair : pairs) {
        MappingWorkUnitAttr source = pair.first;
        auto target = llvm::find_if(consumerUnits->second,
                                    [&](MappingWorkUnitAttr candidate) {
                                      return sameResultTile(source, candidate);
                                    });
        assert(target != consumerUnits->second.end());
        auto identity =
            std::make_tuple(producerId, source.getId().getInt(), consumerId,
                            target->getId().getInt(), use.getOperandNumber());
        if (!seen.insert(identity).second)
          continue;
        FailureOr<int64_t> bytes =
            getShardByteSize(source, producerType, use.getOwner());
        if (failed(bytes))
          return failure();
        exactEdges.push_back(MappingWorkUnitEdgeAttr::get(
            function.getContext(), builder.getI64IntegerAttr(producerId),
            source.getId(), builder.getI64IntegerAttr(consumerId),
            target->getId(), builder.getI64IntegerAttr(tensorId),
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
