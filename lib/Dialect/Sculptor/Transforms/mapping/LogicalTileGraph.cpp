#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingRealization.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

using Endpoint = std::pair<int64_t, int64_t>;
using TilePair = std::pair<int64_t, int64_t>;
using DependencyKey =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

DependencyKey getDependencyKey(const LogicalTileDependency &dependency) {
  return {dependency.sourceOperationId, dependency.sourceWorkUnitId,
          dependency.targetOperationId, dependency.targetWorkUnitId,
          dependency.tensorId,          dependency.targetOperandNumber,
          dependency.byteSize};
}

FailureOr<int64_t> checkedAddBytes(int64_t current, int64_t increment,
                                   Operation *anchor) {
  std::optional<int64_t> result = llvm::checkedAdd(current, increment);
  if (!result) {
    anchor->emitError("logical-tile communication byte count overflow");
    return failure();
  }
  return *result;
}

FailureOr<SmallVector<int64_t>>
buildRANodePath(int64_t leafId, int64_t rootId,
                const DenseMap<int64_t, const StructuralRATreeNode *> &nodes,
                Operation *anchor) {
  SmallVector<int64_t> path;
  llvm::SmallDenseSet<int64_t> visited;
  int64_t nodeId = leafId;
  while (nodeId >= 0) {
    if (!visited.insert(nodeId).second) {
      anchor->emitError("cycle in RA-tree ancestry for leaf ") << leafId;
      return failure();
    }
    const StructuralRATreeNode *node = nodes.lookup(nodeId);
    if (!node) {
      anchor->emitError("logical-tile formation cannot resolve RA node ")
          << nodeId;
      return failure();
    }
    path.push_back(nodeId);
    nodeId = node->parentId;
  }
  std::reverse(path.begin(), path.end());
  if (path.empty() || path.front() != rootId || path.back() != leafId) {
    anchor->emitError("RA path for leaf ")
        << leafId << " does not connect the tree root to the leaf";
    return failure();
  }
  return path;
}

ArrayAttr getI64Array(Builder &builder, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attributes;
  attributes.reserve(values.size());
  for (int64_t value : values)
    attributes.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attributes);
}

FailureOr<SmallVector<int64_t>> parseI64Array(ArrayAttr attr, Operation *anchor,
                                              StringRef description) {
  SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (Attribute value : attr) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer) {
      anchor->emitError(description) << " must contain integer attributes";
      return failure();
    }
    values.push_back(integer.getInt());
  }
  return values;
}

LogicalTileAssignmentAttr
serializeAssignment(Builder &builder, const LogicalTileAssignment &assignment) {
  MappingLaneKind laneKind = assignment.laneKind == LogicalLaneKind::Digital
                                 ? MappingLaneKind::Digital
                                 : MappingLaneKind::Analog;
  return LogicalTileAssignmentAttr::get(
      builder.getContext(), builder.getI64IntegerAttr(assignment.leafId),
      builder.getI64IntegerAttr(assignment.operationId),
      builder.getI64IntegerAttr(assignment.workUnitId), laneKind,
      builder.getI64IntegerAttr(assignment.laneIndex),
      getI64Array(builder, assignment.raNodePath),
      builder.getF64FloatAttr(assignment.startNs),
      builder.getF64FloatAttr(assignment.finishNs));
}

LogicalTileDependencyAttr
serializeDependency(Builder &builder, const LogicalTileDependency &dependency) {
  return LogicalTileDependencyAttr::get(
      builder.getContext(),
      builder.getI64IntegerAttr(dependency.sourceOperationId),
      builder.getI64IntegerAttr(dependency.sourceWorkUnitId),
      builder.getI64IntegerAttr(dependency.targetOperationId),
      builder.getI64IntegerAttr(dependency.targetWorkUnitId),
      builder.getI64IntegerAttr(dependency.tensorId),
      builder.getI64IntegerAttr(dependency.targetOperandNumber),
      builder.getI64IntegerAttr(dependency.byteSize));
}

FailureOr<LogicalTileAssignment>
deserializeAssignment(LogicalTileAssignmentAttr attr, Operation *anchor) {
  FailureOr<SmallVector<int64_t>> path = parseI64Array(
      attr.getRaNodePath(), anchor, "logical-tile assignment RA path");
  if (failed(path))
    return failure();
  LogicalTileAssignment assignment;
  assignment.leafId = attr.getLeafId().getInt();
  assignment.operationId = attr.getOperationId().getInt();
  assignment.workUnitId = attr.getWorkUnitId().getInt();
  assignment.laneKind = attr.getLaneKind() == MappingLaneKind::Digital
                            ? LogicalLaneKind::Digital
                            : LogicalLaneKind::Analog;
  assignment.laneIndex = attr.getLaneIndex().getInt();
  assignment.raNodePath = std::move(*path);
  assignment.startNs = attr.getStartNs().getValueAsDouble();
  assignment.finishNs = attr.getFinishNs().getValueAsDouble();
  return assignment;
}

LogicalTileDependency deserializeDependency(LogicalTileDependencyAttr attr) {
  return {attr.getSourceOperationId().getInt(),
          attr.getSourceWorkUnitId().getInt(),
          attr.getTargetOperationId().getInt(),
          attr.getTargetWorkUnitId().getInt(),
          attr.getTensorId().getInt(),
          attr.getTargetOperandNumber().getInt(),
          attr.getByteSize().getInt()};
}

bool assignmentLess(const LogicalTileAssignment &left,
                    const LogicalTileAssignment &right) {
  return std::tie(left.startNs, left.finishNs, left.leafId) <
         std::tie(right.startNs, right.finishNs, right.leafId);
}

bool dependencyLess(const LogicalTileDependency &left,
                    const LogicalTileDependency &right) {
  return getDependencyKey(left) < getDependencyKey(right);
}

class LogicalTileGraphBuilder {
public:
  LogicalTileGraphBuilder(const ComputeGraph &graph,
                          const ResourceAllocationTree &tree,
                          const MappingRealization &realization,
                          Operation *anchor)
      : graph(graph), tree(tree), realization(realization), anchor(anchor) {
    for (const StructuralRATreeNode &node : tree.nodes)
      nodes[node.id] = &node;
  }

  FailureOr<LogicalTileGraph> build() {
    if (realization.logicalTileCount <= 0 ||
        realization.analogLanesPerTile <= 0 ||
        realization.digitalWorkPerTile.size() !=
            static_cast<size_t>(realization.logicalTileCount)) {
      anchor->emitError("cannot form logical tiles from an invalid mapping "
                        "realization shape");
      return failure();
    }

    result.logicalTileCapacity = realization.logicalTileCount;
    result.analogLanesPerTile = realization.analogLanesPerTile;
    workingTiles.reserve(realization.logicalTileCount);
    for (int64_t tileId = 0; tileId < realization.logicalTileCount; ++tileId) {
      LogicalTile tile;
      tile.id = tileId;
      tile.digitalWork = realization.digitalWorkPerTile[tileId];
      tile.analogLanes.reserve(realization.analogLanesPerTile);
      for (int64_t lane = 0; lane < realization.analogLanesPerTile; ++lane)
        tile.analogLanes.push_back({lane, std::nullopt, {}});
      workingTiles.push_back(std::move(tile));
    }

    if (failed(addAssignments()) || failed(addBoundaries()) ||
        failed(addDependencies()))
      return failure();

    for (LogicalTile &tile : workingTiles) {
      bool active = !tile.digitalAssignments.empty() ||
                    llvm::any_of(tile.analogLanes, [](const auto &lane) {
                      return !lane.assignments.empty();
                    });
      if (!active)
        continue;
      llvm::sort(tile.digitalAssignments, assignmentLess);
      for (LogicalTileAnalogLane &lane : tile.analogLanes)
        llvm::sort(lane.assignments, assignmentLess);
      llvm::sort(tile.internalDependencies, dependencyLess);
      llvm::sort(tile.modelInputTensorIds);
      llvm::sort(tile.modelOutputTensorIds);
      result.tileIndexById[tile.id] = result.tiles.size();
      result.tiles.push_back(std::move(tile));
    }

    int64_t edgeId = 0;
    for (auto &[tilePair, edge] : externalEdges) {
      edge.id = edgeId++;
      llvm::sort(edge.dependencies, dependencyLess);
      result.edges.push_back(std::move(edge));
    }
    return result;
  }

private:
  FailureOr<SmallVector<int64_t>> getTilesForEndpoint(int64_t operationId,
                                                      int64_t workUnitId) {
    const std::set<int64_t> *tileSet = nullptr;
    if (workUnitId >= 0) {
      auto found = endpointTiles.find({operationId, workUnitId});
      if (found != endpointTiles.end())
        tileSet = &found->second;
    } else {
      auto found = operationTiles.find(operationId);
      if (found != operationTiles.end())
        tileSet = &found->second;
    }
    if (!tileSet || tileSet->empty()) {
      anchor->emitError("logical-tile dependency endpoint is not assigned: "
                        "operation ")
          << operationId << ", work unit " << workUnitId;
      return failure();
    }
    return SmallVector<int64_t>(tileSet->begin(), tileSet->end());
  }

  LogicalResult addAssignments() {
    std::set<int64_t> seenLeaves;
    for (const MappingLeafAssignment &realized : realization.leafAssignments) {
      if (realized.tileId < 0 ||
          realized.tileId >= realization.logicalTileCount ||
          !seenLeaves.insert(realized.leafId).second) {
        anchor->emitError("mapping realization contains an invalid or "
                          "duplicate logical-tile leaf assignment");
        return failure();
      }
      const StructuralRATreeNode *leaf = nodes.lookup(realized.leafId);
      if (!leaf || leaf->kind != RATreeNodeKind::Leaf ||
          leaf->operationId != realized.operationId ||
          realized.operationId < 0 ||
          realized.operationId >=
              static_cast<int64_t>(graph.operations.size())) {
        anchor->emitError("logical-tile assignment does not match its RA "
                          "leaf or compute operation");
        return failure();
      }
      FailureOr<SmallVector<int64_t>> path =
          buildRANodePath(leaf->id, tree.rootId, nodes, anchor);
      if (failed(path))
        return failure();

      LogicalTileAssignment assignment;
      assignment.leafId = realized.leafId;
      assignment.operationId = realized.operationId;
      assignment.workUnitId = leaf->workUnitId;
      assignment.laneKind = realized.laneKind;
      assignment.laneIndex = realized.laneIndex;
      assignment.raNodePath = std::move(*path);
      assignment.startNs = realized.startNs;
      assignment.finishNs = realized.finishNs;

      LogicalTile &tile = workingTiles[realized.tileId];
      const ComputeOperation &operation =
          graph.operations[realized.operationId];
      if (realized.laneKind == LogicalLaneKind::Digital) {
        if (realized.laneIndex != 0) {
          anchor->emitError("digital logical-tile assignments must use lane "
                            "index zero");
          return failure();
        }
        tile.digitalAssignments.push_back(std::move(assignment));
      } else {
        if (realized.laneIndex < 0 ||
            realized.laneIndex >= realization.analogLanesPerTile) {
          anchor->emitError("analog logical-tile assignment has an invalid "
                            "lane index");
          return failure();
        }
        LogicalTileAnalogLane &lane = tile.analogLanes[realized.laneIndex];
        if (operation.laneBindingGroup) {
          if (lane.laneBindingGroup &&
              lane.laneBindingGroup != operation.laneBindingGroup) {
            anchor->emitError("one logical analog lane contains multiple "
                              "matrix lane-binding groups");
            return failure();
          }
          lane.laneBindingGroup = operation.laneBindingGroup;
        }
        lane.assignments.push_back(std::move(assignment));
      }
      operationTiles[realized.operationId].insert(realized.tileId);
      if (leaf->workUnitId >= 0)
        endpointTiles[{realized.operationId, leaf->workUnitId}].insert(
            realized.tileId);
    }
    return success();
  }

  LogicalResult addBoundaries() {
    for (const ComputeTensor &tensor : graph.tensors) {
      if (tensor.isFunctionInput) {
        std::set<int64_t> consumers;
        for (int64_t operationId : tensor.consumerOperations) {
          auto found = operationTiles.find(operationId);
          if (found != operationTiles.end())
            consumers.insert(found->second.begin(), found->second.end());
        }
        for (int64_t tileId : consumers)
          workingTiles[tileId].modelInputTensorIds.push_back(tensor.id);
      }
      if (tensor.isFunctionOutput) {
        std::set<int64_t> producers;
        for (int64_t operationId : tensor.producerOperations) {
          auto found = operationTiles.find(operationId);
          if (found != operationTiles.end())
            producers.insert(found->second.begin(), found->second.end());
        }
        for (int64_t tileId : producers)
          workingTiles[tileId].modelOutputTensorIds.push_back(tensor.id);
      }
    }
    for (LogicalTile &tile : workingTiles) {
      llvm::sort(tile.modelInputTensorIds);
      tile.modelInputTensorIds.erase(
          std::unique(tile.modelInputTensorIds.begin(),
                      tile.modelInputTensorIds.end()),
          tile.modelInputTensorIds.end());
      llvm::sort(tile.modelOutputTensorIds);
      tile.modelOutputTensorIds.erase(
          std::unique(tile.modelOutputTensorIds.begin(),
                      tile.modelOutputTensorIds.end()),
          tile.modelOutputTensorIds.end());
    }
    return success();
  }

  LogicalResult recordDependency(int64_t sourceTileId, int64_t targetTileId,
                                 const LogicalTileDependency &dependency) {
    if (sourceTileId == targetTileId) {
      std::set<DependencyKey> &keys = internalDependencyKeys[sourceTileId];
      if (keys.insert(getDependencyKey(dependency)).second)
        workingTiles[sourceTileId].internalDependencies.push_back(dependency);
      return success();
    }
    if (dependency.byteSize < 0) {
      anchor->emitError("cross-logical-tile dependency requires a static byte "
                        "size");
      return failure();
    }
    TilePair pair{sourceTileId, targetTileId};
    std::set<DependencyKey> &keys = externalDependencyKeys[pair];
    if (!keys.insert(getDependencyKey(dependency)).second)
      return success();
    LogicalTileEdge &edge = externalEdges[pair];
    edge.sourceTileId = sourceTileId;
    edge.targetTileId = targetTileId;
    FailureOr<int64_t> bytes =
        checkedAddBytes(edge.byteSize, dependency.byteSize, anchor);
    if (failed(bytes))
      return failure();
    edge.byteSize = *bytes;
    edge.dependencies.push_back(dependency);
    return success();
  }

  LogicalResult addDependencies() {
    std::set<std::pair<int64_t, int64_t>> wildcardRefinedEdges;
    std::set<std::tuple<int64_t, int64_t, int64_t>> refinedTensorEdges;
    for (const MappingWorkUnitEdge &edge : tree.workUnitEdges) {
      if (edge.tensorId < 0)
        wildcardRefinedEdges.insert(
            {edge.sourceOperationId, edge.targetOperationId});
      else
        refinedTensorEdges.insert(
            {edge.sourceOperationId, edge.targetOperationId, edge.tensorId});
      FailureOr<SmallVector<int64_t>> sourceTiles =
          getTilesForEndpoint(edge.sourceOperationId, edge.sourceWorkUnitId);
      FailureOr<SmallVector<int64_t>> targetTiles =
          getTilesForEndpoint(edge.targetOperationId, edge.targetWorkUnitId);
      if (failed(sourceTiles) || failed(targetTiles))
        return failure();
      LogicalTileDependency dependency{
          edge.sourceOperationId, edge.sourceWorkUnitId,
          edge.targetOperationId, edge.targetWorkUnitId,
          edge.tensorId,          edge.targetOperandNumber,
          edge.byteSize};
      for (int64_t source : *sourceTiles)
        for (int64_t target : *targetTiles)
          if (failed(recordDependency(source, target, dependency)))
            return failure();
    }

    for (const ComputeTensor &tensor : graph.tensors) {
      for (int64_t producer : tensor.producerOperations) {
        for (int64_t consumer : tensor.consumerOperations) {
          if (producer == consumer ||
              wildcardRefinedEdges.contains({producer, consumer}) ||
              refinedTensorEdges.contains({producer, consumer, tensor.id}))
            continue;
          FailureOr<SmallVector<int64_t>> sourceTiles =
              getTilesForEndpoint(producer, /*workUnitId=*/-1);
          FailureOr<SmallVector<int64_t>> targetTiles =
              getTilesForEndpoint(consumer, /*workUnitId=*/-1);
          if (failed(sourceTiles) || failed(targetTiles))
            return failure();
          LogicalTileDependency dependency{producer,
                                           /*sourceWorkUnitId=*/-1,
                                           consumer,
                                           /*targetWorkUnitId=*/-1,
                                           tensor.id,
                                           /*targetOperandNumber=*/-1,
                                           tensor.byteSize};
          for (int64_t source : *sourceTiles)
            for (int64_t target : *targetTiles)
              if (failed(recordDependency(source, target, dependency)))
                return failure();
        }
      }
    }
    return success();
  }

  const ComputeGraph &graph;
  const ResourceAllocationTree &tree;
  const MappingRealization &realization;
  Operation *anchor;
  LogicalTileGraph result;
  SmallVector<LogicalTile, 0> workingTiles;
  DenseMap<int64_t, const StructuralRATreeNode *> nodes;
  std::map<int64_t, std::set<int64_t>> operationTiles;
  std::map<Endpoint, std::set<int64_t>> endpointTiles;
  std::map<int64_t, std::set<DependencyKey>> internalDependencyKeys;
  std::map<TilePair, std::set<DependencyKey>> externalDependencyKeys;
  std::map<TilePair, LogicalTileEdge> externalEdges;
};

LogicalResult verifyDependency(const LogicalTileDependency &dependency,
                               const ComputeGraph &graph,
                               const ResourceAllocationTree &tree,
                               Operation *anchor) {
  auto validOperation = [&](int64_t id) {
    return id >= 0 && id < static_cast<int64_t>(graph.operations.size());
  };
  if (!validOperation(dependency.sourceOperationId) ||
      !validOperation(dependency.targetOperationId)) {
    anchor->emitError("logical-tile dependency references an unknown compute "
                      "operation");
    return failure();
  }
  if (dependency.tensorId >= 0) {
    if (dependency.tensorId >= static_cast<int64_t>(graph.tensors.size())) {
      anchor->emitError("logical-tile dependency references an unknown tensor");
      return failure();
    }
    const ComputeTensor &tensor = graph.tensors[dependency.tensorId];
    if (!llvm::is_contained(tensor.producerOperations,
                            dependency.sourceOperationId) ||
        !llvm::is_contained(tensor.consumerOperations,
                            dependency.targetOperationId) ||
        (dependency.targetOperandNumber >= 0 &&
         (dependency.targetOperandNumber >=
              static_cast<int64_t>(
                  graph.operations[dependency.targetOperationId]
                      .operation->getNumOperands()) ||
          graph.operations[dependency.targetOperationId].operation->getOperand(
              dependency.targetOperandNumber) != tensor.value)) ||
        ((dependency.sourceWorkUnitId < 0 && dependency.targetWorkUnitId < 0)
             ? tensor.byteSize != dependency.byteSize
             : dependency.byteSize > tensor.byteSize)) {
      anchor->emitError("logical-tile dependency does not match its compute "
                        "tensor");
      return failure();
    }
    if (dependency.sourceWorkUnitId >= 0 || dependency.targetWorkUnitId >= 0) {
      bool matches = llvm::any_of(tree.workUnitEdges, [&](const auto &edge) {
        return edge.sourceOperationId == dependency.sourceOperationId &&
               edge.sourceWorkUnitId == dependency.sourceWorkUnitId &&
               edge.targetOperationId == dependency.targetOperationId &&
               edge.targetWorkUnitId == dependency.targetWorkUnitId &&
               edge.tensorId == dependency.tensorId &&
               edge.targetOperandNumber == dependency.targetOperandNumber &&
               edge.byteSize == dependency.byteSize;
      });
      if (!matches) {
        anchor->emitError(
            "logical-tile shard dependency has no exact RA-tree edge");
        return failure();
      }
    }
    return success();
  }
  bool matches = llvm::any_of(tree.workUnitEdges, [&](const auto &edge) {
    return edge.sourceOperationId == dependency.sourceOperationId &&
           edge.sourceWorkUnitId == dependency.sourceWorkUnitId &&
           edge.targetOperationId == dependency.targetOperationId &&
           edge.targetWorkUnitId == dependency.targetWorkUnitId &&
           edge.tensorId == dependency.tensorId &&
           edge.targetOperandNumber == dependency.targetOperandNumber &&
           edge.byteSize == dependency.byteSize;
  });
  if (!matches) {
    anchor->emitError("logical-tile dependency does not match an RA work-unit "
                      "edge");
    return failure();
  }
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<LogicalTileGraph>
buildLogicalTileGraph(const ComputeGraph &graph,
                      const ResourceAllocationTree &tree,
                      const MappingRealization &realization,
                      const MappingHardwareModel &hardware, Operation *anchor) {
  LogicalTileGraphBuilder builder(graph, tree, realization, anchor);
  FailureOr<LogicalTileGraph> tileGraph = builder.build();
  FailureOr<int64_t> coreCount = hardware.getCoreCount(anchor);
  if (failed(tileGraph) || failed(coreCount))
    return failure();
  tileGraph->version = 3;
  tileGraph->plannedMeshRows = hardware.meshRows;
  tileGraph->plannedMeshCols = hardware.meshCols;
  if (tileGraph->logicalTileCapacity != *coreCount ||
      tileGraph->analogLanesPerTile != hardware.arraysPerCore) {
    anchor->emitError(
        "logical-tile realization does not match its planning hardware");
    return failure();
  }
  if (failed(verifyLogicalTileGraph(*tileGraph, graph, tree, anchor)))
    return failure();
  return tileGraph;
}

LogicalResult verifyLogicalTileGraph(const LogicalTileGraph &tileGraph,
                                     const ComputeGraph &graph,
                                     const ResourceAllocationTree &tree,
                                     Operation *anchor) {
  auto plannedCapacity =
      llvm::checkedMul(tileGraph.plannedMeshRows, tileGraph.plannedMeshCols);
  if (tileGraph.version != 3 || tileGraph.plannedMeshRows <= 0 ||
      tileGraph.plannedMeshCols <= 0 || !plannedCapacity ||
      *plannedCapacity != tileGraph.logicalTileCapacity ||
      tileGraph.logicalTileCapacity <= 0 || tileGraph.analogLanesPerTile <= 0 ||
      tileGraph.tiles.size() >
          static_cast<size_t>(tileGraph.logicalTileCapacity)) {
    anchor->emitError("invalid logical-tile graph shape or version");
    return failure();
  }

  DenseMap<int64_t, const StructuralRATreeNode *> nodes;
  int64_t treeLeafCount = 0;
  for (const StructuralRATreeNode &node : tree.nodes) {
    nodes[node.id] = &node;
    if (node.kind == RATreeNodeKind::Leaf)
      ++treeLeafCount;
  }

  std::set<int64_t> tileIds;
  std::set<int64_t> leafIds;
  for (const LogicalTile &tile : tileGraph.tiles) {
    if (tile.id < 0 || tile.id >= tileGraph.logicalTileCapacity ||
        !tileIds.insert(tile.id).second || tile.digitalWork < 0 ||
        tile.analogLanes.size() !=
            static_cast<size_t>(tileGraph.analogLanesPerTile)) {
      anchor->emitError("invalid or duplicate logical tile");
      return failure();
    }
    auto verifyAssignment = [&](const LogicalTileAssignment &assignment,
                                LogicalLaneKind expectedLane,
                                int64_t expectedLaneIndex) -> LogicalResult {
      const StructuralRATreeNode *leaf = nodes.lookup(assignment.leafId);
      if (!leaf || leaf->kind != RATreeNodeKind::Leaf ||
          leaf->operationId != assignment.operationId ||
          leaf->workUnitId != assignment.workUnitId ||
          !leafIds.insert(assignment.leafId).second ||
          assignment.operationId < 0 ||
          assignment.operationId >=
              static_cast<int64_t>(graph.operations.size()) ||
          assignment.laneKind != expectedLane ||
          assignment.laneIndex != expectedLaneIndex ||
          assignment.finishNs < assignment.startNs) {
        anchor->emitError("invalid logical-tile leaf assignment");
        return failure();
      }
      FailureOr<SmallVector<int64_t>> expectedPath =
          buildRANodePath(assignment.leafId, tree.rootId, nodes, anchor);
      if (failed(expectedPath) || assignment.raNodePath != *expectedPath) {
        anchor->emitError("logical-tile assignment has a stale RA-node path");
        return failure();
      }
      const ComputeOperation &operation =
          graph.operations[assignment.operationId];
      if (operation.requiredLane && *operation.requiredLane != expectedLane) {
        operation.operation->emitError(
            "logical-tile assignment violates the operation lane requirement");
        return failure();
      }
      return success();
    };

    for (const LogicalTileAssignment &assignment : tile.digitalAssignments)
      if (failed(verifyAssignment(assignment, LogicalLaneKind::Digital, 0)))
        return failure();
    for (auto indexedLane : llvm::enumerate(tile.analogLanes)) {
      const LogicalTileAnalogLane &lane = indexedLane.value();
      if (lane.laneIndex != static_cast<int64_t>(indexedLane.index())) {
        anchor->emitError("logical analog lanes must be ordered by lane index");
        return failure();
      }
      for (const LogicalTileAssignment &assignment : lane.assignments) {
        if (failed(verifyAssignment(assignment, LogicalLaneKind::Analog,
                                    lane.laneIndex)))
          return failure();
        const ComputeOperation &operation =
            graph.operations[assignment.operationId];
        if (operation.laneBindingGroup != lane.laneBindingGroup) {
          operation.operation->emitError(
              "logical analog lane does not preserve its binding group");
          return failure();
        }
      }
    }
    for (int64_t tensorId : tile.modelInputTensorIds) {
      if (tensorId < 0 ||
          tensorId >= static_cast<int64_t>(graph.tensors.size()) ||
          !graph.tensors[tensorId].isFunctionInput) {
        anchor->emitError("logical tile references an invalid model input");
        return failure();
      }
    }
    for (int64_t tensorId : tile.modelOutputTensorIds) {
      if (tensorId < 0 ||
          tensorId >= static_cast<int64_t>(graph.tensors.size()) ||
          !graph.tensors[tensorId].isFunctionOutput) {
        anchor->emitError("logical tile references an invalid model output");
        return failure();
      }
    }
    for (const LogicalTileDependency &dependency : tile.internalDependencies)
      if (failed(verifyDependency(dependency, graph, tree, anchor)))
        return failure();
  }
  if (static_cast<int64_t>(leafIds.size()) != treeLeafCount) {
    anchor->emitError("logical-tile graph does not contain every RA leaf");
    return failure();
  }

  for (auto indexedEdge : llvm::enumerate(tileGraph.edges)) {
    const LogicalTileEdge &edge = indexedEdge.value();
    if (edge.id != static_cast<int64_t>(indexedEdge.index()) ||
        edge.sourceTileId == edge.targetTileId ||
        !tileIds.contains(edge.sourceTileId) ||
        !tileIds.contains(edge.targetTileId) || edge.byteSize < 0) {
      anchor->emitError("invalid logical-tile communication edge");
      return failure();
    }
    int64_t totalBytes = 0;
    for (const LogicalTileDependency &dependency : edge.dependencies) {
      if (dependency.byteSize < 0 ||
          failed(verifyDependency(dependency, graph, tree, anchor)))
        return failure();
      FailureOr<int64_t> updated =
          checkedAddBytes(totalBytes, dependency.byteSize, anchor);
      if (failed(updated))
        return failure();
      totalBytes = *updated;
    }
    if (totalBytes != edge.byteSize) {
      anchor->emitError("logical-tile edge byte total does not match its "
                        "dependencies");
      return failure();
    }
  }
  return success();
}

LogicalTileGraphAttr serializeLogicalTileGraph(MLIRContext *context,
                                               const LogicalTileGraph &graph) {
  Builder builder(context);
  SmallVector<Attribute> tiles;
  tiles.reserve(graph.tiles.size());
  for (const LogicalTile &tile : graph.tiles) {
    SmallVector<Attribute> digitalAssignments;
    for (const LogicalTileAssignment &assignment : tile.digitalAssignments)
      digitalAssignments.push_back(serializeAssignment(builder, assignment));

    SmallVector<Attribute> analogLanes;
    for (const LogicalTileAnalogLane &lane : tile.analogLanes) {
      SmallVector<Attribute> assignments;
      for (const LogicalTileAssignment &assignment : lane.assignments)
        assignments.push_back(serializeAssignment(builder, assignment));
      analogLanes.push_back(LogicalTileAnalogLaneAttr::get(
          context, builder.getI64IntegerAttr(lane.laneIndex),
          builder.getI64IntegerAttr(lane.laneBindingGroup.value_or(-1)),
          builder.getArrayAttr(assignments)));
    }

    SmallVector<Attribute> internalDependencies;
    for (const LogicalTileDependency &dependency : tile.internalDependencies)
      internalDependencies.push_back(serializeDependency(builder, dependency));
    tiles.push_back(
        LogicalTileAttr::get(context, builder.getI64IntegerAttr(tile.id),
                             builder.getI64IntegerAttr(tile.digitalWork),
                             builder.getArrayAttr(digitalAssignments),
                             builder.getArrayAttr(analogLanes),
                             getI64Array(builder, tile.modelInputTensorIds),
                             getI64Array(builder, tile.modelOutputTensorIds),
                             builder.getArrayAttr(internalDependencies)));
  }

  SmallVector<Attribute> edges;
  edges.reserve(graph.edges.size());
  for (const LogicalTileEdge &edge : graph.edges) {
    SmallVector<Attribute> dependencies;
    for (const LogicalTileDependency &dependency : edge.dependencies)
      dependencies.push_back(serializeDependency(builder, dependency));
    edges.push_back(
        LogicalTileEdgeAttr::get(context, builder.getI64IntegerAttr(edge.id),
                                 builder.getI64IntegerAttr(edge.sourceTileId),
                                 builder.getI64IntegerAttr(edge.targetTileId),
                                 builder.getI64IntegerAttr(edge.byteSize),
                                 builder.getArrayAttr(dependencies)));
  }

  return LogicalTileGraphAttr::get(
      context, builder.getI64IntegerAttr(graph.version),
      builder.getI64IntegerAttr(graph.plannedMeshRows),
      builder.getI64IntegerAttr(graph.plannedMeshCols),
      builder.getI64IntegerAttr(graph.logicalTileCapacity),
      builder.getI64IntegerAttr(graph.analogLanesPerTile),
      builder.getArrayAttr(tiles), builder.getArrayAttr(edges));
}

FailureOr<LogicalTileGraph> deserializeLogicalTileGraph(
    LogicalTileGraphAttr attr, const ComputeGraph &graph,
    const ResourceAllocationTree &tree, Operation *anchor) {
  LogicalTileGraph tileGraph;
  tileGraph.version = attr.getVersion().getInt();
  tileGraph.plannedMeshRows = attr.getPlannedMeshRows().getInt();
  tileGraph.plannedMeshCols = attr.getPlannedMeshCols().getInt();
  tileGraph.logicalTileCapacity = attr.getLogicalTileCapacity().getInt();
  tileGraph.analogLanesPerTile = attr.getAnalogLanesPerTile().getInt();

  for (Attribute tileValue : attr.getTiles()) {
    auto tileAttr = dyn_cast<LogicalTileAttr>(tileValue);
    if (!tileAttr) {
      anchor->emitError("logical-tile graph tiles must contain typed logical "
                        "tile attributes");
      return failure();
    }
    LogicalTile tile;
    tile.id = tileAttr.getTileId().getInt();
    tile.digitalWork = tileAttr.getDigitalWork().getInt();
    FailureOr<SmallVector<int64_t>> inputs = parseI64Array(
        tileAttr.getModelInputTensorIds(), anchor, "logical-tile model inputs");
    FailureOr<SmallVector<int64_t>> outputs =
        parseI64Array(tileAttr.getModelOutputTensorIds(), anchor,
                      "logical-tile model outputs");
    if (failed(inputs) || failed(outputs))
      return failure();
    tile.modelInputTensorIds = std::move(*inputs);
    tile.modelOutputTensorIds = std::move(*outputs);

    for (Attribute assignmentValue : tileAttr.getDigitalAssignments()) {
      auto assignmentAttr =
          dyn_cast<LogicalTileAssignmentAttr>(assignmentValue);
      if (!assignmentAttr) {
        anchor->emitError("logical-tile digital assignments must be typed");
        return failure();
      }
      FailureOr<LogicalTileAssignment> assignment =
          deserializeAssignment(assignmentAttr, anchor);
      if (failed(assignment))
        return failure();
      tile.digitalAssignments.push_back(std::move(*assignment));
    }
    for (Attribute laneValue : tileAttr.getAnalogLanes()) {
      auto laneAttr = dyn_cast<LogicalTileAnalogLaneAttr>(laneValue);
      if (!laneAttr) {
        anchor->emitError("logical-tile analog lanes must be typed");
        return failure();
      }
      LogicalTileAnalogLane lane;
      lane.laneIndex = laneAttr.getLaneIndex().getInt();
      int64_t bindingGroup = laneAttr.getLaneBindingGroup().getInt();
      if (bindingGroup >= 0)
        lane.laneBindingGroup = bindingGroup;
      for (Attribute assignmentValue : laneAttr.getAssignments()) {
        auto assignmentAttr =
            dyn_cast<LogicalTileAssignmentAttr>(assignmentValue);
        if (!assignmentAttr) {
          anchor->emitError("logical analog lane assignments must be typed");
          return failure();
        }
        FailureOr<LogicalTileAssignment> assignment =
            deserializeAssignment(assignmentAttr, anchor);
        if (failed(assignment))
          return failure();
        lane.assignments.push_back(std::move(*assignment));
      }
      tile.analogLanes.push_back(std::move(lane));
    }
    for (Attribute dependencyValue : tileAttr.getInternalDependencies()) {
      auto dependencyAttr =
          dyn_cast<LogicalTileDependencyAttr>(dependencyValue);
      if (!dependencyAttr) {
        anchor->emitError("logical-tile internal dependencies must be typed");
        return failure();
      }
      tile.internalDependencies.push_back(
          deserializeDependency(dependencyAttr));
    }
    if (tileGraph.tileIndexById.contains(tile.id)) {
      anchor->emitError("logical-tile graph contains a duplicate tile ID");
      return failure();
    }
    tileGraph.tileIndexById[tile.id] = tileGraph.tiles.size();
    tileGraph.tiles.push_back(std::move(tile));
  }

  for (Attribute edgeValue : attr.getEdges()) {
    auto edgeAttr = dyn_cast<LogicalTileEdgeAttr>(edgeValue);
    if (!edgeAttr) {
      anchor->emitError("logical-tile graph edges must be typed");
      return failure();
    }
    LogicalTileEdge edge;
    edge.id = edgeAttr.getEdgeId().getInt();
    edge.sourceTileId = edgeAttr.getSourceTileId().getInt();
    edge.targetTileId = edgeAttr.getTargetTileId().getInt();
    edge.byteSize = edgeAttr.getByteSize().getInt();
    for (Attribute dependencyValue : edgeAttr.getDependencies()) {
      auto dependencyAttr =
          dyn_cast<LogicalTileDependencyAttr>(dependencyValue);
      if (!dependencyAttr) {
        anchor->emitError("logical-tile edge dependencies must be typed");
        return failure();
      }
      edge.dependencies.push_back(deserializeDependency(dependencyAttr));
    }
    tileGraph.edges.push_back(std::move(edge));
  }

  if (failed(verifyLogicalTileGraph(tileGraph, graph, tree, anchor)))
    return failure();
  return tileGraph;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
