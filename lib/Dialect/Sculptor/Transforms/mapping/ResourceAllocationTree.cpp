#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

LogicalResult emitTreeError(Operation *anchor, const Twine &message) {
  anchor->emitError("invalid Resource Allocation Tree: ") << message;
  return failure();
}

using LeafEndpoint = std::pair<int64_t, int64_t>;

void collectSubtreeEndpoints(int64_t nodeId, const ResourceAllocationTree &tree,
                             const DenseMap<int64_t, size_t> &nodeIndices,
                             SmallVectorImpl<LeafEndpoint> &endpoints) {
  const StructuralRATreeNode &node = tree.nodes[nodeIndices.lookup(nodeId)];
  if (node.kind == RATreeNodeKind::Leaf) {
    endpoints.push_back({node.operationId, node.workUnitId});
    return;
  }
  for (int64_t childId : node.childIds)
    collectSubtreeEndpoints(childId, tree, nodeIndices, endpoints);
}

LogicalResult verifyTemporalOrders(const ResourceAllocationTree &tree,
                                   const DenseMap<int64_t, size_t> &nodeIndices,
                                   const DenseMap<int64_t,
                                                  SmallVector<int64_t>>
                                       &graphConsumersByProducer,
                                   Operation *anchor) {
  DenseSet<std::pair<int64_t, int64_t>> refinedOperationEdges;
  DenseMap<int64_t, SmallVector<const MappingWorkUnitEdge *>>
      workUnitEdgesBySource;
  for (const MappingWorkUnitEdge &edge : tree.workUnitEdges) {
    refinedOperationEdges.insert(
        {edge.sourceOperationId, edge.targetOperationId});
    workUnitEdgesBySource[edge.sourceOperationId].push_back(&edge);
  }

  for (const StructuralRATreeNode &node : tree.nodes) {
    if (node.kind != RATreeNodeKind::TemporalCut)
      continue;

    DenseMap<LeafEndpoint, int64_t> endpointToChild;
    DenseMap<int64_t, SmallVector<int64_t>> operationToChildren;
    for (auto [childOrder, childId] : llvm::enumerate(node.childIds)) {
      int64_t order = static_cast<int64_t>(childOrder);
      SmallVector<LeafEndpoint> descendants;
      collectSubtreeEndpoints(childId, tree, nodeIndices, descendants);
      for (LeafEndpoint endpoint : descendants) {
        endpointToChild[endpoint] = order;
        SmallVector<int64_t> &children = operationToChildren[endpoint.first];
        if (children.empty() || children.back() != order)
          children.push_back(order);
      }
    }

    auto verifyForward = [&](int64_t sourceOperationId,
                             int64_t sourceWorkUnitId,
                             int64_t targetOperationId,
                             int64_t targetWorkUnitId) -> LogicalResult {
      SmallVector<int64_t, 1> exactSource;
      SmallVector<int64_t, 1> exactTarget;
      ArrayRef<int64_t> sourceChildren;
      ArrayRef<int64_t> targetChildren;
      if (sourceWorkUnitId < 0) {
        auto source = operationToChildren.find(sourceOperationId);
        if (source == operationToChildren.end())
          return success();
        sourceChildren = source->second;
      } else {
        auto source =
            endpointToChild.find({sourceOperationId, sourceWorkUnitId});
        if (source == endpointToChild.end())
          return success();
        exactSource.push_back(source->second);
        sourceChildren = exactSource;
      }
      if (targetWorkUnitId < 0) {
        auto target = operationToChildren.find(targetOperationId);
        if (target == operationToChildren.end())
          return success();
        targetChildren = target->second;
      } else {
        auto target =
            endpointToChild.find({targetOperationId, targetWorkUnitId});
        if (target == endpointToChild.end())
          return success();
        exactTarget.push_back(target->second);
        targetChildren = exactTarget;
      }
      for (int64_t source : sourceChildren) {
        for (int64_t target : targetChildren) {
          if (source <= target)
            continue;
          return emitTreeError(anchor, Twine("T-Cut node ") + Twine(node.id) +
                                           " orders mapping dependency " +
                                           Twine(sourceOperationId) + ":" +
                                           Twine(sourceWorkUnitId) + " -> " +
                                           Twine(targetOperationId) + ":" +
                                           Twine(targetWorkUnitId) +
                                           " backwards");
        }
      }
      return success();
    };

    // Only dependencies whose source occurs below this cut can constrain its
    // child order.  Indexing by source avoids rescanning the complete graph for
    // every temporal cut, which is quadratic for models with many MVM waves.
    for (const auto &[producerId, children] : operationToChildren) {
      (void)children;
      auto refined = workUnitEdgesBySource.find(producerId);
      if (refined != workUnitEdgesBySource.end()) {
        for (const MappingWorkUnitEdge *edge : refined->second) {
          if (failed(verifyForward(
                  edge->sourceOperationId, edge->sourceWorkUnitId,
                  edge->targetOperationId, edge->targetWorkUnitId)))
            return failure();
        }
      }
      auto consumers = graphConsumersByProducer.find(producerId);
      if (consumers == graphConsumersByProducer.end())
        continue;
      for (int64_t consumerId : consumers->second) {
        if (refinedOperationEdges.contains({producerId, consumerId}))
          continue;
        if (failed(verifyForward(producerId, /*sourceWorkUnitId=*/-1,
                                 consumerId, /*targetWorkUnitId=*/-1)))
          return failure();
      }
    }
  }
  return success();
}

LogicalResult visitTreeNode(int64_t nodeId, const ResourceAllocationTree &tree,
                            const DenseMap<int64_t, size_t> &nodeIndices,
                            DenseMap<int64_t, unsigned> &visitState,
                            Operation *anchor) {
  unsigned &state = visitState[nodeId];
  if (state == 1)
    return emitTreeError(anchor, Twine("cycle at node ") + Twine(nodeId));
  if (state == 2)
    return emitTreeError(anchor, Twine("node ") + Twine(nodeId) +
                                     " is reachable more than once");

  state = 1;
  const StructuralRATreeNode &node = tree.nodes[nodeIndices.lookup(nodeId)];
  for (int64_t childId : node.childIds) {
    if (failed(visitTreeNode(childId, tree, nodeIndices, visitState, anchor)))
      return failure();
  }
  state = 2;
  return success();
}

std::string digestToHex(const std::array<uint8_t, 32> &digest) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (uint8_t byte : digest) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

void printOperationStructure(Operation *operation, raw_ostream &stream,
                             unsigned depth = 0) {
  stream.indent(depth * 2) << operation->getName().getStringRef();
  for (Type type : operation->getOperandTypes())
    stream << " in:" << type;
  for (Type type : operation->getResultTypes())
    stream << " out:" << type;
  for (NamedAttribute attribute : operation->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name.starts_with("sculptor.semantic.") ||
        name.starts_with("sculptor.mapping.") ||
        name == mapping::kRATreeAttrName)
      continue;
    stream << " attr:" << name << '=' << attribute.getValue();
  }
  stream << '\n';
  for (Region &region : operation->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block)
        printOperationStructure(&nested, stream, depth + 1);
    }
  }
}

FailureOr<SmallVector<int64_t>>
parseI64Array(ArrayAttr values, Operation *anchor, StringRef description) {
  SmallVector<int64_t> result;
  result.reserve(values.size());
  for (Attribute value : values) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer) {
      anchor->emitError(description) << " must contain integer attributes";
      return failure();
    }
    result.push_back(integer.getInt());
  }
  return result;
}

FailureOr<int64_t> checkedVolume(ArrayRef<int64_t> sizes, Operation *anchor,
                                 StringRef description) {
  int64_t volume = 1;
  for (int64_t size : sizes) {
    if (size <= 0) {
      anchor->emitError(description) << " must contain positive sizes";
      return failure();
    }
    std::optional<int64_t> next = llvm::checkedMul(volume, size);
    if (!next) {
      anchor->emitError(description) << " volume overflows int64";
      return failure();
    }
    volume = *next;
  }
  return volume;
}

bool staticTilesOverlap(const MappingWorkUnit &lhs,
                        const MappingWorkUnit &rhs) {
  assert(lhs.resultOffsets.size() == rhs.resultOffsets.size());
  for (size_t dimension = 0; dimension < lhs.resultOffsets.size();
       ++dimension) {
    int64_t lhsEnd = lhs.resultOffsets[dimension] + lhs.resultSizes[dimension];
    int64_t rhsEnd = rhs.resultOffsets[dimension] + rhs.resultSizes[dimension];
    if (lhsEnd <= rhs.resultOffsets[dimension] ||
        rhsEnd <= lhs.resultOffsets[dimension])
      return false;
  }
  return true;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<ResourceAllocationTree>
buildTemporalBaselineRATree(const ComputeGraph &graph, Operation *anchor) {
  if (graph.operations.empty()) {
    anchor->emitError("cannot build an RA Tree without supported compute "
                      "operations");
    return failure();
  }

  ResourceAllocationTree tree;
  auto appendNode = [&](RATreeNodeKind kind, int64_t parentId,
                        int64_t operationId) -> int64_t {
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    tree.nodes.push_back({id, kind, parentId, /*childIds=*/{}, operationId,
                          /*workUnitId=*/-1, /*workGroupCount=*/1});
    if (parentId >= 0)
      tree.nodes[parentId].childIds.push_back(id);
    return id;
  };

  if (graph.operations.size() == 1) {
    tree.rootId = appendNode(RATreeNodeKind::Leaf, /*parentId=*/-1,
                             graph.topologicalOrder.front());
    return tree;
  }

  auto appendCut = [&](RATreeNodeKind kind, int64_t parentId) -> int64_t {
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    StructuralRATreeNode node;
    node.id = id;
    node.kind = kind;
    node.parentId = parentId;
    tree.nodes.push_back(std::move(node));
    if (parentId >= 0)
      tree.nodes[parentId].childIds.push_back(id);
    return id;
  };

  int64_t modelRoot = -1;
  if (graph.layerRegions.size() > 1)
    modelRoot = appendCut(RATreeNodeKind::TemporalCut, /*parentId=*/-1);

  for (int64_t regionId : graph.topologicalLayerRegionOrder) {
    const LayerRegion &region = graph.layerRegions[regionId];
    int64_t regionRoot = modelRoot;
    if (region.operationIds.size() > 1)
      regionRoot = appendCut(RATreeNodeKind::TemporalCut, modelRoot);
    for (int64_t operationId : region.operationIds)
      appendNode(RATreeNodeKind::Leaf, regionRoot, operationId);
    if (modelRoot < 0)
      modelRoot = region.operationIds.size() == 1
                      ? tree.nodes.back().id
                      : regionRoot;
  }
  tree.rootId = modelRoot;
  return tree;
}

FailureOr<ResourceAllocationTree>
deserializeResourceAllocationTree(RATreeAttr attr, const ComputeGraph &graph,
                                  Operation *anchor) {
  if (attr.getVersion().getInt() != 4) {
    anchor->emitError("unsupported Resource Allocation Tree version ")
        << attr.getVersion().getInt() << "; expected version 4";
    return failure();
  }
  if (attr.getOperationCount().getInt() !=
      static_cast<int64_t>(graph.operations.size())) {
    anchor->emitError("Resource Allocation Tree operation count does not match "
                      "the rebuilt compute graph");
    return failure();
  }
  if (attr.getTensorCount().getInt() !=
      static_cast<int64_t>(graph.tensors.size())) {
    anchor->emitError(
        "Resource Allocation Tree tensor count does not match the "
        "rebuilt compute graph");
    return failure();
  }

  std::string fingerprint = computeGraphFingerprint(graph);
  if (attr.getGraphFingerprint().getValue() != fingerprint) {
    anchor->emitError("Resource Allocation Tree graph fingerprint does not "
                      "match the current compute graph");
    return failure();
  }

  ResourceAllocationTree tree;
  tree.rootId = attr.getRootId().getInt();
  tree.workUnits.reserve(attr.getWorkUnits().size());
  for (Attribute workUnitAttribute : attr.getWorkUnits()) {
    auto workUnitAttr = dyn_cast<MappingWorkUnitAttr>(workUnitAttribute);
    if (!workUnitAttr) {
      anchor->emitError("Resource Allocation Tree work units must use "
                        "#sculptor.mapping_work_unit");
      return failure();
    }
    MappingWorkUnit workUnit;
    workUnit.id = workUnitAttr.getId().getInt();
    workUnit.operationId = workUnitAttr.getOperationId().getInt();
    workUnit.resultNumber = workUnitAttr.getResultNumber().getInt();
    FailureOr<SmallVector<int64_t>> resultOffsets = parseI64Array(
        workUnitAttr.getResultOffsets(), anchor, "work-unit result offsets");
    FailureOr<SmallVector<int64_t>> resultSizes = parseI64Array(
        workUnitAttr.getResultSizes(), anchor, "work-unit result sizes");
    FailureOr<SmallVector<int64_t>> iterationOffsets =
        parseI64Array(workUnitAttr.getIterationOffsets(), anchor,
                      "work-unit iteration offsets");
    FailureOr<SmallVector<int64_t>> iterationSizes = parseI64Array(
        workUnitAttr.getIterationSizes(), anchor, "work-unit iteration sizes");
    if (failed(resultOffsets) || failed(resultSizes) ||
        failed(iterationOffsets) || failed(iterationSizes))
      return failure();
    workUnit.resultOffsets = std::move(*resultOffsets);
    workUnit.resultSizes = std::move(*resultSizes);
    workUnit.iterationOffsets = std::move(*iterationOffsets);
    workUnit.iterationSizes = std::move(*iterationSizes);
    workUnit.shardGroupId = workUnitAttr.getShardGroupId().getInt();
    workUnit.shardIndex = workUnitAttr.getShardIndex().getInt();
    workUnit.shardCount = workUnitAttr.getShardCount().getInt();
    tree.workUnits.push_back(std::move(workUnit));
  }
  tree.workUnitEdges.reserve(attr.getWorkUnitEdges().size());
  for (Attribute edgeAttribute : attr.getWorkUnitEdges()) {
    auto edgeAttr = dyn_cast<MappingWorkUnitEdgeAttr>(edgeAttribute);
    if (!edgeAttr) {
      anchor->emitError("Resource Allocation Tree work-unit edges must use "
                        "#sculptor.mapping_work_unit_edge");
      return failure();
    }
    tree.workUnitEdges.push_back({edgeAttr.getSourceOperationId().getInt(),
                                  edgeAttr.getSourceWorkUnitId().getInt(),
                                  edgeAttr.getTargetOperationId().getInt(),
                                  edgeAttr.getTargetWorkUnitId().getInt(),
                                  edgeAttr.getTensorId().getInt(),
                                  edgeAttr.getSourceResultNumber().getInt(),
                                  edgeAttr.getTargetOperandNumber().getInt(),
                                  edgeAttr.getByteSize().getInt()});
  }
  tree.nodes.reserve(attr.getNodes().size());
  for (Attribute nodeAttribute : attr.getNodes()) {
    auto nodeAttr = dyn_cast<RATreeNodeAttr>(nodeAttribute);
    if (!nodeAttr) {
      anchor->emitError(
          "Resource Allocation Tree nodes must use #sculptor.ra_tree_node");
      return failure();
    }

    StructuralRATreeNode node;
    node.id = nodeAttr.getId().getInt();
    node.kind = nodeAttr.getKind();
    node.parentId = nodeAttr.getParentId().getInt();
    node.operationId = nodeAttr.getOperationId().getInt();
    node.workUnitId = nodeAttr.getWorkUnitId().getInt();
    node.workGroupCount = nodeAttr.getWorkGroupCount().getInt();
    node.childIds.reserve(nodeAttr.getChildIds().size());
    for (Attribute childAttribute : nodeAttr.getChildIds()) {
      auto childId = dyn_cast<IntegerAttr>(childAttribute);
      if (!childId) {
        anchor->emitError(
            "Resource Allocation Tree child IDs must be integer attributes");
        return failure();
      }
      node.childIds.push_back(childId.getInt());
    }
    tree.nodes.push_back(std::move(node));
  }

  if (failed(verifyResourceAllocationTree(tree, graph, anchor)))
    return failure();
  return tree;
}

ResourceAllocationTree
cloneResourceAllocationTree(const ResourceAllocationTree &tree) {
  return tree;
}

FailureOr<ResourceAllocationTree>
stripLayerRegionNodes(const ResourceAllocationTree &tree, Operation *anchor) {
  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  for (const StructuralRATreeNode &node : tree.nodes) {
    if (!nodesById.try_emplace(node.id, &node).second) {
      anchor->emitError("cannot strip Layer nodes from a Resource Allocation "
                        "Tree with duplicate node ID ")
          << node.id;
      return failure();
    }
  }

  ResourceAllocationTree result;
  result.workUnits = tree.workUnits;
  result.workUnitEdges = tree.workUnitEdges;
  std::function<FailureOr<int64_t>(int64_t)> rebuild =
      [&](int64_t oldId) -> FailureOr<int64_t> {
    const StructuralRATreeNode *oldNode = nodesById.lookup(oldId);
    if (!oldNode) {
      anchor->emitError("cannot strip Layer nodes with unknown child node ")
          << oldId;
      return failure();
    }
    if (oldNode->kind == RATreeNodeKind::Layer) {
      if (oldNode->childIds.size() != 1) {
        anchor->emitError("cannot strip malformed Layer RA node ")
            << oldNode->id;
        return failure();
      }
      return rebuild(oldNode->childIds.front());
    }

    StructuralRATreeNode node = *oldNode;
    node.id = static_cast<int64_t>(result.nodes.size());
    node.parentId = -1;
    node.childIds.clear();
    const int64_t newId = node.id;
    result.nodes.push_back(std::move(node));
    for (int64_t childId : oldNode->childIds) {
      FailureOr<int64_t> rebuiltChild = rebuild(childId);
      if (failed(rebuiltChild))
        return failure();
      result.nodes[newId].childIds.push_back(*rebuiltChild);
      result.nodes[*rebuiltChild].parentId = newId;
    }
    return newId;
  };

  FailureOr<int64_t> root = rebuild(tree.rootId);
  if (failed(root))
    return failure();
  result.rootId = *root;
  return reindexResourceAllocationTree(result, anchor);
}

FailureOr<ResourceAllocationTree>
materializeLayerRegionNodes(const ResourceAllocationTree &tree,
                            const ComputeGraph &graph, Operation *anchor) {
  FailureOr<ResourceAllocationTree> stripped =
      stripLayerRegionNodes(tree, anchor);
  if (failed(stripped))
    return failure();

  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  for (const StructuralRATreeNode &node : stripped->nodes)
    nodesById[node.id] = &node;

  DenseMap<int64_t, std::optional<int64_t>> uniformRegions;
  std::function<std::optional<int64_t>(int64_t)> inferUniformRegion =
      [&](int64_t nodeId) -> std::optional<int64_t> {
    auto cached = uniformRegions.find(nodeId);
    if (cached != uniformRegions.end())
      return cached->second;
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    if (!node)
      return std::nullopt;
    if (node->kind == RATreeNodeKind::Leaf) {
      const ComputeOperation &operation = graph.operations[node->operationId];
      std::optional<int64_t> region;
      if (operation.kind != ComputeOperationKind::MatrixSetup &&
          operation.layerRegionId >= 0)
        region = operation.layerRegionId;
      uniformRegions[nodeId] = region;
      return region;
    }

    std::optional<int64_t> region;
    bool homogeneous = true;
    for (int64_t childId : node->childIds) {
      std::optional<int64_t> childRegion = inferUniformRegion(childId);
      if (!childRegion) {
        homogeneous = false;
        continue;
      }
      if (region && *region != *childRegion)
        homogeneous = false;
      else if (!region)
        region = childRegion;
    }
    if (!homogeneous)
      region = std::nullopt;
    uniformRegions[nodeId] = region;
    return region;
  };
  inferUniformRegion(stripped->rootId);

  ResourceAllocationTree result;
  result.workUnits = stripped->workUnits;
  result.workUnitEdges = stripped->workUnitEdges;
  std::function<FailureOr<int64_t>(int64_t, std::optional<int64_t>)> rebuild =
      [&](int64_t oldId,
          std::optional<int64_t> parentRegion) -> FailureOr<int64_t> {
    const StructuralRATreeNode *oldNode = nodesById.lookup(oldId);
    if (!oldNode) {
      anchor->emitError("cannot materialize Layer nodes with unknown RA node ")
          << oldId;
      return failure();
    }
    std::optional<int64_t> region = uniformRegions.lookup(oldId);

    StructuralRATreeNode node = *oldNode;
    node.id = static_cast<int64_t>(result.nodes.size());
    node.parentId = -1;
    node.childIds.clear();
    const int64_t newId = node.id;
    result.nodes.push_back(std::move(node));
    for (int64_t childId : oldNode->childIds) {
      FailureOr<int64_t> rebuiltChild = rebuild(childId, region);
      if (failed(rebuiltChild))
        return failure();
      result.nodes[newId].childIds.push_back(*rebuiltChild);
      result.nodes[*rebuiltChild].parentId = newId;
    }

    if (!region || region == parentRegion)
      return newId;

    StructuralRATreeNode layer;
    layer.id = static_cast<int64_t>(result.nodes.size());
    layer.kind = RATreeNodeKind::Layer;
    layer.childIds.push_back(newId);
    result.nodes[newId].parentId = layer.id;
    result.nodes.push_back(std::move(layer));
    return result.nodes.back().id;
  };

  FailureOr<int64_t> root = rebuild(stripped->rootId, std::nullopt);
  if (failed(root))
    return failure();
  result.rootId = *root;
  FailureOr<ResourceAllocationTree> reindexed =
      reindexResourceAllocationTree(result, anchor);
  if (failed(reindexed) ||
      failed(verifyResourceAllocationTree(*reindexed, graph, anchor)))
    return failure();
  return reindexed;
}

FailureOr<ResourceAllocationTree>
reindexResourceAllocationTree(const ResourceAllocationTree &tree,
                              Operation *anchor) {
  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  for (const StructuralRATreeNode &node : tree.nodes) {
    if (!nodesById.try_emplace(node.id, &node).second) {
      anchor->emitError(
          "cannot reindex Resource Allocation Tree with duplicate "
          "node ID ")
          << node.id;
      return failure();
    }
  }
  if (!nodesById.contains(tree.rootId)) {
    anchor->emitError("cannot reindex Resource Allocation Tree with missing "
                      "root node ")
        << tree.rootId;
    return failure();
  }

  SmallVector<int64_t> preorder;
  llvm::SmallSet<int64_t, 16> visited;
  std::function<LogicalResult(int64_t)> visit = [&](int64_t nodeId) {
    auto node = nodesById.find(nodeId);
    if (node == nodesById.end()) {
      anchor->emitError("cannot reindex Resource Allocation Tree with unknown "
                        "child node ")
          << nodeId;
      return failure();
    }
    if (!visited.insert(nodeId).second) {
      anchor->emitError("cannot reindex cyclic or multiply referenced Resource "
                        "Allocation Tree node ")
          << nodeId;
      return failure();
    }
    preorder.push_back(nodeId);
    for (int64_t childId : node->second->childIds) {
      if (failed(visit(childId)))
        return failure();
    }
    return success();
  };
  if (failed(visit(tree.rootId)))
    return failure();
  if (preorder.size() != tree.nodes.size()) {
    anchor->emitError(
        "cannot reindex Resource Allocation Tree with unreachable nodes");
    return failure();
  }

  DenseMap<int64_t, int64_t> newIds;
  for (auto [newId, oldId] : llvm::enumerate(preorder))
    newIds[oldId] = static_cast<int64_t>(newId);

  ResourceAllocationTree result;
  result.rootId = 0;
  result.workUnits = tree.workUnits;
  result.workUnitEdges = tree.workUnitEdges;
  result.nodes.reserve(tree.nodes.size());
  for (int64_t oldId : preorder) {
    const StructuralRATreeNode &oldNode = *nodesById.lookup(oldId);
    StructuralRATreeNode node = oldNode;
    node.id = newIds.lookup(oldId);
    node.parentId = oldNode.parentId < 0 ? -1 : newIds.lookup(oldNode.parentId);
    node.childIds.clear();
    for (int64_t oldChildId : oldNode.childIds)
      node.childIds.push_back(newIds.lookup(oldChildId));
    result.nodes.push_back(std::move(node));
  }
  return result;
}

LogicalResult verifyResourceAllocationTree(const ResourceAllocationTree &tree,
                                           const ComputeGraph &graph,
                                           Operation *anchor) {
  if (tree.nodes.empty())
    return emitTreeError(anchor, "tree has no nodes");

  DenseMap<int64_t, size_t> nodeIndices;
  int64_t rootCount = 0;
  for (auto [index, node] : llvm::enumerate(tree.nodes)) {
    if (node.id < 0)
      return emitTreeError(anchor, "node ID must be non-negative");
    if (!nodeIndices.try_emplace(node.id, index).second)
      return emitTreeError(anchor,
                           Twine("duplicate node ID ") + Twine(node.id));
    if (node.parentId < 0)
      ++rootCount;
    if (node.workGroupCount <= 0) {
      return emitTreeError(anchor, Twine("node ") + Twine(node.id) +
                                       " has non-positive work-group count");
    }
  }

  if (rootCount != 1)
    return emitTreeError(anchor, "tree must contain exactly one root");
  auto root = nodeIndices.find(tree.rootId);
  if (root == nodeIndices.end())
    return emitTreeError(anchor, "root ID does not resolve to a node");
  if (tree.nodes[root->second].parentId >= 0)
    return emitTreeError(anchor, "root node must not have a parent");

  DenseMap<int64_t, const MappingWorkUnit *> workUnitsById;
  DenseMap<int64_t, SmallVector<const MappingWorkUnit *>> workUnitsByOperation;
  for (const MappingWorkUnit &workUnit : tree.workUnits) {
    if (workUnit.id < 0)
      return emitTreeError(anchor, "work-unit ID must be non-negative");
    if (!workUnitsById.try_emplace(workUnit.id, &workUnit).second)
      return emitTreeError(anchor, Twine("duplicate work-unit ID ") +
                                       Twine(workUnit.id));
    if (workUnit.operationId < 0 ||
        workUnit.operationId >= static_cast<int64_t>(graph.operations.size())) {
      return emitTreeError(anchor, Twine("work unit ") + Twine(workUnit.id) +
                                       " references unknown operation " +
                                       Twine(workUnit.operationId));
    }
    const ComputeOperation &operation = graph.operations[workUnit.operationId];
    if (workUnit.resultNumber < 0 ||
        workUnit.resultNumber >=
            static_cast<int64_t>(operation.operation->getNumResults())) {
      return emitTreeError(anchor, Twine("work unit ") + Twine(workUnit.id) +
                                       " references unknown result " +
                                       Twine(workUnit.resultNumber));
    }
    auto resultType = dyn_cast<RankedTensorType>(
        operation.operation->getResult(workUnit.resultNumber).getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return emitTreeError(
          anchor, Twine("work unit ") + Twine(workUnit.id) +
                      " requires a statically shaped ranked tensor result");
    }
    if (workUnit.resultOffsets.size() !=
            static_cast<size_t>(resultType.getRank()) ||
        workUnit.resultSizes.size() !=
            static_cast<size_t>(resultType.getRank())) {
      return emitTreeError(anchor, Twine("work unit ") + Twine(workUnit.id) +
                                       " result tile rank mismatch");
    }
    for (int64_t dimension = 0; dimension < resultType.getRank(); ++dimension) {
      int64_t offset = workUnit.resultOffsets[dimension];
      int64_t size = workUnit.resultSizes[dimension];
      if (offset < 0 || size <= 0 ||
          offset > resultType.getDimSize(dimension) - size) {
        return emitTreeError(anchor, Twine("work unit ") + Twine(workUnit.id) +
                                         " result tile is out of bounds");
      }
    }
    if (workUnit.iterationOffsets.size() != operation.iterationDomain.size() ||
        workUnit.iterationSizes.size() != operation.iterationDomain.size()) {
      return emitTreeError(anchor, Twine("work unit ") + Twine(workUnit.id) +
                                       " iteration tile rank mismatch");
    }
    for (size_t dimension = 0; dimension < operation.iterationDomain.size();
         ++dimension) {
      int64_t extent = operation.iterationDomain[dimension].staticExtent;
      int64_t offset = workUnit.iterationOffsets[dimension];
      int64_t size = workUnit.iterationSizes[dimension];
      if (ShapedType::isDynamic(extent) || offset < 0 || size <= 0 ||
          offset > extent - size) {
        return emitTreeError(anchor, Twine("work unit ") + Twine(workUnit.id) +
                                         " iteration tile is invalid");
      }
    }
    bool hasShardIdentity = workUnit.shardGroupId >= 0 ||
                            workUnit.shardIndex >= 0 ||
                            workUnit.shardCount >= 0;
    if (hasShardIdentity &&
        (workUnit.shardGroupId < 0 || workUnit.shardIndex < 0 ||
         workUnit.shardCount <= 0 ||
         workUnit.shardIndex >= workUnit.shardCount)) {
      return emitTreeError(anchor, Twine("work unit ") + Twine(workUnit.id) +
                                       " has incomplete shard identity");
    }
    workUnitsByOperation[workUnit.operationId].push_back(&workUnit);
  }

  auto endpointIsValid = [&](int64_t operationId, int64_t workUnitId) {
    if (operationId < 0 ||
        operationId >= static_cast<int64_t>(graph.operations.size()))
      return false;
    if (workUnitId < 0)
      return workUnitId == -1;
    const MappingWorkUnit *workUnit = workUnitsById.lookup(workUnitId);
    return workUnit && workUnit->operationId == operationId;
  };
  DenseSet<std::pair<int64_t, int64_t>> graphOperationEdges;
  DenseMap<int64_t, SmallVector<int64_t>> graphConsumersByProducer;
  for (const ComputeTensor &tensor : graph.tensors) {
    for (int64_t producerId : tensor.producerOperations) {
      for (int64_t consumerId : tensor.consumerOperations) {
        if (graphOperationEdges.insert({producerId, consumerId}).second)
          graphConsumersByProducer[producerId].push_back(consumerId);
      }
    }
  }
  std::set<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>> uniqueEdges;
  for (const MappingWorkUnitEdge &edge : tree.workUnitEdges) {
    if (!endpointIsValid(edge.sourceOperationId, edge.sourceWorkUnitId)) {
      return emitTreeError(anchor,
                           "work-unit edge has an invalid source endpoint");
    }
    if (!endpointIsValid(edge.targetOperationId, edge.targetWorkUnitId)) {
      return emitTreeError(anchor,
                           "work-unit edge has an invalid target endpoint");
    }
    if (edge.sourceWorkUnitId < 0 && edge.targetWorkUnitId < 0) {
      return emitTreeError(
          anchor, "work-unit edge must refine at least one tiled endpoint");
    }
    if (edge.byteSize < 0)
      return emitTreeError(anchor,
                           "work-unit edge byte size must be non-negative");
    if (!graphOperationEdges.contains(
            {edge.sourceOperationId, edge.targetOperationId})) {
      return emitTreeError(anchor,
                           Twine("work-unit edge refines nonexistent operation "
                                 "dependency ") +
                               Twine(edge.sourceOperationId) + " -> " +
                               Twine(edge.targetOperationId));
    }
    if (edge.sourceResultNumber < 0 ||
        edge.sourceResultNumber >=
            static_cast<int64_t>(graph.operations[edge.sourceOperationId]
                                     .operation->getNumResults()) ||
        edge.targetOperandNumber < 0 ||
        edge.targetOperandNumber >=
            static_cast<int64_t>(graph.operations[edge.targetOperationId]
                                     .operation->getNumOperands())) {
      return emitTreeError(
          anchor, "work-unit edge has invalid result or operand index");
    }
    if (edge.tensorId < -1 ||
        edge.tensorId >= static_cast<int64_t>(graph.tensors.size()))
      return emitTreeError(anchor, "work-unit edge has an invalid tensor ID");
    if (edge.tensorId >= 0) {
      const ComputeTensor &tensor = graph.tensors[edge.tensorId];
      if (!llvm::is_contained(tensor.producerOperations,
                              edge.sourceOperationId) ||
          !llvm::is_contained(tensor.consumerOperations,
                              edge.targetOperationId) ||
          graph.operations[edge.targetOperationId].operation->getOperand(
              edge.targetOperandNumber) != tensor.value ||
          edge.byteSize > getProducerContributionByteSize(
                              tensor, edge.sourceOperationId)) {
        return emitTreeError(
            anchor, "work-unit edge does not match its compute tensor");
      }
    }
    auto identity = std::make_tuple(
        edge.sourceOperationId, edge.sourceWorkUnitId, edge.targetOperationId,
        edge.targetWorkUnitId, edge.targetOperandNumber);
    if (!uniqueEdges.insert(identity).second)
      return emitTreeError(anchor, "duplicate work-unit edge");
  }

  llvm::SmallSet<int64_t, 16> coveredOperations;
  llvm::SmallSet<int64_t, 16> referencedWorkUnits;
  DenseMap<int64_t, bool> operationUsesWholeLeaf;
  for (const StructuralRATreeNode &node : tree.nodes) {
    if (node.kind == RATreeNodeKind::Leaf) {
      if (!node.childIds.empty()) {
        return emitTreeError(anchor, Twine("leaf node ") + Twine(node.id) +
                                         " must not have children");
      }
      if (node.operationId < 0 ||
          node.operationId >= static_cast<int64_t>(graph.operations.size())) {
        return emitTreeError(anchor, Twine("leaf node ") + Twine(node.id) +
                                         " references unknown operation " +
                                         Twine(node.operationId));
      }
      if (node.workUnitId < 0) {
        if (operationUsesWholeLeaf.contains(node.operationId) ||
            workUnitsByOperation.contains(node.operationId)) {
          return emitTreeError(anchor, Twine("operation ") +
                                           Twine(node.operationId) +
                                           " mixes whole and tiled leaves");
        }
        operationUsesWholeLeaf[node.operationId] = true;
      } else {
        const MappingWorkUnit *workUnit = workUnitsById.lookup(node.workUnitId);
        if (!workUnit || workUnit->operationId != node.operationId) {
          return emitTreeError(anchor, Twine("leaf node ") + Twine(node.id) +
                                           " references an invalid work unit");
        }
        if (!referencedWorkUnits.insert(node.workUnitId).second) {
          return emitTreeError(anchor, Twine("work unit ") +
                                           Twine(node.workUnitId) +
                                           " appears in multiple leaves");
        }
      }
      coveredOperations.insert(node.operationId);
    } else {
      if (node.kind == RATreeNodeKind::Layer) {
        if (node.childIds.size() != 1) {
          return emitTreeError(anchor, Twine("Layer node ") + Twine(node.id) +
                                           " must have exactly one child");
        }
      } else if (node.childIds.size() < 2) {
        return emitTreeError(anchor, Twine("cut node ") + Twine(node.id) +
                                         " must have at least two children");
      }
      if (node.operationId >= 0) {
        return emitTreeError(anchor, Twine("internal node ") + Twine(node.id) +
                                         " must not reference an operation");
      }
      if (node.workUnitId >= 0) {
        return emitTreeError(anchor, Twine("internal node ") + Twine(node.id) +
                                         " must not reference a work unit");
      }
    }

    for (int64_t childId : node.childIds) {
      auto child = nodeIndices.find(childId);
      if (child == nodeIndices.end()) {
        return emitTreeError(anchor, Twine("node ") + Twine(node.id) +
                                         " references unknown child " +
                                         Twine(childId));
      }
      if (tree.nodes[child->second].parentId != node.id) {
        return emitTreeError(anchor, Twine("parent mismatch for child ") +
                                         Twine(childId));
      }
    }
  }

  if (coveredOperations.size() != graph.operations.size())
    return emitTreeError(anchor,
                         "not every supported compute operation has one leaf");
  if (referencedWorkUnits.size() != tree.workUnits.size())
    return emitTreeError(anchor, "not every mapping work unit has one leaf");

  for (const auto &[operationId, workUnits] : workUnitsByOperation) {
    if (operationUsesWholeLeaf.contains(operationId))
      return emitTreeError(anchor, Twine("operation ") + Twine(operationId) +
                                       " mixes whole and tiled leaves");
    const MappingWorkUnit &first = *workUnits.front();
    const ComputeOperation &operation = graph.operations[operationId];
    auto resultType = cast<RankedTensorType>(
        operation.operation->getResult(first.resultNumber).getType());
    int64_t coveredVolume = 0;
    for (auto [index, workUnit] : llvm::enumerate(workUnits)) {
      if (workUnit->resultNumber != first.resultNumber)
        return emitTreeError(anchor, Twine("operation ") + Twine(operationId) +
                                         " tiles multiple results");
      FailureOr<int64_t> volume =
          checkedVolume(workUnit->resultSizes, anchor, "work-unit result tile");
      if (failed(volume))
        return failure();
      std::optional<int64_t> next = llvm::checkedAdd(coveredVolume, *volume);
      if (!next)
        return emitTreeError(anchor,
                             "work-unit coverage volume overflows int64");
      coveredVolume = *next;
      for (size_t other = 0; other < index; ++other) {
        if (staticTilesOverlap(*workUnit, *workUnits[other]))
          return emitTreeError(anchor, Twine("operation ") +
                                           Twine(operationId) +
                                           " has overlapping result tiles");
      }
    }
    if (coveredVolume != resultType.getNumElements()) {
      return emitTreeError(anchor, Twine("operation ") + Twine(operationId) +
                                       " result tiles do not cover the result");
    }
  }

  DenseMap<int64_t, unsigned> visitState;
  if (failed(visitTreeNode(tree.rootId, tree, nodeIndices, visitState, anchor)))
    return failure();
  if (visitState.size() != tree.nodes.size())
    return emitTreeError(anchor,
                         "tree contains nodes unreachable from the root");

  for (const StructuralRATreeNode &layer : tree.nodes) {
    if (layer.kind != RATreeNodeKind::Layer)
      continue;
    std::optional<int64_t> layerRegionId;
    SmallVector<int64_t> pending(layer.childIds.begin(), layer.childIds.end());
    while (!pending.empty()) {
      const StructuralRATreeNode &descendant =
          tree.nodes[nodeIndices.lookup(pending.pop_back_val())];
      if (descendant.kind != RATreeNodeKind::Leaf) {
        pending.append(descendant.childIds.begin(), descendant.childIds.end());
        continue;
      }
      const ComputeOperation &operation =
          graph.operations[descendant.operationId];
      if (operation.kind == ComputeOperationKind::MatrixSetup) {
        return emitTreeError(anchor, Twine("Layer node ") + Twine(layer.id) +
                                         " must not contain matrix setup");
      }
      if (operation.layerRegionId < 0) {
        return emitTreeError(anchor, Twine("Layer node ") + Twine(layer.id) +
                                         " contains an unassigned layer "
                                         "region");
      }
      if (layerRegionId && *layerRegionId != operation.layerRegionId) {
        return emitTreeError(anchor, Twine("Layer node ") + Twine(layer.id) +
                                         " spans multiple layer regions");
      }
      layerRegionId = operation.layerRegionId;
    }
    if (!layerRegionId) {
      return emitTreeError(anchor, Twine("Layer node ") + Twine(layer.id) +
                                       " contains no compute operations");
    }
  }

  return verifyTemporalOrders(tree, nodeIndices, graphConsumersByProducer,
                              anchor);
}

std::string computeGraphFingerprint(const ComputeGraph &graph) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "compute-graph-v3 function " << graph.functionSymbol << '\n';
  for (const ComputeOperation &computeOperation : graph.operations) {
    stream << "operation " << computeOperation.id << " kind "
           << stringifyComputeOperationKind(computeOperation.kind) << '\n';
    if (computeOperation.laneBindingGroup)
      stream << "  lane-binding-group " << *computeOperation.laneBindingGroup
             << '\n';
    if (computeOperation.mvmWaveId)
      stream << "  mvm-wave " << *computeOperation.mvmWaveId << " member "
             << computeOperation.mvmWaveMember.value_or(-1) << " size "
             << computeOperation.mvmWaveSize.value_or(-1) << '\n';
    if (computeOperation.analogMVM) {
      stream << "  mvm rows " << computeOperation.analogMVM->outputRows
             << " columns " << computeOperation.analogMVM->inputColumns << '\n';
    }
    for (Operation *member : computeOperation.members)
      printOperationStructure(member, stream);
    for (const ComputeIterationDimension &dimension :
         computeOperation.iterationDomain) {
      stream << "  iterator " << dimension.loopIndex << ' '
             << stringifyComputeIteratorKind(dimension.kind) << ' '
             << dimension.staticExtent << '\n';
    }
  }
  for (const LaneBindingGroup &group : graph.laneBindingGroups) {
    stream << "lane-binding-group " << group.id << " setup "
           << group.setupOperationId << " operations";
    for (int64_t operationId : group.operationIds)
      stream << ' ' << operationId;
    stream << '\n';
  }
  for (const MVMWave &wave : graph.mvmWaves) {
    stream << "mvm-wave " << wave.id << " vector-tiles";
    for (int64_t operationId : wave.vectorTileOperationIds)
      stream << ' ' << operationId;
    stream << " physical-mvms";
    for (int64_t operationId : wave.physicalMVMOperationIds)
      stream << ' ' << operationId;
    stream << " recombine " << wave.recombineOperationId.value_or(-1)
           << " bias-add " << wave.biasAddOperationId.value_or(-1) << '\n';
  }
  for (const ComputeTensor &tensor : graph.tensors) {
    stream << "tensor " << tensor.id << " producers";
    for (int64_t producer : tensor.producerOperations)
      stream << ' ' << producer << ':'
             << getProducerContributionByteSize(tensor, producer);
    stream << " consumers";
    for (int64_t consumer : tensor.consumerOperations)
      stream << ' ' << consumer;
    stream << " type " << tensor.type << " bytes " << tensor.byteSize
           << " input " << tensor.isFunctionInput << " output "
           << tensor.isFunctionOutput << '\n';
  }
  stream.flush();

  llvm::SHA256 hasher;
  hasher.update(description);
  return digestToHex(hasher.final());
}

std::string computeRATreeFingerprint(const ResourceAllocationTree &tree) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  stream << "ra-tree-v4 root " << tree.rootId << '\n';
  for (const MappingWorkUnit &workUnit : tree.workUnits) {
    stream << "work-unit " << workUnit.id << " operation "
           << workUnit.operationId << " result " << workUnit.resultNumber
           << " result-offsets";
    for (int64_t value : workUnit.resultOffsets)
      stream << ' ' << value;
    stream << " result-sizes";
    for (int64_t value : workUnit.resultSizes)
      stream << ' ' << value;
    stream << " iteration-offsets";
    for (int64_t value : workUnit.iterationOffsets)
      stream << ' ' << value;
    stream << " iteration-sizes";
    for (int64_t value : workUnit.iterationSizes)
      stream << ' ' << value;
    stream << " shard " << workUnit.shardGroupId << ':' << workUnit.shardIndex
           << '/' << workUnit.shardCount;
    stream << '\n';
  }
  for (const MappingWorkUnitEdge &edge : tree.workUnitEdges) {
    stream << "work-unit-edge source " << edge.sourceOperationId << ':'
           << edge.sourceWorkUnitId << " target " << edge.targetOperationId
           << ':' << edge.targetWorkUnitId << " tensor " << edge.tensorId
           << " source-result " << edge.sourceResultNumber << " target-operand "
           << edge.targetOperandNumber << " bytes " << edge.byteSize << '\n';
  }
  for (const StructuralRATreeNode &node : tree.nodes) {
    stream << "node " << node.id << " kind " << static_cast<uint32_t>(node.kind)
           << " parent " << node.parentId << " operation " << node.operationId
           << " work-unit " << node.workUnitId << " groups "
           << node.workGroupCount << " children";
    for (int64_t childId : node.childIds)
      stream << ' ' << childId;
    stream << '\n';
  }
  stream.flush();

  llvm::SHA256 hasher;
  hasher.update(description);
  return digestToHex(hasher.final());
}

RATreeAttr serializeResourceAllocationTree(MLIRContext *context,
                                           const ResourceAllocationTree &tree,
                                           const ComputeGraph &graph,
                                           StringRef graphFingerprint) {
  Builder builder(context);
  auto buildI64Array = [&](ArrayRef<int64_t> values) {
    SmallVector<Attribute> attributes;
    attributes.reserve(values.size());
    for (int64_t value : values)
      attributes.push_back(builder.getI64IntegerAttr(value));
    return builder.getArrayAttr(attributes);
  };
  SmallVector<Attribute> workUnitAttrs;
  workUnitAttrs.reserve(tree.workUnits.size());
  for (const MappingWorkUnit &workUnit : tree.workUnits) {
    workUnitAttrs.push_back(MappingWorkUnitAttr::get(
        context, builder.getI64IntegerAttr(workUnit.id),
        builder.getI64IntegerAttr(workUnit.operationId),
        builder.getI64IntegerAttr(workUnit.resultNumber),
        buildI64Array(workUnit.resultOffsets),
        buildI64Array(workUnit.resultSizes),
        buildI64Array(workUnit.iterationOffsets),
        buildI64Array(workUnit.iterationSizes),
        builder.getI64IntegerAttr(workUnit.shardGroupId),
        builder.getI64IntegerAttr(workUnit.shardIndex),
        builder.getI64IntegerAttr(workUnit.shardCount)));
  }
  SmallVector<Attribute> nodeAttrs;
  nodeAttrs.reserve(tree.nodes.size());
  for (const StructuralRATreeNode &node : tree.nodes) {
    SmallVector<Attribute> childIds;
    childIds.reserve(node.childIds.size());
    for (int64_t childId : node.childIds)
      childIds.push_back(builder.getI64IntegerAttr(childId));
    nodeAttrs.push_back(
        RATreeNodeAttr::get(context, builder.getI64IntegerAttr(node.id),
                            node.kind, builder.getI64IntegerAttr(node.parentId),
                            builder.getArrayAttr(childIds),
                            builder.getI64IntegerAttr(node.operationId),
                            builder.getI64IntegerAttr(node.workUnitId),
                            builder.getI64IntegerAttr(node.workGroupCount)));
  }

  SmallVector<Attribute> workUnitEdgeAttrs;
  workUnitEdgeAttrs.reserve(tree.workUnitEdges.size());
  for (const MappingWorkUnitEdge &edge : tree.workUnitEdges) {
    workUnitEdgeAttrs.push_back(MappingWorkUnitEdgeAttr::get(
        context, builder.getI64IntegerAttr(edge.sourceOperationId),
        builder.getI64IntegerAttr(edge.sourceWorkUnitId),
        builder.getI64IntegerAttr(edge.targetOperationId),
        builder.getI64IntegerAttr(edge.targetWorkUnitId),
        builder.getI64IntegerAttr(edge.tensorId),
        builder.getI64IntegerAttr(edge.sourceResultNumber),
        builder.getI64IntegerAttr(edge.targetOperandNumber),
        builder.getI64IntegerAttr(edge.byteSize)));
  }

  return RATreeAttr::get(context, builder.getI64IntegerAttr(4),
                         builder.getI64IntegerAttr(tree.rootId),
                         builder.getArrayAttr(nodeAttrs),
                         builder.getArrayAttr(workUnitAttrs),
                         builder.getArrayAttr(workUnitEdgeAttrs),
                         builder.getI64IntegerAttr(graph.operations.size()),
                         builder.getI64IntegerAttr(graph.tensors.size()),
                         builder.getStringAttr(graphFingerprint));
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
