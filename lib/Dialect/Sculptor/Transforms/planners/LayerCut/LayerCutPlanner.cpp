#include "LayerCutPlanner.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <optional>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

class LayerCutTreeBuilder {
public:
  explicit LayerCutTreeBuilder(const ResourceAllocationTree &source) {
    tree.workUnits = source.workUnits;
    tree.workUnitEdges = source.workUnitEdges;
  }

  int64_t addLeaf(const StructuralRATreeNode &source) {
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    tree.nodes.push_back({id,
                          RATreeNodeKind::Leaf,
                          -1,
                          {},
                          source.operationId,
                          source.workUnitId,
                          source.workGroupCount});
    return id;
  }

  int64_t addCut(RATreeNodeKind kind, ArrayRef<int64_t> children,
                 int64_t workGroupCount = 1) {
    assert(children.size() >= 2 && "cuts require at least two children");
    assert(kind != RATreeNodeKind::Leaf && kind != RATreeNodeKind::Layer &&
           "layer-cut only constructs scheduling cuts");
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    StructuralRATreeNode node;
    node.id = id;
    node.kind = kind;
    node.childIds.assign(children.begin(), children.end());
    node.workGroupCount = workGroupCount;
    tree.nodes.push_back(std::move(node));
    for (int64_t childId : children)
      tree.nodes[childId].parentId = id;
    return id;
  }

  LogicalResult multiplyWorkGroups(int64_t nodeId, int64_t multiplier,
                                   Operation *anchor) {
    std::optional<int64_t> product =
        llvm::checkedMul(tree.nodes[nodeId].workGroupCount, multiplier);
    if (!product) {
      anchor->emitError("layer-cut work-group normalization overflows a "
                        "signed 64-bit integer");
      return failure();
    }
    tree.nodes[nodeId].workGroupCount = *product;
    return success();
  }

  ResourceAllocationTree finish(int64_t rootId) {
    tree.rootId = rootId;
    tree.nodes[rootId].parentId = -1;
    return std::move(tree);
  }

private:
  ResourceAllocationTree tree;
};

enum class SelectedSubtree { MatrixSetup, LayerRegion };

FailureOr<std::optional<int64_t>> cloneSelectedSubtree(
    int64_t nodeId,
    const DenseMap<int64_t, const StructuralRATreeNode *> &sourceNodes,
    const ComputeGraph &graph, SelectedSubtree selection, int64_t regionId,
    LayerCutTreeBuilder &builder, Operation *anchor) {
  const StructuralRATreeNode *node = sourceNodes.lookup(nodeId);
  if (!node) {
    anchor->emitError("layer-cut cannot resolve delegated RA node ") << nodeId;
    return failure();
  }

  if (node->kind == RATreeNodeKind::Leaf) {
    const ComputeOperation &operation = graph.operations[node->operationId];
    bool selected = selection == SelectedSubtree::MatrixSetup
                        ? operation.kind == ComputeOperationKind::MatrixSetup
                        : operation.kind != ComputeOperationKind::MatrixSetup &&
                              operation.layerRegionId == regionId;
    if (!selected)
      return std::optional<int64_t>();
    return std::optional<int64_t>(builder.addLeaf(*node));
  }

  SmallVector<int64_t> children;
  children.reserve(node->childIds.size());
  for (int64_t childId : node->childIds) {
    FailureOr<std::optional<int64_t>> child = cloneSelectedSubtree(
        childId, sourceNodes, graph, selection, regionId, builder, anchor);
    if (failed(child))
      return failure();
    if (*child)
      children.push_back(**child);
  }

  if (children.empty())
    return std::optional<int64_t>();
  if (children.size() == 1) {
    if (failed(builder.multiplyWorkGroups(children.front(),
                                          node->workGroupCount, anchor)))
      return failure();
    return std::optional<int64_t>(children.front());
  }
  if (node->kind == RATreeNodeKind::Layer) {
    anchor->emitError("layer-cut received an unstripped Layer RA node");
    return failure();
  }
  return std::optional<int64_t>(
      builder.addCut(node->kind, children, node->workGroupCount));
}

FailureOr<SmallVector<SmallVector<int64_t>>>
buildLayerWaves(const ComputeGraph &graph, ArrayRef<int64_t> activeRegions,
                Operation *anchor) {
  DenseSet<int64_t> active(activeRegions.begin(), activeRegions.end());
  DenseMap<int64_t, SmallVector<int64_t>> successors;
  DenseMap<int64_t, int64_t> indegree;
  for (int64_t regionId : activeRegions)
    indegree[regionId] = 0;

  DenseMap<int64_t, int64_t> topologicalRank;
  for (auto [rank, regionId] :
       llvm::enumerate(graph.topologicalLayerRegionOrder))
    topologicalRank[regionId] = static_cast<int64_t>(rank);

  auto addDependency = [&](int64_t predecessor, int64_t successor) {
    if (predecessor == successor || !active.contains(predecessor) ||
        !active.contains(successor) ||
        llvm::is_contained(successors[predecessor], successor))
      return;
    successors[predecessor].push_back(successor);
    ++indegree[successor];
  };

  for (const ComputeTensor &tensor : graph.tensors) {
    for (int64_t producerId : tensor.producerOperations) {
      const ComputeOperation &producer = graph.operations[producerId];
      if (producer.kind == ComputeOperationKind::MatrixSetup)
        continue;
      int64_t producerRegion = producer.layerRegionId;
      if (!active.contains(producerRegion))
        continue;
      for (int64_t consumerId : tensor.consumerOperations) {
        const ComputeOperation &consumer = graph.operations[consumerId];
        if (consumer.kind == ComputeOperationKind::MatrixSetup)
          continue;
        int64_t consumerRegion = consumer.layerRegionId;
        addDependency(producerRegion, consumerRegion);
      }
    }
  }

  // A lane-binding group is persistent matrix state: every physical MVM in
  // the group must remain on one analog lane. Independent layer regions that
  // reuse that state therefore cannot be siblings of the same spatial cut.
  // Serialize those regions in their existing topological order while still
  // allowing unrelated regions to occupy the same parallel wave.
  for (const LaneBindingGroup &group : graph.laneBindingGroups) {
    SmallVector<int64_t> regions;
    for (int64_t operationId : group.operationIds) {
      const ComputeOperation &operation = graph.operations[operationId];
      if (operation.kind == ComputeOperationKind::MatrixSetup ||
          !active.contains(operation.layerRegionId) ||
          llvm::is_contained(regions, operation.layerRegionId))
        continue;
      regions.push_back(operation.layerRegionId);
    }
    llvm::sort(regions, [&](int64_t left, int64_t right) {
      return std::make_pair(topologicalRank.lookup(left), left) <
             std::make_pair(topologicalRank.lookup(right), right);
    });
    for (auto adjacent : llvm::zip(regions, llvm::drop_begin(regions)))
      addDependency(std::get<0>(adjacent), std::get<1>(adjacent));
  }

  auto orderRegions = [&](SmallVectorImpl<int64_t> &regions) {
    llvm::sort(regions, [&](int64_t left, int64_t right) {
      return std::make_pair(topologicalRank.lookup(left), left) <
             std::make_pair(topologicalRank.lookup(right), right);
    });
  };

  SmallVector<int64_t> ready;
  for (int64_t regionId : activeRegions) {
    if (indegree.lookup(regionId) == 0)
      ready.push_back(regionId);
  }
  orderRegions(ready);

  SmallVector<SmallVector<int64_t>> waves;
  int64_t visited = 0;
  while (!ready.empty()) {
    SmallVector<int64_t> wave = std::move(ready);
    ready.clear();
    visited += static_cast<int64_t>(wave.size());
    SmallVector<int64_t> next;
    for (int64_t regionId : wave) {
      for (int64_t successor : successors[regionId]) {
        int64_t &degree = indegree[successor];
        if (--degree == 0)
          next.push_back(successor);
      }
    }
    orderRegions(next);
    waves.push_back(std::move(wave));
    ready = std::move(next);
  }

  if (visited != static_cast<int64_t>(activeRegions.size())) {
    anchor->emitError("layer-cut found a cycle in the layer dependency graph");
    return failure();
  }
  return waves;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<MappingPlan>
LayerCutPlanner::refine(const MappingProblem &problem,
                        const MappingEvaluator &evaluator) const {
  DenseMap<int64_t, const StructuralRATreeNode *> sourceNodes;
  for (const StructuralRATreeNode &node : problem.currentTree.nodes)
    sourceNodes[node.id] = &node;

  LayerCutTreeBuilder builder(problem.currentTree);
  FailureOr<std::optional<int64_t>> setupRoot = cloneSelectedSubtree(
      problem.currentTree.rootId, sourceNodes, problem.graph,
      SelectedSubtree::MatrixSetup, /*regionId=*/-1, builder, problem.anchor);
  if (failed(setupRoot))
    return failure();

  SmallVector<int64_t> activeRegions;
  DenseMap<int64_t, int64_t> regionRoots;
  for (int64_t regionId : problem.graph.topologicalLayerRegionOrder) {
    FailureOr<std::optional<int64_t>> regionRoot = cloneSelectedSubtree(
        problem.currentTree.rootId, sourceNodes, problem.graph,
        SelectedSubtree::LayerRegion, regionId, builder, problem.anchor);
    if (failed(regionRoot))
      return failure();
    if (!*regionRoot)
      continue;
    activeRegions.push_back(regionId);
    regionRoots[regionId] = **regionRoot;
  }
  if (activeRegions.empty()) {
    problem.anchor->emitError("layer-cut requires at least one compute layer");
    return failure();
  }

  FailureOr<SmallVector<SmallVector<int64_t>>> waves =
      buildLayerWaves(problem.graph, activeRegions, problem.anchor);
  if (failed(waves) || waves->empty())
    return failure();

  SmallVector<int64_t> waveRoots;
  waveRoots.reserve(waves->size());
  for (const SmallVector<int64_t> &wave : *waves) {
    SmallVector<int64_t> children;
    children.reserve(wave.size());
    for (int64_t regionId : wave)
      children.push_back(regionRoots.lookup(regionId));
    waveRoots.push_back(
        children.size() == 1
            ? children.front()
            : builder.addCut(RATreeNodeKind::SpatialCut, children));
  }
  int64_t computeRoot =
      waveRoots.size() == 1
          ? waveRoots.front()
          : builder.addCut(RATreeNodeKind::TemporalCut, waveRoots);
  int64_t root = *setupRoot ? builder.addCut(RATreeNodeKind::TemporalCut,
                                             {**setupRoot, computeRoot})
                            : computeRoot;
  ResourceAllocationTree tree = builder.finish(root);
  FailureOr<ResourceAllocationTree> reindexed =
      reindexResourceAllocationTree(tree, problem.anchor);
  if (failed(reindexed) || failed(verifyResourceAllocationTree(
                               *reindexed, problem.graph, problem.anchor)))
    return failure();

  FailureOr<MappingEvaluation> evaluation =
      evaluator.evaluate(problem, *reindexed);
  if (failed(evaluation))
    return failure();
  MappingPlan plan;
  plan.plannerName = getName().str();
  plan.objective = problem.objective;
  plan.selectedTree = std::move(*reindexed);
  plan.evaluation = *evaluation;
  plan.candidates.push_back(
      {/*name=*/"layer-cut", *evaluation, /*selected=*/true});
  return plan;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
