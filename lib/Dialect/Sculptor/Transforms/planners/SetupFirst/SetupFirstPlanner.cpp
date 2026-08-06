#include "SetupFirstPlanner.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <optional>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

class SetupFirstTreeBuilder {
public:
  explicit SetupFirstTreeBuilder(const ResourceAllocationTree &source) {
    tree.workUnits = source.workUnits;
    tree.workUnitEdges = source.workUnitEdges;
  }

  int64_t addLeaf(int64_t operationId, int64_t workGroupCount = 1,
                  int64_t workUnitId = -1) {
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    tree.nodes.push_back({id,
                          RATreeNodeKind::Leaf,
                          -1,
                          {},
                          operationId,
                          workUnitId,
                          workGroupCount});
    return id;
  }

  int64_t addCut(RATreeNodeKind kind, ArrayRef<int64_t> children,
                 int64_t workGroupCount = 1) {
    assert(children.size() >= 2 && "cuts require at least two children");
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
    std::optional<int64_t> count =
        llvm::checkedMul(tree.nodes[nodeId].workGroupCount, multiplier);
    if (!count) {
      anchor->emitError(
          "setup-first work-group normalization overflows a signed 64-bit "
          "integer");
      return failure();
    }
    tree.nodes[nodeId].workGroupCount = *count;
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

FailureOr<std::optional<int64_t>> cloneComputeSubtree(
    int64_t nodeId,
    const DenseMap<int64_t, const StructuralRATreeNode *> &nodes,
    const ComputeGraph &graph, SetupFirstTreeBuilder &builder,
    Operation *anchor) {
  const StructuralRATreeNode *node = nodes.lookup(nodeId);
  if (!node) {
    anchor->emitError("setup-first planner cannot resolve delegated RA node ")
        << nodeId;
    return failure();
  }

  if (node->kind == RATreeNodeKind::Leaf) {
    if (graph.operations[node->operationId].kind ==
        ComputeOperationKind::MatrixSetup)
      return std::optional<int64_t>();
    return std::optional<int64_t>(builder.addLeaf(
        node->operationId, node->workGroupCount, node->workUnitId));
  }

  SmallVector<int64_t> children;
  children.reserve(node->childIds.size());
  for (int64_t childId : node->childIds) {
    FailureOr<std::optional<int64_t>> child =
        cloneComputeSubtree(childId, nodes, graph, builder, anchor);
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
  return std::optional<int64_t>(
      builder.addCut(node->kind, children, node->workGroupCount));
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<MappingPlan>
SetupFirstPlanner::refine(const MappingProblem &problem,
                          const MappingEvaluator &evaluator) const {
  DenseMap<int64_t, const StructuralRATreeNode *> delegatedNodes;
  for (const StructuralRATreeNode &node : problem.currentTree.nodes)
    delegatedNodes[node.id] = &node;

  SetupFirstTreeBuilder builder(problem.currentTree);
  FailureOr<std::optional<int64_t>> computeRoot =
      cloneComputeSubtree(problem.currentTree.rootId, delegatedNodes,
                          problem.graph, builder, problem.anchor);
  if (failed(computeRoot))
    return failure();
  if (!*computeRoot) {
    problem.anchor->emitError(
        "setup-first planner requires at least one non-setup operation");
    return failure();
  }

  SmallVector<int64_t> setupLeaves;
  for (int64_t operationId : problem.graph.topologicalOrder) {
    if (problem.graph.operations[operationId].kind ==
        ComputeOperationKind::MatrixSetup)
      setupLeaves.push_back(builder.addLeaf(operationId));
  }
  if (setupLeaves.empty()) {
    problem.anchor->emitError(
        "setup-first planner requires at least one matrix setup operation");
    return failure();
  }

  int64_t setupRoot =
      setupLeaves.size() == 1
          ? setupLeaves.front()
          : builder.addCut(RATreeNodeKind::SpatialCut, setupLeaves);
  int64_t root =
      builder.addCut(RATreeNodeKind::TemporalCut, {setupRoot, **computeRoot});
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
      {/*name=*/"setup-first", *evaluation, /*selected=*/true});
  return plan;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
