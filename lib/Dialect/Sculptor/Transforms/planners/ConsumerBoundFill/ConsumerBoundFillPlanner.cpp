#include "ConsumerBoundFillPlanner.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <optional>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

bool isFill(const ComputeOperation &operation) {
  return operation.operation->getName().getStringRef() == "linalg.fill";
}

SmallVector<int64_t> findDirectConsumers(const ComputeOperation &operation,
                                         const ComputeGraph &graph) {
  DenseSet<int64_t> seen;
  SmallVector<int64_t> consumers;
  for (int64_t tensorId : operation.outputTensors) {
    for (int64_t consumerId : graph.tensors[tensorId].consumerOperations) {
      if (consumerId == operation.id || !seen.insert(consumerId).second)
        continue;
      consumers.push_back(consumerId);
    }
  }
  llvm::sort(consumers);
  return consumers;
}

class ConsumerBoundFillTreeBuilder {
public:
  ConsumerBoundFillTreeBuilder(
      const MappingProblem &problem,
      const DenseMap<int64_t, SmallVector<int64_t>> &fillLeavesByTarget,
      const DenseSet<int64_t> &movedFillLeaves)
      : problem(problem), fillLeavesByTarget(fillLeavesByTarget),
        movedFillLeaves(movedFillLeaves) {
    tree.workUnits = problem.currentTree.workUnits;
    tree.workUnitEdges = problem.currentTree.workUnitEdges;
    for (const StructuralRATreeNode &node : problem.currentTree.nodes)
      sourceNodes[node.id] = &node;
  }

  FailureOr<ResourceAllocationTree> build() {
    FailureOr<std::optional<int64_t>> root = clone(problem.currentTree.rootId);
    if (failed(root))
      return failure();
    if (!*root) {
      problem.anchor->emitError(
          "consumer-bound-fill removed the complete RA hierarchy");
      return failure();
    }
    tree.rootId = **root;
    tree.nodes[tree.rootId].parentId = -1;
    return std::move(tree);
  }

private:
  int64_t addLeaf(const StructuralRATreeNode &source) {
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    tree.nodes.push_back({id, RATreeNodeKind::Leaf, -1, {},
                          source.operationId, source.workUnitId,
                          source.workGroupCount});
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

  LogicalResult multiplyWorkGroups(int64_t nodeId, int64_t multiplier) {
    std::optional<int64_t> count =
        llvm::checkedMul(tree.nodes[nodeId].workGroupCount, multiplier);
    if (!count) {
      problem.anchor->emitError(
          "consumer-bound-fill work-group count overflows int64");
      return failure();
    }
    tree.nodes[nodeId].workGroupCount = *count;
    return success();
  }

  FailureOr<std::optional<int64_t>> clone(int64_t sourceNodeId) {
    const StructuralRATreeNode *source = sourceNodes.lookup(sourceNodeId);
    if (!source) {
      problem.anchor->emitError(
          "consumer-bound-fill cannot resolve RA node ")
          << sourceNodeId;
      return failure();
    }

    std::optional<int64_t> cloned;
    if (source->kind == RATreeNodeKind::Leaf) {
      if (!movedFillLeaves.contains(sourceNodeId))
        cloned = addLeaf(*source);
    } else {
      SmallVector<int64_t> children;
      for (int64_t childId : source->childIds) {
        FailureOr<std::optional<int64_t>> child = clone(childId);
        if (failed(child))
          return failure();
        if (*child)
          children.push_back(**child);
      }
      if (children.size() == 1) {
        if (failed(multiplyWorkGroups(children.front(),
                                      source->workGroupCount)))
          return failure();
        cloned = children.front();
      } else if (!children.empty()) {
        cloned = addCut(source->kind, children, source->workGroupCount);
      }
    }

    auto boundFills = fillLeavesByTarget.find(sourceNodeId);
    if (boundFills == fillLeavesByTarget.end())
      return cloned;
    if (!cloned) {
      problem.anchor->emitError(
          "consumer-bound-fill target disappeared while rebuilding the RA "
          "hierarchy");
      return failure();
    }

    SmallVector<int64_t> phases;
    phases.reserve(boundFills->second.size() + 1);
    for (int64_t fillLeafId : boundFills->second) {
      const StructuralRATreeNode *fillLeaf = sourceNodes.lookup(fillLeafId);
      if (!fillLeaf || fillLeaf->kind != RATreeNodeKind::Leaf) {
        problem.anchor->emitError(
            "consumer-bound-fill cannot resolve a fill leaf");
        return failure();
      }
      phases.push_back(addLeaf(*fillLeaf));
    }
    phases.push_back(*cloned);
    return std::optional<int64_t>(
        addCut(RATreeNodeKind::TemporalCut, phases));
  }

  const MappingProblem &problem;
  const DenseMap<int64_t, SmallVector<int64_t>> &fillLeavesByTarget;
  const DenseSet<int64_t> &movedFillLeaves;
  ResourceAllocationTree tree;
  DenseMap<int64_t, const StructuralRATreeNode *> sourceNodes;
};

FailureOr<ResourceAllocationTree>
bindFillsToConsumers(const MappingProblem &problem) {
  DenseMap<int64_t, SmallVector<int64_t>> leavesByOperation;
  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  for (const StructuralRATreeNode &node : problem.currentTree.nodes) {
    nodesById[node.id] = &node;
    if (node.kind == RATreeNodeKind::Leaf)
      leavesByOperation[node.operationId].push_back(node.id);
  }

  DenseMap<int64_t, const MappingWorkUnit *> workUnitsById;
  for (const MappingWorkUnit &workUnit : problem.currentTree.workUnits)
    workUnitsById[workUnit.id] = &workUnit;

  auto tilesMatch = [&](const StructuralRATreeNode &fill,
                        const StructuralRATreeNode &consumer) {
    if (fill.workUnitId < 0 || consumer.workUnitId < 0)
      return fill.workUnitId < 0 && consumer.workUnitId < 0;
    const MappingWorkUnit *fillWorkUnit =
        workUnitsById.lookup(fill.workUnitId);
    const MappingWorkUnit *consumerWorkUnit =
        workUnitsById.lookup(consumer.workUnitId);
    return fillWorkUnit && consumerWorkUnit &&
           fillWorkUnit->resultOffsets == consumerWorkUnit->resultOffsets &&
           fillWorkUnit->resultSizes == consumerWorkUnit->resultSizes;
  };

  DenseMap<int64_t, int64_t> topologicalRank;
  for (auto [rank, operationId] : llvm::enumerate(problem.graph.topologicalOrder))
    topologicalRank[operationId] = static_cast<int64_t>(rank);

  DenseMap<int64_t, SmallVector<int64_t>> fillLeavesByTarget;
  DenseSet<int64_t> movedFillLeaves;
  for (const ComputeOperation &operation : problem.graph.operations) {
    if (!isFill(operation))
      continue;

    SmallVector<int64_t> consumers =
        findDirectConsumers(operation, problem.graph);
    if (consumers.size() != 1 || isFill(problem.graph.operations[consumers[0]]))
      continue;

    auto fillLeaves = leavesByOperation.find(operation.id);
    auto consumerLeaves = leavesByOperation.find(consumers[0]);
    if (fillLeaves == leavesByOperation.end() ||
        consumerLeaves == leavesByOperation.end() ||
        fillLeaves->second.size() != consumerLeaves->second.size())
      continue;

    SmallVector<std::pair<int64_t, int64_t>> pairs;
    DenseSet<int64_t> matchedConsumers;
    for (int64_t fillLeafId : fillLeaves->second) {
      const StructuralRATreeNode *fillLeaf = nodesById.lookup(fillLeafId);
      int64_t matchingConsumer = -1;
      for (int64_t consumerLeafId : consumerLeaves->second) {
        if (matchedConsumers.contains(consumerLeafId))
          continue;
        const StructuralRATreeNode *consumerLeaf =
            nodesById.lookup(consumerLeafId);
        if (fillLeaf && consumerLeaf && tilesMatch(*fillLeaf, *consumerLeaf)) {
          matchingConsumer = consumerLeafId;
          break;
        }
      }
      if (matchingConsumer < 0) {
        pairs.clear();
        break;
      }
      matchedConsumers.insert(matchingConsumer);
      pairs.push_back({fillLeafId, matchingConsumer});
    }
    if (pairs.size() != fillLeaves->second.size())
      continue;

    for (auto [fillLeafId, consumerLeafId] : pairs) {
      movedFillLeaves.insert(fillLeafId);
      fillLeavesByTarget[consumerLeafId].push_back(fillLeafId);
    }
  }

  if (movedFillLeaves.empty())
    return cloneResourceAllocationTree(problem.currentTree);

  for (auto &[targetId, fillLeaves] : fillLeavesByTarget) {
    llvm::sort(fillLeaves, [&](int64_t left, int64_t right) {
      int64_t leftOperation = nodesById.lookup(left)->operationId;
      int64_t rightOperation = nodesById.lookup(right)->operationId;
      return std::pair(topologicalRank.lookup(leftOperation), leftOperation) <
             std::pair(topologicalRank.lookup(rightOperation), rightOperation);
    });
  }

  ConsumerBoundFillTreeBuilder builder(problem, fillLeavesByTarget,
                                       movedFillLeaves);
  return builder.build();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<MappingPlan>
ConsumerBoundFillPlanner::refine(const MappingProblem &problem,
                                 const MappingEvaluator &evaluator) const {
  FailureOr<ResourceAllocationTree> tree = bindFillsToConsumers(problem);
  if (failed(tree))
    return failure();
  FailureOr<ResourceAllocationTree> reindexed =
      reindexResourceAllocationTree(*tree, problem.anchor);
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
      {/*name=*/getName().str(), *evaluation, /*selected=*/true});
  return plan;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
