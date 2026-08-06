#include "FanOutCutPlanner.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

struct PlannedNode {
  int64_t treeNodeId = -1;
  int64_t digitalLanes = 0;
  int64_t anonymousAnalogLanes = 0;
  std::set<int64_t> bindingGroups;
};

struct SchedulingUnit {
  int64_t id = -1;
  int64_t sourceRootId = -1;
  SmallVector<int64_t> operationIds;
  int64_t topologicalRank = std::numeric_limits<int64_t>::max();
  bool preparation = false;
};

struct FanOutStructure {
  int64_t setupRootId = -1;
  SmallVector<SchedulingUnit> units;
  SmallVector<int64_t> preparationUnitIds;
  SmallVector<SmallVector<int64_t, 0>, 0> computeLevels;
};

bool isRootFill(const ComputeOperation &operation,
                const ComputeGraph &graph) {
  if (operation.operation->getName().getStringRef() != "linalg.fill")
    return false;
  for (int64_t tensorId : operation.inputTensors) {
    for (int64_t producerId : graph.tensors[tensorId].producerOperations) {
      if (producerId != operation.id &&
          graph.operations[producerId].kind !=
              ComputeOperationKind::MatrixSetup)
        return false;
    }
  }
  return true;
}

void collectSubtreeOperations(
    int64_t nodeId,
    const DenseMap<int64_t, const StructuralRATreeNode *> &nodes,
    SmallVectorImpl<int64_t> &operationIds) {
  const StructuralRATreeNode *node = nodes.lookup(nodeId);
  assert(node && "verified RA tree must contain every referenced node");
  if (node->kind == RATreeNodeKind::Leaf) {
    if (!llvm::is_contained(operationIds, node->operationId))
      operationIds.push_back(node->operationId);
    return;
  }
  for (int64_t childId : node->childIds)
    collectSubtreeOperations(childId, nodes, operationIds);
}

bool containsExactlySetups(
    int64_t nodeId,
    const DenseMap<int64_t, const StructuralRATreeNode *> &nodes,
    const ComputeGraph &graph) {
  DenseSet<int64_t> expected;
  for (const ComputeOperation &operation : graph.operations) {
    if (operation.kind == ComputeOperationKind::MatrixSetup)
      expected.insert(operation.id);
  }
  if (expected.empty())
    return false;

  SmallVector<int64_t> actual;
  collectSubtreeOperations(nodeId, nodes, actual);
  if (actual.size() != expected.size())
    return false;
  return llvm::all_of(actual,
                      [&](int64_t operationId) {
                        return expected.contains(operationId);
                      });
}

FailureOr<FanOutStructure> buildFanOutStructure(const MappingProblem &problem) {
  DenseMap<int64_t, const StructuralRATreeNode *> nodes;
  for (const StructuralRATreeNode &node : problem.currentTree.nodes)
    nodes[node.id] = &node;

  const StructuralRATreeNode *root = nodes.lookup(problem.currentTree.rootId);
  if (!root) {
    problem.anchor->emitError("fan-out-cut cannot resolve the RA-tree root");
    return failure();
  }

  FanOutStructure result;
  int64_t computeRootId = root->id;
  if (root->kind == RATreeNodeKind::TemporalCut &&
      root->childIds.size() == 2 &&
      containsExactlySetups(root->childIds.front(), nodes, problem.graph)) {
    result.setupRootId = root->childIds.front();
    computeRootId = root->childIds.back();
  }

  const StructuralRATreeNode *computeRoot = nodes.lookup(computeRootId);
  if (!computeRoot) {
    problem.anchor->emitError("fan-out-cut cannot resolve the compute root");
    return failure();
  }
  SmallVector<int64_t> unitRootIds =
      computeRoot->kind == RATreeNodeKind::TemporalCut
          ? computeRoot->childIds
          : SmallVector<int64_t>{computeRootId};

  DenseMap<int64_t, int64_t> operationRanks;
  for (auto [rank, operationId] : llvm::enumerate(problem.graph.topologicalOrder))
    operationRanks[operationId] = static_cast<int64_t>(rank);

  DenseMap<int64_t, int64_t> operationToUnit;
  for (int64_t unitRootId : unitRootIds) {
    SchedulingUnit unit;
    unit.id = static_cast<int64_t>(result.units.size());
    unit.sourceRootId = unitRootId;
    collectSubtreeOperations(unitRootId, nodes, unit.operationIds);
    llvm::sort(unit.operationIds, [&](int64_t left, int64_t right) {
      return std::pair(operationRanks.lookup(left), left) <
             std::pair(operationRanks.lookup(right), right);
    });
    if (unit.operationIds.empty()) {
      problem.anchor->emitError("fan-out-cut found an empty scheduling unit");
      return failure();
    }
    for (int64_t operationId : unit.operationIds) {
      const ComputeOperation &operation = problem.graph.operations[operationId];
      if (operation.kind == ComputeOperationKind::MatrixSetup) {
        operation.operation->emitError(
            "fan-out-cut found matrix setup work inside the compute subtree");
        return failure();
      }
      if (!operationToUnit.try_emplace(operationId, unit.id).second) {
        operation.operation->emitError(
            "fan-out-cut found one operation in multiple scheduling units");
        return failure();
      }
      unit.topologicalRank =
          std::min(unit.topologicalRank, operationRanks.lookup(operationId));
    }
    unit.preparation = llvm::all_of(unit.operationIds, [&](int64_t operationId) {
      return isRootFill(problem.graph.operations[operationId], problem.graph);
    });
    result.units.push_back(std::move(unit));
  }

  for (const ComputeOperation &operation : problem.graph.operations) {
    if (operation.kind == ComputeOperationKind::MatrixSetup)
      continue;
    if (!operationToUnit.contains(operation.id)) {
      operation.operation->emitError(
          "fan-out-cut current hierarchy does not cover this operation");
      return failure();
    }
  }

  SmallVector<SmallVector<int64_t>> predecessors(result.units.size());
  SmallVector<SmallVector<int64_t>> successors(result.units.size());
  for (const ComputeTensor &tensor : problem.graph.tensors) {
    for (int64_t producerId : tensor.producerOperations) {
      if (!operationToUnit.contains(producerId))
        continue;
      int64_t producerUnit = operationToUnit.lookup(producerId);
      for (int64_t consumerId : tensor.consumerOperations) {
        if (!operationToUnit.contains(consumerId))
          continue;
        int64_t consumerUnit = operationToUnit.lookup(consumerId);
        if (producerUnit == consumerUnit ||
            result.units[producerUnit].preparation ||
            result.units[consumerUnit].preparation ||
            llvm::is_contained(successors[producerUnit], consumerUnit))
          continue;
        successors[producerUnit].push_back(consumerUnit);
        predecessors[consumerUnit].push_back(producerUnit);
      }
    }
  }

  SmallVector<int64_t> indegree(result.units.size(), 0);
  std::set<std::pair<int64_t, int64_t>> ready;
  int64_t computeUnitCount = 0;
  for (const SchedulingUnit &unit : result.units) {
    if (unit.preparation) {
      result.preparationUnitIds.push_back(unit.id);
      continue;
    }
    indegree[unit.id] = static_cast<int64_t>(predecessors[unit.id].size());
    if (indegree[unit.id] == 0)
      ready.insert({unit.topologicalRank, unit.id});
    ++computeUnitCount;
  }
  llvm::sort(result.preparationUnitIds, [&](int64_t left, int64_t right) {
    return std::pair(result.units[left].topologicalRank, left) <
           std::pair(result.units[right].topologicalRank, right);
  });

  SmallVector<int64_t> unitOrder;
  while (!ready.empty()) {
    int64_t unitId = ready.begin()->second;
    ready.erase(ready.begin());
    unitOrder.push_back(unitId);
    for (int64_t successor : successors[unitId]) {
      if (--indegree[successor] == 0)
        ready.insert({result.units[successor].topologicalRank, successor});
    }
  }
  if (unitOrder.size() != static_cast<size_t>(computeUnitCount)) {
    problem.anchor->emitError(
        "fan-out-cut collapsed scheduling-unit graph contains a cycle");
    return failure();
  }

  SmallVector<int64_t> unitLevels(result.units.size(), -1);
  int64_t maximumLevel = -1;
  for (int64_t unitId : unitOrder) {
    int64_t level = 0;
    for (int64_t predecessor : predecessors[unitId]) {
      std::optional<int64_t> next =
          llvm::checkedAdd(unitLevels[predecessor], int64_t{1});
      if (!next) {
        problem.anchor->emitError("fan-out-cut level overflows int64");
        return failure();
      }
      level = std::max(level, *next);
    }
    unitLevels[unitId] = level;
    maximumLevel = std::max(maximumLevel, level);
  }
  if (maximumLevel < 0) {
    problem.anchor->emitError("fan-out-cut requires at least one compute unit");
    return failure();
  }

  result.computeLevels.resize(maximumLevel + 1);
  for (int64_t unitId : unitOrder)
    result.computeLevels[unitLevels[unitId]].push_back(unitId);
  result.computeLevels.erase(
      std::remove_if(result.computeLevels.begin(), result.computeLevels.end(),
                     [](const auto &level) { return level.empty(); }),
      result.computeLevels.end());
  return result;
}

class FanOutTreeBuilder {
public:
  explicit FanOutTreeBuilder(const MappingProblem &problem) : problem(problem) {
    tree.workUnits = problem.currentTree.workUnits;
    tree.workUnitEdges = problem.currentTree.workUnitEdges;
    for (const StructuralRATreeNode &node : problem.currentTree.nodes)
      sourceNodes[node.id] = &node;
  }

  FailureOr<ResourceAllocationTree> build(const FanOutStructure &structure) {
    SmallVector<PlannedNode> phases;
    if (structure.setupRootId >= 0) {
      FailureOr<PlannedNode> setup = cloneSubtree(structure.setupRootId);
      if (failed(setup))
        return failure();
      phases.push_back(std::move(*setup));
    }

    if (!structure.preparationUnitIds.empty()) {
      FailureOr<SmallVector<PlannedNode>> preparation =
          cloneUnits(structure, structure.preparationUnitIds);
      if (failed(preparation))
        return failure();
      phases.push_back(makeTemporal(*preparation));
    }

    size_t previousWidth = 0;
    for (const auto &level : structure.computeLevels) {
      FailureOr<SmallVector<PlannedNode>> units = cloneUnits(structure, level);
      if (failed(units))
        return failure();
      if (previousWidth == 1 && units->size() > 1) {
        FailureOr<PlannedNode> fanOut = makeParallel(*units);
        if (failed(fanOut))
          return failure();
        phases.push_back(std::move(*fanOut));
      } else {
        phases.push_back(makeTemporal(*units));
      }
      previousWidth = level.size();
    }

    PlannedNode root = makeTemporal(phases);
    tree.rootId = root.treeNodeId;
    tree.nodes[tree.rootId].parentId = -1;
    return std::move(tree);
  }

private:
  PlannedNode addLeaf(const StructuralRATreeNode &source) {
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    tree.nodes.push_back({id, RATreeNodeKind::Leaf, -1, {}, source.operationId,
                          source.workUnitId, source.workGroupCount});

    PlannedNode result;
    result.treeNodeId = id;
    const ComputeOperation &operation =
        problem.graph.operations[source.operationId];
    LogicalLaneKind lane =
        operation.requiredLane.value_or(LogicalLaneKind::Digital);
    if (lane == LogicalLaneKind::Digital) {
      result.digitalLanes = 1;
    } else if (operation.laneBindingGroup) {
      result.bindingGroups.insert(*operation.laneBindingGroup);
    } else {
      result.anonymousAnalogLanes = 1;
    }
    return result;
  }

  PlannedNode addCut(RATreeNodeKind kind, ArrayRef<PlannedNode> children,
                     int64_t workGroupCount = 1) {
    assert(children.size() >= 2 && "cuts require at least two children");
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    StructuralRATreeNode node;
    node.id = id;
    node.kind = kind;
    node.workGroupCount = workGroupCount;
    for (const PlannedNode &child : children) {
      node.childIds.push_back(child.treeNodeId);
      tree.nodes[child.treeNodeId].parentId = id;
    }
    tree.nodes.push_back(std::move(node));

    PlannedNode result;
    result.treeNodeId = id;
    for (const PlannedNode &child : children) {
      result.bindingGroups.insert(child.bindingGroups.begin(),
                                  child.bindingGroups.end());
      if (kind == RATreeNodeKind::TemporalCut) {
        result.digitalLanes = std::max(result.digitalLanes, child.digitalLanes);
        result.anonymousAnalogLanes =
            std::max(result.anonymousAnalogLanes,
                     child.anonymousAnalogLanes);
      } else {
        result.digitalLanes += child.digitalLanes;
        result.anonymousAnalogLanes += child.anonymousAnalogLanes;
      }
    }
    return result;
  }

  PlannedNode makeTemporal(ArrayRef<PlannedNode> children) {
    assert(!children.empty());
    return children.size() == 1
               ? children.front()
               : addCut(RATreeNodeKind::TemporalCut, children);
  }

  FailureOr<PlannedNode> cloneSubtree(int64_t sourceNodeId) {
    const StructuralRATreeNode *source = sourceNodes.lookup(sourceNodeId);
    if (!source) {
      problem.anchor->emitError("fan-out-cut cannot clone missing RA node ")
          << sourceNodeId;
      return failure();
    }
    if (source->kind == RATreeNodeKind::Leaf)
      return addLeaf(*source);

    SmallVector<PlannedNode> children;
    for (int64_t childId : source->childIds) {
      FailureOr<PlannedNode> child = cloneSubtree(childId);
      if (failed(child))
        return failure();
      children.push_back(std::move(*child));
    }
    return addCut(source->kind, children, source->workGroupCount);
  }

  bool canShareSpatialBatch(ArrayRef<PlannedNode> batch,
                            const PlannedNode &candidate) const {
    int64_t digitalLanes = candidate.digitalLanes;
    int64_t anonymousAnalogLanes = candidate.anonymousAnalogLanes;
    std::set<int64_t> bindingGroups = candidate.bindingGroups;
    for (const PlannedNode &child : batch) {
      digitalLanes += child.digitalLanes;
      anonymousAnalogLanes += child.anonymousAnalogLanes;
      for (int64_t binding : child.bindingGroups) {
        if (!bindingGroups.insert(binding).second)
          return false;
      }
    }
    int64_t coreCount = problem.hardware.meshRows * problem.hardware.meshCols;
    int64_t analogCapacity = coreCount * problem.hardware.arraysPerCore;
    return digitalLanes <= coreCount &&
           anonymousAnalogLanes + static_cast<int64_t>(bindingGroups.size()) <=
               analogCapacity;
  }

  FailureOr<PlannedNode> makeParallel(ArrayRef<PlannedNode> children) {
    assert(!children.empty());
    if (children.size() == 1)
      return children.front();

    SmallVector<PlannedNode> batches;
    SmallVector<PlannedNode> batch;
    auto flush = [&]() {
      if (batch.empty())
        return;
      batches.push_back(batch.size() == 1
                            ? batch.front()
                            : addCut(RATreeNodeKind::SpatialCut, batch));
      batch.clear();
    };
    for (const PlannedNode &child : children) {
      if (!canShareSpatialBatch({}, child)) {
        problem.anchor->emitError(
            "fan-out-cut unit exceeds available logical resources");
        return failure();
      }
      if (!batch.empty() && !canShareSpatialBatch(batch, child))
        flush();
      batch.push_back(child);
    }
    flush();
    return makeTemporal(batches);
  }

  FailureOr<SmallVector<PlannedNode>>
  cloneUnits(const FanOutStructure &structure, ArrayRef<int64_t> unitIds) {
    SmallVector<PlannedNode> nodes;
    nodes.reserve(unitIds.size());
    for (int64_t unitId : unitIds) {
      FailureOr<PlannedNode> node =
          cloneSubtree(structure.units[unitId].sourceRootId);
      if (failed(node))
        return failure();
      nodes.push_back(std::move(*node));
    }
    return nodes;
  }

  const MappingProblem &problem;
  ResourceAllocationTree tree;
  DenseMap<int64_t, const StructuralRATreeNode *> sourceNodes;
};

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<MappingPlan>
FanOutCutPlanner::refine(const MappingProblem &problem,
                         const MappingEvaluator &evaluator) const {
  FailureOr<FanOutStructure> structure = buildFanOutStructure(problem);
  if (failed(structure))
    return failure();

  FanOutTreeBuilder builder(problem);
  FailureOr<ResourceAllocationTree> tree = builder.build(*structure);
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
