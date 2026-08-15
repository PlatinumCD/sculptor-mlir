#include "MVMWavePlanner.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

class MVMWaveTreeBuilder {
public:
  MVMWaveTreeBuilder() = default;

  explicit MVMWaveTreeBuilder(const ResourceAllocationTree &source) {
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

  ResourceAllocationTree finish(int64_t rootId) {
    tree.rootId = rootId;
    tree.nodes[rootId].parentId = -1;
    return std::move(tree);
  }

private:
  ResourceAllocationTree tree;
};

struct SchedulingUnit {
  std::optional<int64_t> waveId;
  SmallVector<int64_t> operationIds;
  int64_t topologicalRank = std::numeric_limits<int64_t>::max();
};

FailureOr<int64_t>
buildWaveSubtree(const MVMWave &wave, const ComputeGraph &graph,
                 const DenseMap<int64_t, int64_t> &topologicalRank,
                 MVMWaveTreeBuilder &builder, Operation *anchor) {
  auto byTopologicalRank = [&](int64_t lhs, int64_t rhs) {
    return std::pair(topologicalRank.lookup(lhs), lhs) <
           std::pair(topologicalRank.lookup(rhs), rhs);
  };

  SmallVector<int64_t> vectorTiles = wave.vectorTileOperationIds;
  llvm::sort(vectorTiles, byTopologicalRank);

  SmallVector<int64_t> physicalMVMs = wave.physicalMVMOperationIds;
  llvm::sort(physicalMVMs, [&](int64_t lhs, int64_t rhs) {
    const ComputeOperation &left = graph.operations[lhs];
    const ComputeOperation &right = graph.operations[rhs];
    return std::pair(left.mvmWaveMember.value_or(-1), lhs) <
           std::pair(right.mvmWaveMember.value_or(-1), rhs);
  });
  if (physicalMVMs.empty()) {
    anchor->emitError("MVM-wave planner found a wave without physical MVMs");
    return failure();
  }

  SmallVector<int64_t> phases;
  phases.reserve(vectorTiles.size() + 3);
  for (int64_t operationId : vectorTiles)
    phases.push_back(builder.addLeaf(operationId));

  SmallVector<int64_t> spatialBranches;
  spatialBranches.reserve(physicalMVMs.size());
  for (int64_t physicalMVMId : physicalMVMs) {
    const ComputeOperation &physicalMVM = graph.operations[physicalMVMId];
    if (!physicalMVM.mvmWaveMember) {
      physicalMVM.operation->emitError(
          "physical MVM is missing its wave-member identity");
      return failure();
    }
    spatialBranches.push_back(builder.addLeaf(physicalMVMId));
  }
  phases.push_back(
      spatialBranches.size() == 1
          ? spatialBranches.front()
          : builder.addCut(RATreeNodeKind::SpatialCut, spatialBranches));

  if (wave.recombineOperationId)
    phases.push_back(builder.addLeaf(*wave.recombineOperationId));
  if (wave.biasAddOperationId)
    phases.push_back(builder.addLeaf(*wave.biasAddOperationId));

  return phases.size() == 1
             ? phases.front()
             : builder.addCut(RATreeNodeKind::TemporalCut, phases);
}

FailureOr<ResourceAllocationTree>
buildMVMWaveTree(const MappingProblem &problem,
                 ArrayRef<int64_t> includedOperationIds) {
  if (includedOperationIds.empty()) {
    problem.anchor->emitError(
        "MVM-wave planner requires at least one compute operation");
    return failure();
  }
  if (problem.graph.mvmWaves.empty()) {
    problem.anchor->emitError(
        "MVM-wave planner requires at least one expanded MVM wave");
    return failure();
  }

  DenseSet<int64_t> includedOperations;
  for (int64_t operationId : includedOperationIds) {
    if (operationId < 0 ||
        operationId >= static_cast<int64_t>(problem.graph.operations.size())) {
      problem.anchor->emitError(
          "MVM-wave refinement references unknown operation ")
          << operationId;
      return failure();
    }
    if (!includedOperations.insert(operationId).second) {
      problem.anchor->emitError(
          "MVM-wave refinement contains duplicate operation ")
          << operationId;
      return failure();
    }
  }

  DenseMap<int64_t, int64_t> topologicalRank;
  for (auto [rank, operationId] :
       llvm::enumerate(problem.graph.topologicalOrder))
    topologicalRank[operationId] = static_cast<int64_t>(rank);

  DenseMap<int64_t, SmallVector<const MappingWorkUnit *>> workUnitsByOperation;
  for (const MappingWorkUnit &workUnit : problem.currentTree.workUnits)
    workUnitsByOperation[workUnit.operationId].push_back(&workUnit);
  for (auto &[operationId, workUnits] : workUnitsByOperation) {
    llvm::sort(workUnits,
               [](const MappingWorkUnit *left,
                  const MappingWorkUnit *right) { return left->id < right->id; });
  }

  std::map<std::pair<int64_t, int64_t>, int64_t> workGroupCounts;
  for (const StructuralRATreeNode &node : problem.currentTree.nodes) {
    if (node.kind == RATreeNodeKind::Leaf)
      workGroupCounts[{node.operationId, node.workUnitId}] =
          node.workGroupCount;
  }
  auto getWorkGroupCount = [&](int64_t operationId, int64_t workUnitId) {
    auto count = workGroupCounts.find({operationId, workUnitId});
    return count == workGroupCounts.end() ? int64_t{1} : count->second;
  };

  DenseMap<int64_t, const MVMWave *> wavesById;
  for (const MVMWave &wave : problem.graph.mvmWaves) {
    if (!wavesById.try_emplace(wave.id, &wave).second) {
      problem.anchor->emitError("MVM-wave planner found duplicate wave ID ")
          << wave.id;
      return failure();
    }
  }

  SmallVector<SchedulingUnit> units;
  DenseMap<int64_t, int64_t> operationToUnit;
  DenseMap<int64_t, int64_t> waveToUnit;
  for (int64_t operationId : problem.graph.topologicalOrder) {
    if (!includedOperations.contains(operationId))
      continue;
    const ComputeOperation &operation = problem.graph.operations[operationId];
    if (!operation.mvmWaveId) {
      int64_t unitId = static_cast<int64_t>(units.size());
      SchedulingUnit unit;
      unit.operationIds.push_back(operationId);
      unit.topologicalRank = topologicalRank.lookup(operationId);
      units.push_back(std::move(unit));
      operationToUnit[operationId] = unitId;
      continue;
    }

    if (waveToUnit.contains(*operation.mvmWaveId))
      continue;
    const MVMWave *wave = wavesById.lookup(*operation.mvmWaveId);
    if (!wave) {
      operation.operation->emitError("operation references unknown MVM wave ")
          << *operation.mvmWaveId;
      return failure();
    }

    int64_t unitId = static_cast<int64_t>(units.size());
    SchedulingUnit unit;
    unit.waveId = wave->id;
    unit.operationIds.append(wave->vectorTileOperationIds.begin(),
                             wave->vectorTileOperationIds.end());
    unit.operationIds.append(wave->physicalMVMOperationIds.begin(),
                             wave->physicalMVMOperationIds.end());
    if (wave->recombineOperationId)
      unit.operationIds.push_back(*wave->recombineOperationId);
    if (wave->biasAddOperationId)
      unit.operationIds.push_back(*wave->biasAddOperationId);
    llvm::sort(unit.operationIds);
    unit.operationIds.erase(
        std::unique(unit.operationIds.begin(), unit.operationIds.end()),
        unit.operationIds.end());
    for (int64_t memberId : unit.operationIds) {
      if (memberId < 0 ||
          memberId >= static_cast<int64_t>(problem.graph.operations.size())) {
        problem.anchor->emitError("MVM wave references unknown operation ")
            << memberId;
        return failure();
      }
      if (!includedOperations.contains(memberId)) {
        operation.operation->emitError(
            "MVM-wave strategy cannot refine only part of wave ")
            << wave->id;
        return failure();
      }
      if (!operationToUnit.try_emplace(memberId, unitId).second) {
        problem.graph.operations[memberId].operation->emitError(
            "operation belongs to multiple MVM scheduling units");
        return failure();
      }
      unit.topologicalRank =
          std::min(unit.topologicalRank, topologicalRank.lookup(memberId));
    }
    units.push_back(std::move(unit));
    waveToUnit[wave->id] = unitId;
  }

  if (operationToUnit.size() != includedOperations.size()) {
    problem.anchor->emitError(
        "MVM-wave planner did not assign every operation to a scheduling "
        "unit");
    return failure();
  }

  SmallVector<SmallVector<int64_t>> successors(units.size());
  SmallVector<int64_t> indegree(units.size(), 0);
  DenseSet<std::pair<int64_t, int64_t>> unitEdges;
  for (const ComputeTensor &tensor : problem.graph.tensors) {
    for (int64_t producerId : tensor.producerOperations) {
      if (!includedOperations.contains(producerId))
        continue;
      int64_t producerUnit = operationToUnit.lookup(producerId);
      for (int64_t consumerId : tensor.consumerOperations) {
        if (!includedOperations.contains(consumerId))
          continue;
        int64_t consumerUnit = operationToUnit.lookup(consumerId);
        if (producerUnit == consumerUnit ||
            !unitEdges.insert({producerUnit, consumerUnit}).second)
          continue;
        successors[producerUnit].push_back(consumerUnit);
        ++indegree[consumerUnit];
      }
    }
  }

  std::set<std::pair<int64_t, int64_t>> ready;
  for (auto [unitId, degree] : llvm::enumerate(indegree)) {
    if (degree == 0)
      ready.insert(
          {units[unitId].topologicalRank, static_cast<int64_t>(unitId)});
  }
  SmallVector<int64_t> unitOrder;
  while (!ready.empty()) {
    int64_t unitId = ready.begin()->second;
    ready.erase(ready.begin());
    unitOrder.push_back(unitId);
    for (int64_t successor : successors[unitId]) {
      if (--indegree[successor] == 0)
        ready.insert({units[successor].topologicalRank, successor});
    }
  }
  if (unitOrder.size() != units.size()) {
    problem.anchor->emitError(
        "MVM-wave grouping creates a cycle between scheduling units");
    return failure();
  }

  MVMWaveTreeBuilder builder(problem.currentTree);
  SmallVector<int64_t> unitRoots;
  unitRoots.reserve(unitOrder.size());
  for (int64_t unitId : unitOrder) {
    const SchedulingUnit &unit = units[unitId];
    if (!unit.waveId) {
      int64_t operationId = unit.operationIds.front();
      auto workUnits = workUnitsByOperation.find(operationId);
      if (workUnits == workUnitsByOperation.end()) {
        unitRoots.push_back(builder.addLeaf(
            operationId, getWorkGroupCount(operationId, -1)));
        continue;
      }

      SmallVector<int64_t> workerLeaves;
      workerLeaves.reserve(workUnits->second.size());
      for (const MappingWorkUnit *workUnit : workUnits->second) {
        workerLeaves.push_back(builder.addLeaf(
            operationId, getWorkGroupCount(operationId, workUnit->id),
            workUnit->id));
      }
      unitRoots.push_back(
          workerLeaves.size() == 1
              ? workerLeaves.front()
              : builder.addCut(RATreeNodeKind::SpatialCut, workerLeaves));
      continue;
    }
    for (int64_t operationId : unit.operationIds) {
      if (workUnitsByOperation.contains(operationId)) {
        problem.graph.operations[operationId].operation->emitError(
            "MVM-wave operations cannot carry expanded digital work units");
        return failure();
      }
    }
    FailureOr<int64_t> waveRoot =
        buildWaveSubtree(*wavesById.lookup(*unit.waveId), problem.graph,
                         topologicalRank, builder, problem.anchor);
    if (failed(waveRoot))
      return failure();
    unitRoots.push_back(*waveRoot);
  }

  int64_t rootId = unitRoots.size() == 1
                       ? unitRoots.front()
                       : builder.addCut(RATreeNodeKind::TemporalCut, unitRoots);
  ResourceAllocationTree tree = builder.finish(rootId);
  FailureOr<ResourceAllocationTree> reindexed =
      reindexResourceAllocationTree(tree, problem.anchor);
  if (failed(reindexed))
    return failure();
  if (includedOperations.size() == problem.graph.operations.size() &&
      failed(verifyResourceAllocationTree(*reindexed, problem.graph,
                                          problem.anchor)))
    return failure();
  return *reindexed;
}

void collectSubtreeOperations(
    int64_t nodeId,
    const DenseMap<int64_t, const StructuralRATreeNode *> &nodes,
    DenseSet<int64_t> &operationIds) {
  const StructuralRATreeNode *node = nodes.lookup(nodeId);
  assert(node && "verified RA tree must contain every referenced node");
  if (node->kind == RATreeNodeKind::Leaf) {
    operationIds.insert(node->operationId);
    return;
  }
  for (int64_t childId : node->childIds)
    collectSubtreeOperations(childId, nodes, operationIds);
}

struct SetupFirstPartition {
  int64_t setupRootId = -1;
  SmallVector<int64_t> computeOperationIds;
};

std::optional<SetupFirstPartition>
findSetupFirstPartition(const MappingProblem &problem) {
  DenseMap<int64_t, const StructuralRATreeNode *> nodes;
  for (const StructuralRATreeNode &node : problem.currentTree.nodes)
    nodes[node.id] = &node;

  const StructuralRATreeNode *root = nodes.lookup(problem.currentTree.rootId);
  if (!root || root->kind != RATreeNodeKind::TemporalCut ||
      root->childIds.size() != 2)
    return std::nullopt;

  DenseSet<int64_t> expectedSetups;
  for (const ComputeOperation &operation : problem.graph.operations) {
    if (operation.kind == ComputeOperationKind::MatrixSetup)
      expectedSetups.insert(operation.id);
  }
  if (expectedSetups.empty())
    return std::nullopt;

  DenseSet<int64_t> actualSetups;
  collectSubtreeOperations(root->childIds.front(), nodes, actualSetups);
  if (actualSetups.size() != expectedSetups.size())
    return std::nullopt;
  for (int64_t operationId : expectedSetups) {
    if (!actualSetups.contains(operationId))
      return std::nullopt;
  }

  SetupFirstPartition partition;
  partition.setupRootId = root->childIds.front();
  for (int64_t operationId : problem.graph.topologicalOrder) {
    if (!expectedSetups.contains(operationId))
      partition.computeOperationIds.push_back(operationId);
  }
  return partition;
}

FailureOr<int64_t>
cloneSubtree(int64_t nodeId,
             const DenseMap<int64_t, const StructuralRATreeNode *> &nodes,
             MVMWaveTreeBuilder &builder, Operation *anchor) {
  const StructuralRATreeNode *node = nodes.lookup(nodeId);
  if (!node) {
    anchor->emitError("cannot clone missing RA-tree node ") << nodeId;
    return failure();
  }
  if (node->kind == RATreeNodeKind::Leaf)
    return builder.addLeaf(node->operationId, node->workGroupCount,
                           node->workUnitId);

  SmallVector<int64_t> children;
  for (int64_t childId : node->childIds) {
    FailureOr<int64_t> child = cloneSubtree(childId, nodes, builder, anchor);
    if (failed(child))
      return failure();
    children.push_back(*child);
  }
  return builder.addCut(node->kind, children, node->workGroupCount);
}

FailureOr<ResourceAllocationTree>
buildMVMWaveRefinement(const MappingProblem &problem) {
  std::optional<SetupFirstPartition> setupFirst =
      findSetupFirstPartition(problem);
  if (!setupFirst)
    return buildMVMWaveTree(problem, problem.graph.topologicalOrder);

  FailureOr<ResourceAllocationTree> computeTree =
      buildMVMWaveTree(problem, setupFirst->computeOperationIds);
  if (failed(computeTree))
    return failure();

  DenseMap<int64_t, const StructuralRATreeNode *> baselineNodes;
  for (const StructuralRATreeNode &node : problem.currentTree.nodes)
    baselineNodes[node.id] = &node;
  DenseMap<int64_t, const StructuralRATreeNode *> computeNodes;
  for (const StructuralRATreeNode &node : computeTree->nodes)
    computeNodes[node.id] = &node;

  MVMWaveTreeBuilder builder(problem.currentTree);
  FailureOr<int64_t> setupRoot = cloneSubtree(
      setupFirst->setupRootId, baselineNodes, builder, problem.anchor);
  FailureOr<int64_t> computeRoot =
      cloneSubtree(computeTree->rootId, computeNodes, builder, problem.anchor);
  if (failed(setupRoot) || failed(computeRoot))
    return failure();

  int64_t root =
      builder.addCut(RATreeNodeKind::TemporalCut, {*setupRoot, *computeRoot});
  ResourceAllocationTree tree = builder.finish(root);
  FailureOr<ResourceAllocationTree> reindexed =
      reindexResourceAllocationTree(tree, problem.anchor);
  if (failed(reindexed) || failed(verifyResourceAllocationTree(
                               *reindexed, problem.graph, problem.anchor)))
    return failure();
  return *reindexed;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<MappingPlan>
MVMWavePlanner::refine(const MappingProblem &problem,
                       const MappingEvaluator &evaluator) const {
  FailureOr<ResourceAllocationTree> tree = buildMVMWaveRefinement(problem);
  if (failed(tree))
    return failure();
  FailureOr<MappingEvaluation> evaluation = evaluator.evaluate(problem, *tree);
  if (failed(evaluation))
    return failure();

  MappingPlan plan;
  plan.plannerName = getName().str();
  plan.objective = problem.objective;
  plan.selectedTree = std::move(*tree);
  plan.evaluation = *evaluation;
  plan.candidates.push_back(
      {/*name=*/"mvm-wave", *evaluation, /*selected=*/true});
  return plan;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
