#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphDigitalShardPlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphReductionPlacement.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace {

static mlir::InFlightDiagnostic emitPlacementError(
    const mlir::sculptor::task_schedulers::TaskGraphPlacementProblem &problem) {
  return problem.diagnosticOp->emitError();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

LogicalResult validatePlacementPlan(const TaskGraphPlacementProblem &problem,
                                    const IslandPlacementPlan &plan) {
  if (!problem.taskGraphFunc || !problem.diagnosticOp)
    return failure();

  if (plan.placements.size() != problem.islandGraph.islands.size()) {
    emitPlacementError(problem)
        << "expected placement plan to contain one placement per logical "
           "island";
    return failure();
  }

  llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<int64_t, int64_t>, 4>>
      laneCoresByTree;
  llvm::DenseMap<int64_t, unsigned> rootCountByTree;
  llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<int64_t, int64_t>, 8>>
      shardCoresByGroup;
  for (auto indexedIsland : llvm::enumerate(problem.islandGraph.islands)) {
    const auto &island = indexedIsland.value();
    const LogicalIslandPlacement &placement =
        plan.placements[indexedIsland.index()];
    if (placement.coreId < 0 || placement.coreId >= problem.budget.numCores) {
      emitPlacementError(problem)
          << "expected placement plan core " << placement.coreId
          << " to belong to the hardware budget";
      return failure();
    }

    if (task_graph::isAnalogIsland(island)) {
      if (!placement.physicalArrayId) {
        emitPlacementError(problem)
            << "expected analog island " << island.islandIndex
            << " to own a physical array";
        return failure();
      }
      int64_t physicalArrayId = *placement.physicalArrayId;
      if (physicalArrayId < 0 ||
          physicalArrayId >= problem.budget.numAnalogArrays ||
          !llvm::is_contained(problem.budget.analogArrays, physicalArrayId) ||
          physicalArrayId / problem.budget.arraysPerCore != placement.coreId) {
        emitPlacementError(problem)
            << "expected analog island physical array " << physicalArrayId
            << " to belong to its assigned core and hardware budget";
        return failure();
      }
      continue;
    }

    if (task_graph::isDigitalShardIsland(island)) {
      if (placement.physicalArrayId || !island.distributionGroupId ||
          !island.distributionShardId || !island.distributionShardCount ||
          !island.distributionPlacement) {
        emitPlacementError(problem)
            << "expected digital shard island " << island.islandIndex
            << " to own a core, no analog array, and complete distribution "
               "metadata";
        return failure();
      }
      shardCoresByGroup[*island.distributionGroupId].push_back(
          {*island.distributionShardId, placement.coreId});
      continue;
    }

    if (placement.physicalArrayId || !island.reductionTreeId ||
        !island.reductionLevel || !island.reductionWidth) {
      emitPlacementError(problem)
          << "expected reduction island " << island.islandIndex
          << " to own a core, no analog array, and complete tree metadata";
      return failure();
    }
    if (task_graph::isReductionRootIsland(island)) {
      ++rootCountByTree[*island.reductionTreeId];
      continue;
    }
    if (!task_graph::isReductionLaneIsland(island) || !island.reductionLane) {
      emitPlacementError(problem)
          << "expected a reduction island to be a first-stage lane or root";
      return failure();
    }
    laneCoresByTree[*island.reductionTreeId].push_back(
        {*island.reductionLane, placement.coreId});
  }

  for (const auto &entry : rootCountByTree) {
    if (entry.second != 1) {
      emitPlacementError(problem)
          << "expected reduction tree " << entry.first
          << " to contain one independently placed root";
      return failure();
    }
  }
  for (const auto &entry : laneCoresByTree) {
    llvm::DenseSet<int64_t> seenCores;
    llvm::DenseSet<int64_t> seenLanes;
    for (const auto &[lane, core] : entry.second) {
      if (!seenLanes.insert(lane).second) {
        emitPlacementError(problem) << "expected one island for reduction tree "
                                    << entry.first << " lane " << lane;
        return failure();
      }
      if (!seenCores.insert(core).second) {
        emitPlacementError(problem)
            << "expected active lanes in reduction tree " << entry.first
            << " to use distinct cores";
        return failure();
      }
    }
    if (rootCountByTree.lookup(entry.first) != 1) {
      emitPlacementError(problem)
          << "expected reduction tree " << entry.first
          << " to contain one independently placed root";
      return failure();
    }
  }

  for (const DistributedShardPlacementConstraint &group :
       problem.constraints.distributedShardGroups) {
    auto placements = shardCoresByGroup.find(group.groupId);
    if (placements == shardCoresByGroup.end() ||
        static_cast<int64_t>(placements->second.size()) != group.shardCount) {
      emitPlacementError(problem)
          << "expected one placement for every shard in digital distribution "
             "group "
          << group.groupId;
      return failure();
    }
    llvm::DenseSet<int64_t> seenShardIds;
    llvm::DenseSet<int64_t> seenCores;
    for (const auto &[shardId, coreId] : placements->second) {
      if (!seenShardIds.insert(shardId).second) {
        emitPlacementError(problem)
            << "expected unique shard IDs in digital distribution group "
            << group.groupId;
        return failure();
      }
      if (group.policy == DistributionPlacementPolicy::RequireDistinct &&
          !seenCores.insert(coreId).second) {
        emitPlacementError(problem)
            << "expected digital distribution group " << group.groupId
            << " to use distinct cores";
        return failure();
      }
    }
  }

  return success();
}

FailureOr<IslandPlacementPlan> buildPlacementPlanFromPhysicalArrayOrder(
    const TaskGraphPlacementProblem &problem,
    llvm::ArrayRef<int64_t> physicalArrayOrder) {
  auto resources = buildIslandPlacementResources(problem, physicalArrayOrder);
  if (failed(resources))
    return failure();

  llvm::DenseMap<unsigned, int64_t> physicalArrayByAnalogIsland;
  size_t nextArray = 0;
  for (const auto &island : problem.islandGraph.islands) {
    if (!task_graph::isAnalogIsland(island))
      continue;
    if (resources->analogPhysicalArrayOrder.empty()) {
      emitPlacementError(problem)
          << "expected analog island placement to have at least one physical "
             "array";
      return failure();
    }
    physicalArrayByAnalogIsland[island.islandIndex] =
        resources->analogPhysicalArrayOrder
            [nextArray++ % resources->analogPhysicalArrayOrder.size()];
  }
  return buildPlacementPlanFromAnalogPlacements(problem, *resources,
                                                physicalArrayByAnalogIsland);
}

FailureOr<IslandPlacementResources>
buildIslandPlacementResources(const TaskGraphPlacementProblem &problem,
                              llvm::ArrayRef<int64_t> physicalArrayOrder) {
  bool hasAnalogIsland =
      llvm::any_of(problem.islandGraph.islands, [](const auto &island) {
        return task_graph::isAnalogIsland(island);
      });
  if (hasAnalogIsland && physicalArrayOrder.empty()) {
    emitPlacementError(problem)
        << "expected analog island placement to have a physical array order";
    return failure();
  }

  IslandPlacementResources resources;
  resources.analogPhysicalArrayOrder.append(physicalArrayOrder.begin(),
                                            physicalArrayOrder.end());
  return resources;
}

FailureOr<IslandPlacementPlan> buildPlacementPlanFromAnalogPlacements(
    const TaskGraphPlacementProblem &problem,
    const IslandPlacementResources &resources,
    const llvm::DenseMap<unsigned, int64_t> &physicalArrayByAnalogIsland) {
  llvm::DenseMap<unsigned, int64_t> coreByAnalogIsland;
  for (const auto &island : problem.islandGraph.islands) {
    if (!task_graph::isAnalogIsland(island))
      continue;
    auto physicalArray = physicalArrayByAnalogIsland.find(island.islandIndex);
    if (physicalArray == physicalArrayByAnalogIsland.end() ||
        !llvm::is_contained(resources.analogPhysicalArrayOrder,
                            physicalArray->second)) {
      emitPlacementError(problem)
          << "expected a physical array for analog island "
          << island.islandIndex;
      return failure();
    }
    coreByAnalogIsland[island.islandIndex] =
        physicalArray->second / problem.budget.arraysPerCore;
  }

  auto reductionCoreByIsland =
      buildSpatialReductionCorePlacements(problem, coreByAnalogIsland);
  if (failed(reductionCoreByIsland))
    return failure();
  llvm::DenseMap<unsigned, int64_t> initialCoreByIsland = coreByAnalogIsland;
  for (const auto &entry : *reductionCoreByIsland)
    initialCoreByIsland[entry.first] = entry.second;
  auto digitalShardCoreByIsland =
      buildDigitalShardCorePlacements(problem, initialCoreByIsland);
  if (failed(digitalShardCoreByIsland))
    return failure();

  IslandPlacementPlan plan;
  plan.placements.reserve(problem.islandGraph.islands.size());
  for (const auto &island : problem.islandGraph.islands) {
    if (task_graph::isReductionIsland(island)) {
      auto core = reductionCoreByIsland->find(island.islandIndex);
      if (core == reductionCoreByIsland->end()) {
        emitPlacementError(problem)
            << "expected a core for reduction island " << island.islandIndex;
        return failure();
      }
      plan.placements.push_back(
          LogicalIslandPlacement{core->second, std::nullopt});
      continue;
    }

    if (task_graph::isDigitalShardIsland(island)) {
      auto core = digitalShardCoreByIsland->find(island.islandIndex);
      if (core == digitalShardCoreByIsland->end()) {
        emitPlacementError(problem)
            << "expected a core for digital shard island "
            << island.islandIndex;
        return failure();
      }
      plan.placements.push_back(
          LogicalIslandPlacement{core->second, std::nullopt});
      continue;
    }

    auto physicalArray = physicalArrayByAnalogIsland.find(island.islandIndex);
    if (physicalArray == physicalArrayByAnalogIsland.end()) {
      emitPlacementError(problem)
          << "expected a physical array for analog island "
          << island.islandIndex;
      return failure();
    }
    int64_t coreId = physicalArray->second / problem.budget.arraysPerCore;
    plan.placements.push_back(
        LogicalIslandPlacement{coreId, physicalArray->second});
  }

  if (failed(validatePlacementPlan(problem, plan)))
    return failure();
  return plan;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
