#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphPlacementPlan.h"

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

  llvm::DenseMap<int64_t, int64_t> reductionCoreByLane;
  llvm::DenseSet<int64_t> reductionCores;
  llvm::DenseSet<int64_t> analogCores;
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
      analogCores.insert(placement.coreId);
      continue;
    }

    if (placement.physicalArrayId || !island.reductionLane) {
      emitPlacementError(problem)
          << "expected reduction island " << island.islandIndex
          << " to own a core but no analog array";
      return failure();
    }
    auto laneCore = reductionCoreByLane.try_emplace(*island.reductionLane,
                                                    placement.coreId);
    if (!laneCore.second && laneCore.first->second != placement.coreId) {
      emitPlacementError(problem)
          << "expected reduction lane " << *island.reductionLane
          << " to reuse one dedicated core";
      return failure();
    }
    reductionCores.insert(placement.coreId);
  }

  if (reductionCores.size() != reductionCoreByLane.size()) {
    emitPlacementError(problem)
        << "expected distinct reduction lanes to use distinct cores";
    return failure();
  }
  for (int64_t core : reductionCores) {
    if (analogCores.contains(core)) {
      emitPlacementError(problem) << "expected reduction core " << core
                                  << " to remain exclusive from analog islands";
      return failure();
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
             "array outside the reduction core pool";
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
  if (!problem.islandGraph.islands.empty() && physicalArrayOrder.empty()) {
    emitPlacementError(problem)
        << "expected logical island placement to have a physical array order";
    return failure();
  }

  int64_t reductionWidth = 0;
  for (const auto &island : problem.islandGraph.islands) {
    if (!task_graph::isReductionIsland(island))
      continue;
    if (!island.reductionWidth || !island.reductionLane) {
      emitPlacementError(problem)
          << "expected reduction island to carry width and lane metadata";
      return failure();
    }
    reductionWidth = std::max(reductionWidth, *island.reductionWidth);
  }

  llvm::SmallVector<int64_t, 8> reductionCoreByLane;
  llvm::DenseSet<int64_t> seenCores;
  for (int64_t physicalArrayId : physicalArrayOrder) {
    int64_t coreId = physicalArrayId / problem.budget.arraysPerCore;
    if (!seenCores.insert(coreId).second)
      continue;
    if (static_cast<int64_t>(reductionCoreByLane.size()) < reductionWidth)
      reductionCoreByLane.push_back(coreId);
  }
  if (static_cast<int64_t>(reductionCoreByLane.size()) != reductionWidth) {
    emitPlacementError(problem) << "expected at least " << reductionWidth
                                << " cores for the reduction core pool";
    return failure();
  }

  IslandPlacementResources resources;
  llvm::DenseSet<int64_t> reductionCores(reductionCoreByLane.begin(),
                                         reductionCoreByLane.end());
  for (int64_t physicalArrayId : physicalArrayOrder) {
    int64_t coreId = physicalArrayId / problem.budget.arraysPerCore;
    if (!reductionCores.contains(coreId))
      resources.analogPhysicalArrayOrder.push_back(physicalArrayId);
  }
  for (const auto &island : problem.islandGraph.islands) {
    if (!task_graph::isReductionIsland(island))
      continue;
    if (*island.reductionLane >=
        static_cast<int64_t>(reductionCoreByLane.size())) {
      emitPlacementError(problem)
          << "expected reduction lane to fit the reduction core pool";
      return failure();
    }
    resources.reductionCoreByIsland[island.islandIndex] =
        reductionCoreByLane[*island.reductionLane];
  }
  return resources;
}

FailureOr<IslandPlacementPlan> buildPlacementPlanFromAnalogPlacements(
    const TaskGraphPlacementProblem &problem,
    const IslandPlacementResources &resources,
    const llvm::DenseMap<unsigned, int64_t> &physicalArrayByAnalogIsland) {
  IslandPlacementPlan plan;
  plan.placements.reserve(problem.islandGraph.islands.size());
  for (const auto &island : problem.islandGraph.islands) {
    if (task_graph::isReductionIsland(island)) {
      auto core = resources.reductionCoreByIsland.find(island.islandIndex);
      if (core == resources.reductionCoreByIsland.end()) {
        emitPlacementError(problem)
            << "expected a core for reduction island " << island.islandIndex;
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
