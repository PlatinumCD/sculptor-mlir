#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"

#include "mlir/IR/Builders.h"

#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

FailureOr<int64_t> getPhysicalTileCapacity(const PhysicalMeshGeometry &mesh,
                                           Operation *anchor) {
  if (mesh.rows <= 0 || mesh.columns <= 0 || mesh.arraysPerCore <= 0) {
    anchor->emitError("physical logical-tile placement requires positive mesh "
                      "dimensions and arrays per core");
    return failure();
  }
  std::optional<int64_t> capacity = llvm::checkedMul(mesh.rows, mesh.columns);
  if (!capacity) {
    anchor->emitError("physical mesh capacity overflow");
    return failure();
  }
  return *capacity;
}

PhysicalTileLocation getLocation(int64_t physicalTileId,
                                 const PhysicalMeshGeometry &mesh) {
  return {physicalTileId, physicalTileId / mesh.columns,
          physicalTileId % mesh.columns};
}

int64_t getManhattanDistance(const PhysicalTileLocation &source,
                             const PhysicalTileLocation &target) {
  return std::abs(source.row - target.row) +
         std::abs(source.column - target.column);
}

FailureOr<int64_t> checkedTransferCost(int64_t byteSize, int64_t hops,
                                       Operation *anchor) {
  if (byteSize < 0 || hops < 0) {
    anchor->emitError("logical-tile transfer cost requires nonnegative bytes "
                      "and Manhattan hops");
    return failure();
  }
  std::optional<int64_t> result = llvm::checkedMul(byteSize, hops);
  if (!result) {
    anchor->emitError("logical-tile transfer cost overflow");
    return failure();
  }
  return *result;
}

FailureOr<int64_t> checkedAddCost(int64_t current, int64_t increment,
                                  Operation *anchor) {
  std::optional<int64_t> result = llvm::checkedAdd(current, increment);
  if (!result) {
    anchor->emitError("logical-tile placement score overflow");
    return failure();
  }
  return *result;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

LogicalResult initializeLogicalTilePlacementProblem(
    const ComputeGraph &computeGraph,
    const ResourceAllocationTree &resourceAllocationTree,
    MappingCostProfile &costProfileStorage,
    LogicalTilePlacementProblem &problem) {
  if (!problem.anchor)
    return failure();

  MappingHardwareModel hardware;
  hardware.meshRows = problem.mesh.rows;
  hardware.meshCols = problem.mesh.columns;
  hardware.arraysPerCore = problem.mesh.arraysPerCore;
  auto profileAttr = problem.anchor->getAttrOfType<DictionaryAttr>(
      kMappingCostProfileAttrName);
  if (profileAttr) {
    if (auto clock = profileAttr.getAs<IntegerAttr>("clock_frequency_hz"))
      hardware.clockFrequencyHz = clock.getInt();
    if (auto network = profileAttr.getAs<DictionaryAttr>("network")) {
      if (auto wordBits = network.getAs<IntegerAttr>("word_bits"))
        hardware.networkWordBits = wordBits.getInt();
    }
  }

  FailureOr<MappingCostProfile> resolvedProfile =
      profileAttr
          ? deserializeMappingCostProfile(profileAttr, hardware, problem.anchor)
          : FailureOr<MappingCostProfile>(
                getLegacyMappingCostProfile(hardware));
  if (failed(resolvedProfile))
    return failure();

  costProfileStorage = std::move(*resolvedProfile);
  problem.computeGraph = &computeGraph;
  problem.raTree = &resourceAllocationTree;
  problem.costProfile = &costProfileStorage;
  return success();
}

LogicalResult initializeLogicalTilePlacementProblemFromPlan(
    LogicalTilePlacementAttr placementAttr, const ComputeGraph &computeGraph,
    const ResourceAllocationTree &resourceAllocationTree,
    MappingCostProfile &costProfileStorage,
    LogicalTilePlacementProblem &problem) {
  problem.mesh = {placementAttr.getMeshRows().getInt(),
                  placementAttr.getMeshCols().getInt(),
                  placementAttr.getArraysPerCore().getInt()};
  FailureOr<PlacementObjectiveKind> objective = parsePlacementObjective(
      placementAttr.getObjective().getValue(), problem.anchor);
  FailureOr<TemporalNetworkMode> networkMode = parseTemporalNetworkMode(
      placementAttr.getNetworkMode().getValue(), problem.anchor);
  FailureOr<TemporalTimingScope> timingScope = parseTemporalTimingScope(
      placementAttr.getTimingScope().getValue(), problem.anchor);
  if (failed(objective) || failed(networkMode) || failed(timingScope) ||
      failed(initializeLogicalTilePlacementProblem(
          computeGraph, resourceAllocationTree, costProfileStorage, problem)))
    return failure();

  problem.objective = *objective;
  problem.networkMode = *networkMode;
  problem.timingScope = *timingScope;
  return success();
}

FailureOr<LogicalTileScheduleKind>
parseLogicalTileSchedule(StringRef value, Operation *anchor,
                         bool allowAnnealing) {
  if (value == "random")
    return LogicalTileScheduleKind::Random;
  if (value == "snake")
    return LogicalTileScheduleKind::Snake;
  if (value == "greedy")
    return LogicalTileScheduleKind::Greedy;
  if (value == "annealing" && allowAnnealing)
    return LogicalTileScheduleKind::Annealing;
  anchor->emitError("unknown logical-tile placement schedule '")
      << value << "'";
  return failure();
}

StringRef stringifyLogicalTileSchedule(LogicalTileScheduleKind schedule) {
  switch (schedule) {
  case LogicalTileScheduleKind::Random:
    return "random";
  case LogicalTileScheduleKind::Snake:
    return "snake";
  case LogicalTileScheduleKind::Greedy:
    return "greedy";
  case LogicalTileScheduleKind::Annealing:
    return "annealing";
  }
  llvm_unreachable("unknown logical-tile placement schedule");
}

FailureOr<GreedyTileOrder> parseGreedyTileOrder(StringRef value,
                                                Operation *anchor) {
  if (value == "sequential")
    return GreedyTileOrder::Sequential;
  if (value == "priority")
    return GreedyTileOrder::Priority;
  anchor->emitError("unknown greedy logical-tile order '") << value << "'";
  return failure();
}

StringRef stringifyGreedyTileOrder(GreedyTileOrder order) {
  switch (order) {
  case GreedyTileOrder::Sequential:
    return "sequential";
  case GreedyTileOrder::Priority:
    return "priority";
  }
  llvm_unreachable("unknown greedy logical-tile order");
}

FailureOr<GreedyPriorityMode> parseGreedyPriorityMode(StringRef value,
                                                      Operation *anchor) {
  if (value == "sum")
    return GreedyPriorityMode::Sum;
  if (value == "max")
    return GreedyPriorityMode::Max;
  anchor->emitError("unknown greedy priority mode '") << value << "'";
  return failure();
}

StringRef stringifyGreedyPriorityMode(GreedyPriorityMode mode) {
  switch (mode) {
  case GreedyPriorityMode::Sum:
    return "sum";
  case GreedyPriorityMode::Max:
    return "max";
  }
  llvm_unreachable("unknown greedy priority mode");
}

FailureOr<GreedyCandidateScope> parseGreedyCandidateScope(StringRef value,
                                                          Operation *anchor) {
  if (value == "cardinal")
    return GreedyCandidateScope::Cardinal;
  if (value == "diagonal")
    return GreedyCandidateScope::Diagonal;
  if (value == "frontier")
    return GreedyCandidateScope::Frontier;
  anchor->emitError("unknown greedy candidate scope '") << value << "'";
  return failure();
}

StringRef stringifyGreedyCandidateScope(GreedyCandidateScope scope) {
  switch (scope) {
  case GreedyCandidateScope::Cardinal:
    return "cardinal";
  case GreedyCandidateScope::Diagonal:
    return "diagonal";
  case GreedyCandidateScope::Frontier:
    return "frontier";
  }
  llvm_unreachable("unknown greedy candidate scope");
}

LogicalResult validateLogicalTilePlacementProblem(
    const LogicalTilePlacementProblem &problem) {
  FailureOr<int64_t> capacity =
      getPhysicalTileCapacity(problem.mesh, problem.anchor);
  if (failed(capacity))
    return failure();
  if (problem.tileGraph.tiles.empty()) {
    problem.anchor->emitError("logical-tile placement requires an active tile");
    return failure();
  }
  if (problem.mesh.rows != problem.tileGraph.plannedMeshRows ||
      problem.mesh.columns != problem.tileGraph.plannedMeshCols) {
    problem.anchor->emitError("physical placement mesh ")
        << problem.mesh.rows << "x" << problem.mesh.columns
        << " does not match logical planning mesh "
        << problem.tileGraph.plannedMeshRows << "x"
        << problem.tileGraph.plannedMeshCols;
    return failure();
  }
  if (problem.mesh.arraysPerCore != problem.tileGraph.analogLanesPerTile) {
    problem.anchor->emitError("physical placement arrays per core ")
        << problem.mesh.arraysPerCore
        << " does not match logical planning arrays per core "
        << problem.tileGraph.analogLanesPerTile;
    return failure();
  }
  if (problem.tileGraph.tiles.size() > static_cast<size_t>(*capacity)) {
    problem.anchor->emitError("physical mesh has ")
        << *capacity << " tiles but the mapping requires "
        << problem.tileGraph.tiles.size() << " active logical tiles";
    return failure();
  }
  for (const LogicalTile &tile : problem.tileGraph.tiles) {
    if (tile.analogLanes.size() >
        static_cast<size_t>(problem.mesh.arraysPerCore)) {
      problem.anchor->emitError("logical tile ")
          << tile.id << " exceeds physical analog-lane capacity";
      return failure();
    }
  }
  return success();
}

FailureOr<int64_t>
scoreLogicalTilePlacement(const LogicalTilePlacementProblem &problem,
                          ArrayRef<int64_t> physicalTileByLogicalTileIndex) {
  if (failed(validateLogicalTilePlacementProblem(problem)))
    return failure();
  if (physicalTileByLogicalTileIndex.size() != problem.tileGraph.tiles.size()) {
    problem.anchor->emitError(
        "logical-tile placement score requires one location per active tile");
    return failure();
  }
  FailureOr<int64_t> capacity =
      getPhysicalTileCapacity(problem.mesh, problem.anchor);
  if (failed(capacity))
    return failure();
  std::set<int64_t> usedPhysicalTiles;
  for (int64_t physicalTileId : physicalTileByLogicalTileIndex) {
    if (physicalTileId < 0 || physicalTileId >= *capacity ||
        !usedPhysicalTiles.insert(physicalTileId).second) {
      problem.anchor->emitError(
          "logical-tile placement contains an invalid or duplicate physical "
          "tile");
      return failure();
    }
  }

  int64_t total = 0;
  for (const LogicalTileEdge &edge : problem.tileGraph.edges) {
    auto source = problem.tileGraph.tileIndexById.find(edge.sourceTileId);
    auto target = problem.tileGraph.tileIndexById.find(edge.targetTileId);
    if (source == problem.tileGraph.tileIndexById.end() ||
        target == problem.tileGraph.tileIndexById.end()) {
      problem.anchor->emitError(
          "logical-tile placement edge references an inactive tile");
      return failure();
    }
    PhysicalTileLocation sourceLocation = getLocation(
        physicalTileByLogicalTileIndex[source->second], problem.mesh);
    PhysicalTileLocation targetLocation = getLocation(
        physicalTileByLogicalTileIndex[target->second], problem.mesh);
    FailureOr<int64_t> cost = checkedTransferCost(
        edge.byteSize, getManhattanDistance(sourceLocation, targetLocation),
        problem.anchor);
    if (failed(cost))
      return failure();
    FailureOr<int64_t> updated = checkedAddCost(total, *cost, problem.anchor);
    if (failed(updated))
      return failure();
    total = *updated;
  }
  return total;
}

FailureOr<LogicalTilePlacementPlan>
buildLogicalTilePlacementPlan(const LogicalTilePlacementProblem &problem,
                              ArrayRef<int64_t> physicalTileByLogicalTileIndex,
                              StringRef schedule, int64_t initialScore,
                              int64_t evaluations) {
  FailureOr<PlacementObjectiveEvaluation> objective =
      evaluatePlacementObjective(problem, physicalTileByLogicalTileIndex);
  if (failed(objective))
    return failure();

  LogicalTilePlacementPlan plan;
  plan.schedule = schedule.str();
  plan.objective = problem.objective;
  plan.networkMode = problem.networkMode;
  plan.timingScope = problem.timingScope;
  if (problem.costProfile) {
    plan.costProfileName = problem.costProfile->name;
    plan.costProfileHash = problem.costProfile->contentHash;
  }
  plan.mesh = problem.mesh;
  plan.initialScore = initialScore;
  plan.objectiveScore = objective->score;
  plan.totalTransferCost = objective->transferCost;
  plan.predictedMakespanNs = objective->makespanNs;
  plan.evaluations = evaluations;
  plan.assignments.reserve(problem.tileGraph.tiles.size());
  for (auto indexedTile : llvm::enumerate(problem.tileGraph.tiles)) {
    LogicalTilePhysicalAssignment assignment;
    assignment.logicalTileId = indexedTile.value().id;
    assignment.location = getLocation(
        physicalTileByLogicalTileIndex[indexedTile.index()], problem.mesh);
    plan.assignmentIndexByTileId[assignment.logicalTileId] =
        plan.assignments.size();
    plan.assignments.push_back(assignment);
  }

  plan.edges.reserve(problem.tileGraph.edges.size());
  for (const LogicalTileEdge &edge : problem.tileGraph.edges) {
    const LogicalTilePhysicalAssignment &source =
        plan.assignments[plan.assignmentIndexByTileId.lookup(
            edge.sourceTileId)];
    const LogicalTilePhysicalAssignment &target =
        plan.assignments[plan.assignmentIndexByTileId.lookup(
            edge.targetTileId)];
    int64_t hops = getManhattanDistance(source.location, target.location);
    FailureOr<int64_t> cost =
        checkedTransferCost(edge.byteSize, hops, problem.anchor);
    if (failed(cost))
      return failure();
    plan.edges.push_back({edge.id, edge.sourceTileId, edge.targetTileId,
                          edge.byteSize, hops, *cost});
  }
  if (failed(verifyLogicalTilePlacementPlan(problem, plan)))
    return failure();
  return plan;
}

LogicalResult
verifyLogicalTilePlacementPlan(const LogicalTilePlacementProblem &problem,
                               const LogicalTilePlacementPlan &plan) {
  if (failed(validateLogicalTilePlacementProblem(problem)))
    return failure();
  if (plan.version != 2)
    return problem.anchor->emitError(
               "logical-tile placement version mismatch: ")
           << plan.version << " versus 2";
  if (plan.schedule.empty())
    return problem.anchor->emitError(
        "logical-tile placement schedule must not be empty");
  if (plan.objective != problem.objective)
    return problem.anchor->emitError(
               "logical-tile placement objective mismatch: '")
           << stringifyPlacementObjective(plan.objective) << "' versus '"
           << stringifyPlacementObjective(problem.objective) << "'";
  if (plan.networkMode != problem.networkMode)
    return problem.anchor->emitError(
               "logical-tile placement network mode mismatch: '")
           << stringifyTemporalNetworkMode(plan.networkMode) << "' versus '"
           << stringifyTemporalNetworkMode(problem.networkMode) << "'";
  if (plan.timingScope != problem.timingScope)
    return problem.anchor->emitError(
               "logical-tile placement timing scope mismatch: '")
           << stringifyTemporalTimingScope(plan.timingScope) << "' versus '"
           << stringifyTemporalTimingScope(problem.timingScope) << "'";
  if (plan.routePolicy != "xy")
    return problem.anchor->emitError(
               "logical-tile placement route policy mismatch: '")
           << plan.routePolicy << "' versus 'xy'";
  if (problem.costProfile && plan.costProfileName != problem.costProfile->name)
    return problem.anchor->emitError(
               "logical-tile placement cost profile name mismatch: '")
           << plan.costProfileName << "' versus '" << problem.costProfile->name
           << "'";
  if (problem.costProfile &&
      plan.costProfileHash != problem.costProfile->contentHash)
    return problem.anchor->emitError(
        "logical-tile placement cost profile hash mismatch");
  if (plan.mesh.rows != problem.mesh.rows)
    return problem.anchor->emitError(
               "logical-tile placement mesh-row mismatch: ")
           << plan.mesh.rows << " versus " << problem.mesh.rows;
  if (plan.mesh.columns != problem.mesh.columns)
    return problem.anchor->emitError(
               "logical-tile placement mesh-column mismatch: ")
           << plan.mesh.columns << " versus " << problem.mesh.columns;
  if (plan.mesh.arraysPerCore != problem.mesh.arraysPerCore)
    return problem.anchor->emitError(
               "logical-tile placement arrays-per-core mismatch: ")
           << plan.mesh.arraysPerCore << " versus "
           << problem.mesh.arraysPerCore;
  if (plan.initialScore < 0)
    return problem.anchor->emitError(
        "logical-tile placement initial score must be nonnegative");
  if (plan.objectiveScore < 0)
    return problem.anchor->emitError(
        "logical-tile placement objective score must be nonnegative");
  if (plan.totalTransferCost < 0)
    return problem.anchor->emitError(
        "logical-tile placement transfer cost must be nonnegative");
  if (!std::isfinite(plan.predictedMakespanNs) ||
      plan.predictedMakespanNs < 0.0)
    return problem.anchor->emitError(
        "logical-tile placement predicted makespan must be finite and "
        "nonnegative");
  if (plan.evaluations < 0)
    return problem.anchor->emitError(
        "logical-tile placement evaluation count must be nonnegative");
  if (plan.assignments.size() != problem.tileGraph.tiles.size())
    return problem.anchor->emitError(
               "logical-tile placement assignment-count mismatch: ")
           << plan.assignments.size() << " versus "
           << problem.tileGraph.tiles.size();
  if (plan.edges.size() != problem.tileGraph.edges.size())
    return problem.anchor->emitError(
               "logical-tile placement edge-count mismatch: ")
           << plan.edges.size() << " versus " << problem.tileGraph.edges.size();

  FailureOr<int64_t> capacity =
      getPhysicalTileCapacity(problem.mesh, problem.anchor);
  if (failed(capacity))
    return failure();
  std::set<int64_t> logicalTiles;
  std::set<int64_t> physicalTiles;
  DenseMap<int64_t, PhysicalTileLocation> locations;
  for (const LogicalTilePhysicalAssignment &assignment : plan.assignments) {
    if (!problem.tileGraph.tileIndexById.contains(assignment.logicalTileId) ||
        !logicalTiles.insert(assignment.logicalTileId).second ||
        assignment.location.physicalTileId < 0 ||
        assignment.location.physicalTileId >= *capacity ||
        !physicalTiles.insert(assignment.location.physicalTileId).second ||
        assignment.location.row !=
            assignment.location.physicalTileId / problem.mesh.columns ||
        assignment.location.column !=
            assignment.location.physicalTileId % problem.mesh.columns) {
      problem.anchor->emitError(
          "invalid or duplicate logical-to-physical tile assignment");
      return failure();
    }
    locations[assignment.logicalTileId] = assignment.location;
  }

  int64_t total = 0;
  for (auto indexedEdge : llvm::enumerate(plan.edges)) {
    const PlacedLogicalTileEdge &placed = indexedEdge.value();
    const LogicalTileEdge &logical =
        problem.tileGraph.edges[indexedEdge.index()];
    if (placed.edgeId != logical.id ||
        placed.sourceTileId != logical.sourceTileId ||
        placed.targetTileId != logical.targetTileId ||
        placed.byteSize != logical.byteSize) {
      problem.anchor->emitError(
          "placed edge does not match its logical-tile edge");
      return failure();
    }
    int64_t expectedHops =
        getManhattanDistance(locations.lookup(placed.sourceTileId),
                             locations.lookup(placed.targetTileId));
    FailureOr<int64_t> expectedCost =
        checkedTransferCost(placed.byteSize, expectedHops, problem.anchor);
    if (failed(expectedCost) || placed.manhattanHops != expectedHops ||
        placed.transferCost != *expectedCost) {
      problem.anchor->emitError("placed edge has an invalid distance or cost");
      return failure();
    }
    FailureOr<int64_t> updated =
        checkedAddCost(total, placed.transferCost, problem.anchor);
    if (failed(updated))
      return failure();
    total = *updated;
  }
  if (total != plan.totalTransferCost) {
    problem.anchor->emitError(
        "logical-tile placement total does not match its edges");
    return failure();
  }
  if (plan.objective == PlacementObjectiveKind::TransferCost &&
      (plan.objectiveScore != plan.totalTransferCost ||
       plan.predictedMakespanNs != 0.0)) {
    problem.anchor->emitError(
        "transfer-cost placement has inconsistent objective metadata");
    return failure();
  }
  if (plan.objective == PlacementObjectiveKind::Makespan) {
    FailureOr<PlacementObjectiveEvaluation> objective =
        evaluatePlacementObjective(problem, [&]() {
          SmallVector<int64_t> physical;
          physical.reserve(problem.tileGraph.tiles.size());
          for (const LogicalTile &tile : problem.tileGraph.tiles)
            physical.push_back(locations.lookup(tile.id).physicalTileId);
          return physical;
        }());
    if (failed(objective) || objective->score != plan.objectiveScore ||
        objective->makespanNs != plan.predictedMakespanNs)
      return problem.anchor->emitError(
          "makespan placement has inconsistent objective metadata");
  }
  return success();
}

LogicalTilePlacementAttr
serializeLogicalTilePlacement(MLIRContext *context,
                              const LogicalTilePlacementPlan &plan) {
  Builder builder(context);
  SmallVector<Attribute> assignments;
  assignments.reserve(plan.assignments.size());
  for (const LogicalTilePhysicalAssignment &assignment : plan.assignments) {
    assignments.push_back(PhysicalTileAssignmentAttr::get(
        context, builder.getI64IntegerAttr(assignment.logicalTileId),
        builder.getI64IntegerAttr(assignment.location.physicalTileId),
        builder.getI64IntegerAttr(assignment.location.row),
        builder.getI64IntegerAttr(assignment.location.column)));
  }
  SmallVector<Attribute> edges;
  edges.reserve(plan.edges.size());
  for (const PlacedLogicalTileEdge &edge : plan.edges) {
    edges.push_back(PlacedLogicalTileEdgeAttr::get(
        context, builder.getI64IntegerAttr(edge.edgeId),
        builder.getI64IntegerAttr(edge.sourceTileId),
        builder.getI64IntegerAttr(edge.targetTileId),
        builder.getI64IntegerAttr(edge.byteSize),
        builder.getI64IntegerAttr(edge.manhattanHops),
        builder.getI64IntegerAttr(edge.transferCost)));
  }
  return LogicalTilePlacementAttr::get(
      context, builder.getI64IntegerAttr(plan.version),
      builder.getStringAttr(plan.schedule),
      builder.getStringAttr(stringifyPlacementObjective(plan.objective)),
      builder.getStringAttr(stringifyTemporalNetworkMode(plan.networkMode)),
      builder.getStringAttr(stringifyTemporalTimingScope(plan.timingScope)),
      builder.getStringAttr(plan.routePolicy),
      builder.getStringAttr(plan.costProfileName),
      builder.getStringAttr(plan.costProfileHash),
      builder.getI64IntegerAttr(plan.mesh.rows),
      builder.getI64IntegerAttr(plan.mesh.columns),
      builder.getI64IntegerAttr(plan.mesh.arraysPerCore),
      builder.getI64IntegerAttr(plan.initialScore),
      builder.getI64IntegerAttr(plan.objectiveScore),
      builder.getI64IntegerAttr(plan.totalTransferCost),
      builder.getF64FloatAttr(plan.predictedMakespanNs),
      builder.getI64IntegerAttr(plan.evaluations),
      builder.getArrayAttr(assignments), builder.getArrayAttr(edges));
}

FailureOr<LogicalTilePlacementPlan>
deserializeLogicalTilePlacement(LogicalTilePlacementAttr attr,
                                const LogicalTilePlacementProblem &problem) {
  LogicalTilePlacementPlan plan;
  plan.version = attr.getVersion().getInt();
  plan.schedule = attr.getSchedule().getValue().str();
  FailureOr<PlacementObjectiveKind> objective =
      parsePlacementObjective(attr.getObjective().getValue(), problem.anchor);
  if (failed(objective))
    return failure();
  plan.objective = *objective;
  FailureOr<TemporalNetworkMode> networkMode = parseTemporalNetworkMode(
      attr.getNetworkMode().getValue(), problem.anchor);
  FailureOr<TemporalTimingScope> timingScope = parseTemporalTimingScope(
      attr.getTimingScope().getValue(), problem.anchor);
  if (failed(networkMode) || failed(timingScope))
    return failure();
  plan.networkMode = *networkMode;
  plan.timingScope = *timingScope;
  plan.routePolicy = attr.getRoutePolicy().getValue().str();
  plan.costProfileName = attr.getCostProfileName().getValue().str();
  plan.costProfileHash = attr.getCostProfileHash().getValue().str();
  plan.mesh = {attr.getMeshRows().getInt(), attr.getMeshCols().getInt(),
               attr.getArraysPerCore().getInt()};
  plan.initialScore = attr.getInitialScore().getInt();
  plan.objectiveScore = attr.getObjectiveScore().getInt();
  plan.totalTransferCost = attr.getTotalTransferCost().getInt();
  plan.predictedMakespanNs = attr.getPredictedMakespanNs().getValueAsDouble();
  plan.evaluations = attr.getEvaluations().getInt();
  for (Attribute value : attr.getAssignments()) {
    auto assignmentAttr = dyn_cast<PhysicalTileAssignmentAttr>(value);
    if (!assignmentAttr) {
      problem.anchor->emitError(
          "logical-tile placement assignments must be typed");
      return failure();
    }
    LogicalTilePhysicalAssignment assignment{
        assignmentAttr.getLogicalTileId().getInt(),
        {assignmentAttr.getPhysicalTileId().getInt(),
         assignmentAttr.getRow().getInt(),
         assignmentAttr.getColumn().getInt()}};
    if (plan.assignmentIndexByTileId.contains(assignment.logicalTileId)) {
      problem.anchor->emitError(
          "logical-tile placement contains a duplicate logical tile");
      return failure();
    }
    plan.assignmentIndexByTileId[assignment.logicalTileId] =
        plan.assignments.size();
    plan.assignments.push_back(assignment);
  }
  for (Attribute value : attr.getEdges()) {
    auto edgeAttr = dyn_cast<PlacedLogicalTileEdgeAttr>(value);
    if (!edgeAttr) {
      problem.anchor->emitError("logical-tile placed edges must be typed");
      return failure();
    }
    plan.edges.push_back(
        {edgeAttr.getEdgeId().getInt(), edgeAttr.getSourceTileId().getInt(),
         edgeAttr.getTargetTileId().getInt(), edgeAttr.getByteSize().getInt(),
         edgeAttr.getManhattanHops().getInt(),
         edgeAttr.getTransferCost().getInt()});
  }
  if (failed(verifyLogicalTilePlacementPlan(problem, plan)))
    return failure();
  return plan;
}

LogicalTileAnnealingTraceAttr
serializeLogicalTileAnnealingTrace(MLIRContext *context,
                                   const LogicalTileAnnealingTrace &trace) {
  Builder builder(context);
  SmallVector<Attribute> samples;
  samples.reserve(trace.samples.size());
  for (const LogicalTileAnnealingSample &sample : trace.samples) {
    samples.push_back(AnnealingScoreSampleAttr::get(
        context, builder.getI64IntegerAttr(sample.iteration),
        builder.getI64IntegerAttr(sample.candidateScore),
        builder.getI64IntegerAttr(sample.currentScore),
        builder.getI64IntegerAttr(sample.bestScore),
        builder.getBoolAttr(sample.accepted)));
  }
  return LogicalTileAnnealingTraceAttr::get(
      context, builder.getI64IntegerAttr(trace.version),
      builder.getI64IntegerAttr(trace.initialScore),
      builder.getI64IntegerAttr(trace.finalScore),
      builder.getI64IntegerAttr(trace.evaluations),
      builder.getArrayAttr(samples));
}

FailureOr<LogicalTileAnnealingTrace>
deserializeLogicalTileAnnealingTrace(LogicalTileAnnealingTraceAttr attr,
                                     const LogicalTilePlacementPlan &plan,
                                     Operation *anchor) {
  LogicalTileAnnealingTrace trace;
  trace.version = attr.getVersion().getInt();
  trace.initialScore = attr.getInitialScore().getInt();
  trace.finalScore = attr.getFinalScore().getInt();
  trace.evaluations = attr.getEvaluations().getInt();
  for (Attribute value : attr.getSamples()) {
    auto sampleAttr = dyn_cast<AnnealingScoreSampleAttr>(value);
    if (!sampleAttr) {
      anchor->emitError("logical-tile annealing samples must be typed");
      return failure();
    }
    trace.samples.push_back({sampleAttr.getIteration().getInt(),
                             sampleAttr.getCandidateScore().getInt(),
                             sampleAttr.getCurrentScore().getInt(),
                             sampleAttr.getBestScore().getInt(),
                             sampleAttr.getAccepted().getValue()});
  }

  if (trace.version != 1 || plan.schedule != "annealing" ||
      trace.initialScore != plan.initialScore ||
      trace.finalScore != plan.objectiveScore ||
      trace.evaluations != plan.evaluations || trace.evaluations < 0 ||
      trace.samples.empty() || trace.samples.front().iteration != 0 ||
      trace.samples.back().iteration != trace.evaluations) {
    anchor->emitError("annealing trace does not match its placement plan");
    return failure();
  }

  int64_t previousCurrent = trace.initialScore;
  int64_t previousBest = trace.initialScore;
  for (auto indexedSample : llvm::enumerate(trace.samples)) {
    const LogicalTileAnnealingSample &sample = indexedSample.value();
    if ((indexedSample.index() > 0 &&
         sample.iteration <=
             trace.samples[indexedSample.index() - 1].iteration) ||
        sample.iteration < 0 || sample.iteration > trace.evaluations ||
        sample.candidateScore < 0 || sample.currentScore < 0 ||
        sample.bestScore < 0) {
      anchor->emitError("invalid annealing score sample metadata");
      return failure();
    }
    if (indexedSample.index() == 0) {
      if (!sample.accepted || sample.candidateScore != trace.initialScore ||
          sample.currentScore != trace.initialScore ||
          sample.bestScore != trace.initialScore) {
        anchor->emitError("invalid initial annealing score sample");
        return failure();
      }
      continue;
    }
    bool consecutive = sample.iteration ==
                       trace.samples[indexedSample.index() - 1].iteration + 1;
    int64_t expectedCurrent =
        sample.accepted ? sample.candidateScore : previousCurrent;
    int64_t expectedBest = std::min(previousBest, expectedCurrent);
    if ((consecutive && (sample.currentScore != expectedCurrent ||
                         sample.bestScore != expectedBest)) ||
        (!consecutive && (sample.bestScore > previousBest ||
                          sample.bestScore > sample.currentScore))) {
      anchor->emitError("annealing score trajectory is inconsistent");
      return failure();
    }
    previousCurrent = sample.currentScore;
    previousBest = sample.bestScore;
  }
  if (previousBest != trace.finalScore) {
    anchor->emitError("annealing trace final score is inconsistent");
    return failure();
  }
  return trace;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
