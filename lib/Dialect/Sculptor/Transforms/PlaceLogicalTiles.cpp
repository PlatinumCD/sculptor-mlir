#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlaceLogicalTiles.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir {
namespace sculptor {

void PlaceLogicalTilesPass::runOnOperation() {
  ModuleOp module = getOperation();
  FailureOr<mapping::LogicalTileScheduleKind> parsedSchedule =
      mapping::parseLogicalTileSchedule(schedule, module);
  FailureOr<mapping::PlacementObjectiveKind> parsedObjective =
      mapping::parsePlacementObjective(placementObjective, module);
  FailureOr<mapping::TemporalNetworkMode> parsedNetworkMode =
      mapping::parseTemporalNetworkMode(networkMode, module);
  FailureOr<mapping::TemporalTimingScope> parsedTimingScope =
      mapping::parseTemporalTimingScope(timingScope, module);
  FailureOr<mapping::LogicalTileScheduleKind> parsedInitial =
      mapping::parseLogicalTileSchedule(annealingInitialSchedule, module,
                                        /*allowAnnealing=*/false);
  FailureOr<mapping::GreedyTileOrder> parsedGreedyTileOrder =
      mapping::parseGreedyTileOrder(greedyTileOrder, module);
  FailureOr<mapping::GreedyPriorityMode> parsedGreedyPriorityMode =
      mapping::parseGreedyPriorityMode(greedyPriorityMode, module);
  FailureOr<mapping::GreedyCandidateScope> parsedGreedyCandidateScope =
      mapping::parseGreedyCandidateScope(greedyCandidateScope, module);
  if (failed(parsedSchedule) || failed(parsedObjective) ||
      failed(parsedNetworkMode) || failed(parsedTimingScope) ||
      failed(parsedInitial) || failed(parsedGreedyTileOrder) ||
      failed(parsedGreedyPriorityMode) || failed(parsedGreedyCandidateScope)) {
    signalPassFailure();
    return;
  }

  std::unique_ptr<llvm::raw_fd_ostream> summary;
  if (!summaryOutput.empty()) {
    std::error_code error;
    summary = std::make_unique<llvm::raw_fd_ostream>(summaryOutput, error,
                                                     llvm::sys::fs::OF_Text);
    if (error) {
      module.emitError("cannot open logical-tile placement summary '")
          << summaryOutput << "': " << error.message();
      signalPassFailure();
      return;
    }
    *summary << "function,schedule,objective,network_mode,timing_scope,"
                "cost_profile_name,cost_profile_hash,"
                "mesh_rows,mesh_cols,arrays_per_core,"
                "greedy_tile_order,greedy_priority_mode,"
                "greedy_candidate_scope,greedy_lookahead,digital_workers,"
                "matrix_duplication,matrix_setups,logical_tiles,logical_edges,"
                "initial_score,"
                "objective_score,total_transfer_cost,predicted_makespan_ns,"
                "evaluations,estimated_latency_ns,"
                "crossing_bytes,estimated_communication_ns,"
                "required_resource_units,pipeline_stages\n";
  }

  bool placedFunction = false;
  bool foundTerminallyMaterializedPlan = false;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    auto tileGraphAttr = function->getAttrOfType<LogicalTileGraphAttr>(
        mapping::kLogicalTileGraphAttrName);
    if (!tileGraphAttr) {
      foundTerminallyMaterializedPlan |=
          function->hasAttr("sculptor.mapping.applied_work_unit_count");
      continue;
    }
    auto treeAttr =
        function->getAttrOfType<RATreeAttr>(mapping::kRATreeAttrName);
    if (!treeAttr) {
      function.emitError(
          "logical-tile placement requires the selected RA tree");
      signalPassFailure();
      return;
    }

    FailureOr<mapping::ComputeGraph> graph =
        mapping::buildComputeGraph(function);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    FailureOr<mapping::ResourceAllocationTree> tree =
        mapping::deserializeResourceAllocationTree(treeAttr, *graph, function);
    if (failed(tree)) {
      signalPassFailure();
      return;
    }
    FailureOr<mapping::LogicalTileGraph> tileGraph =
        mapping::deserializeLogicalTileGraph(tileGraphAttr, *graph, *tree,
                                             function);
    if (failed(tileGraph)) {
      signalPassFailure();
      return;
    }

    int64_t resolvedMeshRows = meshRows == 0 ? tileGraph->plannedMeshRows
                                             : static_cast<int64_t>(meshRows);
    int64_t resolvedMeshCols = meshCols == 0 ? tileGraph->plannedMeshCols
                                             : static_cast<int64_t>(meshCols);
    int64_t resolvedArraysPerCore = arraysPerCore == 0
                                        ? tileGraph->analogLanesPerTile
                                        : static_cast<int64_t>(arraysPerCore);
    mapping::LogicalTilePlacementProblem problem{
        *tileGraph,
        {resolvedMeshRows, resolvedMeshCols, resolvedArraysPerCore},
        function};
    mapping::MappingCostProfile resolvedProfile;
    if (failed(mapping::initializeLogicalTilePlacementProblem(
            *graph, *tree, resolvedProfile, problem))) {
      signalPassFailure();
      return;
    }
    problem.objective = *parsedObjective;
    problem.networkMode = *parsedNetworkMode;
    problem.timingScope = *parsedTimingScope;
    problem.temporalCandidateLimit = temporalCandidateLimit;
    mapping::LogicalTilePlacementConfig config;
    config.schedule = *parsedSchedule;
    config.greedy.tileOrder = *parsedGreedyTileOrder;
    config.greedy.priorityMode = *parsedGreedyPriorityMode;
    config.greedy.candidateScope = *parsedGreedyCandidateScope;
    config.greedy.lookahead = greedyLookahead;
    config.annealingInitialSchedule = *parsedInitial;
    config.randomSeed = randomSeed;
    config.annealingIterations = annealingIterations;
    config.annealingInitialTemperature = annealingInitialTemperature;
    config.annealingCoolingRate = annealingCoolingRate;
    config.annealingTraceSampleInterval = annealingTraceSampleInterval;

    FailureOr<mapping::LogicalTilePlacementPlan> placement =
        mapping::scheduleLogicalTiles(problem, config);
    if (failed(placement) ||
        (verifyPlacement && failed(mapping::verifyLogicalTilePlacementPlan(
                                problem, *placement)))) {
      signalPassFailure();
      return;
    }
    function->setAttr(
        mapping::kLogicalTilePlacementAttrName,
        mapping::serializeLogicalTilePlacement(&getContext(), *placement));
    bool usesGreedy =
        *parsedSchedule == mapping::LogicalTileScheduleKind::Greedy ||
        (*parsedSchedule == mapping::LogicalTileScheduleKind::Annealing &&
         *parsedInitial == mapping::LogicalTileScheduleKind::Greedy);
    if (usesGreedy) {
      function->setAttr(
          mapping::kLogicalTileGreedyTileOrderAttrName,
          StringAttr::get(&getContext(), mapping::stringifyGreedyTileOrder(
                                             *parsedGreedyTileOrder)));
      function->setAttr(
          mapping::kLogicalTileGreedyPriorityModeAttrName,
          StringAttr::get(&getContext(), mapping::stringifyGreedyPriorityMode(
                                             *parsedGreedyPriorityMode)));
      function->setAttr(
          mapping::kLogicalTileGreedyCandidateScopeAttrName,
          StringAttr::get(&getContext(), mapping::stringifyGreedyCandidateScope(
                                             *parsedGreedyCandidateScope)));
      function->setAttr(mapping::kLogicalTileGreedyLookaheadAttrName,
                        IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                         greedyLookahead));
    } else {
      function->removeAttr(mapping::kLogicalTileGreedyTileOrderAttrName);
      function->removeAttr(mapping::kLogicalTileGreedyPriorityModeAttrName);
      function->removeAttr(mapping::kLogicalTileGreedyCandidateScopeAttrName);
      function->removeAttr(mapping::kLogicalTileGreedyLookaheadAttrName);
    }
    if (placement->annealingTrace) {
      function->setAttr(mapping::kLogicalTileAnnealingTraceAttrName,
                        mapping::serializeLogicalTileAnnealingTrace(
                            &getContext(), *placement->annealingTrace));
    } else {
      function->removeAttr(mapping::kLogicalTileAnnealingTraceAttrName);
    }
    if (summary) {
      int64_t matrixSetupCount = 0;
      bool matrixDuplication = false;
      function.walk([&](Operation *operation) {
        if (operation->getName().getStringRef() != "sculptor.array.set")
          return;
        ++matrixSetupCount;
        matrixDuplication |= operation->hasAttr("sculptor.matrix_replica_id");
      });
      auto digitalWorkers = function->getAttrOfType<IntegerAttr>(
          "sculptor.mapping.digital_parallel_workers");
      auto mappingPlan = function->getAttrOfType<MappingPlanAttr>(
          mapping::kMappingPlanAttrName);
      *summary
          << function.getSymName() << ',' << placement->schedule << ','
          << mapping::stringifyPlacementObjective(placement->objective) << ','
          << mapping::stringifyTemporalNetworkMode(*parsedNetworkMode) << ','
          << mapping::stringifyTemporalTimingScope(*parsedTimingScope) << ','
          << placement->costProfileName << ',' << placement->costProfileHash
          << ',' << placement->mesh.rows << ',' << placement->mesh.columns
          << ',' << placement->mesh.arraysPerCore << ','
          << mapping::stringifyGreedyTileOrder(*parsedGreedyTileOrder) << ','
          << mapping::stringifyGreedyPriorityMode(*parsedGreedyPriorityMode)
          << ','
          << mapping::stringifyGreedyCandidateScope(*parsedGreedyCandidateScope)
          << ',' << greedyLookahead << ','
          << (digitalWorkers ? digitalWorkers.getInt() : 1) << ','
          << (matrixDuplication ? "on" : "off") << ',' << matrixSetupCount
          << ',' << tileGraph->tiles.size() << ',' << tileGraph->edges.size()
          << ',' << placement->initialScore << ',' << placement->objectiveScore
          << ',' << placement->totalTransferCost << ','
          << placement->predictedMakespanNs << ',' << placement->evaluations
          << ','
          << (mappingPlan
                  ? mappingPlan.getEstimatedLatencyNs().getValueAsDouble()
                  : 0.0)
          << ',' << (mappingPlan ? mappingPlan.getCrossingBytes().getInt() : 0)
          << ','
          << (mappingPlan
                  ? mappingPlan.getEstimatedCommunicationNs().getValueAsDouble()
                  : 0.0)
          << ','
          << (mappingPlan ? mappingPlan.getRequiredResourceUnits().getInt() : 0)
          << ',' << (mappingPlan ? mappingPlan.getPipelineStages().getInt() : 0)
          << '\n';
    }
    placedFunction = true;
  }

  if (!placedFunction) {
    if (foundTerminallyMaterializedPlan) {
      module.emitError(
          "cannot place after --sculptor-apply-mapping-plan: that terminal "
          "utility consumes the RA tree and logical-tile graph; run "
          "--sculptor-place-logical-tiles directly after "
          "--sculptor-plan-mapping");
      signalPassFailure();
      return;
    }
    module.emitError(
        "expected at least one function with a logical-tile graph");
    signalPassFailure();
  }
}

void registerPlaceLogicalTilesPass() {
  PassRegistration<PlaceLogicalTilesPass>();
}

} // namespace sculptor
} // namespace mlir
