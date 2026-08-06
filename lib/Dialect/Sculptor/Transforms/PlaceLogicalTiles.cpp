#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlaceLogicalTiles.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
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
  FailureOr<mapping::LogicalTileScheduleKind> parsedInitial =
      mapping::parseLogicalTileSchedule(annealingInitialSchedule, module,
                                        /*allowAnnealing=*/false);
  if (failed(parsedSchedule) || failed(parsedInitial)) {
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
    *summary << "function,schedule,mesh_rows,mesh_cols,arrays_per_core,"
                "lookahead,beam_width,digital_workers,matrix_duplication,"
                "matrix_setups,logical_tiles,logical_edges,initial_score,"
                "total_transfer_cost,evaluations,estimated_latency_ns,"
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
    mapping::LogicalTilePlacementConfig config;
    config.schedule = *parsedSchedule;
    config.greedyLookahead = lookahead;
    config.greedyBeamWidth = beamWidth;
    config.annealingInitialSchedule = *parsedInitial;
    config.randomSeed = randomSeed;
    config.annealingIterations = annealingIterations;
    config.annealingInitialTemperature = annealingInitialTemperature;
    config.annealingCoolingRate = annealingCoolingRate;

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
    if (*parsedSchedule == mapping::LogicalTileScheduleKind::GreedyBeam) {
      function->setAttr(
          mapping::kLogicalTileGreedyLookaheadAttrName,
          IntegerAttr::get(IntegerType::get(&getContext(), 64), lookahead));
      function->setAttr(
          mapping::kLogicalTileGreedyBeamWidthAttrName,
          IntegerAttr::get(IntegerType::get(&getContext(), 64), beamWidth));
    } else {
      function->removeAttr(mapping::kLogicalTileGreedyLookaheadAttrName);
      function->removeAttr(mapping::kLogicalTileGreedyBeamWidthAttrName);
    }
    if (placement->annealingTrace) {
      function->setAttr(mapping::kLogicalTileAnnealingTraceAttrName,
                        mapping::serializeLogicalTileAnnealingTrace(
                            &getContext(), *placement->annealingTrace));
    } else {
      function->removeAttr(mapping::kLogicalTileAnnealingTraceAttrName);
    }
    if (summary) {
      bool usesBeamSearch =
          *parsedSchedule == mapping::LogicalTileScheduleKind::GreedyBeam ||
          (*parsedSchedule == mapping::LogicalTileScheduleKind::Annealing &&
           *parsedInitial == mapping::LogicalTileScheduleKind::GreedyBeam);
      int64_t reportedLookahead = usesBeamSearch ? lookahead : 1;
      int64_t reportedBeamWidth = usesBeamSearch ? beamWidth : 1;
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
          << placement->mesh.rows << ',' << placement->mesh.columns << ','
          << placement->mesh.arraysPerCore << ',' << reportedLookahead << ','
          << reportedBeamWidth << ','
          << (digitalWorkers ? digitalWorkers.getInt() : 1)
          << ',' << (matrixDuplication ? "on" : "off") << ','
          << matrixSetupCount << ',' << tileGraph->tiles.size() << ','
          << tileGraph->edges.size() << ',' << placement->initialScore << ','
          << placement->totalTransferCost << ',' << placement->evaluations
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
