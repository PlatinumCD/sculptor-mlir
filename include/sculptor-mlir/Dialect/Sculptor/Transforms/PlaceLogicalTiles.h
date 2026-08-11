#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLACELOGICALTILES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLACELOGICALTILES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {

struct PlaceLogicalTilesPass
    : public PassWrapper<PlaceLogicalTilesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlaceLogicalTilesPass)

  Option<std::string> schedule{
      *this, "schedule",
      llvm::cl::desc(
          "Physical placement schedule: random, snake, greedy, or annealing"),
      llvm::cl::init("greedy")};

  Option<std::string> placementObjective{
      *this, "objective",
      llvm::cl::desc("Physical objective: transfer-cost or makespan"),
      llvm::cl::init("transfer-cost")};

  Option<std::string> networkMode{
      *this, "network-mode",
      llvm::cl::desc("Temporal network model: ideal, finite, or full"),
      llvm::cl::init("finite")};

  Option<std::string> timingScope{
      *this, "timing-scope",
      llvm::cl::desc("Temporal timing scope: warm or cold"),
      llvm::cl::init("warm")};

  Option<int64_t> temporalCandidateLimit{
      *this, "temporal-candidate-limit",
      llvm::cl::desc("Maximum greedy candidates evaluated with makespan"),
      llvm::cl::init(8)};

  Option<int64_t> meshRows{*this, "mesh-rows",
                           llvm::cl::desc("Physical mesh row count; zero "
                                          "inherits the logical plan"),
                           llvm::cl::init(0)};

  Option<int64_t> meshCols{*this, "mesh-cols",
                           llvm::cl::desc("Physical mesh column count; zero "
                                          "inherits the logical plan"),
                           llvm::cl::init(0)};

  Option<int64_t> arraysPerCore{
      *this, "arrays-per-core",
      llvm::cl::desc("Physical arrays per tile; zero inherits the logical "
                     "tile shape"),
      llvm::cl::init(0)};

  Option<std::string> greedyTileOrder{
      *this, "greedy-tile-order",
      llvm::cl::desc("Greedy logical-tile order: sequential or priority"),
      llvm::cl::init("sequential")};

  Option<std::string> greedyPriorityMode{
      *this, "greedy-priority-mode",
      llvm::cl::desc("Priority queue affinity mode: sum or max"),
      llvm::cl::init("sum")};

  Option<std::string> greedyCandidateScope{
      *this, "greedy-candidate-scope",
      llvm::cl::desc(
          "Greedy physical candidate scope: cardinal, diagonal, or frontier"),
      llvm::cl::init("cardinal")};

  Option<int64_t> greedyLookahead{
      *this, "greedy-lookahead",
      llvm::cl::desc("Number of placements included in each greedy rollout"),
      llvm::cl::init(1)};

  Option<int64_t> randomSeed{*this, "random-seed",
                             llvm::cl::desc("Deterministic random seed"),
                             llvm::cl::init(0)};

  Option<std::string> annealingInitialSchedule{
      *this, "annealing-initial-schedule",
      llvm::cl::desc("Annealing initial schedule: random, snake, or greedy"),
      llvm::cl::init("greedy")};

  Option<int64_t> annealingIterations{
      *this, "annealing-iterations",
      llvm::cl::desc("Number of simulated-annealing candidate evaluations"),
      llvm::cl::init(1000)};

  Option<double> annealingInitialTemperature{
      *this, "annealing-initial-temperature",
      llvm::cl::desc("Initial annealing temperature; zero selects an "
                     "automatic score-relative value"),
      llvm::cl::init(0.0)};

  Option<double> annealingCoolingRate{
      *this, "annealing-cooling-rate",
      llvm::cl::desc("Multiplicative annealing cooling rate"),
      llvm::cl::init(0.995)};

  Option<int64_t> annealingTraceSampleInterval{
      *this, "annealing-trace-sample-interval",
      llvm::cl::desc("Record one annealing trajectory sample per N "
                     "evaluations"),
      llvm::cl::init(1)};

  Option<bool> verifyPlacement{
      *this, "verify-placement",
      llvm::cl::desc("Verify the physical logical-tile placement"),
      llvm::cl::init(true)};

  Option<std::string> summaryOutput{
      *this, "summary-output",
      llvm::cl::desc("Optional CSV path for a compact logical-tile placement "
                     "summary"),
      llvm::cl::init("")};

  PlaceLogicalTilesPass() = default;
  PlaceLogicalTilesPass(const PlaceLogicalTilesPass &pass) : PassWrapper(pass) {
    schedule = pass.schedule;
    placementObjective = pass.placementObjective;
    networkMode = pass.networkMode;
    timingScope = pass.timingScope;
    temporalCandidateLimit = pass.temporalCandidateLimit;
    meshRows = pass.meshRows;
    meshCols = pass.meshCols;
    arraysPerCore = pass.arraysPerCore;
    greedyTileOrder = pass.greedyTileOrder;
    greedyPriorityMode = pass.greedyPriorityMode;
    greedyCandidateScope = pass.greedyCandidateScope;
    greedyLookahead = pass.greedyLookahead;
    randomSeed = pass.randomSeed;
    annealingInitialSchedule = pass.annealingInitialSchedule;
    annealingIterations = pass.annealingIterations;
    annealingInitialTemperature = pass.annealingInitialTemperature;
    annealingCoolingRate = pass.annealingCoolingRate;
    annealingTraceSampleInterval = pass.annealingTraceSampleInterval;
    verifyPlacement = pass.verifyPlacement;
    summaryOutput = pass.summaryOutput;
  }

  StringRef getArgument() const final { return "sculptor-place-logical-tiles"; }

  StringRef getDescription() const final {
    return "Place RA-planned logical tiles on a physical mesh";
  }

  void runOnOperation() override;
};

void registerPlaceLogicalTilesPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLACELOGICALTILES_H
