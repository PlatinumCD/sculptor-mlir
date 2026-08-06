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
      llvm::cl::desc("Physical placement schedule: random, snake, greedy, "
                     "greedy-beam, or annealing"),
      llvm::cl::init("greedy")};

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

  Option<int64_t> lookahead{
      *this, "lookahead",
      llvm::cl::desc("Number of logical-tile placements evaluated by "
                     "greedy-beam before committing the current placement"),
      llvm::cl::init(1)};

  Option<int64_t> beamWidth{
      *this, "beam-width",
      llvm::cl::desc("Number of partial placements retained at each "
                     "greedy-beam lookahead depth"),
      llvm::cl::init(8)};

  Option<int64_t> randomSeed{*this, "random-seed",
                             llvm::cl::desc("Deterministic random seed"),
                             llvm::cl::init(0)};

  Option<std::string> annealingInitialSchedule{
      *this, "annealing-initial-schedule",
      llvm::cl::desc(
          "Annealing initial schedule: random, snake, greedy, or greedy-beam"),
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
    meshRows = pass.meshRows;
    meshCols = pass.meshCols;
    arraysPerCore = pass.arraysPerCore;
    lookahead = pass.lookahead;
    beamWidth = pass.beamWidth;
    randomSeed = pass.randomSeed;
    annealingInitialSchedule = pass.annealingInitialSchedule;
    annealingIterations = pass.annealingIterations;
    annealingInitialTemperature = pass.annealingInitialTemperature;
    annealingCoolingRate = pass.annealingCoolingRate;
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
