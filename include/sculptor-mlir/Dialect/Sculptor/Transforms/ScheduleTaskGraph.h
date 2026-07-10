#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SCHEDULETASKGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SCHEDULETASKGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {

// Attaches the hardware budget and runs the selected task-graph scheduler.
struct ScheduleTaskGraphPass
    : public mlir::PassWrapper<ScheduleTaskGraphPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTaskGraphPass)

  Option<int64_t> cores{
      *this, "cores",
      llvm::cl::desc("Number of available analog/digital cores"),
      llvm::cl::init(1)};

  Option<int64_t> arraysPerCore{
      *this, "arrays-per-core",
      llvm::cl::desc("Number of physical analog arrays attached to each core"),
      llvm::cl::init(1)};

  Option<std::string> topology{
      *this, "topology",
      llvm::cl::desc("Hardware topology used by the task graph scheduler"),
      llvm::cl::init("mesh")};

  Option<int64_t> meshRows{*this, "mesh-rows",
                           llvm::cl::desc("Number of rows in a mesh topology"),
                           llvm::cl::init(1)};

  Option<int64_t> meshCols{
      *this, "mesh-cols",
      llvm::cl::desc(
          "Number of columns in a mesh topology, or 0 to infer from cores"),
      llvm::cl::init(0)};

  Option<std::string> schedule{
      *this, "schedule",
      llvm::cl::desc("Registered task graph scheduling algorithm to run"),
      llvm::cl::init("")};

  Option<int64_t> randomSeed{
      *this, "random-seed",
      llvm::cl::desc("Seed used by randomized task graph schedulers"),
      llvm::cl::init(0)};

  Option<std::string> greedyHeuristic{
      *this, "greedy-heuristic",
      llvm::cl::desc("Heuristic terms used by the greedy task scheduler, "
                     "joined with ',': transfer-cost, boundary-regret, "
                     "compact-region, lookahead=N, beam=N, scope=NAME"),
      llvm::cl::init("transfer-cost")};

  Option<std::string> annealingInitialSchedule{
      *this, "annealing-initial-schedule",
      llvm::cl::desc("Initial placement schedule used by the annealing task "
                     "scheduler: identity, random, snake, or greedy"),
      llvm::cl::init("snake")};

  Option<std::string> annealingMoveSet{
      *this, "annealing-move-set",
      llvm::cl::desc("Comma-separated simulated annealing perturbation moves "
                     "or presets: basic, basic-wide, all, "
                     "move-one-position, move-one-relocation, "
                     "swap-two-positions, adjacent-swap, segment-reverse, "
                     "segment-relocation, block-swap"),
      llvm::cl::init("basic")};

  Option<int64_t> annealingMoveRadius{
      *this, "annealing-move-radius",
      llvm::cl::desc("Maximum physical-array-order index distance for "
                     "single-position annealing moves, or 0 for unbounded"),
      llvm::cl::init(0)};

  Option<double> annealingInitialTemperature{
      *this, "annealing-initial-temperature",
      llvm::cl::desc("Initial temperature used by the annealing task "
                     "scheduler, or 0 to infer it from the initial score"),
      llvm::cl::init(0.0)};

  Option<double> annealingFinalTemperature{
      *this, "annealing-final-temperature",
      llvm::cl::desc("Final temperature used by the annealing task scheduler"),
      llvm::cl::init(1.0)};

  Option<double> annealingCoolingRate{
      *this, "annealing-cooling-rate",
      llvm::cl::desc("Multiplicative cooling rate used by the annealing task "
                     "scheduler"),
      llvm::cl::init(0.9)};

  Option<int64_t> annealingStepsPerTemperature{
      *this, "annealing-steps-per-temperature",
      llvm::cl::desc("Number of perturbation attempts per annealing "
                     "temperature"),
      llvm::cl::init(32)};

  Option<std::string> summaryOutput{
      *this, "summary-output",
      llvm::cl::desc("Optional CSV path where the scheduler appends a compact "
                     "result summary"),
      llvm::cl::init("")};

  ScheduleTaskGraphPass() = default;

  ScheduleTaskGraphPass(const ScheduleTaskGraphPass &pass)
      : PassWrapper(pass),
        cores(*this, "cores",
              llvm::cl::desc("Number of available analog/digital cores"),
              llvm::cl::init(1)),
        arraysPerCore(
            *this, "arrays-per-core",
            llvm::cl::desc(
                "Number of physical analog arrays attached to each core"),
            llvm::cl::init(1)),
        topology(*this, "topology",
                 llvm::cl::desc(
                     "Hardware topology used by the task graph scheduler"),
                 llvm::cl::init("mesh")),
        meshRows(*this, "mesh-rows",
                 llvm::cl::desc("Number of rows in a mesh topology"),
                 llvm::cl::init(1)),
        meshCols(*this, "mesh-cols",
                 llvm::cl::desc("Number of columns in a mesh topology, or 0 "
                                "to infer from cores"),
                 llvm::cl::init(0)),
        schedule(
            *this, "schedule",
            llvm::cl::desc("Registered task graph scheduling algorithm to run"),
            llvm::cl::init("")),
        randomSeed(
            *this, "random-seed",
            llvm::cl::desc("Seed used by randomized task graph schedulers"),
            llvm::cl::init(0)),
        greedyHeuristic(
            *this, "greedy-heuristic",
            llvm::cl::desc("Heuristic terms used by the greedy task "
                           "scheduler, joined with ',': transfer-cost, "
                           "boundary-regret, compact-region, lookahead=N, "
                           "beam=N, scope=NAME"),
            llvm::cl::init("transfer-cost")),
        annealingInitialSchedule(
            *this, "annealing-initial-schedule",
            llvm::cl::desc("Initial placement schedule used by the annealing "
                           "task scheduler: identity, random, snake, or "
                           "greedy"),
            llvm::cl::init("snake")),
        annealingMoveSet(
            *this, "annealing-move-set",
            llvm::cl::desc("Comma-separated simulated annealing perturbation "
                           "moves or presets: basic, basic-wide, all, "
                           "move-one-position, move-one-relocation, "
                           "swap-two-positions, adjacent-swap, "
                           "segment-reverse, segment-relocation, block-swap"),
            llvm::cl::init("basic")),
        annealingMoveRadius(
            *this, "annealing-move-radius",
            llvm::cl::desc("Maximum physical-array-order index distance for "
                           "single-position annealing moves, or 0 for "
                           "unbounded"),
            llvm::cl::init(0)),
        annealingInitialTemperature(
            *this, "annealing-initial-temperature",
            llvm::cl::desc("Initial temperature used by the annealing task "
                           "scheduler, or 0 to infer it from the initial "
                           "score"),
            llvm::cl::init(0.0)),
        annealingFinalTemperature(
            *this, "annealing-final-temperature",
            llvm::cl::desc(
                "Final temperature used by the annealing task scheduler"),
            llvm::cl::init(1.0)),
        annealingCoolingRate(
            *this, "annealing-cooling-rate",
            llvm::cl::desc("Multiplicative cooling rate used by the annealing "
                           "task scheduler"),
            llvm::cl::init(0.9)),
        annealingStepsPerTemperature(
            *this, "annealing-steps-per-temperature",
            llvm::cl::desc("Number of perturbation attempts per annealing "
                           "temperature"),
            llvm::cl::init(32)),
        summaryOutput(
            *this, "summary-output",
            llvm::cl::desc("Optional CSV path where the scheduler appends a "
                           "compact result summary"),
            llvm::cl::init("")) {
    cores = pass.cores;
    arraysPerCore = pass.arraysPerCore;
    topology = pass.topology;
    meshRows = pass.meshRows;
    meshCols = pass.meshCols;
    schedule = pass.schedule;
    randomSeed = pass.randomSeed;
    greedyHeuristic = pass.greedyHeuristic;
    annealingInitialSchedule = pass.annealingInitialSchedule;
    annealingMoveSet = pass.annealingMoveSet;
    annealingMoveRadius = pass.annealingMoveRadius;
    annealingInitialTemperature = pass.annealingInitialTemperature;
    annealingFinalTemperature = pass.annealingFinalTemperature;
    annealingCoolingRate = pass.annealingCoolingRate;
    annealingStepsPerTemperature = pass.annealingStepsPerTemperature;
    summaryOutput = pass.summaryOutput;
  }

  mlir::StringRef getArgument() const final {
    return "sculptor-schedule-task-graph";
  }

  mlir::StringRef getDescription() const final {
    return "Schedule Sculptor task graphs onto a hardware budget";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerScheduleTaskGraphPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SCHEDULETASKGRAPH_H
