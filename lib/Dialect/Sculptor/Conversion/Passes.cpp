#include "sculptor-mlir/Dialect/Sculptor/Conversion/Passes.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/EmitRuntimeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/FinalizeGolemIntrinsics.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/LowerGolemToLLVMShims.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/AssembleTaskGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/ExpandMVMToGolem.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/MaterializeTasks.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ScheduleTaskGraph.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>
#include <string>

namespace mlir {
namespace sculptor {

namespace {

struct SculptorLowerToGolemPipelineOptions
    : public PassPipelineOptions<SculptorLowerToGolemPipelineOptions> {
  PassOptions::Option<int64_t> arrayRows{
      *this, "array-rows", llvm::cl::desc("Number of rows in an analog array"),
      llvm::cl::init(0)};

  PassOptions::Option<int64_t> arrayCols{
      *this, "array-cols",
      llvm::cl::desc("Number of columns in an analog array"),
      llvm::cl::init(0)};
};

struct SculptorLowerGolemToTaskGraphPipelineOptions
    : public PassPipelineOptions<SculptorLowerGolemToTaskGraphPipelineOptions> {
  PassOptions::Option<int64_t> cores{
      *this, "cores",
      llvm::cl::desc("Number of available analog/digital cores"),
      llvm::cl::init(1)};

  PassOptions::Option<int64_t> arraysPerCore{
      *this, "arrays-per-core",
      llvm::cl::desc("Number of physical analog arrays attached to each core"),
      llvm::cl::init(1)};

  PassOptions::Option<std::string> topology{
      *this, "topology",
      llvm::cl::desc("Hardware topology used by the task graph scheduler"),
      llvm::cl::init("mesh")};

  PassOptions::Option<int64_t> meshRows{
      *this, "mesh-rows", llvm::cl::desc("Number of rows in a mesh topology"),
      llvm::cl::init(1)};

  PassOptions::Option<int64_t> meshCols{
      *this, "mesh-cols",
      llvm::cl::desc(
          "Number of columns in a mesh topology, or 0 to infer from cores"),
      llvm::cl::init(0)};

  PassOptions::Option<std::string> schedule{
      *this, "schedule",
      llvm::cl::desc("Registered task graph scheduling algorithm to run"),
      llvm::cl::init("")};

  PassOptions::Option<std::string> greedyHeuristic{
      *this, "greedy-heuristic",
      llvm::cl::desc("Heuristic terms used by the greedy task scheduler, "
                     "joined with ',': transfer-cost, boundary-regret, "
                     "compact-region, lookahead=N, beam=N, scope=NAME"),
      llvm::cl::init("transfer-cost")};

  PassOptions::Option<std::string> annealingInitialSchedule{
      *this, "annealing-initial-schedule",
      llvm::cl::desc("Initial placement schedule used by the annealing task "
                     "scheduler: identity, random, snake, or greedy"),
      llvm::cl::init("snake")};

  PassOptions::Option<std::string> annealingMoveSet{
      *this, "annealing-move-set",
      llvm::cl::desc("Comma-separated simulated annealing perturbation moves "
                     "or presets: basic, basic-wide, all, "
                     "move-one-position, move-one-relocation, "
                     "swap-two-positions, adjacent-swap, segment-reverse, "
                     "segment-relocation, block-swap"),
      llvm::cl::init("basic")};

  PassOptions::Option<int64_t> annealingMoveRadius{
      *this, "annealing-move-radius",
      llvm::cl::desc("Maximum physical-array-order index distance for "
                     "single-position annealing moves, or 0 for unbounded"),
      llvm::cl::init(0)};

  PassOptions::Option<double> annealingInitialTemperature{
      *this, "annealing-initial-temperature",
      llvm::cl::desc("Initial temperature used by the annealing task "
                     "scheduler, or 0 to infer it from the initial score"),
      llvm::cl::init(0.0)};

  PassOptions::Option<double> annealingFinalTemperature{
      *this, "annealing-final-temperature",
      llvm::cl::desc("Final temperature used by the annealing task scheduler"),
      llvm::cl::init(1.0)};

  PassOptions::Option<double> annealingCoolingRate{
      *this, "annealing-cooling-rate",
      llvm::cl::desc("Multiplicative cooling rate used by the annealing task "
                     "scheduler"),
      llvm::cl::init(0.9)};

  PassOptions::Option<int64_t> annealingStepsPerTemperature{
      *this, "annealing-steps-per-temperature",
      llvm::cl::desc("Number of perturbation attempts per annealing "
                     "temperature"),
      llvm::cl::init(32)};
};

void buildSculptorLowerToGolemPipeline(
    OpPassManager &pm, const SculptorLowerToGolemPipelineOptions &options) {
  pm.addPass(std::make_unique<CanonicalizeLayersPass>());
  pm.addPass(std::make_unique<ExtractLayersPass>());
  pm.addPass(std::make_unique<ConvertLayersPass>());

  auto expandMVMToGolemPass = std::make_unique<ExpandMVMToGolemPass>();
  expandMVMToGolemPass->arrayRows = options.arrayRows;
  expandMVMToGolemPass->arrayCols = options.arrayCols;
  pm.addPass(std::move(expandMVMToGolemPass));

  pm.addPass(std::make_unique<MaterializeTasksPass>());
}

void buildSculptorLowerGolemToTaskGraphPipeline(
    OpPassManager &pm,
    const SculptorLowerGolemToTaskGraphPipelineOptions &options) {
  pm.addPass(std::make_unique<AssembleTaskGraphPass>());

  auto scheduleTaskGraphPass = std::make_unique<ScheduleTaskGraphPass>();
  scheduleTaskGraphPass->cores = options.cores;
  scheduleTaskGraphPass->arraysPerCore = options.arraysPerCore;
  scheduleTaskGraphPass->topology = options.topology;
  scheduleTaskGraphPass->meshRows = options.meshRows;
  scheduleTaskGraphPass->meshCols = options.meshCols;
  scheduleTaskGraphPass->schedule = options.schedule;
  scheduleTaskGraphPass->greedyHeuristic = options.greedyHeuristic;
  scheduleTaskGraphPass->annealingInitialSchedule =
      options.annealingInitialSchedule;
  scheduleTaskGraphPass->annealingMoveSet = options.annealingMoveSet;
  scheduleTaskGraphPass->annealingMoveRadius = options.annealingMoveRadius;
  scheduleTaskGraphPass->annealingInitialTemperature =
      options.annealingInitialTemperature;
  scheduleTaskGraphPass->annealingFinalTemperature =
      options.annealingFinalTemperature;
  scheduleTaskGraphPass->annealingCoolingRate = options.annealingCoolingRate;
  scheduleTaskGraphPass->annealingStepsPerTemperature =
      options.annealingStepsPerTemperature;
  pm.addPass(std::move(scheduleTaskGraphPass));

  pm.addPass(std::make_unique<LowerGolemToLLVMShimsPass>());
}

void registerSculptorPassPipelines() {
  PassPipelineRegistration<SculptorLowerToGolemPipelineOptions>(
      "sculptor-lower-to-golem",
      "Lower supported tensor/linalg and sculptor.nn layers to Sculptor Golem "
      "task IR",
      buildSculptorLowerToGolemPipeline);

  PassPipelineRegistration<SculptorLowerGolemToTaskGraphPipelineOptions>(
      "sculptor-lower-golem-to-task-graph",
      "Lower Sculptor Golem task IR to a scheduled task graph and LLVM runtime "
      "shims",
      buildSculptorLowerGolemToTaskGraphPipeline);
}

} // namespace

// Registers the conversion pass bundle exposed by this library entry point.
void registerSculptorConversionPasses() {
  registerLowerGolemToLLVMShimsPass();
  registerFinalizeGolemIntrinsicsPass();
  registerEmitRuntimeGraphPass();
  registerSculptorPassPipelines();
}

} // namespace sculptor
} // namespace mlir
