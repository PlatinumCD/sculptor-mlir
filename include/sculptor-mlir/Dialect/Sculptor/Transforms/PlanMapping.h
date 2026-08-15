#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANMAPPING_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANMAPPING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {

struct PlanMappingPass
    : public PassWrapper<PlanMappingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanMappingPass)

  Option<std::string> strategies{
      *this, "strategies",
      llvm::cl::desc("Ordered comma-separated mapping strategy pipeline; "
                     "setup-first is always prepended"),
      llvm::cl::init("fan-out-cut")};

  Option<std::string> objective{
      *this, "objective",
      llvm::cl::desc("Mapping objective (latency only for now)"),
      llvm::cl::init("latency")};

  Option<std::string> costProfile{
      *this, "cost-profile",
      llvm::cl::desc("JSON mapping cost profile; empty selects legacy-v1"),
      llvm::cl::init("")};

  Option<int64_t> meshRows{*this, "mesh-rows",
                           llvm::cl::desc("Number of mesh rows"),
                           llvm::cl::init(1)};

  Option<int64_t> meshCols{*this, "mesh-cols",
                           llvm::cl::desc("Number of mesh columns"),
                           llvm::cl::init(1)};

  Option<int64_t> arraysPerCore{
      *this, "arrays-per-core",
      llvm::cl::desc("Number of physical analog arrays per core"),
      llvm::cl::init(1)};

  Option<std::string> mvmBodyPolicy{
      *this, "mvm-body-policy",
      llvm::cl::desc("MVM array placement policy: packed fills each logical "
                     "tile, spread uses one logical tile per array, and "
                     "first-use-window binds matrices in scheduled-use order "
                     "inside the active sliding window; first-use-adaptive "
                     "permits the minimum spill needed to preserve RA "
                     "spatial parallelism"),
      llvm::cl::init("spread")};

  Option<std::string> setupBindingPolicy{
      *this, "setup-binding-policy",
      llvm::cl::desc("Matrix setup lane-binding policy: global or "
                     "consumer-anchored"),
      llvm::cl::init("global")};

  Option<bool> balanceDigitalWork{
      *this, "balance-digital-work",
      llvm::cl::desc("Balance weighted digital work across logical cores "
                     "within and across temporal phases"),
      llvm::cl::init(false)};

  Option<std::string> digitalSchedulingPolicy{
      *this, "digital-scheduling-policy",
      llvm::cl::desc("Digital logical-tile scheduling policy: affinity, "
                     "balanced, earliest-finish, progressive, or "
                     "sliding-window"),
      llvm::cl::init("")};

  Option<int64_t> digitalWindowSize{
      *this, "digital-window-size",
      llvm::cl::desc("Number of flexible logical digital lanes visible to "
                     "sliding-window scheduling"),
      llvm::cl::init(0)};

  Option<int64_t> arrayRows{
      *this, "array-rows",
      llvm::cl::desc("Number of rows in each physical analog array"),
      llvm::cl::init(1024)};

  Option<int64_t> arrayCols{
      *this, "array-cols",
      llvm::cl::desc("Number of columns in each physical analog array"),
      llvm::cl::init(512)};

  Option<int64_t> localMemoryBytesPerCore{
      *this, "local-memory-bytes-per-core",
      llvm::cl::desc("Local memory capacity of each core in bytes"),
      llvm::cl::init(64 * 1024 * 1024)};

  Option<int64_t> clockFrequencyHz{
      *this, "clock-frequency-hz",
      llvm::cl::desc("Digital processor clock frequency in hertz"),
      llvm::cl::init(1000000000)};

  Option<int64_t> analogMVMLatencyNs{
      *this, "analog-mvm-latency-ns",
      llvm::cl::desc("Fixed analog MVM execution latency in nanoseconds"),
      llvm::cl::init(100)};

  Option<int64_t> analogIOBitsPerCycle{
      *this, "analog-io-bits-per-cycle",
      llvm::cl::desc("Shared analog load/store bandwidth in bits per cycle"),
      llvm::cl::init(256)};

  Option<std::string> analogIOPolicy{
      *this, "analog-io-policy",
      llvm::cl::desc("Analog I/O bandwidth policy (shared only for now)"),
      llvm::cl::init("shared")};

  Option<std::string> analogArrayExecution{
      *this, "analog-array-execution",
      llvm::cl::desc("Cross-array execution policy (concurrent only for now)"),
      llvm::cl::init("concurrent")};

  Option<int64_t> digitalIssueWidth{
      *this, "digital-issue-width",
      llvm::cl::desc("Maximum scalar digital operations issued per cycle"),
      llvm::cl::init(2)};

  Option<int64_t> digitalVectorBitsPerCycle{
      *this, "digital-vector-bits-per-cycle",
      llvm::cl::desc("Maximum digital vector throughput in bits per cycle"),
      llvm::cl::init(256)};

  Option<int64_t> networkWordBits{
      *this, "network-word-bits",
      llvm::cl::desc("Width of one routed network word in bits"),
      llvm::cl::init(32)};

  Option<int64_t> networkHopCycles{
      *this, "network-hop-cycles",
      llvm::cl::desc("Network forwarding latency per hop in cycles"),
      llvm::cl::init(1)};

  Option<std::string> networkContentionModel{
      *this, "network-contention-model",
      llvm::cl::desc("Network contention model: none or link-serialized"),
      llvm::cl::init("link-serialized")};

  Option<bool> verifyPlan{*this, "verify-plan",
                          llvm::cl::desc("Verify the selected mapping plan"),
                          llvm::cl::init(true)};

  PlanMappingPass() = default;
  PlanMappingPass(const PlanMappingPass &pass) : PassWrapper(pass) {
    strategies = pass.strategies;
    objective = pass.objective;
    costProfile = pass.costProfile;
    meshRows = pass.meshRows;
    meshCols = pass.meshCols;
    arraysPerCore = pass.arraysPerCore;
    mvmBodyPolicy = pass.mvmBodyPolicy;
    setupBindingPolicy = pass.setupBindingPolicy;
    balanceDigitalWork = pass.balanceDigitalWork;
    digitalSchedulingPolicy = pass.digitalSchedulingPolicy;
    digitalWindowSize = pass.digitalWindowSize;
    arrayRows = pass.arrayRows;
    arrayCols = pass.arrayCols;
    localMemoryBytesPerCore = pass.localMemoryBytesPerCore;
    clockFrequencyHz = pass.clockFrequencyHz;
    analogMVMLatencyNs = pass.analogMVMLatencyNs;
    analogIOBitsPerCycle = pass.analogIOBitsPerCycle;
    analogIOPolicy = pass.analogIOPolicy;
    analogArrayExecution = pass.analogArrayExecution;
    digitalIssueWidth = pass.digitalIssueWidth;
    digitalVectorBitsPerCycle = pass.digitalVectorBitsPerCycle;
    networkWordBits = pass.networkWordBits;
    networkHopCycles = pass.networkHopCycles;
    networkContentionModel = pass.networkContentionModel;
    verifyPlan = pass.verifyPlan;
  }

  StringRef getArgument() const final { return "sculptor-plan-mapping"; }

  StringRef getDescription() const final {
    return "Plan an implementation and Resource Allocation Tree mapping";
  }

  void runOnOperation() override;
};

void registerPlanMappingPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANMAPPING_H
