#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_ANALYZETASKGRAPHTIMING_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_ANALYZETASKGRAPHTIMING_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {

struct AnalyzeTaskGraphTimingPass
    : public PassWrapper<AnalyzeTaskGraphTimingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeTaskGraphTimingPass)

  Option<std::string> mvmCostMode{
      *this, "mvm-cost-mode",
      llvm::cl::desc("MVM placement cost model: analog or digital"),
      llvm::cl::init("analog")};

  Option<int64_t> analogMVMLatencyNs{
      *this, "analog-mvm-latency-ns",
      llvm::cl::desc(
          "Fixed latency of one analog MVM operation in nanoseconds"),
      llvm::cl::init(100)};

  Option<int64_t> analogIOBitsPerCycle{
      *this, "analog-io-bits-per-cycle",
      llvm::cl::desc("Analog load/store bandwidth in bits per cycle"),
      llvm::cl::init(256)};

  Option<bool> analogIOShared{
      *this, "analog-io-shared",
      llvm::cl::desc("Whether analog loads and stores share one bandwidth"),
      llvm::cl::init(true)};

  Option<double> digitalClockGHz{
      *this, "digital-clock-ghz",
      llvm::cl::desc("Digital processor clock frequency in gigahertz"),
      llvm::cl::init(1.0)};

  Option<int64_t> digitalIssueWidth{
      *this, "digital-issue-width",
      llvm::cl::desc("Maximum scalar digital operations issued per cycle"),
      llvm::cl::init(2)};

  Option<int64_t> digitalVectorBitsPerCycle{
      *this, "digital-vector-bits-per-cycle",
      llvm::cl::desc("Maximum digital vector throughput in bits per cycle"),
      llvm::cl::init(256)};

  Option<int64_t> fixedRuntimeDispatchCycles{
      *this, "fixed-runtime-dispatch-cycles",
      llvm::cl::desc("Fixed runtime task dispatch and bookkeeping cycles"),
      llvm::cl::init(8)};

  Option<int64_t> fixedTaskEntryCycles{
      *this, "fixed-task-entry-cycles",
      llvm::cl::desc("Fixed task adapter entry overhead in cycles"),
      llvm::cl::init(4)};

  Option<int64_t> fixedTaskExitCycles{
      *this, "fixed-task-exit-cycles",
      llvm::cl::desc("Fixed task adapter exit overhead in cycles"),
      llvm::cl::init(4)};

  Option<int64_t> networkLinkBitsPerCycle{
      *this, "network-link-bits-per-cycle",
      llvm::cl::desc("Network link bandwidth in bits per cycle"),
      llvm::cl::init(32)};

  Option<int64_t> networkHopLatencyCycles{
      *this, "network-hop-latency-cycles",
      llvm::cl::desc("Network forwarding latency per hop in cycles"),
      llvm::cl::init(1)};

  Option<bool> networkPipelined{
      *this, "network-pipelined",
      llvm::cl::desc("Whether communication across network hops is pipelined"),
      llvm::cl::init(true)};

  Option<int64_t> networkLinkWordBits{
      *this, "network-link-word-bits",
      llvm::cl::desc("Width of one routed network word in bits"),
      llvm::cl::init(32)};

  Option<int64_t> protocolWordsPerRoute{
      *this, "protocol-words-per-route",
      llvm::cl::desc("Protocol overhead words carried by every logical route"),
      llvm::cl::init(5)};

  Option<int64_t> nicInjectionWordsPerCycle{
      *this, "nic-injection-words-per-cycle",
      llvm::cl::desc("Source NIC injection throughput in words per cycle"),
      llvm::cl::init(1)};

  Option<int64_t> rxDmaWordsPerCycle{
      *this, "rx-dma-words-per-cycle",
      llvm::cl::desc("Receive DMA throughput in words per cycle"),
      llvm::cl::init(1)};

  Option<std::string> timingBoundary{
      *this, "timing-boundary", llvm::cl::desc("Timing boundary: warm or cold"),
      llvm::cl::init("warm")};

  Option<std::string> runtimeTaskPolicy{
      *this, "runtime-task-policy",
      llvm::cl::desc("Ready-task selection policy"),
      llvm::cl::init("lowest-local-task-index")};

  Option<std::string> runtimeTransmitPolicy{
      *this, "runtime-transmit-policy",
      llvm::cl::desc("Runtime task/transmit policy (overlap-ready-tasks only)"),
      llvm::cl::init("overlap-ready-tasks")};

  Option<std::string> memoryBackend{
      *this, "memory-backend", llvm::cl::desc("Local-memory timing backend"),
      llvm::cl::init("native-untimed")};

  Option<std::string> routingPolicy{*this, "routing-policy",
                                    llvm::cl::desc("Network routing policy"),
                                    llvm::cl::init("xy")};

  AnalyzeTaskGraphTimingPass() = default;

  AnalyzeTaskGraphTimingPass(const AnalyzeTaskGraphTimingPass &pass)
      : PassWrapper(pass),
        mvmCostMode(
            *this, "mvm-cost-mode",
            llvm::cl::desc("MVM placement cost model: analog or digital"),
            llvm::cl::init("analog")),
        analogMVMLatencyNs(
            *this, "analog-mvm-latency-ns",
            llvm::cl::desc(
                "Fixed latency of one analog MVM operation in nanoseconds"),
            llvm::cl::init(100)),
        analogIOBitsPerCycle(
            *this, "analog-io-bits-per-cycle",
            llvm::cl::desc("Analog load/store bandwidth in bits per cycle"),
            llvm::cl::init(256)),
        analogIOShared(
            *this, "analog-io-shared",
            llvm::cl::desc(
                "Whether analog loads and stores share one bandwidth"),
            llvm::cl::init(true)),
        digitalClockGHz(
            *this, "digital-clock-ghz",
            llvm::cl::desc("Digital processor clock frequency in gigahertz"),
            llvm::cl::init(1.0)),
        digitalIssueWidth(
            *this, "digital-issue-width",
            llvm::cl::desc(
                "Maximum scalar digital operations issued per cycle"),
            llvm::cl::init(2)),
        digitalVectorBitsPerCycle(
            *this, "digital-vector-bits-per-cycle",
            llvm::cl::desc(
                "Maximum digital vector throughput in bits per cycle"),
            llvm::cl::init(256)),
        fixedRuntimeDispatchCycles(
            *this, "fixed-runtime-dispatch-cycles",
            llvm::cl::desc(
                "Fixed runtime task dispatch and bookkeeping cycles"),
            llvm::cl::init(8)),
        fixedTaskEntryCycles(
            *this, "fixed-task-entry-cycles",
            llvm::cl::desc("Fixed task adapter entry overhead in cycles"),
            llvm::cl::init(4)),
        fixedTaskExitCycles(
            *this, "fixed-task-exit-cycles",
            llvm::cl::desc("Fixed task adapter exit overhead in cycles"),
            llvm::cl::init(4)),
        networkLinkBitsPerCycle(
            *this, "network-link-bits-per-cycle",
            llvm::cl::desc("Network link bandwidth in bits per cycle"),
            llvm::cl::init(32)),
        networkHopLatencyCycles(
            *this, "network-hop-latency-cycles",
            llvm::cl::desc("Network forwarding latency per hop in cycles"),
            llvm::cl::init(1)),
        networkPipelined(
            *this, "network-pipelined",
            llvm::cl::desc(
                "Whether communication across network hops is pipelined"),
            llvm::cl::init(true)),
        networkLinkWordBits(
            *this, "network-link-word-bits",
            llvm::cl::desc("Width of one routed network word in bits"),
            llvm::cl::init(32)),
        protocolWordsPerRoute(
            *this, "protocol-words-per-route",
            llvm::cl::desc(
                "Protocol overhead words carried by every logical route"),
            llvm::cl::init(5)),
        nicInjectionWordsPerCycle(
            *this, "nic-injection-words-per-cycle",
            llvm::cl::desc(
                "Source NIC injection throughput in words per cycle"),
            llvm::cl::init(1)),
        rxDmaWordsPerCycle(
            *this, "rx-dma-words-per-cycle",
            llvm::cl::desc("Receive DMA throughput in words per cycle"),
            llvm::cl::init(1)),
        timingBoundary(*this, "timing-boundary",
                       llvm::cl::desc("Timing boundary: warm or cold"),
                       llvm::cl::init("warm")),
        runtimeTaskPolicy(*this, "runtime-task-policy",
                          llvm::cl::desc("Ready-task selection policy"),
                          llvm::cl::init("lowest-local-task-index")),
        runtimeTransmitPolicy(
            *this, "runtime-transmit-policy",
            llvm::cl::desc("Runtime task/transmit overlap policy"),
            llvm::cl::init("overlap-ready-tasks")),
        memoryBackend(*this, "memory-backend",
                      llvm::cl::desc("Local-memory timing backend"),
                      llvm::cl::init("native-untimed")),
        routingPolicy(*this, "routing-policy",
                      llvm::cl::desc("Network routing policy"),
                      llvm::cl::init("xy")) {
    mvmCostMode = pass.mvmCostMode;
    analogMVMLatencyNs = pass.analogMVMLatencyNs;
    analogIOBitsPerCycle = pass.analogIOBitsPerCycle;
    analogIOShared = pass.analogIOShared;
    digitalClockGHz = pass.digitalClockGHz;
    digitalIssueWidth = pass.digitalIssueWidth;
    digitalVectorBitsPerCycle = pass.digitalVectorBitsPerCycle;
    fixedRuntimeDispatchCycles = pass.fixedRuntimeDispatchCycles;
    fixedTaskEntryCycles = pass.fixedTaskEntryCycles;
    fixedTaskExitCycles = pass.fixedTaskExitCycles;
    networkLinkBitsPerCycle = pass.networkLinkBitsPerCycle;
    networkHopLatencyCycles = pass.networkHopLatencyCycles;
    networkPipelined = pass.networkPipelined;
    networkLinkWordBits = pass.networkLinkWordBits;
    protocolWordsPerRoute = pass.protocolWordsPerRoute;
    nicInjectionWordsPerCycle = pass.nicInjectionWordsPerCycle;
    rxDmaWordsPerCycle = pass.rxDmaWordsPerCycle;
    timingBoundary = pass.timingBoundary;
    runtimeTaskPolicy = pass.runtimeTaskPolicy;
    runtimeTransmitPolicy = pass.runtimeTransmitPolicy;
    memoryBackend = pass.memoryBackend;
    routingPolicy = pass.routingPolicy;
  }

  StringRef getArgument() const final {
    return "sculptor-analyze-task-graph-timing";
  }

  StringRef getDescription() const final {
    return "Analyze task graph timing before or after physical placement";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerAnalyzeTaskGraphTimingPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_ANALYZETASKGRAPHTIMING_H
