#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_ANALYZETASKGRAPHTIMING_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_ANALYZETASKGRAPHTIMING_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>

namespace mlir {
namespace sculptor {

struct AnalyzeTaskGraphTimingPass
    : public PassWrapper<AnalyzeTaskGraphTimingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeTaskGraphTimingPass)

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

  AnalyzeTaskGraphTimingPass() = default;

  AnalyzeTaskGraphTimingPass(const AnalyzeTaskGraphTimingPass &pass)
      : PassWrapper(pass),
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
            llvm::cl::init(true)) {
    analogMVMLatencyNs = pass.analogMVMLatencyNs;
    analogIOBitsPerCycle = pass.analogIOBitsPerCycle;
    analogIOShared = pass.analogIOShared;
    digitalClockGHz = pass.digitalClockGHz;
    digitalIssueWidth = pass.digitalIssueWidth;
    digitalVectorBitsPerCycle = pass.digitalVectorBitsPerCycle;
    networkLinkBitsPerCycle = pass.networkLinkBitsPerCycle;
    networkHopLatencyCycles = pass.networkHopLatencyCycles;
    networkPipelined = pass.networkPipelined;
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
