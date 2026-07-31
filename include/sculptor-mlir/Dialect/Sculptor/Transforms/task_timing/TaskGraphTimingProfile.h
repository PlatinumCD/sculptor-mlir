#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace sculptor {
namespace task_timing {

enum class MVMCostMode {
  Analog,
  Digital,
};

enum class TaskCostSource {
  StaticAnalysis,
  CalibratedFallback,
  ExplicitMetadata,
};

enum class TaskCostConfidence {
  High,
  Medium,
  Low,
};

llvm::StringRef stringifyTaskCostSource(TaskCostSource source);
llvm::StringRef stringifyTaskCostConfidence(TaskCostConfidence confidence);
std::optional<TaskCostSource> symbolizeTaskCostSource(llvm::StringRef value);
std::optional<TaskCostConfidence>
symbolizeTaskCostConfidence(llvm::StringRef value);

inline llvm::StringRef stringifyMVMCostMode(MVMCostMode mode) {
  switch (mode) {
  case MVMCostMode::Analog:
    return "analog";
  case MVMCostMode::Digital:
    return "digital";
  }
  llvm_unreachable("unknown MVM cost mode");
}

inline std::optional<MVMCostMode> symbolizeMVMCostMode(llvm::StringRef value) {
  if (value == "analog")
    return MVMCostMode::Analog;
  if (value == "digital")
    return MVMCostMode::Digital;
  return std::nullopt;
}

struct TimingModel {
  std::string costModel = "golem-qemu-v1";
  int64_t costModelRevision = 1;
  std::string compilerRevision = "unknown";
  std::string timingBoundary = "warm";
  std::string runtimeTaskPolicy = "lowest-local-task-index";
  std::string runtimeTransmitPolicy = "overlap-ready-tasks";
  std::string memoryBackend = "native-untimed";
  int64_t analogMVMLatencyNs = 100;
  int64_t analogIOBitsPerCycle = 256;
  bool analogIOShared = true;
  double digitalClockGHz = 1.0;
  int64_t digitalIssueWidth = 2;
  int64_t digitalVectorBitsPerCycle = 256;
  int64_t networkLinkBitsPerCycle = 32;
  int64_t networkHopLatencyCycles = 1;
  bool networkPipelined = true;
  int64_t networkLinkWordBits = 32;
  int64_t protocolWordsPerRoute = 5;
  int64_t nicInjectionWordsPerCycle = 1;
  int64_t rxDmaWordsPerCycle = 1;
  std::string routingPolicy = "xy";
  int64_t fixedRuntimeDispatchCycles = 8;
  int64_t fixedTaskEntryCycles = 4;
  int64_t fixedTaskExitCycles = 4;
};

struct TaskWorkloadFeatures {
  unsigned controlPredecessorCount = 0;
  unsigned dataPredecessorCount = 0;
  int64_t incomingDataBytes = 0;
  int64_t outgoingDataBytes = 0;
  int64_t digitalOps = 0;
  int64_t digitalReplacementOps = 0;
  uint64_t analogLoadBytes = 0;
  uint64_t analogExecutionCount = 0;
  uint64_t analogStoreBytes = 0;
  uint64_t staticElements = 0;
  uint64_t localBytesRead = 0;
  uint64_t localBytesWritten = 0;
  uint64_t loopIterations = 0;
};

struct TaskCost {
  uint64_t scalarInstructions = 0;
  uint64_t vectorInstructions = 0;
  uint64_t loadInstructions = 0;
  uint64_t storeInstructions = 0;
  uint64_t controlInstructions = 0;
  uint64_t runtimeDispatchCycles = 0;
  uint64_t taskEntryCycles = 0;
  uint64_t taskExitCycles = 0;
  double predictedCpuCycles = 0.0;
  TaskCostSource source = TaskCostSource::StaticAnalysis;
  TaskCostConfidence confidence = TaskCostConfidence::High;
};

struct TaskTiming {
  unsigned topologicalIndex = 0;
  unsigned localRuntimeIndex = 0;
  unsigned dependencyDepth = 0;
  TaskWorkloadFeatures workload;
  TaskCost cost;
  double analogLoadLatencyNs = 0.0;
  double analogExecuteLatencyNs = 0.0;
  double analogStoreLatencyNs = 0.0;
  double analogPipelineLatencyNs = 0.0;
  double intrinsicLatencyNs = 0.0;
  double earliestStartNs = 0.0;
  double earliestFinishNs = 0.0;
  double criticalPathRemainingNs = 0.0;
  double slackNs = 0.0;
  double incomingNetworkDelayNs = 0.0;
  double coreQueueDelayNs = 0.0;
  int64_t causalInputEdge = -1;
  int64_t causalPreviousTask = -1;
  bool isCritical = false;
};

struct IslandTimingProfile {
  unsigned islandId = 0;
  int64_t taskCount = 0;
  double totalWorkNs = 0.0;
  double analogWorkNs = 0.0;
  double digitalWorkNs = 0.0;
  double earliestStartNs = 0.0;
  double earliestFinishNs = 0.0;
  double criticalPathRemainingNs = 0.0;
  double slackNs = 0.0;
  bool isCritical = false;
};

struct TimedIslandEdge {
  unsigned producerIsland = 0;
  unsigned consumerIsland = 0;
  int64_t bytes = 0;
  double estimatedTransferNsPerHop = 0.0;
  double estimatedAdditionalHopNs = 0.0;
  double criticality = 0.0;
  double producerReadyTimeNs = 0.0;
  double consumerTimingPressure = 0.0;
};

struct SchedulingTimingProfile {
  llvm::SmallVector<TaskTiming, 16> tasks;
  llvm::SmallVector<IslandTimingProfile, 16> islands;
  llvm::SmallVector<TimedIslandEdge, 16> islandEdges;
  double criticalPathNs = 0.0;
  MVMCostMode mvmCostMode = MVMCostMode::Analog;
};

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILE_H
