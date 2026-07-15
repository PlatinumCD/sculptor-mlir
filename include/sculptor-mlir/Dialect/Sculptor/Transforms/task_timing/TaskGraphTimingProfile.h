#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILE_H

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_timing {

struct TimingModel {
  int64_t analogMVMLatencyNs = 100;
  int64_t analogIOBitsPerCycle = 256;
  bool analogIOShared = true;
  double digitalClockGHz = 1.0;
  int64_t digitalIssueWidth = 2;
  int64_t digitalVectorBitsPerCycle = 256;
  int64_t networkLinkBitsPerCycle = 32;
  int64_t networkHopLatencyCycles = 1;
  bool networkPipelined = true;
};

struct TaskTiming {
  unsigned topologicalIndex = 0;
  unsigned dependencyDepth = 0;
  unsigned controlPredecessorCount = 0;
  unsigned dataPredecessorCount = 0;
  int64_t incomingDataBytes = 0;
  int64_t outgoingDataBytes = 0;
  int64_t digitalOps = 0;
  double analogLoadLatencyNs = 0.0;
  double analogExecuteLatencyNs = 0.0;
  double analogStoreLatencyNs = 0.0;
  double intrinsicLatencyNs = 0.0;
  double earliestStartNs = 0.0;
  double earliestFinishNs = 0.0;
  double criticalPathRemainingNs = 0.0;
  double slackNs = 0.0;
  double incomingNetworkDelayNs = 0.0;
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
};

} // namespace task_timing
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_TIMING_TASKGRAPHTIMINGPROFILE_H
