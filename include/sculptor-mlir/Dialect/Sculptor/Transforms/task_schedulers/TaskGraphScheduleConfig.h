#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULECONFIG_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULECONFIG_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

enum class GreedyCandidateScope {
  Cardinal,
  Diagonal,
  ProducerConsumer,
};

struct GreedyScheduleConfig {
  std::string specification = "transfer-cost";
  bool boundaryRegret = false;
  bool compactRegion = false;
  int64_t lookahead = 1;
  int64_t beamWidth = 1;
  GreedyCandidateScope candidateScope = GreedyCandidateScope::Diagonal;
};

enum class AnnealingInitialSchedule {
  Identity,
  Random,
  Snake,
  Greedy,
};

enum class AnnealingMoveKind {
  None,
  MoveOnePosition,
  MoveOneRelocation,
  SwapTwoPositions,
  AdjacentSwap,
  SegmentReverse,
  SegmentRelocation,
  BlockSwap,
};

struct AnnealingScheduleConfig {
  AnnealingInitialSchedule initialSchedule = AnnealingInitialSchedule::Snake;
  std::string moveSetSpecification = "basic";
  llvm::SmallVector<AnnealingMoveKind, 8> moveKinds{
      AnnealingMoveKind::MoveOnePosition, AnnealingMoveKind::SwapTwoPositions};
  int64_t moveRadius = 0;
  double initialTemperature = 0.0;
  double finalTemperature = 1.0;
  double coolingRate = 0.9;
  int64_t stepsPerTemperature = 32;
};

llvm::StringRef stringifyGreedyCandidateScope(GreedyCandidateScope scope);

FailureOr<GreedyScheduleConfig>
parseGreedyScheduleConfig(Operation *diagnosticOp,
                          llvm::StringRef specification);

FailureOr<AnnealingScheduleConfig> parseAnnealingScheduleConfig(
    Operation *diagnosticOp, llvm::StringRef initialSchedule,
    llvm::StringRef moveSet, int64_t moveRadius, double initialTemperature,
    double finalTemperature, double coolingRate, int64_t stepsPerTemperature);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULECONFIG_H
