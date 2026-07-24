#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULECONFIG_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULECONFIG_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <variant>

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
  bool linkPressure = false;
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
  double coolingRate = 0.95;
  int64_t stepsPerTemperature = 64;
  int64_t plateauPatience = 10;
  double improvementThreshold = 0.001;
  int64_t minimumEpochs = 10;
  double plateauAcceptanceRate = 0.01;
  int64_t maximumEvaluations = 100000;
  double maximumRuntimeSeconds = 420.0;
};

struct RandomSchedulerOptions {
  int64_t randomSeed = 0;
};

struct SnakeSchedulerOptions {};

struct GreedySchedulerOptions {
  GreedyScheduleConfig greedy;
};

struct AnnealingSchedulerOptions {
  AnnealingScheduleConfig annealing;
  GreedyScheduleConfig greedyInitialPlacement;
  int64_t randomSeed = 0;
};

using TaskGraphSchedulerOptions =
    std::variant<RandomSchedulerOptions, SnakeSchedulerOptions,
                 GreedySchedulerOptions, AnnealingSchedulerOptions>;

FailureOr<TaskGraphSchedulerOptions> buildTaskGraphSchedulerOptions(
    Operation *diagnosticOp, llvm::StringRef schedule, int64_t randomSeed,
    llvm::StringRef greedyHeuristic,
    llvm::StringRef annealingInitialSchedule, llvm::StringRef annealingMoveSet,
    int64_t annealingMoveRadius, double annealingInitialTemperature,
    double annealingFinalTemperature, double annealingCoolingRate,
    int64_t annealingStepsPerTemperature, int64_t annealingPlateauPatience,
    double annealingImprovementThreshold, int64_t annealingMinimumEpochs,
    double annealingPlateauAcceptanceRate,
    int64_t annealingMaximumEvaluations,
    double annealingMaximumRuntimeSeconds);

void attachTaskGraphSchedulerOptionAttrs(
    Operation *op, Builder &builder,
    const TaskGraphSchedulerOptions &options);

llvm::StringRef stringifyGreedyCandidateScope(GreedyCandidateScope scope);

FailureOr<GreedyScheduleConfig>
parseGreedyScheduleConfig(Operation *diagnosticOp,
                          llvm::StringRef specification);

FailureOr<AnnealingScheduleConfig> parseAnnealingScheduleConfig(
    Operation *diagnosticOp, llvm::StringRef initialSchedule,
    llvm::StringRef moveSet, int64_t moveRadius, double initialTemperature,
    double finalTemperature, double coolingRate, int64_t stepsPerTemperature,
    int64_t plateauPatience, double improvementThreshold,
    int64_t minimumEpochs, double plateauAcceptanceRate,
    int64_t maximumEvaluations, double maximumRuntimeSeconds);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHSCHEDULECONFIG_H
