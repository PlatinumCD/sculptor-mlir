#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduleConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace {

using namespace mlir::sculptor::task_schedulers;

static mlir::FailureOr<int64_t>
parsePositiveGreedyInteger(mlir::Operation *diagnosticOp, llvm::StringRef term,
                           llvm::StringRef value) {
  int64_t parsedValue = 0;
  if (value.getAsInteger(10, parsedValue) || parsedValue < 1) {
    diagnosticOp->emitError("expected Sculptor greedy heuristic term '")
        << term << "' to use a positive integer value";
    return mlir::failure();
  }
  return parsedValue;
}

static void addMoveKind(llvm::SmallVectorImpl<AnnealingMoveKind> &moveKinds,
                        AnnealingMoveKind moveKind) {
  if (!llvm::is_contained(moveKinds, moveKind))
    moveKinds.push_back(moveKind);
}

static void
addBasicMoveKinds(llvm::SmallVectorImpl<AnnealingMoveKind> &moveKinds) {
  addMoveKind(moveKinds, AnnealingMoveKind::MoveOnePosition);
  addMoveKind(moveKinds, AnnealingMoveKind::SwapTwoPositions);
}

static void
addBasicWideMoveKinds(llvm::SmallVectorImpl<AnnealingMoveKind> &moveKinds) {
  addMoveKind(moveKinds, AnnealingMoveKind::MoveOnePosition);
  addMoveKind(moveKinds, AnnealingMoveKind::MoveOneRelocation);
  addMoveKind(moveKinds, AnnealingMoveKind::SwapTwoPositions);
}

static void
addAllMoveKinds(llvm::SmallVectorImpl<AnnealingMoveKind> &moveKinds) {
  addBasicWideMoveKinds(moveKinds);
  addMoveKind(moveKinds, AnnealingMoveKind::AdjacentSwap);
  addMoveKind(moveKinds, AnnealingMoveKind::SegmentReverse);
  addMoveKind(moveKinds, AnnealingMoveKind::SegmentRelocation);
  addMoveKind(moveKinds, AnnealingMoveKind::BlockSwap);
}

static bool parseMoveSetTerm(llvm::StringRef term,
                             llvm::SmallVectorImpl<AnnealingMoveKind> &kinds) {
  if (term == "basic") {
    addBasicMoveKinds(kinds);
  } else if (term == "basic-wide") {
    addBasicWideMoveKinds(kinds);
  } else if (term == "all") {
    addAllMoveKinds(kinds);
  } else if (term == "move-one-position") {
    addMoveKind(kinds, AnnealingMoveKind::MoveOnePosition);
  } else if (term == "move-one-relocation") {
    addMoveKind(kinds, AnnealingMoveKind::MoveOneRelocation);
  } else if (term == "swap-two-positions") {
    addMoveKind(kinds, AnnealingMoveKind::SwapTwoPositions);
  } else if (term == "adjacent-swap") {
    addMoveKind(kinds, AnnealingMoveKind::AdjacentSwap);
  } else if (term == "segment-reverse") {
    addMoveKind(kinds, AnnealingMoveKind::SegmentReverse);
  } else if (term == "segment-relocation") {
    addMoveKind(kinds, AnnealingMoveKind::SegmentRelocation);
  } else if (term == "block-swap") {
    addMoveKind(kinds, AnnealingMoveKind::BlockSwap);
  } else {
    return false;
  }
  return true;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace task_schedulers {

llvm::StringRef stringifyGreedyCandidateScope(GreedyCandidateScope scope) {
  switch (scope) {
  case GreedyCandidateScope::Cardinal:
    return "cardinal";
  case GreedyCandidateScope::Diagonal:
    return "diagonal";
  case GreedyCandidateScope::ProducerConsumer:
    return "producer-consumer";
  }
  llvm_unreachable("unknown greedy candidate scope");
}

FailureOr<GreedyScheduleConfig>
parseGreedyScheduleConfig(Operation *diagnosticOp,
                          llvm::StringRef specification) {
  GreedyScheduleConfig config;
  config.specification = specification.str();

  llvm::SmallVector<llvm::StringRef, 4> terms;
  specification.split(terms, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (terms.empty()) {
    diagnosticOp->emitError(
        "expected Sculptor greedy heuristic to contain at least one "
        "comma-separated term");
    return failure();
  }

  for (llvm::StringRef term : terms) {
    term = term.trim();
    if (term == "transfer-cost")
      continue;
    if (term == "boundary-regret") {
      config.boundaryRegret = true;
      continue;
    }
    if (term == "compact-region") {
      config.compactRegion = true;
      continue;
    }

    if (term.consume_front("lookahead=")) {
      auto parsed = parsePositiveGreedyInteger(diagnosticOp, "lookahead", term);
      if (failed(parsed))
        return failure();
      config.lookahead = *parsed;
      continue;
    }
    if (term.consume_front("beam=") || term.consume_front("beam-width=")) {
      auto parsed = parsePositiveGreedyInteger(diagnosticOp, "beam", term);
      if (failed(parsed))
        return failure();
      config.beamWidth = *parsed;
      continue;
    }
    if (term.consume_front("scope=") ||
        term.consume_front("candidate-scope=")) {
      if (term == "cardinal") {
        config.candidateScope = GreedyCandidateScope::Cardinal;
      } else if (term == "diagonal") {
        config.candidateScope = GreedyCandidateScope::Diagonal;
      } else if (term == "producer-consumer") {
        config.candidateScope = GreedyCandidateScope::ProducerConsumer;
      } else {
        diagnosticOp->emitError("unknown Sculptor greedy candidate scope '")
            << term
            << "'; expected 'cardinal', 'diagonal', or 'producer-consumer'";
        return failure();
      }
      continue;
    }

    diagnosticOp->emitError("unknown Sculptor greedy heuristic term '")
        << term
        << "'; expected comma-separated terms: 'transfer-cost', "
           "'boundary-regret', 'compact-region', 'lookahead=N', 'beam=N', "
           "or 'scope=NAME'";
    return failure();
  }
  return config;
}

FailureOr<AnnealingScheduleConfig> parseAnnealingScheduleConfig(
    Operation *diagnosticOp, llvm::StringRef initialSchedule,
    llvm::StringRef moveSet, int64_t moveRadius, double initialTemperature,
    double finalTemperature, double coolingRate, int64_t stepsPerTemperature) {
  AnnealingScheduleConfig config;
  if (initialSchedule == "identity") {
    config.initialSchedule = AnnealingInitialSchedule::Identity;
  } else if (initialSchedule == "random") {
    config.initialSchedule = AnnealingInitialSchedule::Random;
  } else if (initialSchedule == "snake") {
    config.initialSchedule = AnnealingInitialSchedule::Snake;
  } else if (initialSchedule == "greedy") {
    config.initialSchedule = AnnealingInitialSchedule::Greedy;
  } else {
    diagnosticOp->emitError("unknown Sculptor annealing initial schedule '")
        << initialSchedule
        << "'; expected 'identity', 'random', 'snake', or 'greedy'";
    return failure();
  }

  config.moveSetSpecification = moveSet.str();
  config.moveKinds.clear();
  llvm::SmallVector<llvm::StringRef, 4> terms;
  moveSet.split(terms, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (terms.empty()) {
    diagnosticOp->emitError("expected Sculptor annealing move set to contain "
                            "at least one comma-separated term");
    return failure();
  }
  for (llvm::StringRef term : terms) {
    term = term.trim();
    if (parseMoveSetTerm(term, config.moveKinds))
      continue;
    diagnosticOp->emitError("unknown Sculptor annealing move-set term '")
        << term
        << "'; expected comma-separated terms: 'basic', 'basic-wide', 'all', "
           "'move-one-position', 'move-one-relocation', "
           "'swap-two-positions', 'adjacent-swap', 'segment-reverse', "
           "'segment-relocation', or 'block-swap'";
    return failure();
  }

  if (moveRadius < 0) {
    diagnosticOp->emitError(
        "expected Sculptor annealing move radius to be non-negative");
    return failure();
  }
  if (initialTemperature < 0.0) {
    diagnosticOp->emitError(
        "expected Sculptor annealing initial temperature to be non-negative");
    return failure();
  }
  if (finalTemperature <= 0.0) {
    diagnosticOp->emitError(
        "expected Sculptor annealing final temperature to be positive");
    return failure();
  }
  if (initialTemperature > 0.0 && initialTemperature <= finalTemperature) {
    diagnosticOp->emitError(
        "expected Sculptor annealing initial temperature to be greater than "
        "final temperature when explicitly set");
    return failure();
  }
  if (coolingRate <= 0.0 || coolingRate >= 1.0) {
    diagnosticOp->emitError(
        "expected Sculptor annealing cooling rate to be greater than zero and "
        "less than one");
    return failure();
  }
  if (stepsPerTemperature <= 0) {
    diagnosticOp->emitError(
        "expected Sculptor annealing steps per temperature to be positive");
    return failure();
  }

  config.moveRadius = moveRadius;
  config.initialTemperature = initialTemperature;
  config.finalTemperature = finalTemperature;
  config.coolingRate = coolingRate;
  config.stepsPerTemperature = stepsPerTemperature;
  return config;
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
