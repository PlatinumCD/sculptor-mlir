#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYSEARCHENGINE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYSEARCHENGINE_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <iterator>
#include <utility>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace greedy_detail {

template <typename State, typename Score>
struct LookaheadChoice {
  State immediateState;
  Score leafScore;
};

template <typename State, typename Score, typename IsCompleteFn,
          typename ExpandFn, typename GetScoreFn, typename BetterChoiceFn>
FailureOr<LookaheadChoice<State, Score>> chooseLookaheadState(
    const State &state, int64_t remainingLookahead,
    IsCompleteFn &&isComplete, ExpandFn &&expand, GetScoreFn &&getScore,
    BetterChoiceFn &&isBetterChoice) {
  auto expandedStates = expand(state, remainingLookahead > 1);
  if (expandedStates.empty())
    return failure();

  bool hasBest = false;
  LookaheadChoice<State, Score> best;
  for (State &expandedState : expandedStates) {
    Score leafScore = getScore(expandedState);
    if (remainingLookahead > 1 && !isComplete(expandedState)) {
      auto future = chooseLookaheadState<State, Score>(
          expandedState, remainingLookahead - 1, isComplete, expand, getScore,
          isBetterChoice);
      if (failed(future))
        continue;
      leafScore = future->leafScore;
    }

    if (!isBetterChoice(leafScore, expandedState, hasBest, best.leafScore,
                        best.immediateState, state))
      continue;
    hasBest = true;
    best.immediateState = std::move(expandedState);
    best.leafScore = leafScore;
  }

  if (!hasBest)
    return failure();
  return best;
}

template <typename State, typename Score, typename IsCompleteFn,
          typename ExpandFn, typename GetScoreFn, typename BetterChoiceFn>
FailureOr<State> runLookaheadSearch(State initialState, int64_t lookahead,
                                    IsCompleteFn &&isComplete,
                                    ExpandFn &&expand, GetScoreFn &&getScore,
                                    BetterChoiceFn &&isBetterChoice) {
  State state = std::move(initialState);
  while (!isComplete(state)) {
    auto selected = chooseLookaheadState<State, Score>(
        state, lookahead, isComplete, expand, getScore, isBetterChoice);
    if (failed(selected))
      return failure();
    state = std::move(selected->immediateState);
  }
  return state;
}

template <typename State, typename IsCompleteFn, typename ExpandFn,
          typename PruneFn>
FailureOr<State> runBeamSearch(State initialState, unsigned beamWidth,
                               IsCompleteFn &&isComplete, ExpandFn &&expand,
                               PruneFn &&prune) {
  llvm::SmallVector<State, 8> states;
  states.push_back(std::move(initialState));

  while (!isComplete(states.front())) {
    llvm::SmallVector<State, 16> nextStates;
    for (const State &state : states) {
      auto expandedStates = expand(state, /*recursivePruning=*/true);
      nextStates.append(std::make_move_iterator(expandedStates.begin()),
                        std::make_move_iterator(expandedStates.end()));
    }
    if (nextStates.empty())
      return failure();

    prune(nextStates, beamWidth);
    states = std::move(nextStates);
  }

  return std::move(states.front());
}

} // namespace greedy_detail
} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYSEARCHENGINE_H
