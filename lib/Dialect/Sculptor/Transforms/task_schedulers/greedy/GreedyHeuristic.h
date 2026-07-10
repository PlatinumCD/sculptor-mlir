#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYHEURISTIC_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYHEURISTIC_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct GreedyHeuristicContext {
  const HardwareBudget &budget;
  llvm::ArrayRef<LogicalIslandCommunicationEdge> islandCommunicationEdges;
  const llvm::DenseMap<unsigned, int64_t> &coreByPlacedIsland;
  unsigned activeIsland = 0;
  unsigned activePlacementIndex = 0;
  unsigned totalPlacementCount = 0;
  std::optional<unsigned> firstTaskIsland;
  std::optional<unsigned> lastTaskIsland;
  std::optional<int64_t> bestTransferCost;
};

class GreedyHeuristic {
public:
  virtual ~GreedyHeuristic() = default;

  virtual llvm::StringRef getName() const = 0;
  virtual int64_t evaluate(const GreedyHeuristicContext &context) const = 0;
};

class TransferCostGreedyHeuristic final : public GreedyHeuristic {
public:
  llvm::StringRef getName() const final { return "transfer-cost"; }
  int64_t evaluate(const GreedyHeuristicContext &context) const final;
};

class BoundaryRegretGreedyHeuristic final : public GreedyHeuristic {
public:
  llvm::StringRef getName() const final { return "boundary-regret"; }
  int64_t evaluate(const GreedyHeuristicContext &context) const final;
};

class CompactRegionGreedyHeuristic final : public GreedyHeuristic {
public:
  llvm::StringRef getName() const final { return "compact-region"; }
  int64_t evaluate(const GreedyHeuristicContext &context) const final;
};

class CompositeGreedyHeuristic final : public GreedyHeuristic {
public:
  CompositeGreedyHeuristic(std::string name, bool boundaryRegret,
                           bool compactRegion)
      : name(std::move(name)), boundaryRegret(boundaryRegret),
        compactRegion(compactRegion) {}

  llvm::StringRef getName() const final { return name; }
  int64_t evaluate(const GreedyHeuristicContext &context) const final;

private:
  std::string name;
  bool boundaryRegret = false;
  bool compactRegion = false;
};

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_GREEDY_GREEDYHEURISTIC_H
