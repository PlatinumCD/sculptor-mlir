#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHISLANDS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHISLANDS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphTypes.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct LogicalPlacementIsland {
  unsigned islandIndex = 0;
  unsigned matrixSetupTaskIndex = 0;
  llvm::SmallVector<unsigned, 4> mvmTaskIndices;
  llvm::SmallVector<unsigned, 16> digitalTaskIndices;
  llvm::SmallVector<unsigned, 16> taskIndices;
};

struct LogicalIslandCommunicationEdge {
  unsigned producerIsland = 0;
  unsigned consumerIsland = 0;
  int64_t byteSize = 0;
};

struct LogicalPlacementIslandGraph {
  llvm::SmallVector<LogicalPlacementIsland, 16> islands;
  llvm::SmallVector<LogicalIslandCommunicationEdge, 16> communicationEdges;
  llvm::DenseMap<unsigned, unsigned> islandByTaskIndex;
};

FailureOr<LogicalPlacementIslandGraph>
buildLogicalPlacementIslandGraph(const TaskGraphDAG &dag);

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHISLANDS_H
