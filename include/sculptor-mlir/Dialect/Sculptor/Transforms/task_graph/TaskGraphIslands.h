#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHISLANDS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHISLANDS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace task_graph {

struct LogicalPlacementIsland {
  unsigned islandIndex = 0;
  unsigned matrixSetupTaskIndex = 0;
  llvm::SmallVector<unsigned, 4> mvmTaskIndices;
  llvm::SmallVector<unsigned, 16> digitalTaskIndices;
  llvm::SmallVector<unsigned, 16> taskIndices;
};

struct IslandExecutionEdge {
  unsigned producerIsland = 0;
  unsigned consumerIsland = 0;
  bool controlDependency = false;
  bool dataDependency = false;
  int64_t transferredBytes = 0;
};

struct IslandExecutionGraph {
  llvm::SmallVector<IslandExecutionEdge, 16> edges;
  llvm::DenseMap<unsigned, llvm::SmallVector<unsigned, 4>> predecessors;
  llvm::DenseMap<unsigned, llvm::SmallVector<unsigned, 4>> successors;
};

struct IslandAffinityEdge {
  unsigned firstIsland = 0;
  unsigned secondIsland = 0;
  int64_t byteSize = 0;
};

struct IslandAffinityGraph {
  llvm::SmallVector<IslandAffinityEdge, 16> edges;
};

struct LogicalPlacementIslandGraph {
  llvm::SmallVector<LogicalPlacementIsland, 16> islands;
  llvm::DenseMap<unsigned, unsigned> islandByTaskIndex;
  IslandExecutionGraph executionGraph;
  IslandAffinityGraph affinityGraph;
};

FailureOr<LogicalPlacementIslandGraph>
buildLogicalPlacementIslandGraph(const TaskGraphDAG &dag,
                                 const TaskExecutionGraph &executionGraph);

LogicalResult
attachLogicalPlacementIslandIds(func::FuncOp taskGraphFunc,
                                const TaskGraphDAG &dag,
                                const LogicalPlacementIslandGraph &islandGraph);

FailureOr<LogicalPlacementIslandGraph>
loadLogicalPlacementIslandGraph(const TaskGraphDAG &dag,
                                const TaskExecutionGraph &executionGraph);

} // namespace task_graph
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_GRAPH_TASKGRAPHISLANDS_H
