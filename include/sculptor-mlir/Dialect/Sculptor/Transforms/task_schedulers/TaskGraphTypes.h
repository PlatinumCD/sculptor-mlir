#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHTYPES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHTYPES_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

using task_graph::IslandAffinityEdge;
using task_graph::LogicalPlacementIsland;
using task_graph::LogicalPlacementIslandGraph;
using task_graph::ResourceEdge;
using task_graph::TaskGraphDAG;
using task_graph::TaskGraphNode;

struct HardwareBudget {
  int64_t numCores = 0;
  int64_t arraysPerCore = 0;
  std::string topology;
  int64_t meshRows = 0;
  int64_t meshCols = 0;
  int64_t numAnalogArrays = 0;
  llvm::SmallVector<int64_t, 8> analogArrays;
};

struct LogicalArrayPlacement {
  int64_t logicalArrayIndex = 0;
  int64_t analogArrayIndex = 0;
};

struct PhysicalArrayPlacement {
  int64_t physicalArrayId = 0;
  int64_t coreId = 0;
  int64_t localArrayId = 0;
};

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHTYPES_H
