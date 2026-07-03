#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHTYPES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHTYPES_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {
namespace task_schedulers {

struct HardwareBudget {
  int64_t numCores = 0;
  int64_t arraysPerCore = 0;
  std::string topology;
  int64_t meshRows = 0;
  int64_t meshCols = 0;
  int64_t numAnalogArrays = 0;
  int64_t randomSeed = 0;
  int64_t greedyLookahead = 1;
  std::string greedyCandidateScope = "diagonal";
  llvm::SmallVector<int64_t, 8> analogArrays;
};

struct TaskGraphNode {
  sculptor::TaskCreateOp op;
  unsigned index = 0;
  llvm::SmallVector<unsigned, 4> predecessors;
  llvm::SmallVector<unsigned, 4> successors;
};

struct TaskGraphDAG {
  llvm::SmallVector<TaskGraphNode, 16> nodes;
  llvm::SmallVector<Value, 8> logicalArrayResources;
  llvm::DenseMap<Value, unsigned> nodeIndexByTaskResult;
  unsigned dependencyCount = 0;
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

struct MatrixSetupGroupPlacement {
  unsigned matrixSetupTaskIndex = 0;
  int64_t physicalArrayId = 0;
};

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASK_SCHEDULERS_TASKGRAPHTYPES_H
