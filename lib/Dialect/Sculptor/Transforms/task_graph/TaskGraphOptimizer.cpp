#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphOptimizer.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/ElementwiseFusionOptimizer.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/ElementwiseSlicesOptimizer.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/SegmentedConcatConsumerOptimizer.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/StreamingConvolutionOptimizer.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/VectorizedElementwiseOptimizer.h"

#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace sculptor {
namespace task_graph {

llvm::ArrayRef<TaskGraphOptimizationPattern>
getTaskGraphOptimizationPatterns() {
  static const TaskGraphOptimizationPattern patterns[] = {
      {"streaming-convolution", optimizeStreamingConvolution},
      {"elementwise-slices", optimizeElementwiseSlices},
      {"elementwise-fusion", optimizeElementwiseFusion},
      {"segmented-concat-consumer", optimizeSegmentedConcatConsumer},
      {"vectorized-elementwise", optimizeVectorizedElementwise},
  };
  return patterns;
}

LogicalResult optimizeTaskGraph(
    ModuleOp module, func::FuncOp taskGraphFunc, const TaskGraphDAG &dag,
    llvm::ArrayRef<llvm::StringRef> selectedPatterns, bool &changed) {
  llvm::StringMap<const TaskGraphOptimizationPattern *> patternByName;
  for (const TaskGraphOptimizationPattern &pattern :
       getTaskGraphOptimizationPatterns())
    patternByName.try_emplace(pattern.name, &pattern);

  TaskGraphDAG workingDag = dag;
  changed = false;
  for (llvm::StringRef patternName : selectedPatterns) {
    auto patternIt = patternByName.find(patternName);
    if (patternIt == patternByName.end()) {
      taskGraphFunc.emitError("unknown task-graph optimization pattern '")
          << patternName << "'";
      return failure();
    }

    while (true) {
      bool patternChanged = false;
      if (failed(patternIt->second->apply(module, taskGraphFunc, workingDag,
                                          patternChanged))) {
        taskGraphFunc.emitError("failed task-graph optimization pattern '")
            << patternName << "'";
        return failure();
      }
      if (!patternChanged)
        break;

      changed = true;
      auto nextDag = parseTaskGraphDAG(taskGraphFunc);
      if (failed(nextDag))
        return failure();
      workingDag = std::move(*nextDag);
    }
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
