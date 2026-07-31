#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphRuntimeOrder.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace sculptor {
namespace task_graph {

llvm::SmallVector<unsigned>
buildLocalRuntimeOrder(llvm::ArrayRef<int64_t> coreByTask) {
  llvm::DenseMap<int64_t, unsigned> nextIndexByCore;
  llvm::SmallVector<unsigned> localRuntimeIndexByTask;
  localRuntimeIndexByTask.reserve(coreByTask.size());
  for (int64_t core : coreByTask)
    localRuntimeIndexByTask.push_back(nextIndexByCore[core]++);
  return localRuntimeIndexByTask;
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
