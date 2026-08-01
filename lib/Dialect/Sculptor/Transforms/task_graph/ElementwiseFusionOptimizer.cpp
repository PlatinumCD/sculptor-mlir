#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/ElementwiseFusionOptimizer.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

static bool isStaticF32Tensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape() && tensor.getElementType().isF32();
}

static bool isTaskFunction(func::FuncOp function) {
  return function && function->hasAttr("sculptor.task_kind");
}

static bool hasOnlyParallelLoops(linalg::LinalgOp op) {
  return llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
    return type == utils::IteratorType::parallel;
  });
}

static linalg::GenericOp findElementwiseConsumer(linalg::LinalgOp producer) {
  if (isa<linalg::GenericOp>(producer) || !producer.hasPureTensorSemantics() ||
      producer->getNumResults() != 1 || !producer->getResult(0).hasOneUse() ||
      !isStaticF32Tensor(producer->getResult(0).getType()) ||
      !hasOnlyParallelLoops(producer))
    return {};

  Value value = producer->getResult(0);
  while (value.hasOneUse()) {
    Operation *user = *value.getUsers().begin();
    if (auto consumer = dyn_cast<linalg::GenericOp>(user)) {
      if (!consumer.hasPureTensorSemantics() ||
          consumer->getNumResults() != 1 || !hasOnlyParallelLoops(consumer) ||
          !isStaticF32Tensor(consumer->getResult(0).getType()))
        return {};
      return consumer;
    }

    if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user) ||
        user->getNumResults() != 1 || !user->getResult(0).hasOneUse() ||
        !isStaticF32Tensor(user->getResult(0).getType()))
      return {};
    value = user->getResult(0);
  }
  return {};
}

static bool isEligibleFusionOperand(OpOperand *operand) {
  Operation *producer = operand->get().getDefiningOp();
  Operation *consumer = operand->getOwner();
  if (!producer || !producer->hasOneUse() || !isMemoryEffectFree(producer))
    return false;

  func::FuncOp producerFunction = producer->getParentOfType<func::FuncOp>();
  func::FuncOp consumerFunction = consumer->getParentOfType<func::FuncOp>();
  if (!isTaskFunction(producerFunction) || producerFunction != consumerFunction)
    return false;

  return isStaticF32Tensor(operand->get().getType());
}

static int64_t countLinalgOps(func::FuncOp function) {
  int64_t count = 0;
  function.walk([&](linalg::LinalgOp) { ++count; });
  return count;
}

} // namespace

LogicalResult optimizeElementwiseFusion(ModuleOp module,
                                        func::FuncOp taskGraphFunc,
                                        const TaskGraphDAG &dag,
                                        bool &changed) {
  (void)taskGraphFunc;
  (void)dag;
  changed = false;

  SmallVector<linalg::LinalgOp, 8> producers;
  llvm::SmallPtrSet<Operation *, 8> affectedFunctions;
  module.walk([&](linalg::LinalgOp op) {
    func::FuncOp function = op->getParentOfType<func::FuncOp>();
    linalg::GenericOp consumer = findElementwiseConsumer(op);
    if (!isTaskFunction(function) || !consumer ||
        consumer->getParentOfType<func::FuncOp>() != function)
      return;
    producers.push_back(op);
    affectedFunctions.insert(function.getOperation());
  });

  if (producers.empty())
    return success();

  IRRewriter rewriter(module.getContext());
  for (linalg::LinalgOp producer : producers) {
    if (!producer->getBlock())
      continue;
    rewriter.setInsertionPoint(producer);
    if (succeeded(linalg::generalizeNamedOp(rewriter, producer)))
      changed = true;
  }

  for (Operation *operation : affectedFunctions) {
    auto function = cast<func::FuncOp>(operation);
    int64_t before = countLinalgOps(function);

    RewritePatternSet patterns(module.getContext());
    linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                 isEligibleFusionOperand);
    linalg::populateFoldReshapeOpsByExpansionPatterns(patterns,
                                                      isEligibleFusionOperand);
    tensor::populateBubbleUpExpandShapePatterns(patterns);
    if (failed(applyPatternsGreedily(
            function, std::move(patterns),
            GreedyRewriteConfig().setUseTopDownTraversal())))
      return function.emitError("failed to fuse elementwise task operations");

    changed |= countLinalgOps(function) < before;
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
