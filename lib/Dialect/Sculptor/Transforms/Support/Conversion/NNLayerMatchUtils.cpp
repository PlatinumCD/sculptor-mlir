#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace sculptor {
namespace nn_layer_match {

namespace {

bool returnsOperationResults(func::ReturnOp returnOp, Operation *op) {
  if (!returnOp || !op)
    return false;

  if (returnOp.getNumOperands() != op->getNumResults())
    return false;

  for (unsigned resultIndex = 0, resultCount = op->getNumResults();
       resultIndex < resultCount; ++resultIndex) {
    if (returnOp.getOperand(resultIndex) != op->getResult(resultIndex))
      return false;
  }

  return true;
}

bool isLayerSetupOp(Operation *op) {
  return op && llvm::isa<arith::ConstantOp>(op);
}

bool isUsedOnlyBy(Operation *producer, Operation *consumer) {
  if (!producer || !consumer)
    return false;

  for (Value result : producer->getResults()) {
    if (result.use_empty())
      return false;

    for (Operation *user : result.getUsers()) {
      if (user != consumer)
        return false;
    }
  }

  return true;
}

} // namespace

FailureOr<MatchedNNLayerBody>
matchSingleNNLayerBody(func::FuncOp func, StringRef operationName) {
  if (!func || !isSculptorLayer(func.getOperation()))
    return mlir::failure();

  if (!func.getBody().hasOneBlock())
    return mlir::failure();

  Block &entryBlock = func.front();
  auto returnOp =
      llvm::dyn_cast_or_null<func::ReturnOp>(entryBlock.getTerminator());
  if (!returnOp)
    return mlir::failure();

  Operation *layerOp = nullptr;
  llvm::SmallVector<Operation *, 4> setupOps;

  for (Operation &op : entryBlock) {
    if (&op == returnOp.getOperation())
      continue;

    if (op.getName().getStringRef() == operationName) {
      if (layerOp)
        return mlir::failure();

      layerOp = &op;
      continue;
    }

    if (layerOp)
      return mlir::failure();

    if (!isLayerSetupOp(&op))
      return mlir::failure();

    setupOps.push_back(&op);
  }

  if (!returnsOperationResults(returnOp, layerOp))
    return mlir::failure();

  for (Operation *setupOp : setupOps) {
    if (!isUsedOnlyBy(setupOp, layerOp))
      return mlir::failure();
  }

  return MatchedNNLayerBody{layerOp, returnOp};
}

} // namespace nn_layer_match
} // namespace sculptor
} // namespace mlir
