#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::sculptor;

namespace {

LogicalResult verifyBlockArgsMatchInputs(TaskRegionOp op, Block &body) {
  OperandRange inputs = op.getInputs();
  if (body.getNumArguments() != inputs.size()) {
    return op.emitOpError("expected region argument count (")
           << body.getNumArguments() << ") to match input count ("
           << inputs.size() << ")";
  }

  for (auto [index, input, blockArg] :
       llvm::enumerate(inputs, body.getArguments())) {
    if (input.getType() == blockArg.getType())
      continue;

    return op.emitOpError("expected region argument type at index ")
           << index << " (" << blockArg.getType() << ") to match input type ("
           << input.getType() << ")";
  }

  return success();
}

LogicalResult verifyYieldMatchesResults(TaskRegionOp op, YieldOp yieldOp) {
  OperandRange yieldedValues = yieldOp.getValues();
  ResultRange results = op.getResults();
  if (yieldedValues.size() != results.size()) {
    return op.emitOpError("expected yield value count (")
           << yieldedValues.size() << ") to match result count ("
           << results.size() << ")";
  }

  for (auto [index, yieldedValue, result] :
       llvm::enumerate(yieldedValues, results)) {
    if (yieldedValue.getType() == result.getType())
      continue;

    return op.emitOpError("expected yield type at index ")
           << index << " (" << yieldedValue.getType()
           << ") to match result type (" << result.getType() << ")";
  }

  return success();
}

} // namespace

mlir::LogicalResult mlir::sculptor::TaskRegionOp::verify() {
  if (getKind().empty())
    return emitOpError("expected kind to be a non-empty string");

  Region &region = getBody();
  if (!llvm::hasSingleElement(region))
    return emitOpError("expected exactly one region block");

  Block &block = region.front();
  if (failed(verifyBlockArgsMatchInputs(*this, block)))
    return failure();

  auto yieldOp = llvm::dyn_cast_or_null<YieldOp>(block.getTerminator());
  if (!yieldOp)
    return emitOpError("expected region to terminate with sculptor.yield");

  return verifyYieldMatchesResults(*this, yieldOp);
}

mlir::LogicalResult mlir::sculptor::YieldOp::verify() {
  Operation *parent = (*this)->getParentOp();
  if (!llvm::isa_and_nonnull<TaskRegionOp>(parent))
    return emitOpError("expected to terminate sculptor.task_region");

  return success();
}
