#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_OPERANDRELATIONUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_OPERANDRELATIONUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace sculptor {
namespace layer_utils {
namespace detail {

// Recognizes linalg.generic wrappers whose body computes one op and yields it.
template <typename BodyOpTy>
inline bool genericYieldsSingleOpResult(Operation *op) {
  auto generic = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic)
    return false;

  Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return false;

  Block &block = region.front();
  if (block.empty())
    return false;

  auto it = block.begin();
  auto e = block.end();

  auto bodyOp = llvm::dyn_cast<BodyOpTy>(&*it++);
  if (!bodyOp || it == e)
    return false;

  auto yield = llvm::dyn_cast<linalg::YieldOp>(&*it++);
  if (!yield || it != e)
    return false;

  return yield.getNumOperands() == 1 &&
         yield.getOperand(0) == bodyOp.getResult();
}

} // namespace detail

// Checks the raw operand count while treating null operations as non-matches.
inline bool hasOperands(Operation *op, unsigned count) {
  return op && op->getNumOperands() == count;
}

// Checks DPS input count so destination operands are not counted as data
// inputs.
inline bool hasInputs(Operation *op, unsigned count) {
  auto dpsOp = llvm::dyn_cast_or_null<DestinationStyleOpInterface>(op);
  return dpsOp && dpsOp.getNumDpsInputs() == count;
}

// Checks both DPS data input count and total operand count.
inline bool hasDpsInputsAndOperands(Operation *op, unsigned inputCount,
                                    unsigned operandCount) {
  return hasInputs(op, inputCount) && hasOperands(op, operandCount);
}

// Returns the producer operation for an SSA value in a matcher graph.
inline Operation *producerOf(Value value) { return value.getDefiningOp(); }

// Looks through an operand only when the operation and index are valid.
inline Operation *operandProducer(Operation *op, unsigned operandIndex) {
  if (!op || operandIndex >= op->getNumOperands())
    return nullptr;

  return op->getOperand(operandIndex).getDefiningOp();
}

// Tests whether an operand has a producer before matching its shape.
inline bool hasOperandProducer(Operation *op, unsigned operandIndex) {
  return operandProducer(op, operandIndex) != nullptr;
}

// Returns a static rank-2 tensor type, or null when the value shape is not
// known well enough for layer matching.
inline RankedTensorType getStaticRank2TensorType(Value value) {
  auto type = llvm::dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 2)
    return {};
  return type;
}

// Returns a static rank-2 tensor type for a single-result operation.
inline RankedTensorType getStaticRank2TensorType(Operation *op) {
  if (!op || op->getNumResults() != 1)
    return {};
  return getStaticRank2TensorType(op->getResult(0));
}

// Casts a value producer to the expected op type without caller null checks.
template <typename OpTy> inline OpTy producerOfType(Value value) {
  return llvm::dyn_cast_or_null<OpTy>(producerOf(value));
}

// Fetches and casts an operand producer in one bounds-checked matcher step.
template <typename OpTy>
inline OpTy operandProducerOfType(Operation *op, unsigned operandIndex) {
  return llvm::dyn_cast_or_null<OpTy>(operandProducer(op, operandIndex));
}

// Tests whether an operand is produced by the expected operation type.
template <typename OpTy>
inline bool operandProducerIs(Operation *op, unsigned operandIndex) {
  return static_cast<bool>(operandProducerOfType<OpTy>(op, operandIndex));
}

// Matches two operands against an ordered pair of producer operation types.
template <typename FirstOpTy, typename SecondOpTy>
inline bool operandProducersAre(Operation *op, unsigned firstIndex = 0,
                                unsigned secondIndex = 1) {
  return operandProducerIs<FirstOpTy>(op, firstIndex) &&
         operandProducerIs<SecondOpTy>(op, secondIndex);
}

// Matches two operands against a pair of producer types in either order.
template <typename FirstOpTy, typename SecondOpTy>
inline bool operandProducersAreEither(Operation *op, unsigned firstIndex = 0,
                                      unsigned secondIndex = 1) {
  return operandProducersAre<FirstOpTy, SecondOpTy>(op, firstIndex,
                                                    secondIndex) ||
         operandProducersAre<SecondOpTy, FirstOpTy>(op, firstIndex,
                                                    secondIndex);
}

// Recognizes elementwise addf regions used to encode bias additions.
inline bool isAddfGeneric(Operation *op) {
  return detail::genericYieldsSingleOpResult<arith::AddFOp>(op);
}

// Recognizes elementwise tanh regions used at recurrent layer outputs.
inline bool isTanhGeneric(Operation *op) {
  return detail::genericYieldsSingleOpResult<math::TanhOp>(op);
}

// Recognizes elementwise mulf regions used in recurrent state updates.
inline bool isMulfGeneric(Operation *op) {
  return detail::genericYieldsSingleOpResult<arith::MulFOp>(op);
}

// Recognizes elementwise subf regions used in recurrent state updates.
inline bool isSubfGeneric(Operation *op) {
  return detail::genericYieldsSingleOpResult<arith::SubFOp>(op);
}

// Recognizes elementwise addi regions used by recurrent indexing scaffolds.
inline bool isAddiGeneric(Operation *op) {
  return detail::genericYieldsSingleOpResult<arith::AddIOp>(op);
}

// Recognizes elementwise muli regions used by recurrent indexing scaffolds.
inline bool isMuliGeneric(Operation *op) {
  return detail::genericYieldsSingleOpResult<arith::MulIOp>(op);
}

// Recognizes an arith.constant integer attribute with the expected i64 value.
inline bool constantOpHasI64Value(Operation *op, int64_t expected) {
  auto constant = llvm::dyn_cast_or_null<arith::ConstantOp>(op);
  if (!constant)
    return false;

  auto attr = llvm::dyn_cast<IntegerAttr>(constant.getValue());
  return attr && attr.getInt() == expected;
}

// Recognizes elementwise sigmoid regions lowered as neg-exp-add-div chains.
inline bool isSigmoidGeneric(Operation *op) {
  auto generic = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic)
    return false;

  Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return false;

  Block &block = region.front();
  if (block.empty())
    return false;

  auto it = block.begin();
  auto e = block.end();

  auto neg = llvm::dyn_cast<arith::NegFOp>(&*it++);
  auto exp = (it != e) ? llvm::dyn_cast<math::ExpOp>(&*it++) : math::ExpOp();
  auto add =
      (it != e) ? llvm::dyn_cast<arith::AddFOp>(&*it++) : arith::AddFOp();
  auto div =
      (it != e) ? llvm::dyn_cast<arith::DivFOp>(&*it++) : arith::DivFOp();
  auto yield =
      (it != e) ? llvm::dyn_cast<linalg::YieldOp>(&*it++) : linalg::YieldOp();
  if (!neg || !exp || !add || !div || !yield || it != e)
    return false;

  if (exp.getOperand() != neg.getResult())
    return false;

  Value unitConstant;
  if (add.getLhs() == exp.getResult())
    unitConstant = add.getRhs();
  else if (add.getRhs() == exp.getResult())
    unitConstant = add.getLhs();
  else
    return false;

  return div.getLhs() == unitConstant && div.getRhs() == add.getResult() &&
         yield.getNumOperands() == 1 && yield.getOperand(0) == div.getResult();
}

} // namespace layer_utils
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_OPERANDRELATIONUTILS_H
