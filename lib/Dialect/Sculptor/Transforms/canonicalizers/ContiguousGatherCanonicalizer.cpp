#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <optional>

namespace {

using namespace mlir;

struct LinearForm {
  SmallVector<int64_t> coefficients;
  int64_t constant = 0;
};

struct LinearSequence {
  int64_t slope = 0;
  int64_t offset = 0;
};

std::optional<int64_t> checkedAdd(int64_t left, int64_t right) {
  return llvm::checkedAdd(left, right);
}

std::optional<int64_t> checkedMultiply(int64_t left, int64_t right) {
  return llvm::checkedMul(left, right);
}

std::optional<int64_t> checkedSubtract(int64_t left, int64_t right) {
  return llvm::checkedSub(left, right);
}

std::optional<LinearForm> addForms(const LinearForm &left,
                                   const LinearForm &right, int64_t sign = 1) {
  if (left.coefficients.size() != right.coefficients.size())
    return std::nullopt;
  LinearForm result;
  result.coefficients.reserve(left.coefficients.size());
  for (auto [lhs, rhs] : llvm::zip_equal(left.coefficients,
                                         right.coefficients)) {
    std::optional<int64_t> signedRight = checkedMultiply(rhs, sign);
    if (!signedRight)
      return std::nullopt;
    std::optional<int64_t> coefficient = checkedAdd(lhs, *signedRight);
    if (!coefficient)
      return std::nullopt;
    result.coefficients.push_back(*coefficient);
  }
  std::optional<int64_t> signedConstant =
      checkedMultiply(right.constant, sign);
  if (!signedConstant)
    return std::nullopt;
  std::optional<int64_t> constant =
      checkedAdd(left.constant, *signedConstant);
  if (!constant)
    return std::nullopt;
  result.constant = *constant;
  return result;
}

std::optional<LinearForm> multiplyForm(const LinearForm &form,
                                       int64_t factor) {
  LinearForm result;
  result.coefficients.reserve(form.coefficients.size());
  for (int64_t coefficient : form.coefficients) {
    std::optional<int64_t> product =
        checkedMultiply(coefficient, factor);
    if (!product)
      return std::nullopt;
    result.coefficients.push_back(*product);
  }
  std::optional<int64_t> constant = checkedMultiply(form.constant, factor);
  if (!constant)
    return std::nullopt;
  result.constant = *constant;
  return result;
}

bool isConstantForm(const LinearForm &form) {
  return llvm::all_of(form.coefficients,
                      [](int64_t coefficient) { return coefficient == 0; });
}

std::optional<LinearForm> convertAffineExpression(AffineExpr expression,
                                                  unsigned dimensionCount) {
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= dimensionCount)
      return std::nullopt;
    LinearForm result;
    result.coefficients.assign(dimensionCount, 0);
    result.coefficients[dimension.getPosition()] = 1;
    return result;
  }
  if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    LinearForm result;
    result.coefficients.assign(dimensionCount, 0);
    result.constant = constant.getValue();
    return result;
  }
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return std::nullopt;
  std::optional<LinearForm> lhs =
      convertAffineExpression(binary.getLHS(), dimensionCount);
  std::optional<LinearForm> rhs =
      convertAffineExpression(binary.getRHS(), dimensionCount);
  if (!lhs || !rhs)
    return std::nullopt;
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    return addForms(*lhs, *rhs);
  case AffineExprKind::Mul:
    if (isConstantForm(*lhs))
      return multiplyForm(*rhs, lhs->constant);
    if (isConstantForm(*rhs))
      return multiplyForm(*lhs, rhs->constant);
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

std::optional<LinearForm>
flattenMappedIndices(AffineMap map, RankedTensorType tensorType) {
  if (map.getNumSymbols() != 0 ||
      map.getNumResults() != static_cast<unsigned>(tensorType.getRank()) ||
      !tensorType.hasStaticShape())
    return std::nullopt;
  LinearForm result;
  result.coefficients.assign(map.getNumDims(), 0);
  int64_t stride = 1;
  for (int64_t dimension = tensorType.getRank() - 1; dimension >= 0;
       --dimension) {
    std::optional<LinearForm> index = convertAffineExpression(
        map.getResult(dimension), map.getNumDims());
    if (!index)
      return std::nullopt;
    std::optional<LinearForm> contribution = multiplyForm(*index, stride);
    if (!contribution)
      return std::nullopt;
    std::optional<LinearForm> sum = addForms(result, *contribution);
    if (!sum)
      return std::nullopt;
    result = std::move(*sum);
    std::optional<int64_t> nextStride =
        checkedMultiply(stride, tensorType.getDimSize(dimension));
    if (!nextStride)
      return std::nullopt;
    stride = *nextStride;
  }
  return result;
}

std::optional<int64_t> getIntegerConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  return integer ? std::optional<int64_t>{integer.getInt()} : std::nullopt;
}

class LinearIotaMatcher {
public:
  std::optional<LinearSequence> match(Value value) {
    auto cached = cache.find(value);
    if (cached != cache.end())
      return cached->second;
    if (!active.insert(value).second)
      return std::nullopt;
    std::optional<LinearSequence> result = matchImpl(value);
    active.erase(value);
    cache[value] = result;
    return result;
  }

private:
  std::optional<LinearSequence> matchImpl(Value value) {
    auto resultType = dyn_cast<RankedTensorType>(value.getType());
    if (!resultType || !resultType.hasStaticShape() ||
        !resultType.getElementType().isIntOrIndex())
      return std::nullopt;

    if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
      auto sourceType = dyn_cast<RankedTensorType>(collapse.getSrcType());
      if (!sourceType || !sourceType.hasStaticShape() ||
          sourceType.getNumElements() != resultType.getNumElements())
        return std::nullopt;
      return match(collapse.getSrc());
    }
    if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
      auto sourceType = dyn_cast<RankedTensorType>(expand.getSrcType());
      if (!sourceType || !sourceType.hasStaticShape() ||
          sourceType.getNumElements() != resultType.getNumElements())
        return std::nullopt;
      return match(expand.getSrc());
    }

    auto generic = value.getDefiningOp<linalg::GenericOp>();
    if (!generic || generic->getNumResults() != 1 ||
        generic.getResult(0) != value || !generic.getRegion().hasOneBlock())
      return std::nullopt;
    Block &body = generic.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return std::nullopt;

    unsigned loopCount = generic.getNumLoops();
    DenseMap<Value, LinearForm> forms;
    auto inputOperands = generic.getDpsInputOperands();
    SmallVector<AffineMap> indexingMaps = generic.getIndexingMapsArray();
    for (auto [index, operand] : llvm::enumerate(inputOperands)) {
      std::optional<LinearSequence> sourceSequence = match(operand->get());
      auto sourceType =
          dyn_cast<RankedTensorType>(operand->get().getType());
      if (!sourceSequence || !sourceType || index >= indexingMaps.size())
        return std::nullopt;
      std::optional<LinearForm> form =
          flattenMappedIndices(indexingMaps[index], sourceType);
      if (!form)
        return std::nullopt;
      form = multiplyForm(*form, sourceSequence->slope);
      if (!form)
        return std::nullopt;
      std::optional<int64_t> constant =
          checkedAdd(form->constant, sourceSequence->offset);
      if (!constant)
        return std::nullopt;
      form->constant = *constant;
      forms[body.getArgument(index)] = std::move(*form);
    }

    std::function<std::optional<LinearForm>(Value)> evaluate =
        [&](Value scalar) -> std::optional<LinearForm> {
      auto known = forms.find(scalar);
      if (known != forms.end())
        return known->second;
      Operation *operation = scalar.getDefiningOp();
      if (!operation)
        return std::nullopt;
      LinearForm result;
      result.coefficients.assign(loopCount, 0);
      if (std::optional<int64_t> constant = getIntegerConstant(scalar)) {
        result.constant = *constant;
      } else if (operation->getParentRegion() != &generic.getRegion()) {
        return std::nullopt;
      } else if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        auto integer = dyn_cast<IntegerAttr>(constant.getValue());
        if (!integer)
          return std::nullopt;
        result.constant = integer.getInt();
      } else if (auto index = dyn_cast<linalg::IndexOp>(operation)) {
        if (index.getDim() >= loopCount)
          return std::nullopt;
        result.coefficients[index.getDim()] = 1;
      } else if (auto cast = dyn_cast<arith::IndexCastOp>(operation)) {
        std::optional<LinearForm> input = evaluate(cast.getIn());
        if (!input)
          return std::nullopt;
        result = std::move(*input);
      } else if (auto cast = dyn_cast<arith::IndexCastUIOp>(operation)) {
        std::optional<LinearForm> input = evaluate(cast.getIn());
        if (!input)
          return std::nullopt;
        result = std::move(*input);
      } else if (auto add = dyn_cast<arith::AddIOp>(operation)) {
        std::optional<LinearForm> lhs = evaluate(add.getLhs());
        std::optional<LinearForm> rhs = evaluate(add.getRhs());
        if (!lhs || !rhs)
          return std::nullopt;
        std::optional<LinearForm> sum = addForms(*lhs, *rhs);
        if (!sum)
          return std::nullopt;
        result = std::move(*sum);
      } else if (auto subtract = dyn_cast<arith::SubIOp>(operation)) {
        std::optional<LinearForm> lhs = evaluate(subtract.getLhs());
        std::optional<LinearForm> rhs = evaluate(subtract.getRhs());
        if (!lhs || !rhs)
          return std::nullopt;
        std::optional<LinearForm> difference = addForms(*lhs, *rhs, -1);
        if (!difference)
          return std::nullopt;
        result = std::move(*difference);
      } else if (auto multiply = dyn_cast<arith::MulIOp>(operation)) {
        std::optional<LinearForm> lhs = evaluate(multiply.getLhs());
        std::optional<LinearForm> rhs = evaluate(multiply.getRhs());
        if (!lhs || !rhs)
          return std::nullopt;
        std::optional<LinearForm> product;
        if (isConstantForm(*lhs))
          product = multiplyForm(*rhs, lhs->constant);
        else if (isConstantForm(*rhs))
          product = multiplyForm(*lhs, rhs->constant);
        if (!product)
          return std::nullopt;
        result = std::move(*product);
      } else {
        return std::nullopt;
      }
      forms[scalar] = result;
      return result;
    };

    std::optional<LinearForm> yielded = evaluate(yield.getValues().front());
    if (!yielded)
      return std::nullopt;
    unsigned outputMapIndex = inputOperands.size();
    if (outputMapIndex >= indexingMaps.size())
      return std::nullopt;
    std::optional<LinearForm> expected =
        flattenMappedIndices(indexingMaps[outputMapIndex], resultType);
    if (!expected)
      return std::nullopt;
    SmallVector<int64_t> loopRanges = generic.getStaticLoopRanges();
    if (loopRanges.size() != yielded->coefficients.size())
      return std::nullopt;
    for (auto [dimension, extent] : llvm::enumerate(loopRanges)) {
      // Coefficients of a zero-valued singleton induction variable are
      // semantically irrelevant. Torch-mlir uses these dimensions heavily in
      // broadcasted channel/spatial index construction.
      if (extent == 1) {
        yielded->coefficients[dimension] = 0;
        expected->coefficients[dimension] = 0;
      }
    }
    std::optional<int64_t> slope;
    for (auto [actual, reference] :
         llvm::zip_equal(yielded->coefficients, expected->coefficients)) {
      if (reference == 0) {
        if (actual != 0)
          return std::nullopt;
        continue;
      }
      if (actual % reference != 0)
        return std::nullopt;
      int64_t candidate = actual / reference;
      if (slope && *slope != candidate)
        return std::nullopt;
      slope = candidate;
    }
    int64_t resolvedSlope = slope.value_or(0);
    std::optional<int64_t> scaledExpected =
        checkedMultiply(expected->constant, resolvedSlope);
    if (!scaledExpected)
      return std::nullopt;
    std::optional<int64_t> offset =
        checkedSubtract(yielded->constant, *scaledExpected);
    if (!offset)
      return std::nullopt;
    return LinearSequence{resolvedSlope, *offset};
  }

  DenseMap<Value, std::optional<LinearSequence>> cache;
  DenseSet<Value> active;
};

bool matchesNormalizedIndex(Value value, Value input, int64_t sourceExtent) {
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    value = cast.getIn();
  else if (auto cast = value.getDefiningOp<arith::IndexCastUIOp>())
    value = cast.getIn();
  if (value == input)
    return true;

  auto select = value.getDefiningOp<arith::SelectOp>();
  if (!select || select.getFalseValue() != input)
    return false;
  auto add = select.getTrueValue().getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  Value extentValue;
  if (add.getLhs() == input)
    extentValue = add.getRhs();
  else if (add.getRhs() == input)
    extentValue = add.getLhs();
  else
    return false;
  if (getIntegerConstant(extentValue) != sourceExtent)
    return false;

  auto compare = select.getCondition().getDefiningOp<arith::CmpIOp>();
  return compare && compare.getPredicate() == arith::CmpIPredicate::slt &&
         compare.getLhs() == input && getIntegerConstant(compare.getRhs()) == 0;
}

void eraseDeadProducerTree(ArrayRef<Operation *> roots,
                           IRRewriter &rewriter) {
  SmallVector<Operation *> pending(roots.begin(), roots.end());
  DenseSet<Operation *> erased;
  while (!pending.empty()) {
    Operation *operation = pending.pop_back_val();
    if (!operation || erased.contains(operation))
      continue;
    if (!operation->getBlock() || !isOpTriviallyDead(operation))
      continue;
    SmallVector<Operation *> producers;
    for (Value operand : operation->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        producers.push_back(producer);
    erased.insert(operation);
    rewriter.eraseOp(operation);
    pending.append(producers);
  }
}

} // namespace

namespace mlir {
namespace sculptor {

void canonicalizeContiguousGathers(func::FuncOp function) {
  LinearIotaMatcher iotaMatcher;
  SmallVector<linalg::GenericOp> candidates;
  function.walk([&](linalg::GenericOp generic) {
    if (generic.getNumDpsInputs() == 1 && generic.getNumDpsInits() == 1 &&
        generic->getNumResults() == 1)
      candidates.push_back(generic);
  });

  IRRewriter rewriter(function.getContext());
  for (linalg::GenericOp generic : candidates) {
    if (!generic || !generic->getBlock() || !generic.getRegion().hasOneBlock())
      continue;
    auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    Value indexTensor = generic.getDpsInputOperand(0)->get();
    auto indexType = dyn_cast<RankedTensorType>(indexTensor.getType());
    if (!resultType || !indexType || !resultType.hasStaticShape() ||
        !indexType.hasStaticShape() || resultType.getRank() != 1 ||
        indexType != RankedTensorType::get(resultType.getShape(),
                                           indexType.getElementType(),
                                           indexType.getEncoding()))
      continue;

    std::optional<LinearSequence> sequence = iotaMatcher.match(indexTensor);
    if (!sequence || sequence->slope != 1 || sequence->offset < 0)
      continue;
    int64_t offset = sequence->offset;

    Block &body = generic.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      continue;
    auto extract = yield.getValues().front().getDefiningOp<tensor::ExtractOp>();
    if (!extract || extract.getIndices().size() != 1)
      continue;
    Value source = extract.getTensor();
    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        sourceType.getRank() != 1 ||
        sourceType.getElementType() != resultType.getElementType() ||
        llvm::is_contained(generic->getOperands(), source) ||
        !matchesNormalizedIndex(extract.getIndices().front(),
                                body.getArgument(0),
                                sourceType.getDimSize(0)))
      continue;

    int64_t length = resultType.getDimSize(0);
    std::optional<int64_t> end = checkedAdd(offset, length);
    if (!end || *end > sourceType.getDimSize(0))
      continue;

    Operation *indexProducer = indexTensor.getDefiningOp();
    Operation *initializer =
        generic.getDpsInitOperand(0)->get().getDefiningOp();
    rewriter.setInsertionPoint(generic);
    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(offset)};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(length)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
    auto slice = rewriter.create<tensor::ExtractSliceOp>(
        generic.getLoc(), resultType, source, offsets, sizes, strides);
    generic.getResult(0).replaceAllUsesWith(slice.getResult());
    rewriter.eraseOp(generic);
    eraseDeadProducerTree({indexProducer, initializer}, rewriter);
  }
}

} // namespace sculptor
} // namespace mlir
