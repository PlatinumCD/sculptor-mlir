#include "sculptor-mlir/Dialect/Sculptor/Transforms/FoldInferenceParameters.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cmath>
#include <optional>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
namespace constant_utils = mlir::sculptor::converter_constant;

struct ScalarValue {
  float floating = 0.0f;
  bool boolean = false;
  bool isBoolean = false;

  static ScalarValue getFloat(float value) {
    ScalarValue result;
    result.floating = value;
    return result;
  }

  static ScalarValue getBool(bool value) {
    ScalarValue result;
    result.boolean = value;
    result.isBoolean = true;
    return result;
  }
};

std::optional<ScalarValue> getScalarConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  if (auto attr = llvm::dyn_cast<FloatAttr>(constant.getValue()))
    return ScalarValue::getFloat(static_cast<float>(attr.getValueAsDouble()));
  if (auto attr = llvm::dyn_cast<IntegerAttr>(constant.getValue()))
    return ScalarValue::getBool(attr.getValue().getBoolValue());
  return std::nullopt;
}

std::optional<ScalarValue>
lookupScalar(Value value, const llvm::DenseMap<Value, ScalarValue> &values) {
  auto found = values.find(value);
  if (found != values.end())
    return found->second;
  return getScalarConstant(value);
}

std::optional<bool> evaluateComparison(arith::CmpFPredicate predicate,
                                       float lhs, float rhs) {
  switch (predicate) {
  case arith::CmpFPredicate::AlwaysFalse:
    return false;
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    return lhs == rhs;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    return lhs > rhs;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return lhs >= rhs;
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    return lhs < rhs;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return lhs <= rhs;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return lhs != rhs;
  case arith::CmpFPredicate::ORD:
    return !std::isnan(lhs) && !std::isnan(rhs);
  case arith::CmpFPredicate::UNO:
    return std::isnan(lhs) || std::isnan(rhs);
  case arith::CmpFPredicate::AlwaysTrue:
    return true;
  }
  return std::nullopt;
}

std::optional<float>
evaluatePointwiseBody(linalg::GenericOp operation,
                      ArrayRef<SmallVector<float>> inputValues,
                      int64_t elementIndex) {
  Block &body = operation.getRegion().front();
  llvm::DenseMap<Value, ScalarValue> values;
  unsigned inputCount = operation.getNumDpsInputs();
  if (body.getNumArguments() != inputCount + operation.getNumDpsInits())
    return std::nullopt;

  for (auto [index, arguments] : llvm::enumerate(inputValues))
    values[body.getArgument(index)] =
        ScalarValue::getFloat(arguments[elementIndex]);
  for (unsigned index = inputCount; index < body.getNumArguments(); ++index) {
    if (!body.getArgument(index).use_empty())
      return std::nullopt;
  }

  for (Operation &nested : body.without_terminator()) {
    if (auto constant = llvm::dyn_cast<arith::ConstantOp>(nested)) {
      std::optional<ScalarValue> value =
          getScalarConstant(constant.getResult());
      if (!value)
        return std::nullopt;
      values[constant.getResult()] = *value;
      continue;
    }

    auto getFloat = [&](Value operand) -> std::optional<float> {
      std::optional<ScalarValue> value = lookupScalar(operand, values);
      if (!value || value->isBoolean)
        return std::nullopt;
      return value->floating;
    };

    if (auto trunc = llvm::dyn_cast<arith::TruncFOp>(nested)) {
      std::optional<float> input = getFloat(trunc.getIn());
      if (!input)
        return std::nullopt;
      values[trunc.getResult()] = ScalarValue::getFloat(*input);
      continue;
    }
    if (auto extension = llvm::dyn_cast<arith::ExtFOp>(nested)) {
      std::optional<float> input = getFloat(extension.getIn());
      if (!input)
        return std::nullopt;
      values[extension.getResult()] = ScalarValue::getFloat(*input);
      continue;
    }

    auto evaluateBinary = [&](Value lhsValue, Value rhsValue, Value resultValue,
                              auto function) -> LogicalResult {
      std::optional<float> lhs = getFloat(lhsValue);
      std::optional<float> rhs = getFloat(rhsValue);
      if (!lhs || !rhs)
        return failure();
      values[resultValue] = ScalarValue::getFloat(function(*lhs, *rhs));
      return success();
    };
    if (auto add = llvm::dyn_cast<arith::AddFOp>(nested)) {
      if (failed(
              evaluateBinary(add.getLhs(), add.getRhs(), add.getResult(),
                             [](float lhs, float rhs) { return lhs + rhs; })))
        return std::nullopt;
      continue;
    }
    if (auto subtract = llvm::dyn_cast<arith::SubFOp>(nested)) {
      if (failed(evaluateBinary(
              subtract.getLhs(), subtract.getRhs(), subtract.getResult(),
              [](float lhs, float rhs) { return lhs - rhs; })))
        return std::nullopt;
      continue;
    }
    if (auto multiply = llvm::dyn_cast<arith::MulFOp>(nested)) {
      if (failed(evaluateBinary(
              multiply.getLhs(), multiply.getRhs(), multiply.getResult(),
              [](float lhs, float rhs) { return lhs * rhs; })))
        return std::nullopt;
      continue;
    }
    if (auto divide = llvm::dyn_cast<arith::DivFOp>(nested)) {
      if (failed(evaluateBinary(
              divide.getLhs(), divide.getRhs(), divide.getResult(),
              [](float lhs, float rhs) { return lhs / rhs; })))
        return std::nullopt;
      continue;
    }
    if (auto squareRoot = llvm::dyn_cast<math::SqrtOp>(nested)) {
      std::optional<float> input = getFloat(squareRoot.getOperand());
      if (!input || *input < 0.0f)
        return std::nullopt;
      values[squareRoot.getResult()] = ScalarValue::getFloat(std::sqrt(*input));
      continue;
    }
    if (auto reciprocalSquareRoot = llvm::dyn_cast<math::RsqrtOp>(nested)) {
      std::optional<float> input = getFloat(reciprocalSquareRoot.getOperand());
      if (!input || *input < 0.0f)
        return std::nullopt;
      values[reciprocalSquareRoot.getResult()] =
          ScalarValue::getFloat(1.0f / std::sqrt(*input));
      continue;
    }
    if (auto compare = llvm::dyn_cast<arith::CmpFOp>(nested)) {
      std::optional<float> lhs = getFloat(compare.getLhs());
      std::optional<float> rhs = getFloat(compare.getRhs());
      if (!lhs || !rhs)
        return std::nullopt;
      std::optional<bool> result =
          evaluateComparison(compare.getPredicate(), *lhs, *rhs);
      if (!result)
        return std::nullopt;
      values[compare.getResult()] = ScalarValue::getBool(*result);
      continue;
    }
    if (auto select = llvm::dyn_cast<arith::SelectOp>(nested)) {
      std::optional<ScalarValue> condition =
          lookupScalar(select.getCondition(), values);
      if (!condition || !condition->isBoolean)
        return std::nullopt;
      std::optional<ScalarValue> selected = lookupScalar(
          condition->boolean ? select.getTrueValue() : select.getFalseValue(),
          values);
      if (!selected)
        return std::nullopt;
      values[select.getResult()] = *selected;
      continue;
    }
    if (auto assertion = llvm::dyn_cast<cf::AssertOp>(nested)) {
      std::optional<ScalarValue> condition =
          lookupScalar(assertion.getArg(), values);
      if (!condition || !condition->isBoolean || !condition->boolean)
        return std::nullopt;
      continue;
    }
    return std::nullopt;
  }

  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  std::optional<ScalarValue> result = lookupScalar(yield.getOperand(0), values);
  if (!result || result->isBoolean)
    return std::nullopt;
  return result->floating;
}

struct FoldConstantPointwiseGenericPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp operation,
                                PatternRewriter &rewriter) const override {
    if (operation.getNumResults() != 1 || operation.getNumDpsInits() != 1 ||
        operation.getNumDpsInputs() == 0 ||
        operation.getNumParallelLoops() != operation.getNumLoops())
      return failure();

    auto resultType =
        llvm::dyn_cast<RankedTensorType>(operation.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        !resultType.getElementType().isF32())
      return failure();
    for (AffineMap map : operation.getIndexingMapsArray()) {
      if (!map.isIdentity())
        return failure();
    }

    SmallVector<SmallVector<float>> inputValues;
    inputValues.reserve(operation.getNumDpsInputs());
    for (Value input : operation.getDpsInputs()) {
      auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
      auto constant = input.getDefiningOp<arith::ConstantOp>();
      if (!inputType || inputType != resultType || !constant)
        return failure();
      FailureOr<SmallVector<float>> values =
          constant_utils::getF32ConstantValues(constant);
      if (failed(values) ||
          static_cast<int64_t>(values->size()) != resultType.getNumElements())
        return failure();
      inputValues.push_back(std::move(*values));
    }

    SmallVector<float> results;
    results.reserve(resultType.getNumElements());
    for (int64_t index = 0; index < resultType.getNumElements(); ++index) {
      std::optional<float> result =
          evaluatePointwiseBody(operation, inputValues, index);
      if (!result)
        return failure();
      results.push_back(*result);
    }

    TypedAttr resultAttr = constant_utils::buildF32ElementsAttr(
        resultType, results, "sculptor_folded_parameter_",
        /*useResource=*/results.size() >= 256);
    if (!resultAttr)
      return failure();
    auto constant = rewriter.create<arith::ConstantOp>(operation.getLoc(),
                                                       resultType, resultAttr);
    rewriter.replaceOp(operation, constant.getResult());
    return success();
  }
};

enum class BinaryKind { Add, Subtract, Multiply };

bool hasBinaryBody(linalg::GenericOp operation, BinaryKind kind) {
  if (!operation || !operation.getRegion().hasOneBlock() ||
      operation.getNumDpsInputs() != 2 || operation.getNumDpsInits() != 1)
    return false;
  Block &body = operation.getRegion().front();
  auto yield = llvm::dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  Operation *calculation = yield.getOperand(0).getDefiningOp();
  if (!calculation || calculation->getNumOperands() != 2)
    return false;
  switch (kind) {
  case BinaryKind::Add:
    return llvm::isa<arith::AddFOp>(calculation);
  case BinaryKind::Subtract:
    return llvm::isa<arith::SubFOp>(calculation);
  case BinaryKind::Multiply:
    return llvm::isa<arith::MulFOp>(calculation);
  }
  return false;
}

FailureOr<SmallVector<float>> getChannelValues(Value value,
                                               int64_t channels = -1) {
  while (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>())
    value = expand.getSrc();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto type = llvm::dyn_cast<RankedTensorType>(constant.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
      (channels >= 0 && type.getNumElements() != channels))
    return failure();
  return constant_utils::getF32ConstantValues(constant);
}

struct WeightedProducer {
  Operation *operation = nullptr;
  Value input;
  Value weight;
  Value bias;
  Value result;
};

std::optional<WeightedProducer> getWeightedProducer(Value value) {
  if (auto linear = value.getDefiningOp<NNLinearOp>()) {
    return WeightedProducer{linear.getOperation(), linear.getInput(),
                            linear.getWeight(), linear.getBias(),
                            linear.getResult()};
  }
  if (auto conv = value.getDefiningOp<NNConv1DOp>()) {
    return WeightedProducer{conv.getOperation(), conv.getInput(),
                            conv.getWeight(), conv.getBias(), conv.getResult()};
  }
  if (auto conv = value.getDefiningOp<NNConv2DOp>()) {
    return WeightedProducer{conv.getOperation(), conv.getInput(),
                            conv.getWeight(), conv.getBias(), conv.getResult()};
  }
  if (auto conv = value.getDefiningOp<NNGroupedConv2DOp>()) {
    return WeightedProducer{conv.getOperation(), conv.getInput(),
                            conv.getWeight(), conv.getBias(), conv.getResult()};
  }
  if (auto conv = value.getDefiningOp<NNConv3DOp>()) {
    return WeightedProducer{conv.getOperation(), conv.getInput(),
                            conv.getWeight(), conv.getBias(), conv.getResult()};
  }
  return std::nullopt;
}

FailureOr<Value> foldWeightedAffineProducer(const WeightedProducer &producer,
                                            ArrayRef<float> scale,
                                            ArrayRef<float> affineBias,
                                            PatternRewriter &rewriter) {
  int64_t channels = static_cast<int64_t>(scale.size());
  if (channels <= 0 || affineBias.size() != scale.size())
    return failure();

  auto weightConstant = producer.weight.getDefiningOp<arith::ConstantOp>();
  auto weightType = llvm::dyn_cast<RankedTensorType>(producer.weight.getType());
  if (!weightConstant || !weightType || !weightType.hasStaticShape() ||
      weightType.getRank() < 2 || weightType.getDimSize(0) != channels ||
      !weightType.getElementType().isF32())
    return failure();
  auto producerResultType =
      llvm::dyn_cast<RankedTensorType>(producer.result.getType());
  if (!producerResultType || !producerResultType.hasStaticShape())
    return failure();
  int64_t producerChannelDimension = llvm::isa<NNLinearOp>(producer.operation)
                                         ? producerResultType.getRank() - 1
                                         : 1;
  if (producerChannelDimension < 0 ||
      producerChannelDimension >= producerResultType.getRank() ||
      producerResultType.getDimSize(producerChannelDimension) != channels)
    return failure();

  FailureOr<SmallVector<float>> weightValues =
      constant_utils::getF32ConstantValues(weightConstant);
  if (failed(weightValues) ||
      static_cast<int64_t>(weightValues->size()) != weightType.getNumElements())
    return failure();
  SmallVector<float> biasValues(channels, 0.0f);
  if (producer.bias) {
    auto biasConstant = producer.bias.getDefiningOp<arith::ConstantOp>();
    FailureOr<SmallVector<float>> existingBias =
        constant_utils::getF32ConstantValues(biasConstant);
    if (failed(existingBias) ||
        static_cast<int64_t>(existingBias->size()) != channels)
      return failure();
    biasValues = std::move(*existingBias);
  }

  SmallVector<float> foldedBias(channels);
  for (int64_t channel = 0; channel < channels; ++channel) {
    foldedBias[channel] =
        biasValues[channel] * scale[channel] + affineBias[channel];
  }
  int64_t valuesPerChannel = weightType.getNumElements() / channels;
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t offset = 0; offset < valuesPerChannel; ++offset)
      (*weightValues)[channel * valuesPerChannel + offset] *= scale[channel];
  }

  rewriter.setInsertionPoint(producer.operation);
  TypedAttr weightAttr = constant_utils::buildF32ElementsAttr(
      weightType, *weightValues, "sculptor_folded_conv_weight_",
      /*useResource=*/true);
  auto biasType = RankedTensorType::get({channels}, rewriter.getF32Type());
  TypedAttr biasAttr = constant_utils::buildF32ElementsAttr(
      biasType, foldedBias, "sculptor_folded_conv_bias_",
      /*useResource=*/channels >= 256);
  if (!weightAttr || !biasAttr)
    return failure();
  auto newWeight = rewriter.create<arith::ConstantOp>(
      producer.operation->getLoc(), weightType, weightAttr);
  auto newBias = rewriter.create<arith::ConstantOp>(
      producer.operation->getLoc(), biasType, biasAttr);

  if (auto linear = llvm::dyn_cast<NNLinearOp>(producer.operation)) {
    return rewriter
        .create<NNLinearOp>(linear.getLoc(), linear.getResult().getType(),
                            linear.getInput(), newWeight.getResult(),
                            newBias.getResult(), /*hasBias=*/true)
        .getResult();
  }
  if (auto conv = llvm::dyn_cast<NNConv1DOp>(producer.operation)) {
    return rewriter
        .create<NNConv1DOp>(
            conv.getLoc(), conv.getResult().getType(), conv.getInput(),
            newWeight.getResult(), newBias.getResult(), /*hasBias=*/true,
            conv.getStride(), conv.getPadding(), conv.getDilation())
        .getResult();
  }
  if (auto conv = llvm::dyn_cast<NNConv2DOp>(producer.operation)) {
    return rewriter
        .create<NNConv2DOp>(
            conv.getLoc(), conv.getResult().getType(), conv.getInput(),
            newWeight.getResult(), newBias.getResult(), /*hasBias=*/true,
            conv.getStride(), conv.getPadding(), conv.getDilation())
        .getResult();
  }
  if (auto conv = llvm::dyn_cast<NNGroupedConv2DOp>(producer.operation)) {
    return rewriter
        .create<NNGroupedConv2DOp>(conv.getLoc(), conv.getResult().getType(),
                                   conv.getInput(), newWeight.getResult(),
                                   newBias.getResult(), /*hasBias=*/true,
                                   conv.getGroups(), conv.getStride(),
                                   conv.getPadding(), conv.getDilation())
        .getResult();
  }
  if (auto conv = llvm::dyn_cast<NNConv3DOp>(producer.operation)) {
    return rewriter
        .create<NNConv3DOp>(
            conv.getLoc(), conv.getResult().getType(), conv.getInput(),
            newWeight.getResult(), newBias.getResult(), /*hasBias=*/true,
            conv.getStride(), conv.getPadding(), conv.getDilation())
        .getResult();
  }
  return failure();
}

FailureOr<std::pair<Value, SmallVector<float>>>
splitChainAndChannelParameter(linalg::GenericOp operation, BinaryKind kind,
                              int64_t channels) {
  if (!hasBinaryBody(operation, kind))
    return failure();
  Value first = operation.getDpsInputs()[0];
  Value second = operation.getDpsInputs()[1];
  FailureOr<SmallVector<float>> secondValues =
      getChannelValues(second, channels);
  if (succeeded(secondValues))
    return std::make_pair(first, std::move(*secondValues));
  if (kind == BinaryKind::Add || kind == BinaryKind::Multiply) {
    FailureOr<SmallVector<float>> firstValues =
        getChannelValues(first, channels);
    if (succeeded(firstValues))
      return std::make_pair(second, std::move(*firstValues));
  }
  return failure();
}

struct FoldConvBatchNormPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp finalAdd,
                                PatternRewriter &rewriter) const override {
    auto outputType =
        llvm::dyn_cast<RankedTensorType>(finalAdd.getResult(0).getType());
    if (!outputType || !outputType.hasStaticShape() ||
        !outputType.getElementType().isF32())
      return failure();

    FailureOr<std::pair<Value, SmallVector<float>>> add =
        splitChainAndChannelParameter(finalAdd, BinaryKind::Add,
                                      /*channels=*/-1);
    if (failed(add))
      return failure();
    int64_t channels = static_cast<int64_t>(add->second.size());
    if (channels <= 0)
      return failure();
    auto gammaMultiply = add->first.getDefiningOp<linalg::GenericOp>();
    FailureOr<std::pair<Value, SmallVector<float>>> gamma =
        splitChainAndChannelParameter(gammaMultiply, BinaryKind::Multiply,
                                      channels);
    if (failed(gamma))
      return failure();
    auto inverseStdMultiply = gamma->first.getDefiningOp<linalg::GenericOp>();
    FailureOr<std::pair<Value, SmallVector<float>>> inverseStd =
        splitChainAndChannelParameter(inverseStdMultiply, BinaryKind::Multiply,
                                      channels);
    if (failed(inverseStd))
      return failure();
    auto meanSubtract = inverseStd->first.getDefiningOp<linalg::GenericOp>();
    FailureOr<std::pair<Value, SmallVector<float>>> mean =
        splitChainAndChannelParameter(meanSubtract, BinaryKind::Subtract,
                                      channels);
    if (failed(mean))
      return failure();

    std::optional<WeightedProducer> producer = getWeightedProducer(mean->first);
    if (!producer || !producer->result.hasOneUse() ||
        !meanSubtract.getResult(0).hasOneUse() ||
        !inverseStdMultiply.getResult(0).hasOneUse() ||
        !gammaMultiply.getResult(0).hasOneUse())
      return failure();

    SmallVector<float> scale(channels);
    SmallVector<float> affineBias(channels);
    for (int64_t channel = 0; channel < channels; ++channel) {
      scale[channel] = inverseStd->second[channel] * gamma->second[channel];
      affineBias[channel] =
          add->second[channel] - mean->second[channel] * scale[channel];
    }
    FailureOr<Value> foldedResult =
        foldWeightedAffineProducer(*producer, scale, affineBias, rewriter);
    if (failed(foldedResult))
      return failure();

    rewriter.replaceOp(finalAdd, *foldedResult);
    rewriter.eraseOp(gammaMultiply);
    rewriter.eraseOp(inverseStdMultiply);
    rewriter.eraseOp(meanSubtract);
    rewriter.eraseOp(producer->operation);
    return success();
  }
};

// Fold an inference-time per-channel affine transform directly into the
// weighted producer:
//
//   y = producer(x, weight, bias) * scale + affine_bias
//
// This is the canonical form emitted for frozen batch normalization after its
// parameter-only arithmetic has been evaluated.
struct FoldWeightedAffinePattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp finalAdd,
                                PatternRewriter &rewriter) const override {
    if (finalAdd.getNumResults() != 1)
      return failure();
    auto outputType =
        llvm::dyn_cast<RankedTensorType>(finalAdd.getResult(0).getType());
    if (!outputType || !outputType.hasStaticShape() ||
        !outputType.getElementType().isF32())
      return failure();

    FailureOr<std::pair<Value, SmallVector<float>>> add =
        splitChainAndChannelParameter(finalAdd, BinaryKind::Add,
                                      /*channels=*/-1);
    if (failed(add))
      return failure();
    int64_t channels = static_cast<int64_t>(add->second.size());
    if (channels <= 0)
      return failure();

    auto scaleMultiply = add->first.getDefiningOp<linalg::GenericOp>();
    FailureOr<std::pair<Value, SmallVector<float>>> scale =
        splitChainAndChannelParameter(scaleMultiply, BinaryKind::Multiply,
                                      channels);
    if (failed(scale))
      return failure();

    std::optional<WeightedProducer> producer =
        getWeightedProducer(scale->first);
    if (!producer || !producer->result.hasOneUse() ||
        !scaleMultiply.getResult(0).hasOneUse())
      return failure();

    FailureOr<Value> foldedResult = foldWeightedAffineProducer(
        *producer, scale->second, add->second, rewriter);
    if (failed(foldedResult))
      return failure();

    rewriter.replaceOp(finalAdd, *foldedResult);
    rewriter.eraseOp(scaleMultiply);
    rewriter.eraseOp(producer->operation);
    return success();
  }
};

} // namespace

void mlir::sculptor::FoldInferenceParametersPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<FoldConstantPointwiseGenericPattern, FoldConvBatchNormPattern,
               FoldWeightedAffinePattern>(&getContext());
  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config)))
    signalPassFailure();
}

void mlir::sculptor::registerFoldInferenceParametersPass() {
  PassRegistration<FoldInferenceParametersPass>();
}
