#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/MVMBuildUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/TensorTypeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>

namespace mvm_build = mlir::sculptor::mvm_build;
namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace tensor_type = mlir::sculptor::tensor_type;
namespace converter_constant = mlir::sculptor::converter_constant;

namespace {

using mlir::arith::ConstantOp;
using mlir::sculptor::NNTransformerBlockOp;
using mlir::tensor::EmptyOp;

struct ProjectionLowering {
  mlir::Value weight;
  mlir::Value bias;
  ConstantOp weightConstant;
  ConstantOp biasConstant;
  mlir::RankedTensorType weightTy;
  mlir::RankedTensorType biasTy;
};

struct NormLowering {
  mlir::Value weight;
  mlir::Value bias;
  ConstantOp weightConstant;
  ConstantOp biasConstant;
  bool present = false;
};

struct TransformerBlockLowering {
  NNTransformerBlockOp blockOp;
  mlir::sculptor::TransformerBlockKind blockKind;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType memoryTy;
  mlir::RankedTensorType outputTy;
  mlir::Type elementType;
  ProjectionLowering qkv;
  ProjectionLowering attnOutput;
  ProjectionLowering crossQuery;
  ProjectionLowering crossKeyValue;
  ProjectionLowering crossOutput;
  ProjectionLowering mlpUp;
  ProjectionLowering mlpDown;
  NormLowering attnNorm;
  NormLowering crossNorm;
  NormLowering mlpNorm;
  NormLowering finalNorm;
  int64_t batchSize = 0;
  int64_t sequenceLength = 0;
  int64_t memoryLength = 0;
  int64_t hiddenSize = 0;
  int64_t numHeads = 0;
  int64_t headDim = 0;
  int64_t mlpHiddenSize = 0;
  double layerNormEps = 0.0;
  mlir::StringRef normMode = "post";
  bool hasProjectionBias = false;
  bool hasLayerNormAffine = false;
  bool hasFinalNorm = false;
  bool causal = false;
  bool hasCrossAttention = false;
};

static bool isEncoder(TransformerBlockLowering &match) {
  return match.blockKind == mlir::sculptor::TransformerBlockKind::Encoder;
}

static bool isDecoder(TransformerBlockLowering &match) {
  return match.blockKind == mlir::sculptor::TransformerBlockKind::Decoder;
}

static llvm::SmallVector<mlir::OpFoldResult>
buildIndexAttrs(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::OpFoldResult> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getIndexAttr(value));
  return attrs;
}

static mlir::Value buildIndexConstant(mlir::OpBuilder &builder,
                                      mlir::Location loc, int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
}

static mlir::Value buildF32Constant(mlir::OpBuilder &builder,
                                    mlir::Location loc, double value) {
  auto type = builder.getF32Type();
  return builder.create<mlir::arith::ConstantOp>(
      loc, type, builder.getFloatAttr(type, value));
}

struct StaticScalar {
  enum class Kind { Integer, Float, Bool } kind = Kind::Integer;
  int64_t integer = 0;
  float fp = 0.0f;
  bool flag = false;

  static StaticScalar getInteger(int64_t value) {
    StaticScalar scalar;
    scalar.kind = Kind::Integer;
    scalar.integer = value;
    return scalar;
  }

  static StaticScalar getFloat(float value) {
    StaticScalar scalar;
    scalar.kind = Kind::Float;
    scalar.fp = value;
    return scalar;
  }

  static StaticScalar getBool(bool value) {
    StaticScalar scalar;
    scalar.kind = Kind::Bool;
    scalar.flag = value;
    return scalar;
  }
};

struct StaticTensor {
  mlir::RankedTensorType type;
  llvm::SmallVector<float> f32Values;
  llvm::SmallVector<int64_t> i64Values;
  bool isF32 = false;

  StaticScalar getElement(int64_t linearIndex) const {
    if (isF32)
      return StaticScalar::getFloat(f32Values[linearIndex]);
    return StaticScalar::getInteger(i64Values[linearIndex]);
  }
};

static int64_t linearIndex(llvm::ArrayRef<int64_t> shape,
                           llvm::ArrayRef<int64_t> indices) {
  int64_t offset = 0;
  for (auto [dim, index] : llvm::zip(shape, indices))
    offset = offset * dim + index;
  return offset;
}

static llvm::SmallVector<int64_t>
delinearizeIndex(int64_t index, llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> indices(shape.size(), 0);
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    indices[dim] = index % shape[dim];
    index /= shape[dim];
  }
  return indices;
}

static mlir::FailureOr<int64_t>
evaluateAffineExpr(mlir::AffineExpr expr, llvm::ArrayRef<int64_t> dims) {
  switch (expr.getKind()) {
  case mlir::AffineExprKind::DimId: {
    unsigned position = llvm::cast<mlir::AffineDimExpr>(expr).getPosition();
    if (position >= dims.size())
      return mlir::failure();
    return dims[position];
  }
  case mlir::AffineExprKind::Constant:
    return llvm::cast<mlir::AffineConstantExpr>(expr).getValue();
  case mlir::AffineExprKind::Add: {
    auto binary = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = evaluateAffineExpr(binary.getLHS(), dims);
    auto rhs = evaluateAffineExpr(binary.getRHS(), dims);
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    return *lhs + *rhs;
  }
  case mlir::AffineExprKind::Mul: {
    auto binary = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = evaluateAffineExpr(binary.getLHS(), dims);
    auto rhs = evaluateAffineExpr(binary.getRHS(), dims);
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    return *lhs * *rhs;
  }
  case mlir::AffineExprKind::Mod: {
    auto binary = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = evaluateAffineExpr(binary.getLHS(), dims);
    auto rhs = evaluateAffineExpr(binary.getRHS(), dims);
    if (mlir::failed(lhs) || mlir::failed(rhs) || *rhs == 0)
      return mlir::failure();
    return *lhs % *rhs;
  }
  case mlir::AffineExprKind::FloorDiv: {
    auto binary = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
    auto lhs = evaluateAffineExpr(binary.getLHS(), dims);
    auto rhs = evaluateAffineExpr(binary.getRHS(), dims);
    if (mlir::failed(lhs) || mlir::failed(rhs) || *rhs == 0)
      return mlir::failure();
    return *lhs / *rhs;
  }
  default:
    return mlir::failure();
  }
}

static mlir::FailureOr<llvm::SmallVector<int64_t>>
evaluateMap(mlir::AffineMap map, llvm::ArrayRef<int64_t> loopIndices) {
  llvm::SmallVector<int64_t> indices;
  indices.reserve(map.getNumResults());
  for (mlir::AffineExpr expr : map.getResults()) {
    auto index = evaluateAffineExpr(expr, loopIndices);
    if (mlir::failed(index))
      return mlir::failure();
    indices.push_back(*index);
  }
  return indices;
}

static mlir::FailureOr<StaticTensor>
evaluateStaticTensor(mlir::Value value,
                     llvm::DenseMap<mlir::Value, StaticTensor> &cache);

static mlir::FailureOr<StaticScalar>
evaluateScalarValue(mlir::Value value,
                    llvm::DenseMap<mlir::Value, StaticScalar> &scalars,
                    llvm::DenseMap<mlir::Value, StaticTensor> &cache) {
  auto known = scalars.find(value);
  if (known != scalars.end())
    return known->second;

  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return mlir::failure();

  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
    return StaticScalar::getInteger(intAttr.getInt());

  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(constant.getValue()))
    return StaticScalar::getFloat(floatAttr.getValueAsDouble());

  (void)cache;
  return mlir::failure();
}

static mlir::FailureOr<StaticScalar> evaluateStaticGenericBody(
    mlir::linalg::GenericOp genericOp, llvm::ArrayRef<int64_t> loopIndices,
    llvm::ArrayRef<StaticScalar> inputScalars, StaticScalar outputScalar,
    llvm::DenseMap<mlir::Value, StaticTensor> &cache) {
  if (!genericOp.getRegion().hasOneBlock())
    return mlir::failure();

  mlir::Block &body = genericOp.getRegion().front();
  llvm::DenseMap<mlir::Value, StaticScalar> scalars;
  unsigned argIndex = 0;
  for (StaticScalar scalar : inputScalars)
    scalars[body.getArgument(argIndex++)] = scalar;
  while (argIndex < body.getNumArguments())
    scalars[body.getArgument(argIndex++)] = outputScalar;

  for (mlir::Operation &op : body) {
    if (auto yield = llvm::dyn_cast<mlir::linalg::YieldOp>(op)) {
      if (yield.getNumOperands() != 1)
        return mlir::failure();
      return evaluateScalarValue(yield.getOperand(0), scalars, cache);
    }

    if (auto indexOp = llvm::dyn_cast<mlir::linalg::IndexOp>(op)) {
      if (indexOp.getDim() >= loopIndices.size())
        return mlir::failure();
      scalars[indexOp.getResult()] =
          StaticScalar::getInteger(loopIndices[indexOp.getDim()]);
      continue;
    }

    if (auto constant = llvm::dyn_cast<ConstantOp>(op)) {
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
        scalars[constant.getResult()] =
            StaticScalar::getInteger(intAttr.getInt());
      else if (auto floatAttr =
                   llvm::dyn_cast<mlir::FloatAttr>(constant.getValue()))
        scalars[constant.getResult()] =
            StaticScalar::getFloat(floatAttr.getValueAsDouble());
      else
        return mlir::failure();
      continue;
    }

    if (auto cast = llvm::dyn_cast<mlir::arith::IndexCastOp>(op)) {
      auto operand = evaluateScalarValue(cast.getIn(), scalars, cache);
      if (mlir::failed(operand) || operand->kind != StaticScalar::Kind::Integer)
        return mlir::failure();
      scalars[cast.getOut()] = *operand;
      continue;
    }

    if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(op)) {
      auto lhs = evaluateScalarValue(add.getLhs(), scalars, cache);
      auto rhs = evaluateScalarValue(add.getRhs(), scalars, cache);
      if (mlir::failed(lhs) || mlir::failed(rhs) ||
          lhs->kind != StaticScalar::Kind::Integer ||
          rhs->kind != StaticScalar::Kind::Integer)
        return mlir::failure();
      scalars[add.getResult()] =
          StaticScalar::getInteger(lhs->integer + rhs->integer);
      continue;
    }

    if (auto mul = llvm::dyn_cast<mlir::arith::MulIOp>(op)) {
      auto lhs = evaluateScalarValue(mul.getLhs(), scalars, cache);
      auto rhs = evaluateScalarValue(mul.getRhs(), scalars, cache);
      if (mlir::failed(lhs) || mlir::failed(rhs) ||
          lhs->kind != StaticScalar::Kind::Integer ||
          rhs->kind != StaticScalar::Kind::Integer)
        return mlir::failure();
      scalars[mul.getResult()] =
          StaticScalar::getInteger(lhs->integer * rhs->integer);
      continue;
    }

    if (auto cmp = llvm::dyn_cast<mlir::arith::CmpIOp>(op)) {
      auto lhs = evaluateScalarValue(cmp.getLhs(), scalars, cache);
      auto rhs = evaluateScalarValue(cmp.getRhs(), scalars, cache);
      if (mlir::failed(lhs) || mlir::failed(rhs) ||
          lhs->kind != StaticScalar::Kind::Integer ||
          rhs->kind != StaticScalar::Kind::Integer)
        return mlir::failure();

      bool result = false;
      switch (cmp.getPredicate()) {
      case mlir::arith::CmpIPredicate::slt:
        result = lhs->integer < rhs->integer;
        break;
      case mlir::arith::CmpIPredicate::sgt:
        result = lhs->integer > rhs->integer;
        break;
      case mlir::arith::CmpIPredicate::eq:
        result = lhs->integer == rhs->integer;
        break;
      default:
        return mlir::failure();
      }
      scalars[cmp.getResult()] = StaticScalar::getBool(result);
      continue;
    }

    if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(op)) {
      auto condition =
          evaluateScalarValue(select.getCondition(), scalars, cache);
      auto trueValue =
          evaluateScalarValue(select.getTrueValue(), scalars, cache);
      auto falseValue =
          evaluateScalarValue(select.getFalseValue(), scalars, cache);
      if (mlir::failed(condition) || mlir::failed(trueValue) ||
          mlir::failed(falseValue) ||
          condition->kind != StaticScalar::Kind::Bool)
        return mlir::failure();
      scalars[select.getResult()] = condition->flag ? *trueValue : *falseValue;
      continue;
    }

    if (auto extract = llvm::dyn_cast<mlir::tensor::ExtractOp>(op)) {
      auto tensor = evaluateStaticTensor(extract.getTensor(), cache);
      if (mlir::failed(tensor))
        return mlir::failure();

      llvm::SmallVector<int64_t> indices;
      indices.reserve(extract.getIndices().size());
      for (mlir::Value indexValue : extract.getIndices()) {
        auto index = evaluateScalarValue(indexValue, scalars, cache);
        if (mlir::failed(index) || index->kind != StaticScalar::Kind::Integer)
          return mlir::failure();
        indices.push_back(index->integer);
      }

      int64_t offset = linearIndex(tensor->type.getShape(), indices);
      scalars[extract.getResult()] = tensor->getElement(offset);
      continue;
    }

    return mlir::failure();
  }

  return mlir::failure();
}

static mlir::FailureOr<StaticTensor>
evaluateStaticGeneric(mlir::linalg::GenericOp genericOp,
                      llvm::DenseMap<mlir::Value, StaticTensor> &cache) {
  if (genericOp.getNumResults() != 1 || genericOp.getOutputs().size() != 1)
    return mlir::failure();

  auto resultTy =
      llvm::dyn_cast<mlir::RankedTensorType>(genericOp.getResult(0).getType());
  if (!resultTy || !resultTy.hasStaticShape())
    return mlir::failure();
  if (!resultTy.getElementType().isF32() &&
      !resultTy.getElementType().isInteger(64))
    return mlir::failure();

  for (mlir::utils::IteratorType iterator : genericOp.getIteratorTypesArray()) {
    if (iterator != mlir::utils::IteratorType::parallel)
      return mlir::failure();
  }

  auto maps = genericOp.getIndexingMapsArray();
  unsigned inputCount = genericOp.getInputs().size();
  if (maps.size() != inputCount + genericOp.getOutputs().size())
    return mlir::failure();

  llvm::SmallVector<StaticTensor> inputs;
  inputs.reserve(inputCount);
  for (mlir::Value input : genericOp.getInputs()) {
    auto tensor = evaluateStaticTensor(input, cache);
    if (mlir::failed(tensor))
      return mlir::failure();
    inputs.push_back(*tensor);
  }

  StaticTensor result;
  result.type = resultTy;
  result.isF32 = resultTy.getElementType().isF32();
  int64_t elementCount = resultTy.getNumElements();
  if (result.isF32)
    result.f32Values.assign(elementCount, 0.0f);
  else
    result.i64Values.assign(elementCount, 0);

  StaticScalar outputScalar =
      result.isF32 ? StaticScalar::getFloat(0.0f) : StaticScalar::getInteger(0);
  for (int64_t linear = 0; linear < elementCount; ++linear) {
    llvm::SmallVector<int64_t> loopIndices =
        delinearizeIndex(linear, resultTy.getShape());
    auto outputIndices = evaluateMap(maps[inputCount], loopIndices);
    if (mlir::failed(outputIndices) ||
        static_cast<int64_t>(outputIndices->size()) != resultTy.getRank())
      return mlir::failure();
    for (auto [actual, expected] : llvm::zip(*outputIndices, loopIndices)) {
      if (actual != expected)
        return mlir::failure();
    }

    llvm::SmallVector<StaticScalar> inputScalars;
    inputScalars.reserve(inputCount);
    for (unsigned inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
      auto inputIndices = evaluateMap(maps[inputIndex], loopIndices);
      if (mlir::failed(inputIndices))
        return mlir::failure();
      int64_t inputOffset =
          linearIndex(inputs[inputIndex].type.getShape(), *inputIndices);
      inputScalars.push_back(inputs[inputIndex].getElement(inputOffset));
    }

    auto scalar = evaluateStaticGenericBody(genericOp, loopIndices,
                                            inputScalars, outputScalar, cache);
    if (mlir::failed(scalar))
      return mlir::failure();

    if (result.isF32) {
      if (scalar->kind != StaticScalar::Kind::Float)
        return mlir::failure();
      result.f32Values[linear] = scalar->fp;
    } else {
      if (scalar->kind != StaticScalar::Kind::Integer)
        return mlir::failure();
      result.i64Values[linear] = scalar->integer;
    }
  }

  return result;
}

static mlir::FailureOr<StaticTensor>
evaluateStaticTensor(mlir::Value value,
                     llvm::DenseMap<mlir::Value, StaticTensor> &cache) {
  auto cached = cache.find(value);
  if (cached != cache.end())
    return cached->second;

  auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorTy || !tensorTy.hasStaticShape())
    return mlir::failure();

  StaticTensor result;
  result.type = tensorTy;
  result.isF32 = tensorTy.getElementType().isF32();

  if (auto constant = value.getDefiningOp<ConstantOp>()) {
    if (result.isF32) {
      auto values = converter_constant::getF32ConstantValues(constant);
      if (mlir::failed(values) ||
          static_cast<int64_t>(values->size()) != tensorTy.getNumElements())
        return mlir::failure();
      result.f32Values = std::move(*values);
      cache[value] = result;
      return result;
    }

    if (auto denseInt =
            llvm::dyn_cast<mlir::DenseIntElementsAttr>(constant.getValue())) {
      if (static_cast<int64_t>(denseInt.getNumElements()) !=
          tensorTy.getNumElements())
        return mlir::failure();
      for (llvm::APInt element : denseInt.getValues<llvm::APInt>())
        result.i64Values.push_back(element.getSExtValue());
      cache[value] = result;
      return result;
    }

    return mlir::failure();
  }

  if (auto collapse = value.getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
    auto source = evaluateStaticTensor(collapse.getSrc(), cache);
    if (mlir::failed(source) ||
        source->type.getNumElements() != tensorTy.getNumElements())
      return mlir::failure();
    result.isF32 = source->isF32;
    result.f32Values = source->f32Values;
    result.i64Values = source->i64Values;
    cache[value] = result;
    return result;
  }

  if (auto expand = value.getDefiningOp<mlir::tensor::ExpandShapeOp>()) {
    auto source = evaluateStaticTensor(expand.getSrc(), cache);
    if (mlir::failed(source) ||
        source->type.getNumElements() != tensorTy.getNumElements())
      return mlir::failure();
    result.isF32 = source->isF32;
    result.f32Values = source->f32Values;
    result.i64Values = source->i64Values;
    cache[value] = result;
    return result;
  }

  if (auto generic = value.getDefiningOp<mlir::linalg::GenericOp>()) {
    auto tensor = evaluateStaticGeneric(generic, cache);
    if (mlir::failed(tensor))
      return mlir::failure();
    cache[value] = *tensor;
    return *tensor;
  }

  return mlir::failure();
}

static mlir::FailureOr<llvm::SmallVector<float>>
evaluateStaticF32Tensor(mlir::Value value, mlir::RankedTensorType expectedTy) {
  llvm::DenseMap<mlir::Value, StaticTensor> cache;
  auto tensor = evaluateStaticTensor(value, cache);
  if (mlir::failed(tensor) || !tensor->isF32 ||
      tensor->type.getShape() != expectedTy.getShape())
    return mlir::failure();
  return std::move(tensor->f32Values);
}

static mlir::Block *addTaskRegionBody(mlir::sculptor::TaskRegionOp region,
                                      mlir::ValueRange inputs) {
  mlir::Block *body = new mlir::Block();
  region.getBody().push_back(body);
  for (mlir::Value input : inputs)
    body->addArgument(input.getType(), input.getLoc());
  return body;
}

static mlir::FailureOr<ProjectionLowering>
matchProjection(mlir::Value weight, mlir::Value bias, int64_t outputWidth,
                int64_t inputWidth, bool expectsBias) {
  if (!weight)
    return mlir::failure();

  auto weightTy =
      tensor_type::getPositiveStaticRank2F32Tensor(weight.getType());
  if (mlir::failed(weightTy) ||
      weightTy->getShape() !=
          llvm::ArrayRef<int64_t>({outputWidth, inputWidth}))
    return mlir::failure();

  auto weightConstant = weight.getDefiningOp<ConstantOp>();
  if (!weightConstant &&
      mlir::failed(evaluateStaticF32Tensor(weight, *weightTy)))
    return mlir::failure();

  ProjectionLowering projection;
  projection.weight = weight;
  projection.weightConstant = weightConstant;
  projection.weightTy = *weightTy;

  if (expectsBias) {
    if (!bias)
      return mlir::failure();

    auto biasTy = tensor_type::getPositiveStaticRank1F32Tensor(bias.getType());
    if (mlir::failed(biasTy) ||
        biasTy->getShape() != llvm::ArrayRef<int64_t>({outputWidth}))
      return mlir::failure();

    auto biasConstant = bias.getDefiningOp<ConstantOp>();
    if (!biasConstant && mlir::failed(evaluateStaticF32Tensor(bias, *biasTy)))
      return mlir::failure();

    projection.bias = bias;
    projection.biasConstant = biasConstant;
    projection.biasTy = *biasTy;
  } else if (bias) {
    return mlir::failure();
  }

  return projection;
}

static mlir::FailureOr<NormLowering>
matchNorm(mlir::Value weight, mlir::Value bias, bool sectionPresent,
          bool expectsAffine, int64_t hiddenSize) {
  NormLowering norm;
  norm.present = sectionPresent;

  if (!sectionPresent) {
    if (weight || bias)
      return mlir::failure();
    return norm;
  }

  if (!expectsAffine) {
    if (weight || bias)
      return mlir::failure();
    return norm;
  }

  if (!weight || !bias)
    return mlir::failure();

  auto weightTy =
      tensor_type::getPositiveStaticRank1F32Tensor(weight.getType());
  auto biasTy = tensor_type::getPositiveStaticRank1F32Tensor(bias.getType());
  if (mlir::failed(weightTy) || mlir::failed(biasTy) ||
      weightTy->getShape() != llvm::ArrayRef<int64_t>({hiddenSize}) ||
      biasTy->getShape() != llvm::ArrayRef<int64_t>({hiddenSize}))
    return mlir::failure();

  auto weightConstant = weight.getDefiningOp<ConstantOp>();
  auto biasConstant = bias.getDefiningOp<ConstantOp>();
  if (!weightConstant || !biasConstant)
    return mlir::failure();

  norm.weight = weight;
  norm.bias = bias;
  norm.weightConstant = weightConstant;
  norm.biasConstant = biasConstant;
  return norm;
}

static mlir::FailureOr<NNTransformerBlockOp>
matchSingleTransformerBlockOp(mlir::func::FuncOp func) {
  if (!func || !nn_layer_match::isSculptorLayer(func.getOperation()) ||
      !func.getBody().hasOneBlock())
    return mlir::failure();

  mlir::Block &entryBlock = func.front();
  auto returnOp =
      llvm::dyn_cast_or_null<mlir::func::ReturnOp>(entryBlock.getTerminator());
  if (!returnOp)
    return mlir::failure();

  NNTransformerBlockOp blockOp;
  bool seenBlock = false;
  for (mlir::Operation &op : entryBlock) {
    if (&op == returnOp.getOperation())
      continue;

    if (auto candidate = llvm::dyn_cast<NNTransformerBlockOp>(op)) {
      if (blockOp)
        return mlir::failure();
      blockOp = candidate;
      seenBlock = true;
      continue;
    }

    if (seenBlock)
      return mlir::failure();
  }

  if (!blockOp || returnOp.getNumOperands() != blockOp->getNumResults())
    return mlir::failure();

  for (unsigned index = 0, count = blockOp->getNumResults(); index < count;
       ++index) {
    if (returnOp.getOperand(index) != blockOp->getResult(index))
      return mlir::failure();
  }

  return blockOp;
}

static mlir::LogicalResult
matchBlockFunctionSignature(mlir::func::FuncOp func,
                            NNTransformerBlockOp blockOp,
                            TransformerBlockLowering &match) {
  match.blockKind = blockOp.getBlockKind();

  if (isEncoder(match)) {
    if (!nn_layer_match::hasLayerType(func, "transformer_encoder_block") ||
        func.getNumArguments() != 1 ||
        blockOp.getInput() != func.getArgument(0) || blockOp.getMemory())
      return mlir::failure();
  } else if (isDecoder(match)) {
    if (!nn_layer_match::hasLayerType(func, "transformer_decoder_block") ||
        func.getNumArguments() < 1 ||
        blockOp.getInput() != func.getArgument(0))
      return mlir::failure();

    if (blockOp.getHasCrossAttention()) {
      if (func.getNumArguments() != 2 ||
          blockOp.getMemory() != func.getArgument(1))
        return mlir::failure();
    } else {
      if (func.getNumArguments() != 1 || blockOp.getMemory())
        return mlir::failure();
    }
  } else {
    return mlir::failure();
  }

  if (func.getNumResults() != 1)
    return mlir::failure();

  return mlir::success();
}

static mlir::FailureOr<TransformerBlockLowering>
matchExtractedTransformerBlock(mlir::func::FuncOp func) {
  auto blockOp = matchSingleTransformerBlockOp(func);
  if (mlir::failed(blockOp))
    return mlir::failure();

  TransformerBlockLowering match;
  match.blockOp = *blockOp;
  if (mlir::failed(matchBlockFunctionSignature(func, *blockOp, match)))
    return mlir::failure();

  mlir::StringRef normMode = blockOp->getNormMode();
  if (!blockOp->getBatchFirst() || blockOp->getActivation() != "gelu" ||
      (normMode != "post" && normMode != "pre"))
    return mlir::failure();

  auto inputTy =
      tensor_type::getPositiveStaticF32Tensor(blockOp->getInput().getType(),
                                              /*expectedRank=*/3);
  auto outputTy =
      tensor_type::getPositiveStaticF32Tensor(blockOp->getOutput().getType(),
                                              /*expectedRank=*/3);
  if (mlir::failed(inputTy) || mlir::failed(outputTy))
    return mlir::failure();

  match.inputTy = *inputTy;
  match.outputTy = *outputTy;
  match.elementType = inputTy->getElementType();
  match.batchSize = inputTy->getShape()[0];
  match.sequenceLength = inputTy->getShape()[1];
  match.hiddenSize = blockOp->getHiddenSize();
  match.numHeads = blockOp->getNumHeads();
  match.headDim = blockOp->getHeadDim();
  match.mlpHiddenSize = blockOp->getMlpHiddenSize();
  match.layerNormEps = blockOp->getLayerNormEpsAttr().getValueAsDouble();
  match.normMode = normMode;
  match.hasProjectionBias = blockOp->getHasProjectionBias();
  match.hasLayerNormAffine = blockOp->getHasLayerNormAffine();
  match.hasFinalNorm = blockOp->getHasFinalNorm();
  match.causal = blockOp->getCausal();
  match.hasCrossAttention = blockOp->getHasCrossAttention();

  if (match.hiddenSize <= 0 || match.numHeads <= 0 || match.headDim <= 0 ||
      match.mlpHiddenSize <= 0 ||
      match.hiddenSize != match.numHeads * match.headDim ||
      match.inputTy.getShape()[2] != match.hiddenSize ||
      match.outputTy.getShape() != match.inputTy.getShape())
    return mlir::failure();

  if (isDecoder(match)) {
    if (match.hasCrossAttention && match.normMode == "pre")
      return mlir::failure();

    if (match.hasCrossAttention) {
      auto memoryTy =
          tensor_type::getPositiveStaticF32Tensor(blockOp->getMemory().getType(),
                                                  /*expectedRank=*/3);
      if (mlir::failed(memoryTy) ||
          memoryTy->getShape()[0] != match.batchSize ||
          memoryTy->getShape()[2] != match.hiddenSize)
        return mlir::failure();
      match.memoryTy = *memoryTy;
      match.memoryLength = memoryTy->getShape()[1];
    }
  }

  auto qkv = matchProjection(blockOp->getQkvWeight(), blockOp->getQkvBias(),
                             match.hiddenSize * 3, match.hiddenSize,
                             match.hasProjectionBias);
  auto attnOutput = matchProjection(
      blockOp->getAttnOutputWeight(), blockOp->getAttnOutputBias(),
      match.hiddenSize, match.hiddenSize, match.hasProjectionBias);
  auto mlpUp = matchProjection(blockOp->getMlpUpWeight(),
                               blockOp->getMlpUpBias(), match.mlpHiddenSize,
                               match.hiddenSize, match.hasProjectionBias);
  auto mlpDown = matchProjection(blockOp->getMlpDownWeight(),
                                 blockOp->getMlpDownBias(), match.hiddenSize,
                                 match.mlpHiddenSize, match.hasProjectionBias);
  auto attnNorm = matchNorm(
      blockOp->getAttnNormWeight(), blockOp->getAttnNormBias(),
      /*sectionPresent=*/true, match.hasLayerNormAffine, match.hiddenSize);
  auto mlpNorm = matchNorm(
      blockOp->getMlpNormWeight(), blockOp->getMlpNormBias(),
      /*sectionPresent=*/true, match.hasLayerNormAffine, match.hiddenSize);
  auto finalNorm =
      matchNorm(blockOp->getFinalNormWeight(), blockOp->getFinalNormBias(),
                match.hasFinalNorm, match.hasLayerNormAffine, match.hiddenSize);

  if (mlir::failed(qkv) || mlir::failed(attnOutput) || mlir::failed(mlpUp) ||
      mlir::failed(mlpDown) || mlir::failed(attnNorm) ||
      mlir::failed(mlpNorm) || mlir::failed(finalNorm))
    return mlir::failure();

  match.qkv = *qkv;
  match.attnOutput = *attnOutput;
  match.mlpUp = *mlpUp;
  match.mlpDown = *mlpDown;
  match.attnNorm = *attnNorm;
  match.mlpNorm = *mlpNorm;
  match.finalNorm = *finalNorm;

  if (!match.hasCrossAttention) {
    if (blockOp->getCrossQueryWeight() || blockOp->getCrossQueryBias() ||
        blockOp->getCrossKeyValueWeight() || blockOp->getCrossKeyValueBias() ||
        blockOp->getCrossOutputWeight() || blockOp->getCrossOutputBias() ||
        blockOp->getCrossNormWeight() || blockOp->getCrossNormBias())
      return mlir::failure();
    return match;
  }

  auto crossQuery = matchProjection(
      blockOp->getCrossQueryWeight(), blockOp->getCrossQueryBias(),
      match.hiddenSize, match.hiddenSize, match.hasProjectionBias);
  auto crossKeyValue = matchProjection(
      blockOp->getCrossKeyValueWeight(), blockOp->getCrossKeyValueBias(),
      match.hiddenSize * 2, match.hiddenSize, match.hasProjectionBias);
  auto crossOutput = matchProjection(
      blockOp->getCrossOutputWeight(), blockOp->getCrossOutputBias(),
      match.hiddenSize, match.hiddenSize, match.hasProjectionBias);
  auto crossNorm = matchNorm(
      blockOp->getCrossNormWeight(), blockOp->getCrossNormBias(),
      /*sectionPresent=*/true, match.hasLayerNormAffine, match.hiddenSize);
  if (mlir::failed(crossQuery) || mlir::failed(crossKeyValue) ||
      mlir::failed(crossOutput) || mlir::failed(crossNorm))
    return mlir::failure();

  match.crossQuery = *crossQuery;
  match.crossKeyValue = *crossKeyValue;
  match.crossOutput = *crossOutput;
  match.crossNorm = *crossNorm;
  return match;
}

static mlir::Value buildExtractTokenRegion(TransformerBlockLowering &match,
                                           mlir::Value sequence, int64_t batch,
                                           int64_t step, int64_t width,
                                           llvm::StringRef name,
                                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  mlir::RankedTensorType rowTy =
      mlir::RankedTensorType::get({1, width}, match.elementType);
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{rowTy}, mlir::ValueRange{sequence},
      "digital.token_extract", builder.getStringAttr(name));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange{sequence});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::RankedTensorType sliceTy =
      mlir::RankedTensorType::get({1, 1, width}, match.elementType);
  auto slice = builder.create<mlir::tensor::ExtractSliceOp>(
      loc, sliceTy, body->getArgument(0),
      buildIndexAttrs(builder, {batch, step, 0}),
      buildIndexAttrs(builder, {1, 1, width}),
      buildIndexAttrs(builder, {1, 1, 1}));
  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0, 1},
                                                                    {2}};
  mlir::Value row = builder
                        .create<mlir::tensor::CollapseShapeOp>(
                            loc, rowTy, slice.getResult(), reassociation)
                        .getResult();
  builder.create<mlir::sculptor::YieldOp>(loc, row);
  return region.getResult(0);
}

static mlir::Value buildBiasAddRegion(TransformerBlockLowering &match,
                                      ProjectionLowering &projection,
                                      mlir::RankedTensorType rowResultTy,
                                      mlir::Value mvmResult,
                                      llvm::StringRef name,
                                      mlir::OpBuilder &builder) {
  if (!projection.bias)
    return mvmResult;

  mlir::Location loc = match.blockOp.getLoc();
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{rowResultTy}, mlir::ValueRange{mvmResult},
      "digital.bias_add", builder.getStringAttr(name));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange{mvmResult});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  auto biasConstant =
      builder.create<ConstantOp>(loc, projection.biasConstant.getType(),
                                 projection.biasConstant.getValue());
  llvm::SmallVector<mlir::ReassociationIndices, 1> reassociation = {{0, 1}};
  mlir::Value expandedBias =
      builder
          .create<mlir::tensor::ExpandShapeOp>(
              loc, rowResultTy, biasConstant.getResult(), reassociation)
          .getResult();
  mlir::Value init = builder.create<EmptyOp>(loc, rowResultTy.getShape(),
                                             rowResultTy.getElementType());
  mlir::Value biased =
      builder
          .create<mlir::linalg::AddOp>(
              loc, mlir::ValueRange{body->getArgument(0), expandedBias},
              mlir::ValueRange{init})
          .getResult(0);
  builder.create<mlir::sculptor::YieldOp>(loc, biased);
  return region.getResult(0);
}

static mlir::Value buildOutputRecombineRegion(TransformerBlockLowering &match,
                                              llvm::ArrayRef<mlir::Value> rows,
                                              mlir::RankedTensorType outputTy,
                                              int64_t sequenceLength,
                                              llvm::StringRef name,
                                              mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{outputTy}, mlir::ValueRange(rows),
      "digital.output_recombine", builder.getStringAttr(name));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange(rows));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  int64_t width = outputTy.getShape()[2];
  mlir::RankedTensorType sliceTy =
      mlir::RankedTensorType::get({1, 1, width}, outputTy.getElementType());
  mlir::Value output = builder.create<EmptyOp>(loc, outputTy.getShape(),
                                               outputTy.getElementType());
  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0, 1},
                                                                    {2}};

  unsigned rowIndex = 0;
  for (int64_t batch = 0; batch < match.batchSize; ++batch) {
    for (int64_t step = 0; step < sequenceLength; ++step) {
      mlir::Value expanded =
          builder
              .create<mlir::tensor::ExpandShapeOp>(
                  loc, sliceTy, body->getArgument(rowIndex++), reassociation)
              .getResult();
      output = builder
                   .create<mlir::tensor::InsertSliceOp>(
                       loc, expanded, output,
                       buildIndexAttrs(builder, {batch, step, 0}),
                       buildIndexAttrs(builder, {1, 1, width}),
                       buildIndexAttrs(builder, {1, 1, 1}))
                   .getResult();
    }
  }

  builder.create<mlir::sculptor::YieldOp>(loc, output);
  return region.getResult(0);
}

static mlir::Value
buildSequenceProjection(TransformerBlockLowering &match, mlir::Value sequence,
                        ProjectionLowering &projection, int64_t inputWidth,
                        int64_t outputWidth, int64_t sequenceLength,
                        llvm::StringRef name, mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  mlir::RankedTensorType rowResultTy =
      mlir::RankedTensorType::get({1, outputWidth}, match.elementType);
  llvm::SmallVector<mlir::Value> rows;
  rows.reserve(match.batchSize * sequenceLength);

  for (int64_t batch = 0; batch < match.batchSize; ++batch) {
    for (int64_t step = 0; step < sequenceLength; ++step) {
      std::string suffix =
          "_b" + std::to_string(batch) + "_s" + std::to_string(step);
      mlir::Value token = buildExtractTokenRegion(
          match, sequence, batch, step, inputWidth,
          std::string(name) + "_token_extract" + suffix, builder);
      mlir::Value mvm = mvm_build::buildMVM(loc, rowResultTy, token,
                                            projection.weight, builder);
      rows.push_back(buildBiasAddRegion(
          match, projection, rowResultTy, mvm,
          std::string(name) + "_bias_add" + suffix, builder));
    }
  }

  mlir::RankedTensorType outputTy = mlir::RankedTensorType::get(
      {match.batchSize, sequenceLength, outputWidth}, match.elementType);
  return buildOutputRecombineRegion(match, rows, outputTy, sequenceLength,
                                    std::string(name) + "_output_recombine",
                                    builder);
}

static llvm::SmallVector<mlir::Value, 3>
buildQKVSplitRegion(TransformerBlockLowering &match, mlir::Value qkv,
                    llvm::StringRef name, mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  mlir::RankedTensorType resultTy = mlir::RankedTensorType::get(
      {match.batchSize, match.sequenceLength, match.hiddenSize},
      match.elementType);
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{resultTy, resultTy, resultTy}, mlir::ValueRange{qkv},
      "digital.qkv_split", builder.getStringAttr(name));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange{qkv});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  llvm::SmallVector<mlir::Value, 3> parts;
  for (int64_t part = 0; part < 3; ++part) {
    parts.push_back(
        builder
            .create<mlir::tensor::ExtractSliceOp>(
                loc, resultTy, body->getArgument(0),
                buildIndexAttrs(builder, {0, 0, part * match.hiddenSize}),
                buildIndexAttrs(builder, {match.batchSize, match.sequenceLength,
                                          match.hiddenSize}),
                buildIndexAttrs(builder, {1, 1, 1}))
            .getResult());
  }

  builder.create<mlir::sculptor::YieldOp>(loc, parts);
  return {region.getResult(0), region.getResult(1), region.getResult(2)};
}

static llvm::SmallVector<mlir::Value, 2>
buildCrossKVSplitRegion(TransformerBlockLowering &match, mlir::Value kv,
                        llvm::StringRef name, mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  mlir::RankedTensorType resultTy = mlir::RankedTensorType::get(
      {match.batchSize, match.memoryLength, match.hiddenSize},
      match.elementType);
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{resultTy, resultTy}, mlir::ValueRange{kv},
      "digital.cross_kv_split", builder.getStringAttr(name));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange{kv});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  llvm::SmallVector<mlir::Value, 2> parts;
  for (int64_t part = 0; part < 2; ++part) {
    parts.push_back(
        builder
            .create<mlir::tensor::ExtractSliceOp>(
                loc, resultTy, body->getArgument(0),
                buildIndexAttrs(builder, {0, 0, part * match.hiddenSize}),
                buildIndexAttrs(builder, {match.batchSize, match.memoryLength,
                                          match.hiddenSize}),
                buildIndexAttrs(builder, {1, 1, 1}))
            .getResult());
  }

  builder.create<mlir::sculptor::YieldOp>(loc, parts);
  return {region.getResult(0), region.getResult(1)};
}

static mlir::Value
buildAttentionScoresRegion(TransformerBlockLowering &match, mlir::Value query,
                           mlir::Value key, int64_t queryLength,
                           int64_t keyLength, bool causal, llvm::StringRef name,
                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  mlir::RankedTensorType scoreTy = mlir::RankedTensorType::get(
      {match.batchSize, match.numHeads, queryLength, keyLength},
      match.elementType);
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{scoreTy}, mlir::ValueRange{query, key},
      "digital.attention_scores", builder.getStringAttr(name));
  region->setAttr("head_dim", builder.getI64IntegerAttr(match.headDim));
  region->setAttr("causal", builder.getBoolAttr(causal));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange{query, key});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::Value init = builder.create<EmptyOp>(loc, scoreTy.getShape(),
                                             scoreTy.getElementType());
  double scale = 1.0 / std::sqrt(static_cast<double>(match.headDim));
  mlir::Value scores = init;
  for (int64_t batch = 0; batch < match.batchSize; ++batch) {
    for (int64_t head = 0; head < match.numHeads; ++head) {
      for (int64_t queryIndex = 0; queryIndex < queryLength; ++queryIndex) {
        for (int64_t keyIndex = 0; keyIndex < keyLength; ++keyIndex) {
          mlir::Value sum = buildF32Constant(builder, loc, 0.0);
          for (int64_t dim = 0; dim < match.headDim; ++dim) {
            int64_t feature = head * match.headDim + dim;
            mlir::Value queryValue = builder.create<mlir::tensor::ExtractOp>(
                loc, body->getArgument(0),
                mlir::ValueRange{buildIndexConstant(builder, loc, batch),
                                 buildIndexConstant(builder, loc, queryIndex),
                                 buildIndexConstant(builder, loc, feature)});
            mlir::Value keyValue = builder.create<mlir::tensor::ExtractOp>(
                loc, body->getArgument(1),
                mlir::ValueRange{buildIndexConstant(builder, loc, batch),
                                 buildIndexConstant(builder, loc, keyIndex),
                                 buildIndexConstant(builder, loc, feature)});
            mlir::Value product =
                builder.create<mlir::arith::MulFOp>(loc, queryValue, keyValue);
            sum = builder.create<mlir::arith::AddFOp>(loc, sum, product);
          }
          if (causal && keyIndex > queryIndex) {
            sum = buildF32Constant(builder, loc,
                                   -std::numeric_limits<float>::infinity());
          } else {
            mlir::Value scaleValue = buildF32Constant(builder, loc, scale);
            sum = builder.create<mlir::arith::MulFOp>(loc, sum, scaleValue);
          }
          scores = builder.create<mlir::tensor::InsertOp>(
              loc, sum, scores,
              mlir::ValueRange{buildIndexConstant(builder, loc, batch),
                               buildIndexConstant(builder, loc, head),
                               buildIndexConstant(builder, loc, queryIndex),
                               buildIndexConstant(builder, loc, keyIndex)});
        }
      }
    }
  }

  builder.create<mlir::sculptor::YieldOp>(loc, scores);
  return region.getResult(0);
}

static mlir::Value
buildAttentionSoftmaxRegion(TransformerBlockLowering &match, mlir::Value scores,
                            int64_t queryLength, int64_t keyLength,
                            llvm::StringRef name, mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  mlir::RankedTensorType scoreTy = mlir::RankedTensorType::get(
      {match.batchSize, match.numHeads, queryLength, keyLength},
      match.elementType);
  mlir::RankedTensorType reduceTy = mlir::RankedTensorType::get(
      {match.batchSize, match.numHeads, queryLength, 1}, match.elementType);
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{scoreTy}, mlir::ValueRange{scores},
      "digital.attention_softmax", builder.getStringAttr(name));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange{scores});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::Value negInf =
      buildF32Constant(builder, loc, -std::numeric_limits<float>::infinity());
  mlir::Value maxInit = builder.create<EmptyOp>(loc, reduceTy.getShape(),
                                                reduceTy.getElementType());
  mlir::Value maxFill =
      builder.create<mlir::linalg::FillOp>(loc, negInf, maxInit).getResult(0);

  mlir::MLIRContext *context = builder.getContext();
  mlir::AffineExpr b = builder.getAffineDimExpr(0);
  mlir::AffineExpr h = builder.getAffineDimExpr(1);
  mlir::AffineExpr q = builder.getAffineDimExpr(2);
  mlir::AffineExpr k = builder.getAffineDimExpr(3);
  mlir::AffineMap scoreMap = mlir::AffineMap::get(
      /*dimCount=*/4, /*symbolCount=*/0, {b, h, q, k}, context);
  mlir::AffineMap reduceMap = mlir::AffineMap::get(
      /*dimCount=*/4, /*symbolCount=*/0,
      {b, h, q, builder.getAffineConstantExpr(0)}, context);
  llvm::SmallVector<mlir::utils::IteratorType, 4> reduceIterators = {
      mlir::utils::IteratorType::parallel, mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::reduction};

  mlir::Value maxValue =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, reduceTy, mlir::ValueRange{body->getArgument(0)},
              mlir::ValueRange{maxFill},
              llvm::SmallVector<mlir::AffineMap, 2>{scoreMap, reduceMap},
              reduceIterators,
              [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                 mlir::ValueRange args) {
                mlir::Value max = builder.create<mlir::arith::MaximumFOp>(
                    nestedLoc, args[0], args[1]);
                builder.create<mlir::linalg::YieldOp>(nestedLoc, max);
              })
          .getResult(0);

  mlir::Value expInit = builder.create<EmptyOp>(loc, scoreTy.getShape(),
                                                scoreTy.getElementType());
  llvm::SmallVector<mlir::utils::IteratorType, 4> parallelIterators(
      scoreTy.getRank(), mlir::utils::IteratorType::parallel);
  mlir::Value expValue =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, scoreTy, mlir::ValueRange{body->getArgument(0), maxValue},
              mlir::ValueRange{expInit},
              llvm::SmallVector<mlir::AffineMap, 3>{scoreMap, reduceMap,
                                                    scoreMap},
              parallelIterators,
              [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                 mlir::ValueRange args) {
                mlir::Value shifted = builder.create<mlir::arith::SubFOp>(
                    nestedLoc, args[0], args[1]);
                mlir::Value exp =
                    builder.create<mlir::math::ExpOp>(nestedLoc, shifted);
                builder.create<mlir::linalg::YieldOp>(nestedLoc, exp);
              })
          .getResult(0);

  mlir::Value zero = buildF32Constant(builder, loc, 0.0);
  mlir::Value sumInit = builder.create<EmptyOp>(loc, reduceTy.getShape(),
                                                reduceTy.getElementType());
  mlir::Value sumFill =
      builder.create<mlir::linalg::FillOp>(loc, zero, sumInit).getResult(0);
  mlir::Value sumValue =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, reduceTy, mlir::ValueRange{expValue},
              mlir::ValueRange{sumFill},
              llvm::SmallVector<mlir::AffineMap, 2>{scoreMap, reduceMap},
              reduceIterators,
              [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                 mlir::ValueRange args) {
                mlir::Value sum = builder.create<mlir::arith::AddFOp>(
                    nestedLoc, args[0], args[1]);
                builder.create<mlir::linalg::YieldOp>(nestedLoc, sum);
              })
          .getResult(0);

  mlir::Value probInit = builder.create<EmptyOp>(loc, scoreTy.getShape(),
                                                 scoreTy.getElementType());
  mlir::Value probabilities =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, scoreTy, mlir::ValueRange{expValue, sumValue},
              mlir::ValueRange{probInit},
              llvm::SmallVector<mlir::AffineMap, 3>{scoreMap, reduceMap,
                                                    scoreMap},
              parallelIterators,
              [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                 mlir::ValueRange args) {
                mlir::Value prob = builder.create<mlir::arith::DivFOp>(
                    nestedLoc, args[0], args[1]);
                builder.create<mlir::linalg::YieldOp>(nestedLoc, prob);
              })
          .getResult(0);

  builder.create<mlir::sculptor::YieldOp>(loc, probabilities);
  return region.getResult(0);
}

static mlir::Value
buildAttentionApplyRegion(TransformerBlockLowering &match,
                          mlir::Value probabilities, mlir::Value value,
                          int64_t queryLength, int64_t keyLength,
                          llvm::StringRef name, mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  mlir::RankedTensorType headTy = mlir::RankedTensorType::get(
      {match.batchSize, match.numHeads, queryLength, match.headDim},
      match.elementType);
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{headTy}, mlir::ValueRange{probabilities, value},
      "digital.attention_apply", builder.getStringAttr(name));
  mlir::Block *body =
      addTaskRegionBody(region, mlir::ValueRange{probabilities, value});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::Value init =
      builder.create<EmptyOp>(loc, headTy.getShape(), headTy.getElementType());
  mlir::Value heads = init;
  for (int64_t batch = 0; batch < match.batchSize; ++batch) {
    for (int64_t head = 0; head < match.numHeads; ++head) {
      for (int64_t queryIndex = 0; queryIndex < queryLength; ++queryIndex) {
        for (int64_t dim = 0; dim < match.headDim; ++dim) {
          mlir::Value sum = buildF32Constant(builder, loc, 0.0);
          for (int64_t keyIndex = 0; keyIndex < keyLength; ++keyIndex) {
            int64_t feature = head * match.headDim + dim;
            mlir::Value probability = builder.create<mlir::tensor::ExtractOp>(
                loc, body->getArgument(0),
                mlir::ValueRange{buildIndexConstant(builder, loc, batch),
                                 buildIndexConstant(builder, loc, head),
                                 buildIndexConstant(builder, loc, queryIndex),
                                 buildIndexConstant(builder, loc, keyIndex)});
            mlir::Value value = builder.create<mlir::tensor::ExtractOp>(
                loc, body->getArgument(1),
                mlir::ValueRange{buildIndexConstant(builder, loc, batch),
                                 buildIndexConstant(builder, loc, keyIndex),
                                 buildIndexConstant(builder, loc, feature)});
            mlir::Value product =
                builder.create<mlir::arith::MulFOp>(loc, probability, value);
            sum = builder.create<mlir::arith::AddFOp>(loc, sum, product);
          }
          heads = builder.create<mlir::tensor::InsertOp>(
              loc, sum, heads,
              mlir::ValueRange{buildIndexConstant(builder, loc, batch),
                               buildIndexConstant(builder, loc, head),
                               buildIndexConstant(builder, loc, queryIndex),
                               buildIndexConstant(builder, loc, dim)});
        }
      }
    }
  }

  builder.create<mlir::sculptor::YieldOp>(loc, heads);
  return region.getResult(0);
}

static mlir::Value buildHeadRecombineRegion(TransformerBlockLowering &match,
                                            mlir::Value heads,
                                            int64_t sequenceLength,
                                            llvm::StringRef name,
                                            mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  mlir::RankedTensorType outputTy = mlir::RankedTensorType::get(
      {match.batchSize, sequenceLength, match.hiddenSize}, match.elementType);
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{outputTy}, mlir::ValueRange{heads},
      "digital.head_recombine", builder.getStringAttr(name));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange{heads});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::Value output = builder.create<EmptyOp>(loc, outputTy.getShape(),
                                               outputTy.getElementType());
  for (int64_t batch = 0; batch < match.batchSize; ++batch) {
    for (int64_t head = 0; head < match.numHeads; ++head) {
      for (int64_t step = 0; step < sequenceLength; ++step) {
        for (int64_t dim = 0; dim < match.headDim; ++dim) {
          int64_t feature = head * match.headDim + dim;
          mlir::Value value = builder.create<mlir::tensor::ExtractOp>(
              loc, body->getArgument(0),
              mlir::ValueRange{buildIndexConstant(builder, loc, batch),
                               buildIndexConstant(builder, loc, head),
                               buildIndexConstant(builder, loc, step),
                               buildIndexConstant(builder, loc, dim)});
          output = builder.create<mlir::tensor::InsertOp>(
              loc, value, output,
              mlir::ValueRange{buildIndexConstant(builder, loc, batch),
                               buildIndexConstant(builder, loc, step),
                               buildIndexConstant(builder, loc, feature)});
        }
      }
    }
  }

  builder.create<mlir::sculptor::YieldOp>(loc, output);
  return region.getResult(0);
}

static mlir::Value buildResidualAddRegion(TransformerBlockLowering &match,
                                          mlir::Value residual,
                                          mlir::Value update,
                                          llvm::StringRef name,
                                          mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  auto resultTy = llvm::cast<mlir::RankedTensorType>(residual.getType());
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{resultTy}, mlir::ValueRange{residual, update},
      "digital.residual_add", builder.getStringAttr(name));
  mlir::Block *body =
      addTaskRegionBody(region, mlir::ValueRange{residual, update});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::Value init = builder.create<EmptyOp>(loc, resultTy.getShape(),
                                             resultTy.getElementType());
  llvm::SmallVector<mlir::AffineMap, 3> maps(
      3, builder.getMultiDimIdentityMap(resultTy.getRank()));
  llvm::SmallVector<mlir::utils::IteratorType, 3> iterators(
      resultTy.getRank(), mlir::utils::IteratorType::parallel);
  mlir::Value result =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, resultTy,
              mlir::ValueRange{body->getArgument(0), body->getArgument(1)},
              mlir::ValueRange{init}, maps, iterators,
              [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                 mlir::ValueRange args) {
                mlir::Value sum = builder.create<mlir::arith::AddFOp>(
                    nestedLoc, args[0], args[1]);
                builder.create<mlir::linalg::YieldOp>(nestedLoc, sum);
              })
          .getResult(0);
  builder.create<mlir::sculptor::YieldOp>(loc, result);
  return region.getResult(0);
}

static mlir::Value buildLayerNormRegion(TransformerBlockLowering &match,
                                        mlir::Value input, NormLowering &norm,
                                        llvm::StringRef name,
                                        mlir::OpBuilder &builder) {
  if (!norm.present)
    return input;

  mlir::Location loc = match.blockOp.getLoc();
  auto resultTy = llvm::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<mlir::Value> inputs = {input};
  if (norm.weight) {
    inputs.push_back(norm.weight);
    inputs.push_back(norm.bias);
  }

  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{resultTy}, mlir::ValueRange(inputs),
      "digital.layer_norm", builder.getStringAttr(name));
  region->setAttr("epsilon", builder.getF64FloatAttr(match.layerNormEps));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange(inputs));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::RankedTensorType reduceTy = mlir::RankedTensorType::get(
      {resultTy.getShape()[0], resultTy.getShape()[1], 1},
      resultTy.getElementType());
  mlir::Value zero = buildF32Constant(builder, loc, 0.0);
  mlir::Value invHidden = buildF32Constant(
      builder, loc, 1.0 / static_cast<double>(match.hiddenSize));

  mlir::MLIRContext *context = builder.getContext();
  mlir::AffineExpr b = builder.getAffineDimExpr(0);
  mlir::AffineExpr s = builder.getAffineDimExpr(1);
  mlir::AffineExpr h = builder.getAffineDimExpr(2);
  mlir::AffineMap valueMap = mlir::AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/0, {b, s, h}, context);
  mlir::AffineMap reduceMap =
      mlir::AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0,
                           {b, s, builder.getAffineConstantExpr(0)}, context);
  llvm::SmallVector<mlir::utils::IteratorType, 3> reduceIterators = {
      mlir::utils::IteratorType::parallel, mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::reduction};
  llvm::SmallVector<mlir::utils::IteratorType, 3> parallelIterators(
      resultTy.getRank(), mlir::utils::IteratorType::parallel);

  mlir::Value meanInit = builder.create<EmptyOp>(loc, reduceTy.getShape(),
                                                 reduceTy.getElementType());
  mlir::Value meanFill =
      builder.create<mlir::linalg::FillOp>(loc, zero, meanInit).getResult(0);
  mlir::Value mean =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, reduceTy, mlir::ValueRange{body->getArgument(0)},
              mlir::ValueRange{meanFill},
              llvm::SmallVector<mlir::AffineMap, 2>{valueMap, reduceMap},
              reduceIterators,
              [invHidden](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                          mlir::ValueRange args) {
                mlir::Value scaled = builder.create<mlir::arith::MulFOp>(
                    nestedLoc, args[0], invHidden);
                mlir::Value sum = builder.create<mlir::arith::AddFOp>(
                    nestedLoc, scaled, args[1]);
                builder.create<mlir::linalg::YieldOp>(nestedLoc, sum);
              })
          .getResult(0);

  mlir::Value varianceInit = builder.create<EmptyOp>(loc, reduceTy.getShape(),
                                                     reduceTy.getElementType());
  mlir::Value varianceFill =
      builder.create<mlir::linalg::FillOp>(loc, zero, varianceInit)
          .getResult(0);
  mlir::Value variance =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, reduceTy, mlir::ValueRange{body->getArgument(0), mean},
              mlir::ValueRange{varianceFill},
              llvm::SmallVector<mlir::AffineMap, 3>{valueMap, reduceMap,
                                                    reduceMap},
              reduceIterators,
              [invHidden](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                          mlir::ValueRange args) {
                mlir::Value centered = builder.create<mlir::arith::SubFOp>(
                    nestedLoc, args[0], args[1]);
                mlir::Value squared = builder.create<mlir::arith::MulFOp>(
                    nestedLoc, centered, centered);
                mlir::Value scaled = builder.create<mlir::arith::MulFOp>(
                    nestedLoc, squared, invHidden);
                mlir::Value sum = builder.create<mlir::arith::AddFOp>(
                    nestedLoc, scaled, args[2]);
                builder.create<mlir::linalg::YieldOp>(nestedLoc, sum);
              })
          .getResult(0);

  mlir::Value normalizedInit = builder.create<EmptyOp>(
      loc, resultTy.getShape(), resultTy.getElementType());
  llvm::SmallVector<mlir::AffineMap, 6> normMaps = {valueMap, reduceMap,
                                                    reduceMap};
  llvm::SmallVector<mlir::Value> normInputs = {body->getArgument(0), mean,
                                               variance};
  if (norm.weight) {
    mlir::AffineMap affineMap =
        mlir::AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {h}, context);
    normMaps.push_back(affineMap);
    normMaps.push_back(affineMap);
    normInputs.push_back(body->getArgument(1));
    normInputs.push_back(body->getArgument(2));
  }
  normMaps.push_back(valueMap);

  mlir::Value output =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, resultTy, mlir::ValueRange(normInputs),
              mlir::ValueRange{normalizedInit}, normMaps, parallelIterators,
              [hasAffine = static_cast<bool>(norm.weight),
               eps = match.layerNormEps](mlir::OpBuilder &builder,
                                         mlir::Location nestedLoc,
                                         mlir::ValueRange args) {
                mlir::Value centered = builder.create<mlir::arith::SubFOp>(
                    nestedLoc, args[0], args[1]);
                mlir::Value epsValue =
                    buildF32Constant(builder, nestedLoc, eps);
                mlir::Value varianceEps = builder.create<mlir::arith::AddFOp>(
                    nestedLoc, args[2], epsValue);
                mlir::Value invStd =
                    builder.create<mlir::math::RsqrtOp>(nestedLoc, varianceEps);
                mlir::Value normalized = builder.create<mlir::arith::MulFOp>(
                    nestedLoc, centered, invStd);
                if (hasAffine) {
                  normalized = builder.create<mlir::arith::MulFOp>(
                      nestedLoc, normalized, args[3]);
                  normalized = builder.create<mlir::arith::AddFOp>(
                      nestedLoc, normalized, args[4]);
                }
                builder.create<mlir::linalg::YieldOp>(nestedLoc, normalized);
              })
          .getResult(0);

  builder.create<mlir::sculptor::YieldOp>(loc, output);
  return region.getResult(0);
}

static mlir::Value buildGELURegion(TransformerBlockLowering &match,
                                   mlir::Value input, llvm::StringRef name,
                                   mlir::OpBuilder &builder) {
  mlir::Location loc = match.blockOp.getLoc();
  auto resultTy = llvm::cast<mlir::RankedTensorType>(input.getType());
  auto region = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{resultTy}, mlir::ValueRange{input},
      "digital.activation", builder.getStringAttr(name));
  region->setAttr("activation", builder.getStringAttr("gelu"));
  mlir::Block *body = addTaskRegionBody(region, mlir::ValueRange{input});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::Value init = builder.create<EmptyOp>(loc, resultTy.getShape(),
                                             resultTy.getElementType());
  llvm::SmallVector<mlir::AffineMap, 2> maps(
      2, builder.getMultiDimIdentityMap(resultTy.getRank()));
  llvm::SmallVector<mlir::utils::IteratorType, 3> iterators(
      resultTy.getRank(), mlir::utils::IteratorType::parallel);
  mlir::Value activated =
      builder
          .create<mlir::linalg::GenericOp>(
              loc, resultTy, mlir::ValueRange{body->getArgument(0)},
              mlir::ValueRange{init}, maps, iterators,
              [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                 mlir::ValueRange args) {
                mlir::Value half = buildF32Constant(builder, nestedLoc, 0.5);
                mlir::Value one = buildF32Constant(builder, nestedLoc, 1.0);
                mlir::Value sqrt2 =
                    buildF32Constant(builder, nestedLoc, std::sqrt(2.0));
                mlir::Value scaled = builder.create<mlir::arith::DivFOp>(
                    nestedLoc, args[0], sqrt2);
                mlir::Value erf =
                    builder.create<mlir::math::ErfOp>(nestedLoc, scaled);
                mlir::Value factor =
                    builder.create<mlir::arith::AddFOp>(nestedLoc, one, erf);
                mlir::Value halfInput = builder.create<mlir::arith::MulFOp>(
                    nestedLoc, half, args[0]);
                mlir::Value gelu = builder.create<mlir::arith::MulFOp>(
                    nestedLoc, halfInput, factor);
                builder.create<mlir::linalg::YieldOp>(nestedLoc, gelu);
              })
          .getResult(0);

  builder.create<mlir::sculptor::YieldOp>(loc, activated);
  return region.getResult(0);
}

static mlir::Value buildSelfAttentionSection(TransformerBlockLowering &match,
                                             mlir::Value input, bool causal,
                                             mlir::OpBuilder &builder) {
  mlir::Value qkv = buildSequenceProjection(
      match, input, match.qkv, match.hiddenSize, match.hiddenSize * 3,
      match.sequenceLength, "transformer_block_qkv", builder);
  llvm::SmallVector<mlir::Value, 3> qkvParts =
      buildQKVSplitRegion(match, qkv, "transformer_block_qkv_split", builder);
  mlir::Value scores = buildAttentionScoresRegion(
      match, qkvParts[0], qkvParts[1], match.sequenceLength,
      match.sequenceLength, causal, "transformer_block_self_attention_scores",
      builder);
  mlir::Value probabilities = buildAttentionSoftmaxRegion(
      match, scores, match.sequenceLength, match.sequenceLength,
      "transformer_block_self_attention_softmax", builder);
  mlir::Value headValues = buildAttentionApplyRegion(
      match, probabilities, qkvParts[2], match.sequenceLength,
      match.sequenceLength, "transformer_block_self_attention_apply", builder);
  mlir::Value attention = buildHeadRecombineRegion(
      match, headValues, match.sequenceLength,
      "transformer_block_self_head_recombine", builder);
  mlir::Value outputProjection = buildSequenceProjection(
      match, attention, match.attnOutput, match.hiddenSize, match.hiddenSize,
      match.sequenceLength, "transformer_block_attn_output", builder);
  mlir::Value residual = buildResidualAddRegion(
      match, input, outputProjection,
      "transformer_block_self_attn_residual_add", builder);
  return buildLayerNormRegion(match, residual, match.attnNorm,
                              "transformer_block_attn_norm", builder);
}

static mlir::Value
buildPreNormSelfAttentionSection(TransformerBlockLowering &match,
                                 mlir::Value input, bool causal,
                                 mlir::OpBuilder &builder) {
  mlir::Value normalized = buildLayerNormRegion(
      match, input, match.attnNorm, "transformer_block_attn_norm", builder);
  mlir::Value qkv = buildSequenceProjection(
      match, normalized, match.qkv, match.hiddenSize, match.hiddenSize * 3,
      match.sequenceLength, "transformer_block_qkv", builder);
  llvm::SmallVector<mlir::Value, 3> qkvParts =
      buildQKVSplitRegion(match, qkv, "transformer_block_qkv_split", builder);
  mlir::Value scores = buildAttentionScoresRegion(
      match, qkvParts[0], qkvParts[1], match.sequenceLength,
      match.sequenceLength, causal, "transformer_block_self_attention_scores",
      builder);
  mlir::Value probabilities = buildAttentionSoftmaxRegion(
      match, scores, match.sequenceLength, match.sequenceLength,
      "transformer_block_self_attention_softmax", builder);
  mlir::Value headValues = buildAttentionApplyRegion(
      match, probabilities, qkvParts[2], match.sequenceLength,
      match.sequenceLength, "transformer_block_self_attention_apply", builder);
  mlir::Value attention = buildHeadRecombineRegion(
      match, headValues, match.sequenceLength,
      "transformer_block_self_head_recombine", builder);
  mlir::Value outputProjection = buildSequenceProjection(
      match, attention, match.attnOutput, match.hiddenSize, match.hiddenSize,
      match.sequenceLength, "transformer_block_attn_output", builder);
  return buildResidualAddRegion(match, input, outputProjection,
                                "transformer_block_self_attn_residual_add",
                                builder);
}

static mlir::Value buildCrossAttentionSection(TransformerBlockLowering &match,
                                              mlir::Value decoderState,
                                              mlir::OpBuilder &builder) {
  mlir::Value query = buildSequenceProjection(
      match, decoderState, match.crossQuery, match.hiddenSize, match.hiddenSize,
      match.sequenceLength, "transformer_block_cross_query", builder);
  mlir::Value keyValue = buildSequenceProjection(
      match, match.blockOp.getMemory(), match.crossKeyValue, match.hiddenSize,
      match.hiddenSize * 2, match.memoryLength,
      "transformer_block_cross_key_value", builder);
  llvm::SmallVector<mlir::Value, 2> kvParts = buildCrossKVSplitRegion(
      match, keyValue, "transformer_block_cross_kv_split", builder);
  mlir::Value scores = buildAttentionScoresRegion(
      match, query, kvParts[0], match.sequenceLength, match.memoryLength,
      /*causal=*/false, "transformer_block_cross_attention_scores", builder);
  mlir::Value probabilities = buildAttentionSoftmaxRegion(
      match, scores, match.sequenceLength, match.memoryLength,
      "transformer_block_cross_attention_softmax", builder);
  mlir::Value headValues = buildAttentionApplyRegion(
      match, probabilities, kvParts[1], match.sequenceLength,
      match.memoryLength, "transformer_block_cross_attention_apply", builder);
  mlir::Value attention = buildHeadRecombineRegion(
      match, headValues, match.sequenceLength,
      "transformer_block_cross_head_recombine", builder);
  mlir::Value outputProjection = buildSequenceProjection(
      match, attention, match.crossOutput, match.hiddenSize, match.hiddenSize,
      match.sequenceLength, "transformer_block_cross_output", builder);
  mlir::Value residual = buildResidualAddRegion(
      match, decoderState, outputProjection,
      "transformer_block_cross_attn_residual_add", builder);
  return buildLayerNormRegion(match, residual, match.crossNorm,
                              "transformer_block_cross_norm", builder);
}

static mlir::Value buildMLPSection(TransformerBlockLowering &match,
                                   mlir::Value input,
                                   mlir::OpBuilder &builder) {
  mlir::Value up = buildSequenceProjection(
      match, input, match.mlpUp, match.hiddenSize, match.mlpHiddenSize,
      match.sequenceLength, "transformer_block_mlp_up", builder);
  mlir::Value activated =
      buildGELURegion(match, up, "transformer_block_mlp_gelu", builder);
  mlir::Value down = buildSequenceProjection(
      match, activated, match.mlpDown, match.mlpHiddenSize, match.hiddenSize,
      match.sequenceLength, "transformer_block_mlp_down", builder);
  mlir::Value residual = buildResidualAddRegion(
      match, input, down, "transformer_block_mlp_residual_add", builder);
  return buildLayerNormRegion(match, residual, match.mlpNorm,
                              "transformer_block_mlp_norm", builder);
}

static mlir::Value buildPreNormMLPSection(TransformerBlockLowering &match,
                                          mlir::Value input,
                                          mlir::OpBuilder &builder) {
  mlir::Value normalized = buildLayerNormRegion(
      match, input, match.mlpNorm, "transformer_block_mlp_norm", builder);
  mlir::Value up = buildSequenceProjection(
      match, normalized, match.mlpUp, match.hiddenSize, match.mlpHiddenSize,
      match.sequenceLength, "transformer_block_mlp_up", builder);
  mlir::Value activated =
      buildGELURegion(match, up, "transformer_block_mlp_gelu", builder);
  mlir::Value down = buildSequenceProjection(
      match, activated, match.mlpDown, match.mlpHiddenSize, match.hiddenSize,
      match.sequenceLength, "transformer_block_mlp_down", builder);
  return buildResidualAddRegion(match, input, down,
                                "transformer_block_mlp_residual_add",
                                builder);
}

static mlir::LogicalResult
materializeProjectionOperand(ProjectionLowering &projection,
                             llvm::StringRef name,
                             mlir::RewriterBase &rewriter) {
  if (!projection.weight)
    return mlir::success();

  bool materializeWeight = !projection.weightConstant ||
                           !converter_constant::isResourceBackedF32Constant(
                               projection.weightConstant);
  if (materializeWeight) {
    auto values =
        evaluateStaticF32Tensor(projection.weight, projection.weightTy);
    if (mlir::failed(values))
      return mlir::failure();

    std::string resourcePrefix = (name + "_weight_").str();
    mlir::TypedAttr attr = converter_constant::buildF32ElementsAttr(
        projection.weightTy, *values, resourcePrefix, /*useResource=*/true);
    if (!attr)
      return mlir::failure();

    projection.weightConstant = rewriter.create<ConstantOp>(
        projection.weight.getLoc(), projection.weightTy, attr);
    projection.weight = projection.weightConstant.getResult();
  }

  if (!projection.bias)
    return mlir::success();

  if (!projection.biasConstant) {
    auto values = evaluateStaticF32Tensor(projection.bias, projection.biasTy);
    if (mlir::failed(values))
      return mlir::failure();

    mlir::TypedAttr attr = converter_constant::buildF32ElementsAttr(
        projection.biasTy, *values, name, /*useResource=*/false);
    if (!attr)
      return mlir::failure();

    projection.biasConstant = rewriter.create<ConstantOp>(
        projection.bias.getLoc(), projection.biasTy, attr);
    projection.bias = projection.biasConstant.getResult();
  }

  return mlir::success();
}

static mlir::LogicalResult
materializeProjectionOperands(TransformerBlockLowering &match,
                              mlir::RewriterBase &rewriter) {
  if (mlir::failed(materializeProjectionOperand(
          match.qkv, "transformer_block_qkv", rewriter)) ||
      mlir::failed(materializeProjectionOperand(
          match.attnOutput, "transformer_block_attn_output", rewriter)) ||
      mlir::failed(materializeProjectionOperand(
          match.mlpUp, "transformer_block_mlp_up", rewriter)) ||
      mlir::failed(materializeProjectionOperand(
          match.mlpDown, "transformer_block_mlp_down", rewriter)))
    return mlir::failure();

  if (!match.hasCrossAttention)
    return mlir::success();

  if (mlir::failed(materializeProjectionOperand(
          match.crossQuery, "transformer_block_cross_query", rewriter)) ||
      mlir::failed(materializeProjectionOperand(
          match.crossKeyValue, "transformer_block_cross_key_value",
          rewriter)) ||
      mlir::failed(materializeProjectionOperand(
          match.crossOutput, "transformer_block_cross_output", rewriter)))
    return mlir::failure();

  return mlir::success();
}

static void eraseUnusedProjectionBiases(TransformerBlockLowering &match,
                                        mlir::RewriterBase &rewriter) {
  llvm::SmallPtrSet<mlir::Operation *, 8> constants;
  for (ConstantOp constant :
       {match.qkv.biasConstant, match.attnOutput.biasConstant,
        match.crossQuery.biasConstant, match.crossKeyValue.biasConstant,
        match.crossOutput.biasConstant, match.mlpUp.biasConstant,
        match.mlpDown.biasConstant}) {
    if (constant)
      constants.insert(constant.getOperation());
  }
  for (mlir::Operation *constant : constants) {
    if (constant->use_empty())
      rewriter.eraseOp(constant);
  }
}

static void eraseDeadTopLevelOps(mlir::func::FuncOp func,
                                 mlir::RewriterBase &rewriter) {
  if (!func.getBody().hasOneBlock())
    return;

  mlir::Block &block = func.front();
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *> ops;
    for (mlir::Operation &op : block) {
      if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
        ops.push_back(&op);
    }

    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      mlir::Operation *op = *it;
      if (op->getBlock() && mlir::isOpTriviallyDead(op)) {
        rewriter.eraseOp(op);
        changed = true;
      }
    }
  }
}

static mlir::LogicalResult
lowerTransformerBlockToMVM(mlir::func::FuncOp func,
                           mlir::RewriterBase &rewriter) {
  auto match = matchExtractedTransformerBlock(func);
  if (mlir::failed(match))
    return mlir::failure();

  rewriter.setInsertionPoint(match->blockOp);
  if (mlir::failed(materializeProjectionOperands(*match, rewriter)))
    return mlir::failure();

  mlir::Value state;
  if (match->normMode == "pre") {
    state = buildPreNormSelfAttentionSection(
        *match, match->blockOp.getInput(), match->causal, rewriter);
    state = buildPreNormMLPSection(*match, state, rewriter);
  } else {
    state = buildSelfAttentionSection(*match, match->blockOp.getInput(),
                                      match->causal, rewriter);
    if (match->hasCrossAttention)
      state = buildCrossAttentionSection(*match, state, rewriter);
    state = buildMLPSection(*match, state, rewriter);
  }
  state = buildLayerNormRegion(*match, state, match->finalNorm,
                               "transformer_block_final_norm", rewriter);

  match->blockOp.getOutput().replaceAllUsesWith(state);
  rewriter.eraseOp(match->blockOp);
  eraseUnusedProjectionBiases(*match, rewriter);
  eraseDeadTopLevelOps(func, rewriter);
  return mlir::success();
}

class TransformerBlockConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "transformer_block"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    (void)lowerTransformerBlockToMVM(func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerTransformerBlockConverter(LayerToMVMConverters &converters,
                                       LayerToMVMConverterMap &converterMap,
                                       MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<TransformerBlockConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["transformer_encoder_block"] = converterPtr;
  converterMap["transformer_decoder_block"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
