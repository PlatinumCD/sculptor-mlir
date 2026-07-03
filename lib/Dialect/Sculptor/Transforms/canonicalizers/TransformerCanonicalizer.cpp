#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/TensorTypeUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/RecurrentLayerPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace tensor_type = mlir::sculptor::tensor_type;
namespace linalg_match = mlir::sculptor::linalg_match;
namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace layer_utils = mlir::sculptor::layer_utils;

namespace {

using mlir::arith::ConstantOp;
using mlir::func::ReturnOp;
using mlir::linalg::BatchMatmulOp;
using mlir::linalg::GenericOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::CollapseShapeOp;
using mlir::tensor::ExpandShapeOp;
using mlir::tensor::ExtractSliceOp;
using mlir::sculptor::NNTransformerDecoderOp;
using mlir::sculptor::NNTransformerEncoderOp;
using mlir::sculptor::NNTransformerOp;

enum class TransformerBlockKind { Encoder, Decoder };

enum class ProjectionKind {
  Unknown,
  HiddenToThreeHidden,
  HiddenToHidden,
  HiddenToTwoHidden,
  HiddenToFeedForward,
  FeedForwardToHidden
};

struct TransformerTypes {
  mlir::RankedTensorType srcTy;
  mlir::RankedTensorType tgtTy;
  mlir::RankedTensorType outputTy;
  int64_t batchSize = 0;
  int64_t srcSequenceLength = 0;
  int64_t tgtSequenceLength = 0;
  int64_t hiddenSize = 0;
  int64_t numHeads = 0;
  int64_t headDim = 0;
  int64_t feedForwardSize = 0;
  int64_t encoderLayerCount = 0;
  int64_t decoderLayerCount = 0;
  bool hasBias = true;
  mlir::StringRef activation = "gelu";
};

struct ProjectionMatch {
  ProjectionKind kind = ProjectionKind::Unknown;
  mlir::Value input;
  mlir::Value output;
  mlir::Value weight;
  mlir::Value bias;
  ConstantOp weightConstant;
  ConstantOp biasConstant;
  TransposeOp weightTranspose;
  MatmulOp matmulOp;
  GenericOp biasAddOp;
  mlir::Operation *root = nullptr;
  int64_t inputWidth = 0;
  int64_t outputWidth = 0;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct LayerNormMatch {
  mlir::Value input;
  mlir::Value output;
  ConstantOp weightConstant;
  ConstantOp biasConstant;
  mlir::Operation *meanReduction = nullptr;
  mlir::Operation *centerOp = nullptr;
  mlir::Operation *varianceReduction = nullptr;
  mlir::Operation *epsilonAddOp = nullptr;
  mlir::Operation *rsqrtOp = nullptr;
  mlir::Operation *affineMulOp = nullptr;
  mlir::Operation *affineAddOp = nullptr;
  mlir::Operation *root = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct AttentionCoreMatch {
  mlir::Value query;
  mlir::Value key;
  mlir::Value value;
  mlir::Value probabilities;
  mlir::Value output;
  BatchMatmulOp qkMatmulOp;
  BatchMatmulOp pvMatmulOp;
  mlir::Operation *qkMatmulRoot = nullptr;
  mlir::Operation *pvMatmulRoot = nullptr;
  mlir::Operation *softmaxRoot = nullptr;
  int64_t querySequenceLength = 0;
  int64_t keySequenceLength = 0;
  int64_t numHeads = 0;
  int64_t headDim = 0;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct SelfAttentionMatch {
  ProjectionMatch qkvProjection;
  llvm::SmallVector<ExtractSliceOp> qkvSlices;
  llvm::SmallVector<ExpandShapeOp> qkvExpands;
  llvm::SmallVector<TransposeOp> qkvTransposes;
  llvm::SmallVector<CollapseShapeOp> qkvCollapses;
  AttentionCoreMatch core;
  ProjectionMatch outputProjection;
  mlir::Operation *residualAddOp = nullptr;
  LayerNormMatch norm;
  mlir::Value input;
  mlir::Value output;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct CrossAttentionMatch {
  ProjectionMatch queryProjection;
  ProjectionMatch keyValueProjection;
  llvm::SmallVector<ExtractSliceOp> keyValueSlices;
  llvm::SmallVector<ExpandShapeOp> keyValueExpands;
  llvm::SmallVector<TransposeOp> keyValueTransposes;
  llvm::SmallVector<CollapseShapeOp> keyValueCollapses;
  AttentionCoreMatch core;
  ProjectionMatch outputProjection;
  mlir::Operation *residualAddOp = nullptr;
  LayerNormMatch norm;
  mlir::Value decoderInput;
  mlir::Value memoryInput;
  mlir::Value output;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct FeedForwardMatch {
  ProjectionMatch upProjection;
  mlir::Operation *activationOp = nullptr;
  ProjectionMatch downProjection;
  mlir::Operation *residualAddOp = nullptr;
  LayerNormMatch norm;
  mlir::Value input;
  mlir::Value output;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct EncoderLayerMatch {
  int64_t layerIndex = 0;
  SelfAttentionMatch selfAttention;
  FeedForwardMatch feedForward;
  mlir::Value input;
  mlir::Value output;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct DecoderLayerMatch {
  int64_t layerIndex = 0;
  SelfAttentionMatch selfAttention;
  CrossAttentionMatch crossAttention;
  FeedForwardMatch feedForward;
  mlir::Value input;
  mlir::Value memoryInput;
  mlir::Value output;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct TransformerMatch {
  ReturnOp returnOp;
  llvm::SmallVector<EncoderLayerMatch, 2> encoderLayers;
  llvm::SmallVector<DecoderLayerMatch, 2> decoderLayers;
  LayerNormMatch encoderFinalNorm;
  LayerNormMatch decoderFinalNorm;
  TransformerTypes types;
  llvm::SmallVector<mlir::Operation *> ops;
};

struct TransformerEncoderMatch {
  ReturnOp returnOp;
  llvm::SmallVector<EncoderLayerMatch, 2> layers;
  LayerNormMatch finalNorm;
  bool hasFinalNorm = false;
  bool causal = false;
  mlir::StringRef normMode = "post";
  mlir::Value input;
  TransformerTypes types;
  llvm::SmallVector<mlir::Operation *> ops;
};

static mlir::FailureOr<TransformerTypes>
matchTransformerSignature(mlir::func::FuncOp func) {
  if (func.getNumArguments() != 2 || func.getNumResults() != 1 ||
      !func.getBody().hasOneBlock())
    return mlir::failure();

  auto srcTy =
      tensor_type::getStaticF32Tensor(func.getArgument(0).getType(), 3);
  auto tgtTy =
      tensor_type::getStaticF32Tensor(func.getArgument(1).getType(), 3);
  auto outputTy = tensor_type::getStaticF32Tensor(func.getResultTypes()[0], 3);
  if (mlir::failed(srcTy) || mlir::failed(tgtTy) || mlir::failed(outputTy))
    return mlir::failure();

  int64_t batchSize = srcTy->getDimSize(0);
  int64_t srcSequenceLength = srcTy->getDimSize(1);
  int64_t hiddenSize = srcTy->getDimSize(2);
  int64_t tgtBatchSize = tgtTy->getDimSize(0);
  int64_t tgtSequenceLength = tgtTy->getDimSize(1);
  int64_t tgtHiddenSize = tgtTy->getDimSize(2);
  if (batchSize <= 0 || srcSequenceLength <= 0 ||
      tgtSequenceLength <= 0 || hiddenSize <= 0 ||
      batchSize != tgtBatchSize || hiddenSize != tgtHiddenSize)
    return mlir::failure();

  if (outputTy->getShape() !=
      llvm::ArrayRef<int64_t>({batchSize, tgtSequenceLength, hiddenSize}))
    return mlir::failure();

  TransformerTypes types;
  types.srcTy = *srcTy;
  types.tgtTy = *tgtTy;
  types.outputTy = *outputTy;
  types.batchSize = batchSize;
  types.srcSequenceLength = srcSequenceLength;
  types.tgtSequenceLength = tgtSequenceLength;
  types.hiddenSize = hiddenSize;
  return types;
}

static mlir::FailureOr<TransformerTypes>
matchTransformerEncoderOutputSignature(mlir::func::FuncOp func) {
  if (func.getNumResults() != 1 || !func.getBody().hasOneBlock())
    return mlir::failure();

  auto outputTy = tensor_type::getStaticF32Tensor(func.getResultTypes()[0], 3);
  if (mlir::failed(outputTy))
    return mlir::failure();

  int64_t batchSize = outputTy->getDimSize(0);
  int64_t sequenceLength = outputTy->getDimSize(1);
  int64_t hiddenSize = outputTy->getDimSize(2);
  if (batchSize <= 0 || sequenceLength <= 0 || hiddenSize <= 0)
    return mlir::failure();

  TransformerTypes types;
  types.srcTy = *outputTy;
  types.tgtTy = *outputTy;
  types.outputTy = *outputTy;
  types.batchSize = batchSize;
  types.srcSequenceLength = sequenceLength;
  types.tgtSequenceLength = sequenceLength;
  types.hiddenSize = hiddenSize;
  return types;
}

static void appendIfPresent(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                            mlir::Operation *op) {
  if (op)
    ops.push_back(op);
}

static void appendProjectionOps(ProjectionMatch &projection) {
  appendIfPresent(projection.ops, projection.weightConstant.getOperation());
  appendIfPresent(projection.ops,
                  projection.weight ? projection.weight.getDefiningOp()
                                    : nullptr);
  appendIfPresent(projection.ops, projection.weightTranspose.getOperation());
  appendIfPresent(projection.ops, projection.matmulOp.getOperation());
  appendIfPresent(projection.ops, projection.biasConstant.getOperation());
  appendIfPresent(projection.ops,
                  projection.bias ? projection.bias.getDefiningOp() : nullptr);
  appendIfPresent(projection.ops, projection.biasAddOp.getOperation());
}

static ProjectionKind classifyProjectionKind(const TransformerTypes &types,
                                             int64_t inputWidth,
                                             int64_t outputWidth) {
  int64_t hiddenSize = types.hiddenSize;
  if (inputWidth == hiddenSize && outputWidth == 3 * hiddenSize)
    return ProjectionKind::HiddenToThreeHidden;
  if (inputWidth == hiddenSize && outputWidth == hiddenSize)
    return ProjectionKind::HiddenToHidden;
  if (inputWidth == hiddenSize && outputWidth == 2 * hiddenSize)
    return ProjectionKind::HiddenToTwoHidden;
  if (types.feedForwardSize > 0 && inputWidth == hiddenSize &&
      outputWidth == types.feedForwardSize)
    return ProjectionKind::HiddenToFeedForward;
  if (types.feedForwardSize > 0 && inputWidth == types.feedForwardSize &&
      outputWidth == hiddenSize)
    return ProjectionKind::FeedForwardToHidden;
  return ProjectionKind::Unknown;
}

static void
classifyProjectionCandidates(llvm::SmallVectorImpl<ProjectionMatch> &projections,
                             TransformerTypes &types) {
  int64_t feedForwardSize = 0;
  for (const ProjectionMatch &projection : projections) {
    if (projection.outputWidth == types.hiddenSize &&
        projection.inputWidth > types.hiddenSize)
      feedForwardSize = std::max(feedForwardSize, projection.inputWidth);
  }
  types.feedForwardSize = feedForwardSize;

  for (ProjectionMatch &projection : projections)
    projection.kind = classifyProjectionKind(types, projection.inputWidth,
                                             projection.outputWidth);
}

static mlir::FailureOr<ProjectionMatch>
matchProjectionCandidate(MatmulOp matmul, const TransformerTypes &types) {
  auto resultTy = tensor_type::getStaticF32Tensor(matmul.getResult(0).getType(),
                                                  /*expectedRank=*/2);
  if (mlir::failed(resultTy))
    return mlir::failure();

  int64_t rowCount = resultTy->getDimSize(0);
  int64_t outputWidth = resultTy->getDimSize(1);
  if (rowCount <= 0 || outputWidth <= 0)
    return mlir::failure();

  ProjectionMatch projection;
  for (mlir::Value maybeActivation : matmul.getInputs()) {
    auto activationTy = tensor_type::getStaticF32Tensor(
        maybeActivation.getType(), /*expectedRank=*/2);
    if (mlir::failed(activationTy))
      continue;

    int64_t inputWidth = activationTy->getDimSize(1);
    if (activationTy->getDimSize(0) != rowCount || inputWidth <= 0)
      continue;

    mlir::Value maybeTransposedWeight =
        maybeActivation == matmul.getInputs()[0] ? matmul.getInputs()[1]
                                                 : matmul.getInputs()[0];
    auto transpose =
        layer_utils::producerOfType<TransposeOp>(maybeTransposedWeight);
    if (!transpose ||
        !linalg_match::hasPermutation(transpose, {1, 0}) ||
        !tensor_type::hasStaticF32Shape(maybeTransposedWeight,
                                        {inputWidth, outputWidth}) ||
        !tensor_type::hasStaticF32Shape(transpose.getInput(),
                                        {outputWidth, inputWidth}))
      continue;

    auto matmulTy =
        llvm::cast<mlir::RankedTensorType>(matmul.getResult(0).getType());
    if (!layer_patterns::isZeroInitializedOutput(matmul.getOutputs()[0],
                                                 matmulTy))
      continue;

    projection.input = maybeActivation;
    projection.output = matmul.getResult(0);
    projection.weight = transpose.getInput();
    projection.weightConstant =
        layer_utils::producerOfType<ConstantOp>(projection.weight);
    projection.weightTranspose = transpose;
    projection.matmulOp = matmul;
    projection.root = matmul.getOperation();
    projection.inputWidth = inputWidth;
    projection.outputWidth = outputWidth;
    break;
  }

  if (!projection.matmulOp)
    return mlir::failure();

  llvm::SmallVector<int64_t, 2> outputShape(resultTy->getShape().begin(),
                                            resultTy->getShape().end());
  for (mlir::Operation *user : matmul.getResult(0).getUsers()) {
    auto biasAdd = llvm::dyn_cast<GenericOp>(user);
    if (!biasAdd || biasAdd.getNumResults() != 1)
      continue;

    if (!layer_patterns::isProjectionAddfGeneric(biasAdd) ||
        !tensor_type::hasStaticF32Shape(biasAdd.getResult(0), outputShape) ||
        !tensor_type::hasStaticF32Shape(biasAdd.getOutputs()[0],
                                        outputShape))
      continue;

    mlir::Value first = biasAdd.getInputs()[0];
    mlir::Value second = biasAdd.getInputs()[1];
    mlir::Value projectedValue;
    mlir::Value biasValue;
    unsigned biasInputIndex = 0;
    if (first == matmul.getResult(0) &&
        tensor_type::hasStaticF32Shape(second, {outputWidth})) {
      projectedValue = first;
      biasValue = second;
      biasInputIndex = 1;
    } else if (second == matmul.getResult(0) &&
               tensor_type::hasStaticF32Shape(first, {outputWidth})) {
      projectedValue = second;
      biasValue = first;
      biasInputIndex = 0;
    } else {
      continue;
    }

    if (!linalg_match::hasBiasAddIndexingMaps(
            biasAdd, static_cast<unsigned>(outputShape.size()), biasInputIndex))
      continue;

    projection.output = biasAdd.getResult(0);
    projection.bias = biasValue;
    projection.biasConstant = layer_utils::producerOfType<ConstantOp>(biasValue);
    projection.biasAddOp = biasAdd;
    projection.root = biasAdd.getOperation();
    break;
  }

  projection.kind =
      classifyProjectionKind(types, projection.inputWidth, projection.outputWidth);
  appendProjectionOps(projection);
  return projection;
}

static llvm::SmallVector<ProjectionMatch, 16>
collectProjectionCandidates(mlir::func::FuncOp func, TransformerTypes &types) {
  llvm::SmallVector<ProjectionMatch, 16> projections;
  func.walk([&](MatmulOp matmul) {
    auto projection = matchProjectionCandidate(matmul, types);
    if (mlir::succeeded(projection))
      projections.push_back(*projection);
  });

  classifyProjectionCandidates(projections, types);
  return projections;
}

static bool valueTransitivelyDependsOn(mlir::Value value, mlir::Value target,
                                       llvm::SmallPtrSetImpl<mlir::Operation *>
                                           &visitedOps) {
  if (value == target)
    return true;

  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || !visitedOps.insert(producer).second)
    return false;

  for (mlir::Value operand : producer->getOperands()) {
    if (valueTransitivelyDependsOn(operand, target, visitedOps))
      return true;
  }

  return false;
}

static bool valueTransitivelyDependsOn(mlir::Value value, mlir::Value target) {
  llvm::SmallPtrSet<mlir::Operation *, 32> visitedOps;
  return valueTransitivelyDependsOn(value, target, visitedOps);
}

static bool valueDependsOnThroughViewOps(
    mlir::Value value, mlir::Value target,
    llvm::SmallPtrSetImpl<mlir::Operation *> &visitedOps) {
  if (value == target)
    return true;

  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || !visitedOps.insert(producer).second)
    return false;

  if (!llvm::isa<CollapseShapeOp, ExpandShapeOp, TransposeOp, ExtractSliceOp>(
          producer))
    return false;

  for (mlir::Value operand : producer->getOperands()) {
    if (valueDependsOnThroughViewOps(operand, target, visitedOps))
      return true;
  }

  return false;
}

static bool valueDependsOnThroughViewOps(mlir::Value value,
                                         mlir::Value target) {
  llvm::SmallPtrSet<mlir::Operation *, 16> visitedOps;
  return valueDependsOnThroughViewOps(value, target, visitedOps);
}

static mlir::FailureOr<AttentionCoreMatch>
matchAttentionCore(BatchMatmulOp qkMatmul, BatchMatmulOp pvMatmul,
                   TransformerTypes &types) {
  auto qkQueryTy =
      tensor_type::getStaticF32Tensor(qkMatmul.getInputs()[0].getType(),
                                      /*expectedRank=*/3);
  auto qkKeyTy =
      tensor_type::getStaticF32Tensor(qkMatmul.getInputs()[1].getType(),
                                      /*expectedRank=*/3);
  auto qkOutputTy = tensor_type::getStaticF32Tensor(
      qkMatmul.getResult(0).getType(), /*expectedRank=*/3);
  auto pvProbabilityTy =
      tensor_type::getStaticF32Tensor(pvMatmul.getInputs()[0].getType(),
                                      /*expectedRank=*/3);
  auto pvValueTy =
      tensor_type::getStaticF32Tensor(pvMatmul.getInputs()[1].getType(),
                                      /*expectedRank=*/3);
  auto pvOutputTy = tensor_type::getStaticF32Tensor(
      pvMatmul.getResult(0).getType(), /*expectedRank=*/3);
  if (mlir::failed(qkQueryTy) || mlir::failed(qkKeyTy) ||
      mlir::failed(qkOutputTy) || mlir::failed(pvProbabilityTy) ||
      mlir::failed(pvValueTy) || mlir::failed(pvOutputTy))
    return mlir::failure();

  int64_t flattenedHeads = qkQueryTy->getDimSize(0);
  int64_t querySequenceLength = qkQueryTy->getDimSize(1);
  int64_t headDim = qkQueryTy->getDimSize(2);
  int64_t keySequenceLength = qkKeyTy->getDimSize(2);
  if (flattenedHeads <= 0 || querySequenceLength <= 0 ||
      keySequenceLength <= 0 || headDim <= 0)
    return mlir::failure();

  if (qkKeyTy->getShape() !=
          llvm::ArrayRef<int64_t>({flattenedHeads, headDim,
                                   keySequenceLength}) ||
      qkOutputTy->getShape() !=
          llvm::ArrayRef<int64_t>({flattenedHeads, querySequenceLength,
                                   keySequenceLength}) ||
      pvProbabilityTy->getShape() != qkOutputTy->getShape() ||
      pvValueTy->getShape() !=
          llvm::ArrayRef<int64_t>({flattenedHeads, keySequenceLength,
                                   headDim}) ||
      pvOutputTy->getShape() !=
          llvm::ArrayRef<int64_t>({flattenedHeads, querySequenceLength,
                                   headDim}))
    return mlir::failure();

  if (flattenedHeads % types.batchSize != 0)
    return mlir::failure();

  int64_t numHeads = flattenedHeads / types.batchSize;
  if (numHeads <= 0 || numHeads * headDim != types.hiddenSize)
    return mlir::failure();

  if (!valueTransitivelyDependsOn(pvMatmul.getInputs()[0],
                                  qkMatmul.getResult(0)))
    return mlir::failure();

  if (types.numHeads == 0)
    types.numHeads = numHeads;
  else if (types.numHeads != numHeads)
    return mlir::failure();

  if (types.headDim == 0)
    types.headDim = headDim;
  else if (types.headDim != headDim)
    return mlir::failure();

  AttentionCoreMatch core;
  core.query = qkMatmul.getInputs()[0];
  core.key = qkMatmul.getInputs()[1];
  core.value = pvMatmul.getInputs()[1];
  core.probabilities = pvMatmul.getInputs()[0];
  core.output = pvMatmul.getResult(0);
  core.qkMatmulOp = qkMatmul;
  core.pvMatmulOp = pvMatmul;
  core.qkMatmulRoot = qkMatmul.getOperation();
  core.pvMatmulRoot = pvMatmul.getOperation();
  core.softmaxRoot = pvMatmul.getInputs()[0].getDefiningOp();
  core.querySequenceLength = querySequenceLength;
  core.keySequenceLength = keySequenceLength;
  core.numHeads = numHeads;
  core.headDim = headDim;
  appendIfPresent(core.ops, core.qkMatmulRoot);
  appendIfPresent(core.ops, core.softmaxRoot);
  appendIfPresent(core.ops, core.pvMatmulRoot);
  return core;
}

static llvm::SmallVector<AttentionCoreMatch, 8>
collectAttentionCoreCandidates(mlir::func::FuncOp func,
                               TransformerTypes &types) {
  llvm::SmallVector<BatchMatmulOp, 16> batchMatmuls;
  func.walk([&](BatchMatmulOp batchMatmul) {
    batchMatmuls.push_back(batchMatmul);
  });

  llvm::SmallVector<AttentionCoreMatch, 8> cores;
  llvm::SmallPtrSet<mlir::Operation *, 16> pairedQKMatmuls;
  llvm::SmallPtrSet<mlir::Operation *, 16> pairedPVMatmuls;
  for (auto [qkIndex, qkMatmul] : llvm::enumerate(batchMatmuls)) {
    if (pairedQKMatmuls.contains(qkMatmul.getOperation()))
      continue;

    for (BatchMatmulOp pvMatmul :
         llvm::drop_begin(batchMatmuls, qkIndex + 1)) {
      if (pairedPVMatmuls.contains(pvMatmul.getOperation()))
        continue;

      auto core = matchAttentionCore(qkMatmul, pvMatmul, types);
      if (mlir::failed(core))
        continue;

      pairedQKMatmuls.insert(qkMatmul.getOperation());
      pairedPVMatmuls.insert(pvMatmul.getOperation());
      cores.push_back(*core);
      break;
    }
  }

  return cores;
}

static bool isNegativeInfinityFloat(const llvm::APFloat &value) {
  return value.isInfinity() && value.isNegative();
}

static bool isNegativeInfinityConstantValue(mlir::Attribute attr) {
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr))
    return isNegativeInfinityFloat(floatAttr.getValue());

  if (auto denseAttr = llvm::dyn_cast<mlir::DenseFPElementsAttr>(attr)) {
    for (const llvm::APFloat &value : denseAttr.getValues<llvm::APFloat>()) {
      if (isNegativeInfinityFloat(value))
        return true;
    }
  }

  return false;
}

static bool hasNegativeInfinityInput(GenericOp genericOp) {
  for (mlir::Value input : genericOp.getInputs()) {
    auto constant = layer_utils::producerOfType<ConstantOp>(input);
    if (constant && isNegativeInfinityConstantValue(constant.getValue()))
      return true;
  }
  return false;
}

static bool hasSelectInBody(GenericOp genericOp) {
  if (!genericOp || !genericOp.getRegion().hasOneBlock())
    return false;

  for (mlir::Operation &op : genericOp.getRegion().front().getOperations()) {
    if (llvm::isa<mlir::arith::SelectOp>(op))
      return true;
  }
  return false;
}

static bool hasNegativeInfinitySelect(GenericOp genericOp) {
  return hasSelectInBody(genericOp) && hasNegativeInfinityInput(genericOp);
}

static bool valueTransitivelyHasNegativeInfinitySelect(
    mlir::Value value, llvm::SmallPtrSetImpl<mlir::Operation *> &visitedOps) {
  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || !visitedOps.insert(producer).second)
    return false;

  if (auto generic = llvm::dyn_cast<GenericOp>(producer);
      generic && hasNegativeInfinitySelect(generic))
    return true;

  for (mlir::Value operand : producer->getOperands()) {
    if (valueTransitivelyHasNegativeInfinitySelect(operand, visitedOps))
      return true;
  }

  return false;
}

static bool attentionCoreUsesCausalMask(const AttentionCoreMatch &core) {
  llvm::SmallPtrSet<mlir::Operation *, 32> visitedOps;
  return valueTransitivelyHasNegativeInfinitySelect(core.probabilities,
                                                   visitedOps);
}

static bool isRsqrtGeneric(GenericOp genericOp) {
  if (!genericOp || genericOp.getInputs().size() != 1 ||
      genericOp.getOutputs().size() != 1 || genericOp.getNumResults() != 1 ||
      !genericOp.getRegion().hasOneBlock())
    return false;

  mlir::Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2)
    return false;

  auto rsqrtOp = llvm::dyn_cast<mlir::math::RsqrtOp>(body.front());
  auto yieldOp = llvm::dyn_cast<mlir::linalg::YieldOp>(body.back());
  return rsqrtOp && yieldOp && rsqrtOp.getOperand() == body.getArgument(0) &&
         yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == rsqrtOp.getResult();
}

static bool hasReductionIterator(GenericOp genericOp) {
  if (!genericOp)
    return false;

  return llvm::any_of(genericOp.getIteratorTypesArray(),
                      [](mlir::utils::IteratorType iteratorType) {
                        return iteratorType ==
                               mlir::utils::IteratorType::reduction;
                      });
}

static GenericOp findFirstReductionGeneric(
    mlir::Value value,
    llvm::SmallPtrSetImpl<mlir::Operation *> &visitedOps) {
  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || !visitedOps.insert(producer).second)
    return {};

  if (auto generic = llvm::dyn_cast<GenericOp>(producer);
      generic && hasReductionIterator(generic))
    return generic;

  for (mlir::Value operand : producer->getOperands()) {
    if (auto reduction = findFirstReductionGeneric(operand, visitedOps))
      return reduction;
  }

  return {};
}

static GenericOp findFirstReductionGeneric(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 32> visitedOps;
  return findFirstReductionGeneric(value, visitedOps);
}

static mlir::FailureOr<std::pair<mlir::Value, ConstantOp>>
matchTailAffineMul(GenericOp genericOp, llvm::ArrayRef<int64_t> outputShape,
                   int64_t hiddenSize) {
  if (!layer_utils::isMulfGeneric(genericOp.getOperation()) ||
      !tensor_type::hasStaticF32Shape(genericOp.getResult(0), outputShape) ||
      !tensor_type::hasStaticF32Shape(genericOp.getOutputs()[0], outputShape))
    return mlir::failure();

  auto firstWeight = layer_utils::producerOfType<ConstantOp>(
      genericOp.getInputs()[0]);
  auto secondWeight = layer_utils::producerOfType<ConstantOp>(
      genericOp.getInputs()[1]);
  if (static_cast<bool>(firstWeight) == static_cast<bool>(secondWeight))
    return mlir::failure();

  unsigned weightInputIndex = firstWeight ? 0 : 1;
  if (!linalg_match::hasBiasAddIndexingMaps(
          genericOp, static_cast<unsigned>(outputShape.size()),
          weightInputIndex))
    return mlir::failure();

  ConstantOp weightConstant = firstWeight ? firstWeight : secondWeight;
  mlir::Value normalizedValue =
      firstWeight ? genericOp.getInputs()[1] : genericOp.getInputs()[0];
  if (!tensor_type::hasStaticF32Shape(weightConstant.getResult(),
                                      {hiddenSize}) ||
      !tensor_type::hasStaticF32Shape(normalizedValue, outputShape))
    return mlir::failure();

  return std::make_pair(normalizedValue, weightConstant);
}

static mlir::Value findInputWithShape(GenericOp genericOp,
                                      llvm::ArrayRef<int64_t> shape) {
  for (mlir::Value input : genericOp.getInputs()) {
    if (tensor_type::hasStaticF32Shape(input, shape))
      return input;
  }
  return {};
}

static mlir::FailureOr<LayerNormMatch>
matchLayerNormCandidate(GenericOp affineAddOp, const TransformerTypes &types) {
  auto outputTy = tensor_type::getStaticF32Tensor(
      affineAddOp.getResult(0).getType(), /*expectedRank=*/3);
  if (mlir::failed(outputTy) ||
      outputTy->getDimSize(0) != types.batchSize ||
      outputTy->getDimSize(2) != types.hiddenSize)
    return mlir::failure();

  llvm::SmallVector<int64_t, 3> outputShape(outputTy->getShape().begin(),
                                            outputTy->getShape().end());
  auto biasMatch = layer_patterns::matchProjectionBiasAdd(
      affineAddOp.getResult(0), outputShape, types.hiddenSize);
  if (mlir::failed(biasMatch))
    return mlir::failure();

  auto affineMulOp = layer_utils::producerOfType<GenericOp>(biasMatch->first);
  auto weightMatch =
      matchTailAffineMul(affineMulOp, outputShape, types.hiddenSize);
  if (mlir::failed(weightMatch))
    return mlir::failure();

  auto normalizationMulOp =
      layer_utils::producerOfType<GenericOp>(weightMatch->first);
  if (!layer_utils::isMulfGeneric(normalizationMulOp.getOperation()))
    return mlir::failure();

  GenericOp centerOp;
  GenericOp rsqrtOp;
  for (mlir::Value input : normalizationMulOp.getInputs()) {
    auto producer = layer_utils::producerOfType<GenericOp>(input);
    if (!producer)
      continue;
    if (layer_utils::isSubfGeneric(producer.getOperation()))
      centerOp = producer;
    if (isRsqrtGeneric(producer))
      rsqrtOp = producer;
  }
  if (!centerOp || !rsqrtOp)
    return mlir::failure();

  LayerNormMatch match;
  match.input = findInputWithShape(centerOp, outputShape);
  match.output = affineAddOp.getResult(0);
  match.weightConstant = weightMatch->second;
  match.biasConstant = biasMatch->second;
  match.centerOp = centerOp.getOperation();
  match.meanReduction = findFirstReductionGeneric(centerOp.getInputs()[1]);
  match.rsqrtOp = rsqrtOp.getOperation();
  match.epsilonAddOp = rsqrtOp.getInputs()[0].getDefiningOp();
  if (match.epsilonAddOp)
    match.varianceReduction =
        findFirstReductionGeneric(match.epsilonAddOp->getOperand(0))
            .getOperation();
  match.affineMulOp = affineMulOp.getOperation();
  match.affineAddOp = affineAddOp.getOperation();
  match.root = affineAddOp.getOperation();
  appendIfPresent(match.ops, match.meanReduction);
  appendIfPresent(match.ops, match.centerOp);
  appendIfPresent(match.ops, match.varianceReduction);
  appendIfPresent(match.ops, match.epsilonAddOp);
  appendIfPresent(match.ops, match.rsqrtOp);
  appendIfPresent(match.ops, normalizationMulOp.getOperation());
  appendIfPresent(match.ops, match.weightConstant.getOperation());
  appendIfPresent(match.ops, match.affineMulOp);
  appendIfPresent(match.ops, match.biasConstant.getOperation());
  appendIfPresent(match.ops, match.affineAddOp);
  return match;
}

static llvm::SmallVector<LayerNormMatch, 16>
collectLayerNormCandidates(mlir::func::FuncOp func,
                           const TransformerTypes &types) {
  llvm::SmallVector<LayerNormMatch, 16> layerNorms;
  func.walk([&](GenericOp genericOp) {
    auto layerNorm = matchLayerNormCandidate(genericOp, types);
    if (mlir::succeeded(layerNorm))
      layerNorms.push_back(*layerNorm);
  });
  return layerNorms;
}

static bool isBeforeInBlock(mlir::Operation *before, mlir::Operation *after) {
  return before && after && before->getBlock() == after->getBlock() &&
         before->isBeforeInBlock(after);
}

static bool isGeluGeneric(GenericOp genericOp) {
  if (!genericOp || !genericOp.getRegion().hasOneBlock())
    return false;

  for (mlir::Operation &op : genericOp.getRegion().front().getOperations()) {
    if (llvm::isa<mlir::math::ErfOp>(op))
      return true;
  }
  return false;
}

static GenericOp
findFirstGeluGeneric(mlir::Value value, mlir::Value stop,
                     llvm::SmallPtrSetImpl<mlir::Operation *> &visitedOps) {
  if (value == stop)
    return {};

  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || !visitedOps.insert(producer).second)
    return {};

  if (auto generic = llvm::dyn_cast<GenericOp>(producer);
      generic && isGeluGeneric(generic))
    return generic;

  for (mlir::Value operand : producer->getOperands()) {
    if (auto gelu = findFirstGeluGeneric(operand, stop, visitedOps))
      return gelu;
  }

  return {};
}

static GenericOp findFirstGeluGeneric(mlir::Value value, mlir::Value stop) {
  llvm::SmallPtrSet<mlir::Operation *, 32> visitedOps;
  return findFirstGeluGeneric(value, stop, visitedOps);
}

static mlir::Value getResidualInput(GenericOp residualAddOp,
                                    mlir::Value projectedValue,
                                    llvm::ArrayRef<int64_t> shape) {
  if (!layer_utils::isAddfGeneric(residualAddOp.getOperation()))
    return {};

  for (mlir::Value input : residualAddOp.getInputs()) {
    if (!tensor_type::hasStaticF32Shape(input, shape))
      continue;
    if (!valueTransitivelyDependsOn(input, projectedValue))
      return input;
  }

  return {};
}

static bool genericResultHasShape(GenericOp genericOp,
                                  llvm::ArrayRef<int64_t> shape) {
  return genericOp && genericOp.getNumResults() == 1 &&
         tensor_type::hasStaticF32Shape(genericOp.getResult(0), shape) &&
         tensor_type::hasStaticF32Shape(genericOp.getOutputs()[0], shape);
}

static mlir::FailureOr<std::pair<GenericOp, mlir::Value>>
findResidualAddFromProjection(mlir::Value value, mlir::Value projectedValue,
                              mlir::Value expectedResidual,
                              llvm::ArrayRef<int64_t> shape,
                              llvm::SmallPtrSetImpl<mlir::Operation *> &visited) {
  for (mlir::Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;

    if (auto generic = llvm::dyn_cast<GenericOp>(user);
        layer_utils::isAddfGeneric(user) && genericResultHasShape(generic, shape)) {
      mlir::Value residual = getResidualInput(generic, projectedValue, shape);
      if (residual &&
          (residual == expectedResidual ||
           valueTransitivelyDependsOn(residual, expectedResidual)))
        return std::make_pair(generic, residual);
    }

    for (mlir::Value result : user->getResults()) {
      if (!tensor_type::hasStaticF32Shape(result, shape) &&
          !valueTransitivelyDependsOn(result, projectedValue))
        continue;

      auto nested = findResidualAddFromProjection(
          result, projectedValue, expectedResidual, shape, visited);
      if (mlir::succeeded(nested))
        return nested;
    }
  }

  return mlir::failure();
}

static mlir::FailureOr<std::pair<GenericOp, mlir::Value>>
findResidualAddFromProjection(mlir::Value projectedValue,
                              mlir::Value expectedResidual,
                              llvm::ArrayRef<int64_t> shape) {
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  return findResidualAddFromProjection(projectedValue, projectedValue,
                                       expectedResidual, shape, visited);
}

static mlir::FailureOr<FeedForwardMatch>
matchFeedForwardCandidate(const ProjectionMatch &upProjection,
                          const ProjectionMatch &downProjection,
                          const LayerNormMatch &norm,
                          const TransformerTypes &types) {
  if (types.feedForwardSize <= 0 ||
      upProjection.inputWidth != types.hiddenSize ||
      upProjection.outputWidth != types.feedForwardSize ||
      downProjection.inputWidth != types.feedForwardSize ||
      downProjection.outputWidth != types.hiddenSize ||
      !isBeforeInBlock(upProjection.root, downProjection.root) ||
      !isBeforeInBlock(downProjection.root, norm.root))
    return mlir::failure();

  if (!valueTransitivelyDependsOn(downProjection.input, upProjection.output) ||
      !valueTransitivelyDependsOn(norm.input, downProjection.output))
    return mlir::failure();

  auto activationOp =
      findFirstGeluGeneric(downProjection.input, upProjection.output);
  if (!activationOp)
    return mlir::failure();

  auto residualAddOp = llvm::dyn_cast_or_null<GenericOp>(
      norm.input.getDefiningOp());
  auto normInputTy = tensor_type::getStaticF32Tensor(norm.input.getType(),
                                                     /*expectedRank=*/3);
  if (!residualAddOp || mlir::failed(normInputTy))
    return mlir::failure();

  llvm::SmallVector<int64_t, 3> residualShape(normInputTy->getShape().begin(),
                                              normInputTy->getShape().end());
  mlir::Value residualInput =
      getResidualInput(residualAddOp, downProjection.output, residualShape);
  if (!residualInput)
    return mlir::failure();

  FeedForwardMatch match;
  match.upProjection = upProjection;
  match.activationOp = activationOp.getOperation();
  match.downProjection = downProjection;
  match.residualAddOp = residualAddOp.getOperation();
  match.norm = norm;
  match.input = residualInput;
  match.output = norm.output;
  match.ops.append(upProjection.ops.begin(), upProjection.ops.end());
  appendIfPresent(match.ops, match.activationOp);
  match.ops.append(downProjection.ops.begin(), downProjection.ops.end());
  appendIfPresent(match.ops, match.residualAddOp);
  match.ops.append(norm.ops.begin(), norm.ops.end());
  return match;
}

static llvm::SmallVector<FeedForwardMatch, 8>
collectFeedForwardCandidates(llvm::ArrayRef<ProjectionMatch> projections,
                             llvm::ArrayRef<LayerNormMatch> layerNorms,
                             const TransformerTypes &types) {
  llvm::SmallVector<FeedForwardMatch, 8> feedForwards;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedDownProjections;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedNorms;

  for (const ProjectionMatch &upProjection : projections) {
    if (upProjection.inputWidth != types.hiddenSize ||
        upProjection.outputWidth != types.feedForwardSize)
      continue;

    for (const ProjectionMatch &downProjection : projections) {
      if (usedDownProjections.contains(downProjection.root))
        continue;

      for (const LayerNormMatch &norm : layerNorms) {
        if (usedNorms.contains(norm.root))
          continue;

        auto feedForward =
            matchFeedForwardCandidate(upProjection, downProjection, norm, types);
        if (mlir::failed(feedForward))
          continue;

        usedDownProjections.insert(downProjection.root);
        usedNorms.insert(norm.root);
        feedForwards.push_back(*feedForward);
        break;
      }

      if (usedDownProjections.contains(downProjection.root))
        break;
    }
  }

  return feedForwards;
}

static mlir::FailureOr<FeedForwardMatch>
matchPreNormFeedForwardCandidate(const ProjectionMatch &upProjection,
                                 const ProjectionMatch &downProjection,
                                 const LayerNormMatch &norm,
                                 const TransformerTypes &types) {
  if (types.feedForwardSize <= 0 ||
      upProjection.inputWidth != types.hiddenSize ||
      upProjection.outputWidth != types.feedForwardSize ||
      downProjection.inputWidth != types.feedForwardSize ||
      downProjection.outputWidth != types.hiddenSize ||
      !isBeforeInBlock(norm.root, upProjection.root) ||
      !isBeforeInBlock(upProjection.root, downProjection.root))
    return mlir::failure();

  if (!valueDependsOnThroughViewOps(upProjection.input, norm.output) ||
      !valueTransitivelyDependsOn(downProjection.input, upProjection.output))
    return mlir::failure();

  auto activationOp =
      findFirstGeluGeneric(downProjection.input, upProjection.output);
  if (!activationOp)
    return mlir::failure();

  auto normInputTy = tensor_type::getStaticF32Tensor(norm.input.getType(),
                                                     /*expectedRank=*/3);
  if (mlir::failed(normInputTy))
    return mlir::failure();

  llvm::SmallVector<int64_t, 3> residualShape(normInputTy->getShape().begin(),
                                              normInputTy->getShape().end());
  auto residualAdd =
      findResidualAddFromProjection(downProjection.output, norm.input,
                                    residualShape);
  if (mlir::failed(residualAdd))
    return mlir::failure();

  FeedForwardMatch match;
  match.upProjection = upProjection;
  match.activationOp = activationOp.getOperation();
  match.downProjection = downProjection;
  match.residualAddOp = residualAdd->first.getOperation();
  match.norm = norm;
  match.input = norm.input;
  match.output = residualAdd->first.getResult(0);
  match.ops.append(norm.ops.begin(), norm.ops.end());
  match.ops.append(upProjection.ops.begin(), upProjection.ops.end());
  appendIfPresent(match.ops, match.activationOp);
  match.ops.append(downProjection.ops.begin(), downProjection.ops.end());
  appendIfPresent(match.ops, match.residualAddOp);
  return match;
}

static llvm::SmallVector<FeedForwardMatch, 8>
collectPreNormFeedForwardCandidates(
    llvm::ArrayRef<ProjectionMatch> projections,
    llvm::ArrayRef<LayerNormMatch> layerNorms,
    const TransformerTypes &types) {
  llvm::SmallVector<FeedForwardMatch, 8> feedForwards;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedDownProjections;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedNorms;

  for (const ProjectionMatch &upProjection : projections) {
    if (upProjection.inputWidth != types.hiddenSize ||
        upProjection.outputWidth != types.feedForwardSize)
      continue;

    for (const ProjectionMatch &downProjection : projections) {
      if (usedDownProjections.contains(downProjection.root))
        continue;

      for (const LayerNormMatch &norm : layerNorms) {
        if (usedNorms.contains(norm.root))
          continue;

        auto feedForward = matchPreNormFeedForwardCandidate(
            upProjection, downProjection, norm, types);
        if (mlir::failed(feedForward))
          continue;

        usedDownProjections.insert(downProjection.root);
        usedNorms.insert(norm.root);
        feedForwards.push_back(*feedForward);
        break;
      }

      if (usedDownProjections.contains(downProjection.root))
        break;
    }
  }

  return feedForwards;
}

static mlir::FailureOr<SelfAttentionMatch>
matchSelfAttentionCandidate(const ProjectionMatch &qkvProjection,
                            const AttentionCoreMatch &core,
                            const ProjectionMatch &outputProjection,
                            const LayerNormMatch &norm,
                            const TransformerTypes &types) {
  if (qkvProjection.inputWidth != types.hiddenSize ||
      qkvProjection.outputWidth != 3 * types.hiddenSize ||
      outputProjection.inputWidth != types.hiddenSize ||
      outputProjection.outputWidth != types.hiddenSize ||
      !isBeforeInBlock(qkvProjection.root, core.qkMatmulRoot) ||
      !isBeforeInBlock(core.pvMatmulRoot, outputProjection.root) ||
      !isBeforeInBlock(outputProjection.root, norm.root))
    return mlir::failure();

  if (!valueTransitivelyDependsOn(core.query, qkvProjection.output) ||
      !valueTransitivelyDependsOn(core.key, qkvProjection.output) ||
      !valueTransitivelyDependsOn(core.value, qkvProjection.output) ||
      !valueTransitivelyDependsOn(outputProjection.input, core.output) ||
      !valueTransitivelyDependsOn(norm.input, outputProjection.output))
    return mlir::failure();

  auto residualAddOp =
      llvm::dyn_cast_or_null<GenericOp>(norm.input.getDefiningOp());
  auto normInputTy = tensor_type::getStaticF32Tensor(norm.input.getType(),
                                                     /*expectedRank=*/3);
  if (!residualAddOp || mlir::failed(normInputTy))
    return mlir::failure();

  llvm::SmallVector<int64_t, 3> residualShape(normInputTy->getShape().begin(),
                                              normInputTy->getShape().end());
  mlir::Value residualInput =
      getResidualInput(residualAddOp, outputProjection.output, residualShape);
  if (!residualInput ||
      !valueTransitivelyDependsOn(qkvProjection.input, residualInput))
    return mlir::failure();

  SelfAttentionMatch match;
  match.qkvProjection = qkvProjection;
  match.core = core;
  match.outputProjection = outputProjection;
  match.residualAddOp = residualAddOp.getOperation();
  match.norm = norm;
  match.input = residualInput;
  match.output = norm.output;
  match.ops.append(qkvProjection.ops.begin(), qkvProjection.ops.end());
  match.ops.append(core.ops.begin(), core.ops.end());
  match.ops.append(outputProjection.ops.begin(), outputProjection.ops.end());
  appendIfPresent(match.ops, match.residualAddOp);
  match.ops.append(norm.ops.begin(), norm.ops.end());
  return match;
}

static llvm::SmallVector<SelfAttentionMatch, 8>
collectSelfAttentionCandidates(llvm::ArrayRef<ProjectionMatch> projections,
                               llvm::ArrayRef<AttentionCoreMatch> attentionCores,
                               llvm::ArrayRef<LayerNormMatch> layerNorms,
                               const TransformerTypes &types) {
  llvm::SmallVector<SelfAttentionMatch, 8> selfAttentions;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedCores;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedOutputProjections;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedNorms;

  for (const ProjectionMatch &qkvProjection : projections) {
    if (qkvProjection.inputWidth != types.hiddenSize ||
        qkvProjection.outputWidth != 3 * types.hiddenSize)
      continue;

    for (const AttentionCoreMatch &core : attentionCores) {
      if (usedCores.contains(core.qkMatmulRoot))
        continue;

      for (const ProjectionMatch &outputProjection : projections) {
        if (usedOutputProjections.contains(outputProjection.root))
          continue;

        for (const LayerNormMatch &norm : layerNorms) {
          if (usedNorms.contains(norm.root))
            continue;

          auto selfAttention = matchSelfAttentionCandidate(
              qkvProjection, core, outputProjection, norm, types);
          if (mlir::failed(selfAttention))
            continue;

          usedCores.insert(core.qkMatmulRoot);
          usedOutputProjections.insert(outputProjection.root);
          usedNorms.insert(norm.root);
          selfAttentions.push_back(*selfAttention);
          break;
        }

        if (usedOutputProjections.contains(outputProjection.root))
          break;
      }

      if (usedCores.contains(core.qkMatmulRoot))
        break;
    }
  }

  return selfAttentions;
}

static mlir::FailureOr<SelfAttentionMatch>
matchPreNormSelfAttentionCandidate(const ProjectionMatch &qkvProjection,
                                   const AttentionCoreMatch &core,
                                   const ProjectionMatch &outputProjection,
                                   const LayerNormMatch &norm,
                                   const TransformerTypes &types) {
  if (qkvProjection.inputWidth != types.hiddenSize ||
      qkvProjection.outputWidth != 3 * types.hiddenSize ||
      outputProjection.inputWidth != types.hiddenSize ||
      outputProjection.outputWidth != types.hiddenSize ||
      !isBeforeInBlock(norm.root, qkvProjection.root) ||
      !isBeforeInBlock(qkvProjection.root, core.qkMatmulRoot) ||
      !isBeforeInBlock(core.pvMatmulRoot, outputProjection.root))
    return mlir::failure();

  if (!valueDependsOnThroughViewOps(qkvProjection.input, norm.output) ||
      !valueTransitivelyDependsOn(core.query, qkvProjection.output) ||
      !valueTransitivelyDependsOn(core.key, qkvProjection.output) ||
      !valueTransitivelyDependsOn(core.value, qkvProjection.output) ||
      !valueTransitivelyDependsOn(outputProjection.input, core.output))
    return mlir::failure();

  auto normInputTy = tensor_type::getStaticF32Tensor(norm.input.getType(),
                                                     /*expectedRank=*/3);
  if (mlir::failed(normInputTy))
    return mlir::failure();

  llvm::SmallVector<int64_t, 3> residualShape(normInputTy->getShape().begin(),
                                              normInputTy->getShape().end());
  auto residualAdd =
      findResidualAddFromProjection(outputProjection.output, norm.input,
                                    residualShape);
  if (mlir::failed(residualAdd))
    return mlir::failure();

  SelfAttentionMatch match;
  match.qkvProjection = qkvProjection;
  match.core = core;
  match.outputProjection = outputProjection;
  match.residualAddOp = residualAdd->first.getOperation();
  match.norm = norm;
  match.input = norm.input;
  match.output = residualAdd->first.getResult(0);
  match.ops.append(norm.ops.begin(), norm.ops.end());
  match.ops.append(qkvProjection.ops.begin(), qkvProjection.ops.end());
  match.ops.append(core.ops.begin(), core.ops.end());
  match.ops.append(outputProjection.ops.begin(), outputProjection.ops.end());
  appendIfPresent(match.ops, match.residualAddOp);
  return match;
}

static llvm::SmallVector<SelfAttentionMatch, 8>
collectPreNormSelfAttentionCandidates(
    llvm::ArrayRef<ProjectionMatch> projections,
    llvm::ArrayRef<AttentionCoreMatch> attentionCores,
    llvm::ArrayRef<LayerNormMatch> layerNorms,
    const TransformerTypes &types) {
  llvm::SmallVector<SelfAttentionMatch, 8> selfAttentions;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedCores;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedOutputProjections;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedNorms;

  for (const ProjectionMatch &qkvProjection : projections) {
    if (qkvProjection.inputWidth != types.hiddenSize ||
        qkvProjection.outputWidth != 3 * types.hiddenSize)
      continue;

    for (const AttentionCoreMatch &core : attentionCores) {
      if (usedCores.contains(core.qkMatmulRoot))
        continue;

      for (const ProjectionMatch &outputProjection : projections) {
        if (usedOutputProjections.contains(outputProjection.root))
          continue;

        for (const LayerNormMatch &norm : layerNorms) {
          if (usedNorms.contains(norm.root))
            continue;

          auto selfAttention = matchPreNormSelfAttentionCandidate(
              qkvProjection, core, outputProjection, norm, types);
          if (mlir::failed(selfAttention))
            continue;

          usedCores.insert(core.qkMatmulRoot);
          usedOutputProjections.insert(outputProjection.root);
          usedNorms.insert(norm.root);
          selfAttentions.push_back(*selfAttention);
          break;
        }

        if (usedOutputProjections.contains(outputProjection.root))
          break;
      }

      if (usedCores.contains(core.qkMatmulRoot))
        break;
    }
  }

  return selfAttentions;
}

static mlir::FailureOr<CrossAttentionMatch>
matchCrossAttentionCandidate(const ProjectionMatch &queryProjection,
                             const ProjectionMatch &keyValueProjection,
                             const AttentionCoreMatch &core,
                             const ProjectionMatch &outputProjection,
                             const LayerNormMatch &norm,
                             const TransformerTypes &types) {
  if (queryProjection.inputWidth != types.hiddenSize ||
      queryProjection.outputWidth != types.hiddenSize ||
      keyValueProjection.inputWidth != types.hiddenSize ||
      keyValueProjection.outputWidth != 2 * types.hiddenSize ||
      outputProjection.inputWidth != types.hiddenSize ||
      outputProjection.outputWidth != types.hiddenSize ||
      !isBeforeInBlock(queryProjection.root, core.qkMatmulRoot) ||
      !isBeforeInBlock(keyValueProjection.root, core.qkMatmulRoot) ||
      !isBeforeInBlock(core.pvMatmulRoot, outputProjection.root) ||
      !isBeforeInBlock(outputProjection.root, norm.root))
    return mlir::failure();

  if (!valueTransitivelyDependsOn(core.query, queryProjection.output) ||
      !valueTransitivelyDependsOn(core.key, keyValueProjection.output) ||
      !valueTransitivelyDependsOn(core.value, keyValueProjection.output) ||
      valueTransitivelyDependsOn(core.key, queryProjection.output) ||
      valueTransitivelyDependsOn(core.value, queryProjection.output) ||
      !valueTransitivelyDependsOn(outputProjection.input, core.output) ||
      !valueTransitivelyDependsOn(norm.input, outputProjection.output))
    return mlir::failure();

  auto residualAddOp =
      llvm::dyn_cast_or_null<GenericOp>(norm.input.getDefiningOp());
  auto normInputTy = tensor_type::getStaticF32Tensor(norm.input.getType(),
                                                     /*expectedRank=*/3);
  if (!residualAddOp || mlir::failed(normInputTy))
    return mlir::failure();

  llvm::SmallVector<int64_t, 3> residualShape(normInputTy->getShape().begin(),
                                              normInputTy->getShape().end());
  mlir::Value decoderInput =
      getResidualInput(residualAddOp, outputProjection.output, residualShape);
  if (!decoderInput ||
      !valueTransitivelyDependsOn(queryProjection.input, decoderInput))
    return mlir::failure();

  CrossAttentionMatch match;
  match.queryProjection = queryProjection;
  match.keyValueProjection = keyValueProjection;
  match.core = core;
  match.outputProjection = outputProjection;
  match.residualAddOp = residualAddOp.getOperation();
  match.norm = norm;
  match.decoderInput = decoderInput;
  match.memoryInput = keyValueProjection.input;
  match.output = norm.output;
  match.ops.append(queryProjection.ops.begin(), queryProjection.ops.end());
  match.ops.append(keyValueProjection.ops.begin(), keyValueProjection.ops.end());
  match.ops.append(core.ops.begin(), core.ops.end());
  match.ops.append(outputProjection.ops.begin(), outputProjection.ops.end());
  appendIfPresent(match.ops, match.residualAddOp);
  match.ops.append(norm.ops.begin(), norm.ops.end());
  return match;
}

static llvm::SmallVector<CrossAttentionMatch, 8>
collectCrossAttentionCandidates(
    llvm::ArrayRef<ProjectionMatch> projections,
    llvm::ArrayRef<AttentionCoreMatch> attentionCores,
    llvm::ArrayRef<LayerNormMatch> layerNorms, const TransformerTypes &types) {
  llvm::SmallVector<CrossAttentionMatch, 8> crossAttentions;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedCores;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedOutputProjections;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedNorms;

  for (const ProjectionMatch &queryProjection : projections) {
    if (queryProjection.inputWidth != types.hiddenSize ||
        queryProjection.outputWidth != types.hiddenSize)
      continue;

    for (const ProjectionMatch &keyValueProjection : projections) {
      if (keyValueProjection.inputWidth != types.hiddenSize ||
          keyValueProjection.outputWidth != 2 * types.hiddenSize ||
          queryProjection.root == keyValueProjection.root)
        continue;

      for (const AttentionCoreMatch &core : attentionCores) {
        if (usedCores.contains(core.qkMatmulRoot))
          continue;

        for (const ProjectionMatch &outputProjection : projections) {
          if (usedOutputProjections.contains(outputProjection.root))
            continue;

          for (const LayerNormMatch &norm : layerNorms) {
            if (usedNorms.contains(norm.root))
              continue;

            auto crossAttention = matchCrossAttentionCandidate(
                queryProjection, keyValueProjection, core, outputProjection,
                norm, types);
            if (mlir::failed(crossAttention))
              continue;

            usedCores.insert(core.qkMatmulRoot);
            usedOutputProjections.insert(outputProjection.root);
            usedNorms.insert(norm.root);
            crossAttentions.push_back(*crossAttention);
            break;
          }

          if (usedOutputProjections.contains(outputProjection.root))
            break;
        }

        if (usedCores.contains(core.qkMatmulRoot))
          break;
      }
    }
  }

  return crossAttentions;
}

static llvm::SmallVector<EncoderLayerMatch, 2>
collectEncoderLayerCandidates(llvm::ArrayRef<SelfAttentionMatch> selfAttentions,
                              llvm::ArrayRef<FeedForwardMatch> feedForwards,
                              TransformerTypes &types) {
  llvm::SmallVector<EncoderLayerMatch, 2> layers;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedFeedForwards;

  for (const SelfAttentionMatch &selfAttention : selfAttentions) {
    for (const FeedForwardMatch &feedForward : feedForwards) {
      if (usedFeedForwards.contains(feedForward.residualAddOp) ||
          feedForward.input != selfAttention.output ||
          !isBeforeInBlock(selfAttention.residualAddOp, feedForward.residualAddOp))
        continue;

      EncoderLayerMatch layer;
      layer.layerIndex = static_cast<int64_t>(layers.size());
      layer.selfAttention = selfAttention;
      layer.feedForward = feedForward;
      layer.input = selfAttention.input;
      layer.output = feedForward.output;
      layer.ops.append(selfAttention.ops.begin(), selfAttention.ops.end());
      layer.ops.append(feedForward.ops.begin(), feedForward.ops.end());
      usedFeedForwards.insert(feedForward.residualAddOp);
      layers.push_back(layer);
      break;
    }
  }

  llvm::sort(layers, [](const EncoderLayerMatch &lhs,
                        const EncoderLayerMatch &rhs) {
    return isBeforeInBlock(lhs.selfAttention.residualAddOp,
                           rhs.selfAttention.residualAddOp);
  });

  for (auto [index, layer] : llvm::enumerate(layers))
    layer.layerIndex = static_cast<int64_t>(index);

  types.encoderLayerCount = static_cast<int64_t>(layers.size());
  return layers;
}

static llvm::SmallVector<DecoderLayerMatch, 2> collectDecoderLayerCandidates(
    llvm::ArrayRef<SelfAttentionMatch> selfAttentions,
    llvm::ArrayRef<CrossAttentionMatch> crossAttentions,
    llvm::ArrayRef<FeedForwardMatch> feedForwards, TransformerTypes &types) {
  llvm::SmallVector<DecoderLayerMatch, 2> layers;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedCrossAttentions;
  llvm::SmallPtrSet<mlir::Operation *, 8> usedFeedForwards;

  for (const SelfAttentionMatch &selfAttention : selfAttentions) {
    for (const CrossAttentionMatch &crossAttention : crossAttentions) {
      if (usedCrossAttentions.contains(crossAttention.residualAddOp) ||
          crossAttention.decoderInput != selfAttention.output ||
          !isBeforeInBlock(selfAttention.residualAddOp,
                           crossAttention.residualAddOp))
        continue;

      for (const FeedForwardMatch &feedForward : feedForwards) {
        if (usedFeedForwards.contains(feedForward.residualAddOp) ||
            feedForward.input != crossAttention.output ||
            !isBeforeInBlock(crossAttention.residualAddOp,
                             feedForward.residualAddOp))
          continue;

        DecoderLayerMatch layer;
        layer.layerIndex = static_cast<int64_t>(layers.size());
        layer.selfAttention = selfAttention;
        layer.crossAttention = crossAttention;
        layer.feedForward = feedForward;
        layer.input = selfAttention.input;
        layer.memoryInput = crossAttention.memoryInput;
        layer.output = feedForward.output;
        layer.ops.append(selfAttention.ops.begin(), selfAttention.ops.end());
        layer.ops.append(crossAttention.ops.begin(), crossAttention.ops.end());
        layer.ops.append(feedForward.ops.begin(), feedForward.ops.end());
        usedCrossAttentions.insert(crossAttention.residualAddOp);
        usedFeedForwards.insert(feedForward.residualAddOp);
        layers.push_back(layer);
        break;
      }

      if (usedCrossAttentions.contains(crossAttention.residualAddOp))
        break;
    }
  }

  llvm::sort(layers, [](const DecoderLayerMatch &lhs,
                        const DecoderLayerMatch &rhs) {
    return isBeforeInBlock(lhs.selfAttention.residualAddOp,
                           rhs.selfAttention.residualAddOp);
  });

  for (auto [index, layer] : llvm::enumerate(layers))
    layer.layerIndex = static_cast<int64_t>(index);

  types.decoderLayerCount = static_cast<int64_t>(layers.size());
  return layers;
}

static mlir::FailureOr<LayerNormMatch>
findLayerNormWithInput(llvm::ArrayRef<LayerNormMatch> layerNorms,
                       mlir::Value input) {
  for (const LayerNormMatch &layerNorm : layerNorms) {
    if (layerNorm.input == input)
      return layerNorm;
  }
  return mlir::failure();
}

static mlir::LogicalResult
validateEncoderLayerChain(llvm::ArrayRef<EncoderLayerMatch> encoderLayers,
                          mlir::Value src) {
  mlir::Value current = src;
  for (const EncoderLayerMatch &layer : encoderLayers) {
    if (layer.input != current)
      return mlir::failure();
    current = layer.output;
  }
  return mlir::success();
}

static mlir::LogicalResult
validateDecoderLayerChain(llvm::ArrayRef<DecoderLayerMatch> decoderLayers,
                          mlir::Value tgt) {
  mlir::Value current = tgt;
  for (const DecoderLayerMatch &layer : decoderLayers) {
    if (layer.input != current)
      return mlir::failure();
    current = layer.output;
  }
  return mlir::success();
}

static mlir::FailureOr<TransformerMatch>
matchSupportedTransformer(mlir::func::FuncOp func) {
  auto types = matchTransformerSignature(func);
  if (mlir::failed(types))
    return mlir::failure();

  auto returnOp =
      llvm::dyn_cast<ReturnOp>(func.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1)
    return mlir::failure();

  TransformerTypes transformerTypes = *types;
  auto projections = collectProjectionCandidates(func, transformerTypes);
  auto attentionCores =
      collectAttentionCoreCandidates(func, transformerTypes);
  auto layerNorms = collectLayerNormCandidates(func, transformerTypes);
  auto feedForwards =
      collectFeedForwardCandidates(projections, layerNorms, transformerTypes);
  auto selfAttentions = collectSelfAttentionCandidates(
      projections, attentionCores, layerNorms, transformerTypes);
  auto crossAttentions = collectCrossAttentionCandidates(
      projections, attentionCores, layerNorms, transformerTypes);
  auto encoderLayers =
      collectEncoderLayerCandidates(selfAttentions, feedForwards,
                                    transformerTypes);
  auto decoderLayers = collectDecoderLayerCandidates(
      selfAttentions, crossAttentions, feedForwards, transformerTypes);

  if (encoderLayers.empty() || decoderLayers.empty())
    return mlir::failure();

  if (mlir::failed(validateEncoderLayerChain(encoderLayers,
                                             func.getArgument(0))))
    return mlir::failure();

  if (mlir::failed(validateDecoderLayerChain(decoderLayers,
                                             func.getArgument(1))))
    return mlir::failure();

  auto encoderFinalNorm =
      findLayerNormWithInput(layerNorms, encoderLayers.back().output);
  auto decoderFinalNorm =
      findLayerNormWithInput(layerNorms, decoderLayers.back().output);
  if (mlir::failed(encoderFinalNorm))
    return mlir::failure();
  if (mlir::failed(decoderFinalNorm))
    return mlir::failure();

  for (const DecoderLayerMatch &decoderLayer : decoderLayers) {
    if (decoderLayer.memoryInput == decoderLayer.input)
      return mlir::failure();
  }

  if (returnOp.getOperand(0) != decoderFinalNorm->output)
    return mlir::failure();

  if (static_cast<int64_t>(encoderLayers.size()) !=
          transformerTypes.encoderLayerCount ||
      static_cast<int64_t>(decoderLayers.size()) !=
          transformerTypes.decoderLayerCount ||
      transformerTypes.numHeads <= 0 || transformerTypes.headDim <= 0 ||
      transformerTypes.feedForwardSize <= 0)
    return mlir::failure();

  TransformerMatch match;
  match.returnOp = returnOp;
  match.encoderLayers = encoderLayers;
  match.decoderLayers = decoderLayers;
  match.encoderFinalNorm = *encoderFinalNorm;
  match.decoderFinalNorm = *decoderFinalNorm;
  match.types = transformerTypes;
  for (const EncoderLayerMatch &layer : match.encoderLayers)
    match.ops.append(layer.ops.begin(), layer.ops.end());
  match.ops.append(match.encoderFinalNorm.ops.begin(),
                   match.encoderFinalNorm.ops.end());
  for (const DecoderLayerMatch &layer : match.decoderLayers)
    match.ops.append(layer.ops.begin(), layer.ops.end());
  match.ops.append(match.decoderFinalNorm.ops.begin(),
                   match.decoderFinalNorm.ops.end());
  return match;
}

static mlir::FailureOr<TransformerEncoderMatch>
matchSupportedSingleInputTransformerStack(mlir::func::FuncOp func) {
  auto types = matchTransformerEncoderOutputSignature(func);
  if (mlir::failed(types))
    return mlir::failure();

  auto returnOp =
      llvm::dyn_cast<ReturnOp>(func.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1)
    return mlir::failure();

  TransformerTypes transformerTypes = *types;
  auto projections = collectProjectionCandidates(func, transformerTypes);
  auto attentionCores =
      collectAttentionCoreCandidates(func, transformerTypes);
  auto layerNorms = collectLayerNormCandidates(func, transformerTypes);

  mlir::StringRef normMode = "pre";
  auto feedForwards = collectPreNormFeedForwardCandidates(
      projections, layerNorms, transformerTypes);
  auto selfAttentions = collectPreNormSelfAttentionCandidates(
      projections, attentionCores, layerNorms, transformerTypes);
  auto encoderLayers =
      collectEncoderLayerCandidates(selfAttentions, feedForwards,
                                    transformerTypes);

  if (encoderLayers.empty()) {
    normMode = "post";
    feedForwards =
        collectFeedForwardCandidates(projections, layerNorms, transformerTypes);
    selfAttentions = collectSelfAttentionCandidates(
        projections, attentionCores, layerNorms, transformerTypes);
    encoderLayers =
        collectEncoderLayerCandidates(selfAttentions, feedForwards,
                                      transformerTypes);
  }

  if (encoderLayers.empty())
    return mlir::failure();

  mlir::Value encoderInput = encoderLayers.front().input;
  if (mlir::failed(validateEncoderLayerChain(encoderLayers, encoderInput)))
    return mlir::failure();

  auto encoderInputTy =
      tensor_type::getStaticF32Tensor(encoderInput.getType(), 3);
  if (mlir::failed(encoderInputTy))
    return mlir::failure();
  if (encoderInputTy->getShape() != transformerTypes.outputTy.getShape())
    return mlir::failure();

  bool hasFinalNorm = false;
  LayerNormMatch finalNorm;
  auto maybeFinalNorm =
      findLayerNormWithInput(layerNorms, encoderLayers.back().output);
  if (mlir::succeeded(maybeFinalNorm) &&
      returnOp.getOperand(0) == maybeFinalNorm->output) {
    hasFinalNorm = true;
    finalNorm = *maybeFinalNorm;
  } else if (returnOp.getOperand(0) != encoderLayers.back().output) {
    return mlir::failure();
  }

  if (static_cast<int64_t>(encoderLayers.size()) !=
          transformerTypes.encoderLayerCount ||
      transformerTypes.numHeads <= 0 || transformerTypes.headDim <= 0 ||
      transformerTypes.feedForwardSize <= 0)
    return mlir::failure();

  TransformerEncoderMatch match;
  match.returnOp = returnOp;
  match.layers = encoderLayers;
  match.finalNorm = finalNorm;
  match.hasFinalNorm = hasFinalNorm;
  match.normMode = normMode;
  match.input = encoderInput;
  match.types = transformerTypes;
  match.types.srcTy = *encoderInputTy;
  match.types.tgtTy = *encoderInputTy;
  match.causal = llvm::any_of(match.layers, [](const EncoderLayerMatch &layer) {
    return attentionCoreUsesCausalMask(layer.selfAttention.core);
  });

  for (const EncoderLayerMatch &layer : match.layers)
    match.ops.append(layer.ops.begin(), layer.ops.end());
  if (match.hasFinalNorm)
    match.ops.append(match.finalNorm.ops.begin(), match.finalNorm.ops.end());
  return match;
}

static mlir::FailureOr<TransformerEncoderMatch>
matchSupportedTransformerEncoder(mlir::func::FuncOp func) {
  auto match = matchSupportedSingleInputTransformerStack(func);
  if (mlir::failed(match) || match->causal)
    return mlir::failure();
  return match;
}

static mlir::FailureOr<TransformerEncoderMatch>
matchSupportedTransformerDecoder(mlir::func::FuncOp func) {
  auto match = matchSupportedSingleInputTransformerStack(func);
  if (mlir::failed(match) || !match->causal || match->normMode != "pre")
    return mlir::failure();
  return match;
}

static void appendProjectionParameters(
    const ProjectionMatch &projection,
    llvm::SmallVectorImpl<mlir::Value> &parameters, bool hasBias) {
  parameters.push_back(projection.weight);
  if (hasBias)
    parameters.push_back(projection.bias);
}

static void appendNormParameters(const LayerNormMatch &norm,
                                 llvm::SmallVectorImpl<mlir::Value> &parameters,
                                 bool hasLayerNormAffine) {
  if (!hasLayerNormAffine)
    return;

  parameters.push_back(norm.weightConstant->getResult(0));
  parameters.push_back(norm.biasConstant->getResult(0));
}

static bool hasProjectionBias(const ProjectionMatch &projection) {
  return static_cast<bool>(projection.bias);
}

static bool hasLayerNormAffine(const LayerNormMatch &norm) {
  return static_cast<bool>(norm.weightConstant) &&
         static_cast<bool>(norm.biasConstant);
}

static bool hasCompleteProjectionBias(const TransformerMatch &match) {
  for (const EncoderLayerMatch &layer : match.encoderLayers) {
    if (!hasProjectionBias(layer.selfAttention.qkvProjection) ||
        !hasProjectionBias(layer.selfAttention.outputProjection) ||
        !hasProjectionBias(layer.feedForward.upProjection) ||
        !hasProjectionBias(layer.feedForward.downProjection))
      return false;
  }

  for (const DecoderLayerMatch &layer : match.decoderLayers) {
    if (!hasProjectionBias(layer.selfAttention.qkvProjection) ||
        !hasProjectionBias(layer.selfAttention.outputProjection) ||
        !hasProjectionBias(layer.crossAttention.queryProjection) ||
        !hasProjectionBias(layer.crossAttention.keyValueProjection) ||
        !hasProjectionBias(layer.crossAttention.outputProjection) ||
        !hasProjectionBias(layer.feedForward.upProjection) ||
        !hasProjectionBias(layer.feedForward.downProjection))
      return false;
  }

  return true;
}

static bool hasCompleteLayerNormAffine(const TransformerMatch &match) {
  if (!hasLayerNormAffine(match.encoderFinalNorm) ||
      !hasLayerNormAffine(match.decoderFinalNorm))
    return false;

  for (const EncoderLayerMatch &layer : match.encoderLayers) {
    if (!hasLayerNormAffine(layer.selfAttention.norm) ||
        !hasLayerNormAffine(layer.feedForward.norm))
      return false;
  }

  for (const DecoderLayerMatch &layer : match.decoderLayers) {
    if (!hasLayerNormAffine(layer.selfAttention.norm) ||
        !hasLayerNormAffine(layer.crossAttention.norm) ||
        !hasLayerNormAffine(layer.feedForward.norm))
      return false;
  }

  return true;
}

static bool hasCompleteProjectionBias(const TransformerEncoderMatch &match) {
  for (const EncoderLayerMatch &layer : match.layers) {
    if (!hasProjectionBias(layer.selfAttention.qkvProjection) ||
        !hasProjectionBias(layer.selfAttention.outputProjection) ||
        !hasProjectionBias(layer.feedForward.upProjection) ||
        !hasProjectionBias(layer.feedForward.downProjection))
      return false;
  }

  return true;
}

static bool hasCompleteLayerNormAffine(const TransformerEncoderMatch &match) {
  if (match.hasFinalNorm && !hasLayerNormAffine(match.finalNorm))
    return false;

  for (const EncoderLayerMatch &layer : match.layers) {
    if (!hasLayerNormAffine(layer.selfAttention.norm) ||
        !hasLayerNormAffine(layer.feedForward.norm))
      return false;
  }

  return true;
}

static llvm::SmallVector<mlir::Value, 64>
collectTransformerParameters(const TransformerMatch &match, bool hasBias,
                             bool hasLayerNormAffine) {
  llvm::SmallVector<mlir::Value, 64> parameters;

  for (const EncoderLayerMatch &layer : match.encoderLayers) {
    appendProjectionParameters(layer.selfAttention.qkvProjection, parameters,
                               hasBias);
    appendProjectionParameters(layer.selfAttention.outputProjection,
                               parameters, hasBias);
    appendNormParameters(layer.selfAttention.norm, parameters,
                         hasLayerNormAffine);
    appendProjectionParameters(layer.feedForward.upProjection, parameters,
                               hasBias);
    appendProjectionParameters(layer.feedForward.downProjection, parameters,
                               hasBias);
    appendNormParameters(layer.feedForward.norm, parameters,
                         hasLayerNormAffine);
  }

  for (const DecoderLayerMatch &layer : match.decoderLayers) {
    appendProjectionParameters(layer.selfAttention.qkvProjection, parameters,
                               hasBias);
    appendProjectionParameters(layer.selfAttention.outputProjection,
                               parameters, hasBias);
    appendNormParameters(layer.selfAttention.norm, parameters,
                         hasLayerNormAffine);
    appendProjectionParameters(layer.crossAttention.queryProjection, parameters,
                               hasBias);
    appendProjectionParameters(layer.crossAttention.keyValueProjection,
                               parameters, hasBias);
    appendProjectionParameters(layer.crossAttention.outputProjection,
                               parameters, hasBias);
    appendNormParameters(layer.crossAttention.norm, parameters,
                         hasLayerNormAffine);
    appendProjectionParameters(layer.feedForward.upProjection, parameters,
                               hasBias);
    appendProjectionParameters(layer.feedForward.downProjection, parameters,
                               hasBias);
    appendNormParameters(layer.feedForward.norm, parameters,
                         hasLayerNormAffine);
  }

  appendNormParameters(match.encoderFinalNorm, parameters, hasLayerNormAffine);
  appendNormParameters(match.decoderFinalNorm, parameters, hasLayerNormAffine);
  return parameters;
}

static llvm::SmallVector<mlir::Value, 64>
collectTransformerEncoderParameters(const TransformerEncoderMatch &match,
                                    bool hasBias,
                                    bool hasLayerNormAffine) {
  llvm::SmallVector<mlir::Value, 64> parameters;

  for (const EncoderLayerMatch &layer : match.layers) {
    appendProjectionParameters(layer.selfAttention.qkvProjection, parameters,
                               hasBias);
    appendProjectionParameters(layer.selfAttention.outputProjection,
                               parameters, hasBias);
    appendNormParameters(layer.selfAttention.norm, parameters,
                         hasLayerNormAffine);
    appendProjectionParameters(layer.feedForward.upProjection, parameters,
                               hasBias);
    appendProjectionParameters(layer.feedForward.downProjection, parameters,
                               hasBias);
    appendNormParameters(layer.feedForward.norm, parameters,
                         hasLayerNormAffine);
  }

  if (match.hasFinalNorm)
    appendNormParameters(match.finalNorm, parameters, hasLayerNormAffine);
  return parameters;
}

static void markProducerTree(mlir::Value value,
                             llvm::SmallPtrSetImpl<mlir::Operation *> &keepOps) {
  mlir::Operation *op = value.getDefiningOp();
  if (!op || !keepOps.insert(op).second)
    return;

  for (mlir::Value operand : op->getOperands())
    markProducerTree(operand, keepOps);
}

static void eraseDeadTransformerBody(mlir::func::FuncOp func,
                                     mlir::Operation *newTransformerOp,
                                     mlir::ValueRange keptParameters,
                                     mlir::RewriterBase &rewriter) {
  llvm::SmallPtrSet<mlir::Operation *, 32> keepOps;
  keepOps.insert(newTransformerOp);
  for (mlir::Value parameter : keptParameters) {
    if (parameter.getDefiningOp())
      markProducerTree(parameter, keepOps);
  }

  llvm::SmallVector<mlir::Operation *, 256> ops;
  for (mlir::Operation &op : func.getBody().front().without_terminator())
    ops.push_back(&op);

  for (mlir::Operation *op : llvm::reverse(ops)) {
    if (keepOps.contains(op) || !op->use_empty())
      continue;
    rewriter.eraseOp(op);
  }
}

static void rewriteTransformerMatchToSculptorOp(
    const TransformerMatch &match, mlir::func::FuncOp func,
    mlir::RewriterBase &rewriter) {
  bool hasBias = hasCompleteProjectionBias(match);
  bool hasLayerNormAffine = hasCompleteLayerNormAffine(match);
  if (!hasBias || !hasLayerNormAffine)
    return;

  auto parameters =
      collectTransformerParameters(match, hasBias, hasLayerNormAffine);

  ReturnOp returnOp = match.returnOp;
  rewriter.setInsertionPoint(returnOp);
  auto transformerOp = rewriter.create<NNTransformerOp>(
      returnOp.getLoc(), match.types.outputTy, func.getArgument(0),
      func.getArgument(1), parameters, rewriter.getBoolAttr(true),
      rewriter.getBoolAttr(hasBias),
      rewriter.getBoolAttr(hasLayerNormAffine),
      rewriter.getBoolAttr(/*has_final_norm=*/true),
      rewriter.getBoolAttr(/*causal=*/false), rewriter.getStringAttr("gelu"),
      rewriter.getStringAttr("post"),
      rewriter.getI64IntegerAttr(match.types.hiddenSize),
      rewriter.getI64IntegerAttr(match.types.numHeads),
      rewriter.getI64IntegerAttr(match.types.headDim),
      rewriter.getI64IntegerAttr(match.types.feedForwardSize),
      rewriter.getI64IntegerAttr(match.types.encoderLayerCount),
      rewriter.getI64IntegerAttr(match.types.decoderLayerCount),
      rewriter.getF64FloatAttr(1.0e-5));

  returnOp.setOperand(0, transformerOp.getOutput());
  eraseDeadTransformerBody(func, transformerOp.getOperation(), parameters,
                           rewriter);
}

static void rewriteTransformerEncoderMatchToSculptorOp(
    const TransformerEncoderMatch &match, mlir::func::FuncOp func,
    mlir::RewriterBase &rewriter) {
  bool hasBias = hasCompleteProjectionBias(match);
  bool hasLayerNormAffine = hasCompleteLayerNormAffine(match);
  if (!hasBias || !hasLayerNormAffine)
    return;

  auto parameters =
      collectTransformerEncoderParameters(match, hasBias, hasLayerNormAffine);

  ReturnOp returnOp = match.returnOp;
  rewriter.setInsertionPoint(returnOp);
  auto transformerOp = rewriter.create<NNTransformerEncoderOp>(
      returnOp.getLoc(), match.types.outputTy, match.input, parameters,
      rewriter.getBoolAttr(true), rewriter.getBoolAttr(hasBias),
      rewriter.getBoolAttr(hasLayerNormAffine),
      rewriter.getBoolAttr(match.hasFinalNorm),
      rewriter.getBoolAttr(match.causal), rewriter.getStringAttr("gelu"),
      rewriter.getStringAttr(match.normMode),
      rewriter.getI64IntegerAttr(match.types.hiddenSize),
      rewriter.getI64IntegerAttr(match.types.numHeads),
      rewriter.getI64IntegerAttr(match.types.headDim),
      rewriter.getI64IntegerAttr(match.types.feedForwardSize),
      rewriter.getI64IntegerAttr(match.types.encoderLayerCount),
      rewriter.getF64FloatAttr(1.0e-5));

  returnOp.setOperand(0, transformerOp.getOutput());
  eraseDeadTransformerBody(func, transformerOp.getOperation(), parameters,
                           rewriter);
}

static void rewriteTransformerDecoderMatchToSculptorOp(
    const TransformerEncoderMatch &match, mlir::func::FuncOp func,
    mlir::RewriterBase &rewriter) {
  bool hasBias = hasCompleteProjectionBias(match);
  bool hasLayerNormAffine = hasCompleteLayerNormAffine(match);
  if (!hasBias || !hasLayerNormAffine)
    return;

  auto parameters =
      collectTransformerEncoderParameters(match, hasBias, hasLayerNormAffine);

  ReturnOp returnOp = match.returnOp;
  rewriter.setInsertionPoint(returnOp);
  auto transformerOp = rewriter.create<NNTransformerDecoderOp>(
      returnOp.getLoc(), match.types.outputTy, match.input, parameters,
      rewriter.getBoolAttr(true), rewriter.getBoolAttr(hasBias),
      rewriter.getBoolAttr(hasLayerNormAffine),
      rewriter.getBoolAttr(match.hasFinalNorm),
      rewriter.getBoolAttr(/*causal=*/true),
      rewriter.getBoolAttr(/*has_cross_attention=*/false),
      rewriter.getStringAttr("causal_only"), rewriter.getStringAttr("gelu"),
      rewriter.getStringAttr(match.normMode),
      rewriter.getI64IntegerAttr(match.types.hiddenSize),
      rewriter.getI64IntegerAttr(match.types.numHeads),
      rewriter.getI64IntegerAttr(match.types.headDim),
      rewriter.getI64IntegerAttr(match.types.feedForwardSize),
      rewriter.getI64IntegerAttr(match.types.encoderLayerCount),
      rewriter.getF64FloatAttr(1.0e-5));

  returnOp.setOperand(0, transformerOp.getOutput());
  eraseDeadTransformerBody(func, transformerOp.getOperation(), parameters,
                           rewriter);
}

// Placeholder for torch.nn.Transformer canonicalization.
class TransformerCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit TransformerCanonicalizer(mlir::MLIRContext *context) {
    (void)context;
  }

  mlir::StringRef getName() const override { return "transformer"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    auto match = matchSupportedTransformer(func);
    if (mlir::failed(match))
      return;

    mlir::IRRewriter rewriter(func.getContext());
    rewriteTransformerMatchToSculptorOp(*match, func, rewriter);
  }
};

class TransformerEncoderCanonicalizer
    : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit TransformerEncoderCanonicalizer(mlir::MLIRContext *context) {
    (void)context;
  }

  mlir::StringRef getName() const override { return "transformer_encoder"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    auto match = matchSupportedTransformerEncoder(func);
    if (mlir::failed(match))
      return;

    mlir::IRRewriter rewriter(func.getContext());
    rewriteTransformerEncoderMatchToSculptorOp(*match, func, rewriter);
  }
};

class TransformerDecoderCanonicalizer
    : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit TransformerDecoderCanonicalizer(mlir::MLIRContext *context) {
    (void)context;
  }

  mlir::StringRef getName() const override { return "transformer_decoder"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    auto match = matchSupportedTransformerDecoder(func);
    if (mlir::failed(match))
      return;

    mlir::IRRewriter rewriter(func.getContext());
    rewriteTransformerDecoderMatchToSculptorOp(*match, func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerTransformerCanonicalizer(LayerCanonicalizers &canonicalizers,
                                      MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<TransformerCanonicalizer>(context));
  canonicalizers.push_back(
      std::make_unique<TransformerDecoderCanonicalizer>(context));
  canonicalizers.push_back(
      std::make_unique<TransformerEncoderCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
