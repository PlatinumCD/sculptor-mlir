#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_RECURRENTLAYERPATTERNS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_RECURRENTLAYERPATTERNS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/TensorTypeUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/MatchedSubgraphUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <utility>

namespace mlir {
namespace sculptor {
namespace layer_patterns {

struct RecurrentGateIndexingScaffoldMatch {
  Operation *zeroIndexConstant = nullptr;
  Operation *indexExtentConstant = nullptr;
  Operation *zeroIndexEmpty = nullptr;
  Operation *zeroIndexGeneric = nullptr;
  Operation *zeroIndexExpand = nullptr;
  Operation *baseOffsetEmpty = nullptr;
  Operation *baseOffsetGeneric = nullptr;
  Operation *rangeEmpty = nullptr;
  Operation *rangeGeneric = nullptr;
  Operation *rangeExpand = nullptr;
  Operation *combinedIndicesEmpty = nullptr;
  Operation *combinedIndicesGeneric = nullptr;
  Operation *collapsedIndices = nullptr;
  Operation *gatherEmpty = nullptr;
};

struct RecurrentGateSliceMatch {
  Operation *offsetConstant = nullptr;
  Operation *offsetAdd = nullptr;
  Operation *gather = nullptr;
  Operation *expand = nullptr;
  Operation *activation = nullptr;
};

struct RecurrentTimestepSliceMatch {
  mlir::tensor::CollapseShapeOp collapse;
  mlir::tensor::ExtractSliceOp slice;
  mlir::Value source;
};

inline bool isZeroF32Constant(mlir::Value value);

struct RecurrentWeightProjectionMatch {
  mlir::Value activation;
  mlir::arith::ConstantOp weightConstant;
  mlir::linalg::TransposeOp weightTranspose;
  mlir::linalg::MatmulOp matmulOp;
};

inline bool isProjectionAddfGeneric(mlir::linalg::GenericOp genericOp) {
  if (!mlir::sculptor::linalg_match::hasExpectedBodyShape(genericOp,
                                                        /*inputCount=*/2))
    return false;

  mlir::Block &body = genericOp.getRegion().front();
  auto addOp =
      llvm::dyn_cast<mlir::arith::AddFOp>(body.getOperations().front());
  auto yieldOp =
      llvm::dyn_cast<mlir::linalg::YieldOp>(body.getOperations().back());
  return addOp && yieldOp && yieldOp.getValues().size() == 1 &&
         ((addOp.getOperand(0) == body.getArgument(0) &&
           addOp.getOperand(1) == body.getArgument(1)) ||
          (addOp.getOperand(0) == body.getArgument(1) &&
           addOp.getOperand(1) == body.getArgument(0))) &&
         yieldOp.getValues().front() == addOp.getResult();
}

inline bool isZeroInitializedOutput(mlir::Value output,
                                    mlir::RankedTensorType outputTy) {
  auto fillOp = layer_utils::producerOfType<mlir::linalg::FillOp>(output);
  if (!fillOp || fillOp.getInputs().size() != 1 ||
      fillOp.getOutputs().size() != 1 || fillOp.getResult(0) != output)
    return false;

  auto fillTy = llvm::dyn_cast<mlir::RankedTensorType>(fillOp.getType(0));
  return fillTy && fillTy == outputTy &&
         isZeroF32Constant(fillOp.getInputs()[0]);
}

inline mlir::FailureOr<mlir::arith::ConstantOp>
matchProjectionWeightTranspose(mlir::Value value, int64_t inputWidth,
                               int64_t outputWidth,
                               mlir::linalg::TransposeOp &transposeOut) {
  auto transpose =
      layer_utils::producerOfType<mlir::linalg::TransposeOp>(value);
  if (!transpose || transpose.getResult().front() != value ||
      !mlir::sculptor::linalg_match::hasPermutation(transpose, {1, 0}) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(value,
                                                    {inputWidth, outputWidth}))
    return mlir::failure();

  if (!mlir::sculptor::tensor_type::hasStaticF32Shape(transpose.getInit(),
                                                    {inputWidth, outputWidth}))
    return mlir::failure();

  auto init =
      layer_utils::producerOfType<mlir::tensor::EmptyOp>(transpose.getInit());
  auto weightConstant = layer_utils::producerOfType<mlir::arith::ConstantOp>(
      transpose.getInput());
  if (!init || !weightConstant ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(weightConstant.getResult(),
                                                    {outputWidth, inputWidth}))
    return mlir::failure();

  transposeOut = transpose;
  return weightConstant;
}

inline mlir::FailureOr<RecurrentWeightProjectionMatch>
matchMatmulProjection(mlir::Value value, int64_t rowCount, int64_t inputWidth,
                      int64_t outputWidth) {
  auto matmul = layer_utils::producerOfType<mlir::linalg::MatmulOp>(value);
  if (!matmul || matmul.getInputs().size() != 2 ||
      matmul.getOutputs().size() != 1 ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(value,
                                                    {rowCount, outputWidth}))
    return mlir::failure();

  auto matmulTy =
      llvm::cast<mlir::RankedTensorType>(matmul.getResult(0).getType());
  if (!isZeroInitializedOutput(matmul.getOutputs()[0], matmulTy))
    return mlir::failure();

  mlir::linalg::TransposeOp firstTranspose;
  mlir::linalg::TransposeOp secondTranspose;
  auto firstWeight = matchProjectionWeightTranspose(
      matmul.getInputs()[0], inputWidth, outputWidth, firstTranspose);
  auto secondWeight = matchProjectionWeightTranspose(
      matmul.getInputs()[1], inputWidth, outputWidth, secondTranspose);
  if (mlir::succeeded(firstWeight) == mlir::succeeded(secondWeight))
    return mlir::failure();

  mlir::Value activation = mlir::succeeded(firstWeight) ? matmul.getInputs()[1]
                                                        : matmul.getInputs()[0];
  if (!mlir::sculptor::tensor_type::hasStaticF32Shape(activation,
                                                    {rowCount, inputWidth}))
    return mlir::failure();

  return RecurrentWeightProjectionMatch{
      activation, mlir::succeeded(firstWeight) ? *firstWeight : *secondWeight,
      mlir::succeeded(firstWeight) ? firstTranspose : secondTranspose, matmul};
}

inline mlir::FailureOr<std::pair<mlir::Value, mlir::arith::ConstantOp>>
matchProjectionBiasAdd(mlir::Value value, llvm::ArrayRef<int64_t> outputShape,
                       int64_t biasWidth) {
  auto addOp = layer_utils::producerOfType<mlir::linalg::GenericOp>(value);
  if (!isProjectionAddfGeneric(addOp) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(addOp.getResult(0),
                                                    outputShape) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(addOp.getOutputs()[0],
                                                    outputShape))
    return mlir::failure();

  auto firstBias = layer_utils::producerOfType<mlir::arith::ConstantOp>(
      addOp.getInputs()[0]);
  auto secondBias = layer_utils::producerOfType<mlir::arith::ConstantOp>(
      addOp.getInputs()[1]);
  if (static_cast<bool>(firstBias) == static_cast<bool>(secondBias))
    return mlir::failure();

  unsigned biasInputIndex = firstBias ? 0 : 1;
  if (!mlir::sculptor::linalg_match::hasBiasAddIndexingMaps(
          addOp, static_cast<unsigned>(outputShape.size()), biasInputIndex))
    return mlir::failure();

  mlir::arith::ConstantOp biasConstant = firstBias ? firstBias : secondBias;
  mlir::Value projectedValue =
      firstBias ? addOp.getInputs()[1] : addOp.getInputs()[0];
  if (!mlir::sculptor::tensor_type::hasStaticF32Shape(biasConstant.getResult(),
                                                    {biasWidth}) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(projectedValue,
                                                    outputShape))
    return mlir::failure();

  return std::make_pair(projectedValue, biasConstant);
}

inline mlir::FailureOr<RecurrentTimestepSliceMatch>
matchTimestepProjectionSlice(mlir::Value value, int64_t timestep,
                             llvm::ArrayRef<int64_t> collapsedResultShape,
                             int64_t sequenceLength, int64_t batchSize,
                             int64_t featureWidth) {
  auto collapse =
      layer_utils::producerOfType<mlir::tensor::CollapseShapeOp>(value);
  if (!collapse || !mlir::sculptor::tensor_type::hasStaticF32Shape(
                       collapse.getResult(), collapsedResultShape))
    return mlir::failure();

  auto slice = layer_utils::producerOfType<mlir::tensor::ExtractSliceOp>(
      collapse.getSrc());
  if (!slice ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(
          slice.getResult(), {1, batchSize, featureWidth}) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(
          slice.getSource(), {sequenceLength, batchSize, featureWidth}))
    return mlir::failure();

  llvm::ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  llvm::ArrayRef<int64_t> sizes = slice.getStaticSizes();
  llvm::ArrayRef<int64_t> strides = slice.getStaticStrides();
  if (offsets.size() != 3 || sizes.size() != 3 || strides.size() != 3)
    return mlir::failure();

  if (offsets[0] != timestep || offsets[1] != 0 || offsets[2] != 0 ||
      sizes[0] != 1 || sizes[1] != batchSize || sizes[2] != featureWidth ||
      strides[0] != 1 || strides[1] != 1 || strides[2] != 1)
    return mlir::failure();

  return RecurrentTimestepSliceMatch{collapse, slice, slice.getSource()};
}

inline bool matchesCollapsedRecurrentValue(mlir::Value value,
                                           mlir::Value source,
                                           int64_t batchSize,
                                           int64_t hiddenSize) {
  auto collapse =
      layer_utils::producerOfType<mlir::tensor::CollapseShapeOp>(value);
  return collapse && collapse.getSrc() == source &&
         mlir::sculptor::tensor_type::hasStaticF32Shape(
             source, {1, batchSize, hiddenSize}) &&
         mlir::sculptor::tensor_type::hasStaticF32Shape(collapse.getResult(),
                                                      {batchSize, hiddenSize});
}

inline bool matchesInitialRecurrentStateSlice(mlir::Value value,
                                              mlir::Value state, int64_t layer,
                                              int64_t batchSize,
                                              int64_t hiddenSize) {
  auto slice = layer_utils::producerOfType<mlir::tensor::ExtractSliceOp>(value);
  if (!slice || slice.getSource() != state ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(slice.getResult(),
                                                    {1, batchSize, hiddenSize}))
    return false;

  llvm::ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  llvm::ArrayRef<int64_t> sizes = slice.getStaticSizes();
  llvm::ArrayRef<int64_t> strides = slice.getStaticStrides();
  if (offsets.size() != 3 || sizes.size() != 3 || strides.size() != 3)
    return false;

  return offsets[0] == layer && offsets[1] == 0 && offsets[2] == 0 &&
         sizes[0] == 1 && sizes[1] == batchSize && sizes[2] == hiddenSize &&
         strides[0] == 1 && strides[1] == 1 && strides[2] == 1;
}

inline bool matchesInitialRecurrentStateCollapsed(mlir::Value value,
                                                  mlir::Value state,
                                                  int64_t layer,
                                                  int64_t batchSize,
                                                  int64_t hiddenSize) {
  auto collapse =
      layer_utils::producerOfType<mlir::tensor::CollapseShapeOp>(value);
  if (!collapse || !mlir::sculptor::tensor_type::hasStaticF32Shape(
                       collapse.getResult(), {batchSize, hiddenSize}))
    return false;

  return matchesInitialRecurrentStateSlice(collapse.getSrc(), state, layer,
                                           batchSize, hiddenSize);
}

inline bool matchesLayerOutputConcat(mlir::tensor::ConcatOp concat,
                                     int64_t sequenceLength, int64_t batchSize,
                                     int64_t hiddenSize) {
  if (!concat || concat.getDim() != 0 ||
      concat.getInputs().size() != static_cast<size_t>(sequenceLength) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(
          concat.getResult(), {sequenceLength, batchSize, hiddenSize}))
    return false;

  for (mlir::Value input : concat.getInputs()) {
    if (!mlir::sculptor::tensor_type::hasStaticF32Shape(
            input, {1, batchSize, hiddenSize}))
      return false;
  }

  return true;
}

inline mlir::FailureOr<
    std::pair<mlir::linalg::TransposeOp, mlir::tensor::ConcatOp>>
matchFinalSequenceAssembly(mlir::Value value, int64_t sequenceLength,
                           int64_t batchSize, int64_t hiddenSize) {
  auto transpose =
      layer_utils::producerOfType<mlir::linalg::TransposeOp>(value);
  if (!transpose || transpose.getResult().front() != value ||
      !mlir::sculptor::linalg_match::hasPermutation(transpose, {1, 0, 2}) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(
          transpose.getInput(), {sequenceLength, batchSize, hiddenSize}) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(
          transpose.getInit(), {batchSize, sequenceLength, hiddenSize}))
    return mlir::failure();

  auto init =
      layer_utils::producerOfType<mlir::tensor::EmptyOp>(transpose.getInit());
  auto concat =
      layer_utils::producerOfType<mlir::tensor::ConcatOp>(transpose.getInput());
  if (!init ||
      !matchesLayerOutputConcat(concat, sequenceLength, batchSize, hiddenSize))
    return mlir::failure();

  return std::make_pair(transpose, concat);
}

template <typename FinalStateForLayerFn>
inline mlir::FailureOr<
    std::pair<mlir::tensor::ExpandShapeOp, mlir::tensor::ConcatOp>>
matchFinalRecurrentStateAssembly(mlir::Value value, int64_t layerCount,
                                 int64_t batchSize, int64_t hiddenSize,
                                 FinalStateForLayerFn finalStateForLayer) {
  auto expand = layer_utils::producerOfType<mlir::tensor::ExpandShapeOp>(value);
  if (!expand ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(
          expand.getResult(), {layerCount, batchSize, hiddenSize}) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(
          expand.getSrc(), {layerCount * batchSize, hiddenSize}))
    return mlir::failure();

  auto concat =
      layer_utils::producerOfType<mlir::tensor::ConcatOp>(expand.getSrc());
  if (!concat || concat.getDim() != 0 ||
      concat.getInputs().size() != static_cast<size_t>(layerCount) ||
      !mlir::sculptor::tensor_type::hasStaticF32Shape(
          concat.getResult(), {layerCount * batchSize, hiddenSize}))
    return mlir::failure();

  for (auto [layer, input] : llvm::enumerate(concat.getInputs())) {
    if (!mlir::sculptor::tensor_type::hasStaticF32Shape(input,
                                                      {batchSize, hiddenSize}))
      return mlir::failure();

    mlir::Value expectedState = finalStateForLayer(layer);
    if (!expectedState || !matchesCollapsedRecurrentValue(
                              input, expectedState, batchSize, hiddenSize))
      return mlir::failure();
  }

  return std::make_pair(expand, concat);
}

inline bool isZeroF32Constant(mlir::Value value) {
  auto constant = layer_utils::producerOfType<mlir::arith::ConstantOp>(value);
  if (!constant)
    return false;

  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(constant.getValue()))
    return floatAttr.getValue().isZero();

  if (auto denseAttr =
          llvm::dyn_cast<mlir::DenseElementsAttr>(constant.getValue())) {
    if (!denseAttr.isSplat())
      return false;

    auto splatValue = llvm::dyn_cast<mlir::FloatAttr>(
        denseAttr.getSplatValue<mlir::Attribute>());
    return splatValue && splatValue.getValue().isZero();
  }

  return false;
}

inline Operation *getSigmoidUnitConstant(Operation *op) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic || !generic.getRegion().hasOneBlock())
    return nullptr;

  mlir::Block &block = generic.getRegion().front();
  if (block.empty())
    return nullptr;

  auto it = block.begin();
  auto e = block.end();
  auto neg = llvm::dyn_cast<mlir::arith::NegFOp>(&*it++);
  auto exp = (it != e) ? llvm::dyn_cast<mlir::math::ExpOp>(&*it++)
                       : mlir::math::ExpOp();
  auto add = (it != e) ? llvm::dyn_cast<mlir::arith::AddFOp>(&*it++)
                       : mlir::arith::AddFOp();
  auto div = (it != e) ? llvm::dyn_cast<mlir::arith::DivFOp>(&*it++)
                       : mlir::arith::DivFOp();
  auto yield = (it != e) ? llvm::dyn_cast<mlir::linalg::YieldOp>(&*it++)
                         : mlir::linalg::YieldOp();
  if (!neg || !exp || !add || !div || !yield || it != e ||
      exp.getOperand() != neg.getResult())
    return nullptr;

  mlir::Value unitConstant;
  if (add.getLhs() == exp.getResult())
    unitConstant = add.getRhs();
  else if (add.getRhs() == exp.getResult())
    unitConstant = add.getLhs();
  else
    return nullptr;

  if (div.getLhs() != unitConstant || div.getRhs() != add.getResult() ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != div.getResult())
    return nullptr;

  return layer_utils::producerOf(unitConstant);
}

inline mlir::LogicalResult
matchYieldingConstantGeneric(Operation *op, Operation *expectedConstant) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic || !layer_utils::hasDpsInputsAndOperands(op, 0, 1) ||
      !generic.getRegion().hasOneBlock())
    return mlir::failure();

  mlir::Block &block = generic.getRegion().front();
  if (!llvm::hasSingleElement(block))
    return mlir::failure();

  auto yield = llvm::dyn_cast<mlir::linalg::YieldOp>(&block.front());
  if (!yield || yield.getNumOperands() != 1 ||
      layer_utils::producerOf(yield.getOperand(0)) != expectedConstant)
    return mlir::failure();

  return mlir::success();
}

inline mlir::LogicalResult matchIndexRangeGeneric(Operation *op) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic || !layer_utils::hasDpsInputsAndOperands(op, 0, 1) ||
      !generic.getRegion().hasOneBlock())
    return mlir::failure();

  mlir::Block &block = generic.getRegion().front();
  if (block.empty())
    return mlir::failure();

  auto it = block.begin();
  auto e = block.end();
  auto index = llvm::dyn_cast<mlir::linalg::IndexOp>(&*it++);
  auto cast = (it != e) ? llvm::dyn_cast<mlir::arith::IndexCastOp>(&*it++)
                        : mlir::arith::IndexCastOp();
  auto yield = (it != e) ? llvm::dyn_cast<mlir::linalg::YieldOp>(&*it++)
                         : mlir::linalg::YieldOp();
  if (!index || !cast || !yield || it != e ||
      cast.getIn() != index.getResult() || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != cast.getResult())
    return mlir::failure();

  return mlir::success();
}

inline mlir::LogicalResult
matchMultiplyByConstantGeneric(Operation *op, Operation *expectedConstant) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic || !layer_utils::isMuliGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 1, 2))
    return mlir::failure();

  mlir::Block &block = generic.getRegion().front();
  auto mul = llvm::dyn_cast<mlir::arith::MulIOp>(&block.front());
  if (!mul)
    return mlir::failure();

  Operation *lhsDef = layer_utils::producerOf(mul.getLhs());
  Operation *rhsDef = layer_utils::producerOf(mul.getRhs());
  if (lhsDef != expectedConstant && rhsDef != expectedConstant)
    return mlir::failure();

  return mlir::success();
}

inline mlir::LogicalResult matchOffsetAddGeneric(Operation *op,
                                                 Operation *expectedBaseIndices,
                                                 Operation *expectedOutputEmpty,
                                                 int64_t expectedOffset,
                                                 Operation *&offsetConstant) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic || !layer_utils::isAddiGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 1, 2) ||
      layer_utils::operandProducer(op, 0) != expectedBaseIndices)
    return mlir::failure();

  auto outputEmpty =
      layer_utils::operandProducerOfType<mlir::tensor::EmptyOp>(op, 1);
  if (!outputEmpty || outputEmpty.getOperation() != expectedOutputEmpty)
    return mlir::failure();

  mlir::Block &block = generic.getRegion().front();
  auto add = llvm::dyn_cast<mlir::arith::AddIOp>(&block.front());
  if (!add)
    return mlir::failure();

  offsetConstant = layer_utils::producerOf(add.getLhs());
  if (!offsetConstant)
    offsetConstant = layer_utils::producerOf(add.getRhs());
  if (!layer_utils::constantOpHasI64Value(offsetConstant, expectedOffset))
    return mlir::failure();

  return mlir::success();
}

inline void collectRecurrentGateIndexingOps(
    const RecurrentGateIndexingScaffoldMatch &indexing,
    llvm::SmallVectorImpl<Operation *> &ops) {
  match_utils::appendUniqueOp(ops, indexing.zeroIndexConstant);
  match_utils::appendUniqueOp(ops, indexing.indexExtentConstant);
  match_utils::appendUniqueOp(ops, indexing.zeroIndexEmpty);
  match_utils::appendUniqueOp(ops, indexing.zeroIndexGeneric);
  match_utils::appendUniqueOp(ops, indexing.zeroIndexExpand);
  match_utils::appendUniqueOp(ops, indexing.baseOffsetEmpty);
  match_utils::appendUniqueOp(ops, indexing.baseOffsetGeneric);
  match_utils::appendUniqueOp(ops, indexing.rangeEmpty);
  match_utils::appendUniqueOp(ops, indexing.rangeGeneric);
  match_utils::appendUniqueOp(ops, indexing.rangeExpand);
  match_utils::appendUniqueOp(ops, indexing.combinedIndicesEmpty);
  match_utils::appendUniqueOp(ops, indexing.combinedIndicesGeneric);
  match_utils::appendUniqueOp(ops, indexing.collapsedIndices);
  match_utils::appendUniqueOp(ops, indexing.gatherEmpty);
}

inline void
collectRecurrentGateSliceOps(const RecurrentGateSliceMatch &gate,
                             llvm::SmallVectorImpl<Operation *> &ops) {
  match_utils::appendUniqueOp(ops, gate.offsetConstant);
  match_utils::appendUniqueOp(ops, gate.offsetAdd);
  match_utils::appendUniqueOp(ops, gate.gather);
  match_utils::appendUniqueOp(ops, gate.expand);
  match_utils::appendUniqueOp(ops, gate.activation);
}

} // namespace layer_patterns
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_RECURRENTLAYERPATTERNS_H
