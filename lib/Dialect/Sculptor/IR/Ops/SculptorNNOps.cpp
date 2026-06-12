#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::sculptor;

namespace {

// Static tensor dimensions are the only safe dimensions for verifier equality
// checks; dynamic dimensions remain permissive for canonical sculptor.nn ops.
bool haveMismatchedStaticDims(int64_t lhsDim, int64_t rhsDim) {
  return !ShapedType::isDynamic(lhsDim) && !ShapedType::isDynamic(rhsDim) &&
         lhsDim != rhsDim;
}

LogicalResult verifyMatchingStaticDims(Operation *op, int64_t lhsDim,
                                       int64_t rhsDim, StringRef lhsName,
                                       StringRef rhsName) {
  if (!haveMismatchedStaticDims(lhsDim, rhsDim))
    return success();

  return op->emitOpError("expected ")
         << lhsName << " (" << lhsDim << ") to match " << rhsName << " ("
         << rhsDim << ")";
}

LogicalResult verifyStaticDimDivisibleBy(Operation *op, int64_t dim,
                                         int64_t divisor, StringRef dimName,
                                         StringRef divisorName) {
  if (ShapedType::isDynamic(dim))
    return success();

  if (dim % divisor == 0)
    return success();

  return op->emitOpError("expected ")
         << dimName << " (" << dim << ") to be divisible by " << divisorName
         << " (" << divisor << ")";
}

LogicalResult verifyI64ArrayAttr(Operation *op, ArrayAttr attr,
                                 int64_t expectedSize, StringRef attrName,
                                 int64_t minimumValue,
                                 StringRef minimumDescription,
                                 SmallVectorImpl<int64_t> &values) {
  values.clear();

  if (static_cast<int64_t>(attr.size()) != expectedSize) {
    return op->emitOpError("expected ")
           << attrName << " length (" << attr.size()
           << ") to match convolution spatial rank (" << expectedSize << ")";
  }

  values.reserve(attr.size());
  for (auto [index, attrValue] : llvm::enumerate(attr)) {
    int64_t value = llvm::cast<IntegerAttr>(attrValue).getInt();
    if (value < minimumValue) {
      return op->emitOpError("expected ")
             << attrName << "[" << index << "] to be " << minimumDescription
             << ", got " << value;
    }

    values.push_back(value);
  }

  return success();
}

LogicalResult verifyConvolutionAttributes(
    Operation *op, int64_t spatialRank, ArrayAttr stride, ArrayAttr padding,
    ArrayAttr dilation, SmallVectorImpl<int64_t> &strideValues,
    SmallVectorImpl<int64_t> &paddingValues,
    SmallVectorImpl<int64_t> &dilationValues) {
  if (failed(verifyI64ArrayAttr(op, stride, spatialRank, "stride", 1,
                                "positive", strideValues)))
    return failure();

  if (failed(verifyI64ArrayAttr(op, padding, spatialRank, "padding", 0,
                                "non-negative", paddingValues)))
    return failure();

  if (failed(verifyI64ArrayAttr(op, dilation, spatialRank, "dilation", 1,
                                "positive", dilationValues)))
    return failure();

  return success();
}

std::optional<int64_t> computeStaticConvolutionOutputDim(int64_t inputDim,
                                                         int64_t kernelDim,
                                                         int64_t stride,
                                                         int64_t padding,
                                                         int64_t dilation) {
  if (ShapedType::isDynamic(inputDim) || ShapedType::isDynamic(kernelDim))
    return std::nullopt;

  // Zero-sized kernel dimensions are not introduced by the canonical
  // extractors; leave any such edge case to more specific semantic checks.
  if (kernelDim <= 0)
    return std::nullopt;

  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  if (kernelDim - 1 > (max - 1) / dilation)
    return std::nullopt;

  int64_t effectiveKernel = (kernelDim - 1) * dilation + 1;
  if (padding > (max - inputDim) / 2)
    return std::nullopt;

  int64_t paddedInput = inputDim + 2 * padding;
  return ((paddedInput - effectiveKernel) / stride) + 1;
}

LogicalResult verifyConvolutionSpatialDims(
    Operation *op, RankedTensorType inputType, RankedTensorType weightType,
    RankedTensorType resultType, int64_t spatialRank, ArrayRef<int64_t> stride,
    ArrayRef<int64_t> padding, ArrayRef<int64_t> dilation) {
  for (int64_t dim = 0; dim < spatialRank; ++dim) {
    int64_t resultDim = resultType.getDimSize(dim + 2);
    if (ShapedType::isDynamic(resultDim))
      continue;

    std::optional<int64_t> expectedDim = computeStaticConvolutionOutputDim(
        inputType.getDimSize(dim + 2), weightType.getDimSize(dim + 2),
        stride[dim], padding[dim], dilation[dim]);
    if (!expectedDim)
      continue;

    if (resultDim != *expectedDim) {
      return op->emitOpError("expected result spatial dimension ")
             << dim << " (" << resultDim
             << ") to match computed convolution output dimension ("
             << *expectedDim << ")";
    }
  }

  return success();
}

LogicalResult verifyConvolutionOp(Operation *op, RankedTensorType inputType,
                                  RankedTensorType weightType,
                                  TypedValue<RankedTensorType> bias,
                                  RankedTensorType resultType, bool hasBias,
                                  ArrayAttr stride, ArrayAttr padding,
                                  ArrayAttr dilation, int64_t spatialRank) {
  int64_t expectedTensorRank = spatialRank + 2;
  if (inputType.getRank() != expectedTensorRank) {
    return op->emitOpError("expected input tensor rank to be ")
           << expectedTensorRank;
  }

  if (weightType.getRank() != expectedTensorRank) {
    return op->emitOpError("expected weight tensor rank to be ")
           << expectedTensorRank;
  }

  if (resultType.getRank() != expectedTensorRank) {
    return op->emitOpError("expected result tensor rank to be ")
           << expectedTensorRank;
  }

  if (hasBias && !bias)
    return op->emitOpError(
        "expected has_bias = true to include a bias operand");

  if (!hasBias && bias)
    return op->emitOpError(
        "expected has_bias = false to omit the bias operand");

  if (bias && bias.getType().getRank() != 1)
    return op->emitOpError("expected bias tensor to have rank 1");

  SmallVector<int64_t> strideValues;
  SmallVector<int64_t> paddingValues;
  SmallVector<int64_t> dilationValues;
  if (failed(verifyConvolutionAttributes(op, spatialRank, stride, padding,
                                         dilation, strideValues, paddingValues,
                                         dilationValues)))
    return failure();

  if (failed(verifyMatchingStaticDims(
          op, resultType.getDimSize(0), inputType.getDimSize(0),
          "result batch dimension", "input batch dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(
          op, inputType.getDimSize(1), weightType.getDimSize(1),
          "input channel dimension", "weight input channel dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(op, resultType.getDimSize(1),
                                      weightType.getDimSize(0),
                                      "result output channel dimension",
                                      "weight output channel dimension")))
    return failure();

  if (bias && failed(verifyMatchingStaticDims(
                  op, bias.getType().getDimSize(0), weightType.getDimSize(0),
                  "bias dimension", "weight output channel dimension")))
    return failure();

  return verifyConvolutionSpatialDims(op, inputType, weightType, resultType,
                                      spatialRank, strideValues, paddingValues,
                                      dilationValues);
}

LogicalResult verifyGroupedConv2DOp(Operation *op, RankedTensorType inputType,
                                    RankedTensorType weightType,
                                    TypedValue<RankedTensorType> bias,
                                    RankedTensorType resultType, bool hasBias,
                                    int64_t groups, ArrayAttr stride,
                                    ArrayAttr padding, ArrayAttr dilation) {
  constexpr int64_t spatialRank = 2;
  constexpr int64_t expectedTensorRank = spatialRank + 2;

  if (inputType.getRank() != expectedTensorRank)
    return op->emitOpError("expected input tensor rank to be ")
           << expectedTensorRank;

  if (weightType.getRank() != expectedTensorRank)
    return op->emitOpError("expected weight tensor rank to be ")
           << expectedTensorRank;

  if (resultType.getRank() != expectedTensorRank)
    return op->emitOpError("expected result tensor rank to be ")
           << expectedTensorRank;

  if (hasBias && !bias)
    return op->emitOpError(
        "expected has_bias = true to include a bias operand");

  if (!hasBias && bias)
    return op->emitOpError(
        "expected has_bias = false to omit the bias operand");

  if (bias && bias.getType().getRank() != 1)
    return op->emitOpError("expected bias tensor to have rank 1");

  if (groups <= 1)
    return op->emitOpError("expected groups to be greater than 1");

  SmallVector<int64_t> strideValues;
  SmallVector<int64_t> paddingValues;
  SmallVector<int64_t> dilationValues;
  if (failed(verifyConvolutionAttributes(op, spatialRank, stride, padding,
                                         dilation, strideValues, paddingValues,
                                         dilationValues)))
    return failure();

  if (failed(verifyMatchingStaticDims(
          op, resultType.getDimSize(0), inputType.getDimSize(0),
          "result batch dimension", "input batch dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(op, resultType.getDimSize(1),
                                      weightType.getDimSize(0),
                                      "result output channel dimension",
                                      "weight output channel dimension")))
    return failure();

  if (failed(verifyStaticDimDivisibleBy(op, inputType.getDimSize(1), groups,
                                        "input channel dimension", "groups")))
    return failure();

  if (failed(verifyStaticDimDivisibleBy(op, resultType.getDimSize(1), groups,
                                        "result output channel dimension",
                                        "groups")))
    return failure();

  if (failed(verifyStaticDimDivisibleBy(op, weightType.getDimSize(0), groups,
                                        "weight output channel dimension",
                                        "groups")))
    return failure();

  int64_t inputChannels = inputType.getDimSize(1);
  if (!ShapedType::isDynamic(inputChannels) && inputChannels % groups == 0) {
    int64_t inputChannelsPerGroup = inputChannels / groups;
    if (failed(verifyMatchingStaticDims(
            op, weightType.getDimSize(1), inputChannelsPerGroup,
            "weight input channel dimension", "input channels per group")))
      return failure();
  }

  if (bias && failed(verifyMatchingStaticDims(
                  op, bias.getType().getDimSize(0), weightType.getDimSize(0),
                  "bias dimension", "weight output channel dimension")))
    return failure();

  return verifyConvolutionSpatialDims(op, inputType, weightType, resultType,
                                      spatialRank, strideValues, paddingValues,
                                      dilationValues);
}

LogicalResult verifyMatchingStaticShape(Operation *op, RankedTensorType lhsType,
                                        RankedTensorType rhsType,
                                        StringRef lhsName,
                                        StringRef rhsName) {
  if (lhsType.getRank() != rhsType.getRank()) {
    return op->emitOpError("expected ")
           << lhsName << " rank (" << lhsType.getRank() << ") to match "
           << rhsName << " rank (" << rhsType.getRank() << ")";
  }

  for (int64_t dim = 0, rank = lhsType.getRank(); dim < rank; ++dim) {
    int64_t lhsDim = lhsType.getDimSize(dim);
    int64_t rhsDim = rhsType.getDimSize(dim);
    if (!haveMismatchedStaticDims(lhsDim, rhsDim))
      continue;

    return op->emitOpError("expected ")
           << lhsName << " dimension " << dim << " (" << lhsDim
           << ") to match " << rhsName << " dimension " << dim << " ("
           << rhsDim << ")";
  }

  return success();
}

LogicalResult verifyTensorRank(Operation *op, RankedTensorType type,
                               int64_t expectedRank, StringRef tensorName) {
  if (type.getRank() == expectedRank)
    return success();

  return op->emitOpError("expected ")
         << tensorName << " tensor rank to be " << expectedRank;
}

LogicalResult verifyGateExpandedStaticDim(Operation *op, int64_t dim,
                                          int64_t hiddenSize,
                                          int64_t gateCount,
                                          StringRef dimName) {
  if (ShapedType::isDynamic(dim) || ShapedType::isDynamic(hiddenSize))
    return success();

  if (hiddenSize > std::numeric_limits<int64_t>::max() / gateCount)
    return success();

  int64_t expectedDim = gateCount * hiddenSize;
  if (dim == expectedDim)
    return success();

  return op->emitOpError("expected ")
         << dimName << " (" << dim << ") to match " << gateCount
         << " * hidden state dimension (" << expectedDim << ")";
}

LogicalResult verifyRecurrentCellBiasOperands(
    Operation *op, TypedValue<RankedTensorType> bIh,
    TypedValue<RankedTensorType> bHh, bool hasBias, int64_t hiddenSize,
    int64_t gateCount) {
  if (hasBias && (!bIh || !bHh)) {
    return op->emitOpError(
        "expected has_bias = true to include b_ih and b_hh operands");
  }

  if (!hasBias && (bIh || bHh)) {
    return op->emitOpError(
        "expected has_bias = false to omit bias operands");
  }

  if (bIh && bIh.getType().getRank() != 1)
    return op->emitOpError("expected b_ih tensor to have rank 1");

  if (bHh && bHh.getType().getRank() != 1)
    return op->emitOpError("expected b_hh tensor to have rank 1");

  if (bIh && failed(verifyGateExpandedStaticDim(
                op, bIh.getType().getDimSize(0), hiddenSize, gateCount,
                "b_ih dimension")))
    return failure();

  if (bHh && failed(verifyGateExpandedStaticDim(
                op, bHh.getType().getDimSize(0), hiddenSize, gateCount,
                "b_hh dimension")))
    return failure();

  return success();
}

LogicalResult verifyRecurrentCellOp(Operation *op, RankedTensorType inputType,
                                    RankedTensorType hPrevType,
                                    RankedTensorType wIhType,
                                    RankedTensorType wHhType,
                                    TypedValue<RankedTensorType> bIh,
                                    TypedValue<RankedTensorType> bHh,
                                    RankedTensorType hType, bool hasBias,
                                    int64_t gateCount) {
  if (inputType.getRank() != 2)
    return op->emitOpError("expected input tensor rank to be 2");

  if (hPrevType.getRank() != 2)
    return op->emitOpError("expected h_prev tensor rank to be 2");

  if (wIhType.getRank() != 2)
    return op->emitOpError("expected w_ih tensor rank to be 2");

  if (wHhType.getRank() != 2)
    return op->emitOpError("expected w_hh tensor rank to be 2");

  if (hType.getRank() != 2)
    return op->emitOpError("expected result hidden state tensor rank to be 2");

  int64_t inputFeatureDim = inputType.getDimSize(1);
  int64_t hiddenSize = hPrevType.getDimSize(1);

  if (failed(verifyMatchingStaticDims(
          op, hPrevType.getDimSize(0), inputType.getDimSize(0),
          "h_prev batch dimension", "input batch dimension")))
    return failure();

  if (failed(verifyMatchingStaticShape(op, hType, hPrevType,
                                       "result hidden state",
                                       "previous hidden state")))
    return failure();

  if (failed(verifyGateExpandedStaticDim(
          op, wIhType.getDimSize(0), hiddenSize, gateCount,
          "w_ih output dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(op, wIhType.getDimSize(1),
                                      inputFeatureDim,
                                      "w_ih input dimension",
                                      "input feature dimension")))
    return failure();

  if (failed(verifyGateExpandedStaticDim(
          op, wHhType.getDimSize(0), hiddenSize, gateCount,
          "w_hh output dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(op, wHhType.getDimSize(1), hiddenSize,
                                      "w_hh hidden dimension",
                                      "hidden state dimension")))
    return failure();

  return verifyRecurrentCellBiasOperands(op, bIh, bHh, hasBias, hiddenSize,
                                         gateCount);
}

struct RecurrentVerifierConfig {
  int64_t gateCount = 1;
  bool hasCellState = false;
};

struct RecurrentStateTypes {
  RankedTensorType h0Type;
  RankedTensorType c0Type;
  RankedTensorType hnType;
  RankedTensorType cnType;
};

struct RecurrentLayerParameterTypes {
  RankedTensorType wIhType;
  RankedTensorType wHhType;
  TypedValue<RankedTensorType> bIh;
  TypedValue<RankedTensorType> bHh;
};

RecurrentStateTypes makeHiddenStateTypes(RankedTensorType h0Type,
                                         RankedTensorType hnType) {
  return RecurrentStateTypes{h0Type, RankedTensorType{}, hnType,
                             RankedTensorType{}};
}

RecurrentStateTypes makeHiddenCellStateTypes(RankedTensorType h0Type,
                                             RankedTensorType c0Type,
                                             RankedTensorType hnType,
                                             RankedTensorType cnType) {
  return RecurrentStateTypes{h0Type, c0Type, hnType, cnType};
}

RecurrentLayerParameterTypes
makeRecurrentLayerParameterTypes(RankedTensorType wIhType,
                                 RankedTensorType wHhType,
                                 TypedValue<RankedTensorType> bIh,
                                 TypedValue<RankedTensorType> bHh) {
  return RecurrentLayerParameterTypes{wIhType, wHhType, bIh, bHh};
}

LogicalResult verifyRecurrentPositiveConfig(Operation *op, bool batchFirst,
                                            int64_t hiddenSize,
                                            int64_t numLayers) {
  if (!batchFirst) {
    return op->emitOpError(
        "expected batch_first = true for the current batch-major convention");
  }

  if (numLayers < 1)
    return op->emitOpError("expected num_layers to be at least 1");

  if (hiddenSize < 1)
    return op->emitOpError("expected hidden_size to be at least 1");

  return success();
}

LogicalResult verifyRecurrentLayerIndex(Operation *op, int64_t layerIndex,
                                        int64_t numLayers) {
  if (layerIndex >= 0 && layerIndex < numLayers)
    return success();

  return op->emitOpError("expected layer_index (")
         << layerIndex << ") to be within [0, " << numLayers << ")";
}

LogicalResult verifyRecurrentGateStaticDim(Operation *op, int64_t dim,
                                           int64_t hiddenSize,
                                           RecurrentVerifierConfig config,
                                           StringRef dimName) {
  if (config.gateCount == 1) {
    return verifyMatchingStaticDims(op, dim, hiddenSize, dimName,
                                    "hidden_size");
  }

  return verifyGateExpandedStaticDim(op, dim, hiddenSize, config.gateCount,
                                     dimName);
}

LogicalResult verifyRecurrentOperandCount(Operation *op, ValueRange operands,
                                          bool hasBias, int64_t numLayers) {
  int64_t operandsPerLayer = hasBias ? 4 : 2;
  if (numLayers > std::numeric_limits<int64_t>::max() / operandsPerLayer)
    return op->emitOpError("expected num_layers to be small enough to verify");

  int64_t expectedOperandCount = numLayers * operandsPerLayer;
  if (static_cast<int64_t>(operands.size()) == expectedOperandCount)
    return success();

  return op->emitOpError("expected recurrent operand count (")
         << operands.size() << ") to match " << expectedOperandCount
         << " for " << numLayers << " layers with has_bias = "
         << (hasBias ? "true" : "false");
}

LogicalResult verifyRecurrentWeightPair(Operation *op,
                                        RankedTensorType wIhType,
                                        RankedTensorType wHhType,
                                        int64_t expectedInputSize,
                                        int64_t hiddenSize,
                                        RecurrentVerifierConfig config) {
  if (failed(verifyTensorRank(op, wIhType, 2, "w_ih")))
    return failure();

  if (failed(verifyTensorRank(op, wHhType, 2, "w_hh")))
    return failure();

  if (failed(verifyRecurrentGateStaticDim(op, wIhType.getDimSize(0),
                                          hiddenSize, config,
                                          "w_ih output dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(op, wIhType.getDimSize(1),
                                      expectedInputSize,
                                      "w_ih input dimension",
                                      "expected layer input dimension")))
    return failure();

  if (failed(verifyRecurrentGateStaticDim(op, wHhType.getDimSize(0),
                                          hiddenSize, config,
                                          "w_hh output dimension")))
    return failure();

  return verifyMatchingStaticDims(op, wHhType.getDimSize(1), hiddenSize,
                                  "w_hh hidden dimension", "hidden_size");
}

LogicalResult verifyRecurrentBiasPair(Operation *op,
                                      RankedTensorType bIhType,
                                      RankedTensorType bHhType,
                                      int64_t hiddenSize,
                                      RecurrentVerifierConfig config) {
  if (failed(verifyTensorRank(op, bIhType, 1, "b_ih")))
    return failure();

  if (failed(verifyTensorRank(op, bHhType, 1, "b_hh")))
    return failure();

  if (failed(verifyRecurrentGateStaticDim(
          op, bIhType.getDimSize(0), hiddenSize, config, "b_ih dimension")))
    return failure();

  return verifyRecurrentGateStaticDim(op, bHhType.getDimSize(0), hiddenSize,
                                      config, "b_hh dimension");
}

LogicalResult verifyRecurrentLayerBiasOperands(
    Operation *op, TypedValue<RankedTensorType> bIh,
    TypedValue<RankedTensorType> bHh, bool hasBias, int64_t hiddenSize,
    RecurrentVerifierConfig config) {
  if (hasBias && (!bIh || !bHh)) {
    return op->emitOpError(
        "expected has_bias = true to include b_ih and b_hh operands");
  }

  if (!hasBias && (bIh || bHh)) {
    return op->emitOpError(
        "expected has_bias = false to omit bias operands");
  }

  if (!bIh && !bHh)
    return success();

  return verifyRecurrentBiasPair(op, bIh.getType(), bHh.getType(), hiddenSize,
                                 config);
}

LogicalResult verifyRecurrentRank3Types(Operation *op,
                                        RankedTensorType inputType,
                                        RankedTensorType outputType,
                                        RecurrentStateTypes states,
                                        RecurrentVerifierConfig config) {
  if (failed(verifyTensorRank(op, inputType, 3, "input")))
    return failure();

  if (failed(verifyTensorRank(op, outputType, 3, "output")))
    return failure();

  if (failed(verifyTensorRank(op, states.h0Type, 3, "h0")))
    return failure();

  if (config.hasCellState &&
      failed(verifyTensorRank(op, states.c0Type, 3, "c0")))
    return failure();

  if (failed(verifyTensorRank(op, states.hnType, 3, "hn")))
    return failure();

  if (config.hasCellState &&
      failed(verifyTensorRank(op, states.cnType, 3, "cn")))
    return failure();

  return success();
}

LogicalResult verifyRecurrentOutputShape(Operation *op,
                                         RankedTensorType inputType,
                                         RankedTensorType outputType,
                                         int64_t hiddenSize) {
  if (failed(verifyMatchingStaticDims(op, outputType.getDimSize(0),
                                      inputType.getDimSize(0),
                                      "output batch dimension",
                                      "input batch dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(op, outputType.getDimSize(1),
                                      inputType.getDimSize(1),
                                      "output sequence dimension",
                                      "input sequence dimension")))
    return failure();

  return verifyMatchingStaticDims(op, outputType.getDimSize(2), hiddenSize,
                                  "output hidden dimension", "hidden_size");
}

LogicalResult verifyRecurrentInitialStateShape(Operation *op,
                                               RankedTensorType inputType,
                                               RankedTensorType h0Type,
                                               int64_t hiddenSize,
                                               int64_t expectedLayers,
                                               StringRef layerName) {
  if (failed(verifyMatchingStaticDims(op, h0Type.getDimSize(0),
                                      expectedLayers, "h0 layer dimension",
                                      layerName)))
    return failure();

  if (failed(verifyMatchingStaticDims(op, h0Type.getDimSize(1),
                                      inputType.getDimSize(0),
                                      "h0 batch dimension",
                                      "input batch dimension")))
    return failure();

  return verifyMatchingStaticDims(op, h0Type.getDimSize(2), hiddenSize,
                                  "h0 hidden dimension", "hidden_size");
}

LogicalResult verifyRecurrentFinalStateShape(Operation *op,
                                             RankedTensorType inputType,
                                             RankedTensorType hnType,
                                             int64_t hiddenSize,
                                             int64_t expectedLayers,
                                             StringRef layerName) {
  if (failed(verifyMatchingStaticDims(op, hnType.getDimSize(0),
                                      expectedLayers, "hn layer dimension",
                                      layerName)))
    return failure();

  if (failed(verifyMatchingStaticDims(op, hnType.getDimSize(1),
                                      inputType.getDimSize(0),
                                      "hn batch dimension",
                                      "input batch dimension")))
    return failure();

  return verifyMatchingStaticDims(op, hnType.getDimSize(2), hiddenSize,
                                  "hn hidden dimension", "hidden_size");
}

LogicalResult verifyRecurrentCellStateMatchesHidden(
    Operation *op, RecurrentStateTypes states, RecurrentVerifierConfig config) {
  if (!config.hasCellState)
    return success();

  if (failed(verifyMatchingStaticShape(op, states.c0Type, states.h0Type,
                                       "initial cell state",
                                       "initial hidden state")))
    return failure();

  return verifyMatchingStaticShape(op, states.cnType, states.hnType,
                                   "final cell state", "final hidden state");
}

LogicalResult verifyRecurrentModuleOperands(Operation *op, ValueRange operands,
                                            bool hasBias, int64_t inputSize,
                                            int64_t hiddenSize,
                                            int64_t numLayers,
                                            RecurrentVerifierConfig config) {
  int64_t operandsPerLayer = hasBias ? 4 : 2;
  for (int64_t layer = 0; layer < numLayers; ++layer) {
    int64_t operandBase = layer * operandsPerLayer;
    auto wIhType =
        llvm::cast<RankedTensorType>(operands[operandBase].getType());
    auto wHhType =
        llvm::cast<RankedTensorType>(operands[operandBase + 1].getType());
    int64_t expectedInputSize = layer == 0 ? inputSize : hiddenSize;
    if (failed(verifyRecurrentWeightPair(op, wIhType, wHhType,
                                         expectedInputSize, hiddenSize,
                                         config)))
      return failure();

    if (!hasBias)
      continue;

    auto bIhType =
        llvm::cast<RankedTensorType>(operands[operandBase + 2].getType());
    auto bHhType =
        llvm::cast<RankedTensorType>(operands[operandBase + 3].getType());
    if (failed(verifyRecurrentBiasPair(op, bIhType, bHhType, hiddenSize,
                                       config)))
      return failure();
  }

  return success();
}

LogicalResult verifyRecurrentLayerOp(
    Operation *op, RankedTensorType inputType, RankedTensorType outputType,
    RecurrentStateTypes states, RecurrentLayerParameterTypes parameters,
    bool batchFirst, bool hasBias, int64_t hiddenSize, int64_t layerIndex,
    int64_t numLayers, RecurrentVerifierConfig config) {
  if (failed(verifyRecurrentPositiveConfig(op, batchFirst, hiddenSize,
                                           numLayers)))
    return failure();

  if (failed(verifyRecurrentLayerIndex(op, layerIndex, numLayers)))
    return failure();

  if (failed(verifyRecurrentRank3Types(op, inputType, outputType, states,
                                       config)))
    return failure();

  if (failed(verifyRecurrentWeightPair(op, parameters.wIhType,
                                       parameters.wHhType,
                                       inputType.getDimSize(2), hiddenSize,
                                       config)))
    return failure();

  if (failed(verifyRecurrentLayerBiasOperands(
          op, parameters.bIh, parameters.bHh, hasBias, hiddenSize, config)))
    return failure();

  if (failed(verifyRecurrentCellStateMatchesHidden(op, states, config)))
    return failure();

  if (failed(verifyRecurrentOutputShape(op, inputType, outputType,
                                        hiddenSize)))
    return failure();

  if (failed(verifyRecurrentInitialStateShape(op, inputType, states.h0Type,
                                              hiddenSize, numLayers,
                                              "num_layers")))
    return failure();

  return verifyRecurrentFinalStateShape(op, inputType, states.hnType,
                                        hiddenSize, /*expectedLayers=*/1,
                                        "one layer");
}

LogicalResult verifyRecurrentModuleOp(
    Operation *op, RankedTensorType inputType, ValueRange operands,
    RankedTensorType outputType, RecurrentStateTypes states, bool batchFirst,
    bool hasBias, int64_t hiddenSize, int64_t numLayers,
    RecurrentVerifierConfig config) {
  if (failed(verifyRecurrentPositiveConfig(op, batchFirst, hiddenSize,
                                           numLayers)))
    return failure();

  if (failed(verifyRecurrentRank3Types(op, inputType, outputType, states,
                                       config)))
    return failure();

  if (failed(verifyRecurrentOperandCount(op, operands, hasBias, numLayers)))
    return failure();

  if (failed(verifyRecurrentOutputShape(op, inputType, outputType,
                                        hiddenSize)))
    return failure();

  if (failed(verifyRecurrentCellStateMatchesHidden(op, states, config)))
    return failure();

  if (failed(verifyRecurrentInitialStateShape(op, inputType, states.h0Type,
                                              hiddenSize, numLayers,
                                              "num_layers")))
    return failure();

  if (failed(verifyRecurrentFinalStateShape(op, inputType, states.hnType,
                                            hiddenSize, numLayers,
                                            "num_layers")))
    return failure();

  return verifyRecurrentModuleOperands(op, operands, hasBias,
                                       inputType.getDimSize(2), hiddenSize,
                                       numLayers, config);
}

} // namespace

// Enforces the canonical linear contract while preserving dynamic-dimension
// permissiveness for shapes that cannot be compared statically.
mlir::LogicalResult mlir::sculptor::NNLinearOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType weightType = getWeight().getType();
  RankedTensorType resultType = getResult().getType();
  TypedValue<RankedTensorType> bias = getBias();

  if (inputType.getRank() < 1)
    return emitOpError("expected input tensor to have rank at least 1");

  if (weightType.getRank() != 2)
    return emitOpError("expected weight tensor to have rank 2");

  if (resultType.getRank() != inputType.getRank()) {
    return emitOpError(
        "expected result tensor rank to match input tensor rank");
  }

  if (getHasBias() && !bias)
    return emitOpError("expected has_bias = true to include a bias operand");

  if (!getHasBias() && bias)
    return emitOpError("expected has_bias = false to omit the bias operand");

  if (bias && bias.getType().getRank() != 1)
    return emitOpError("expected bias tensor to have rank 1");

  if (failed(verifyMatchingStaticDims(
          *this, inputType.getDimSize(inputType.getRank() - 1),
          weightType.getDimSize(1), "input feature dimension",
          "weight input dimension")))
    return failure();

  if (failed(verifyMatchingStaticDims(
          *this, resultType.getDimSize(resultType.getRank() - 1),
          weightType.getDimSize(0), "result output dimension",
          "weight output dimension")))
    return failure();

  for (int64_t dim = 0, rank = inputType.getRank() - 1; dim < rank; ++dim) {
    if (failed(verifyMatchingStaticDims(
            *this, resultType.getDimSize(dim), inputType.getDimSize(dim),
            "result batch dimension", "input batch dimension")))
      return failure();
  }

  if (bias && failed(verifyMatchingStaticDims(
                  *this, bias.getType().getDimSize(0), weightType.getDimSize(0),
                  "bias dimension", "weight output dimension")))
    return failure();

  return success();
}

// Enforces the canonical non-grouped Conv1D contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNConv1DOp::verify() {
  return verifyConvolutionOp(*this, getInput().getType(), getWeight().getType(),
                             getBias(), getResult().getType(), getHasBias(),
                             getStride(), getPadding(), getDilation(),
                             /*spatialRank=*/1);
}

// Enforces the canonical non-grouped Conv2D contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNConv2DOp::verify() {
  return verifyConvolutionOp(*this, getInput().getType(), getWeight().getType(),
                             getBias(), getResult().getType(), getHasBias(),
                             getStride(), getPadding(), getDilation(),
                             /*spatialRank=*/2);
}

// Enforces the canonical grouped Conv2D contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNGroupedConv2DOp::verify() {
  return verifyGroupedConv2DOp(*this, getInput().getType(),
                               getWeight().getType(), getBias(),
                               getResult().getType(), getHasBias(),
                               getGroupsAttr().getInt(), getStride(),
                               getPadding(), getDilation());
}

// Enforces the canonical non-grouped Conv3D contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNConv3DOp::verify() {
  return verifyConvolutionOp(*this, getInput().getType(), getWeight().getType(),
                             getBias(), getResult().getType(), getHasBias(),
                             getStride(), getPadding(), getDilation(),
                             /*spatialRank=*/3);
}

// Enforces the canonical RNN cell contract while preserving dynamic-dimension
// permissiveness for shapes that cannot be compared statically.
mlir::LogicalResult mlir::sculptor::NNRNNCellOp::verify() {
  return verifyRecurrentCellOp(*this, getInput().getType(),
                               getHPrev().getType(), getWIh().getType(),
                               getWHh().getType(), getBIh(), getBHh(),
                               getH().getType(), getHasBias(),
                               /*gateCount=*/1);
}

// Enforces the canonical LSTM cell contract while preserving dynamic-dimension
// permissiveness for shapes that cannot be compared statically.
mlir::LogicalResult mlir::sculptor::NNLSTMCellOp::verify() {
  RankedTensorType hPrevType = getHPrev().getType();
  RankedTensorType cPrevType = getCPrev().getType();
  RankedTensorType cType = getC().getType();

  if (cPrevType.getRank() != 2)
    return emitOpError("expected c_prev tensor rank to be 2");

  if (cType.getRank() != 2)
    return emitOpError("expected result cell state tensor rank to be 2");

  if (failed(verifyRecurrentCellOp(
          *this, getInput().getType(), hPrevType, getWIh().getType(),
          getWHh().getType(), getBIh(), getBHh(), getH().getType(),
          getHasBias(), /*gateCount=*/4)))
    return failure();

  if (failed(verifyMatchingStaticShape(*this, cPrevType, hPrevType,
                                       "previous cell state",
                                       "previous hidden state")))
    return failure();

  return verifyMatchingStaticShape(*this, cType, hPrevType,
                                   "result cell state",
                                   "previous hidden state");
}

// Enforces the canonical GRU cell contract while preserving dynamic-dimension
// permissiveness for shapes that cannot be compared statically.
mlir::LogicalResult mlir::sculptor::NNGRUCellOp::verify() {
  return verifyRecurrentCellOp(*this, getInput().getType(),
                               getHPrev().getType(), getWIh().getType(),
                               getWHh().getType(), getBIh(), getBHh(),
                               getH().getType(), getHasBias(),
                               /*gateCount=*/3);
}

// Enforces the canonical GRU contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNGRUOp::verify() {
  return verifyRecurrentModuleOp(
      *this, getInput().getType(), getRecurrentOperands(),
      getOutput().getType(),
      makeHiddenStateTypes(getH0().getType(), getHn().getType()),
      getBatchFirst(), getHasBias(), getHiddenSizeAttr().getInt(),
      getNumLayersAttr().getInt(),
      RecurrentVerifierConfig{/*gateCount=*/3, /*hasCellState=*/false});
}

// Enforces the canonical RNN contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNRNNOp::verify() {
  return verifyRecurrentModuleOp(
      *this, getInput().getType(), getRecurrentOperands(),
      getOutput().getType(),
      makeHiddenStateTypes(getH0().getType(), getHn().getType()),
      getBatchFirst(), getHasBias(), getHiddenSizeAttr().getInt(),
      getNumLayersAttr().getInt(),
      RecurrentVerifierConfig{/*gateCount=*/1, /*hasCellState=*/false});
}

// Enforces the canonical LSTM contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNLSTMOp::verify() {
  return verifyRecurrentModuleOp(
      *this, getInput().getType(), getRecurrentOperands(),
      getOutput().getType(),
      makeHiddenCellStateTypes(getH0().getType(), getC0().getType(),
                               getHn().getType(), getCn().getType()),
      getBatchFirst(), getHasBias(), getHiddenSizeAttr().getInt(),
      getNumLayersAttr().getInt(),
      RecurrentVerifierConfig{/*gateCount=*/4, /*hasCellState=*/true});
}

// Enforces the canonical one-layer RNN contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNRNNLayerOp::verify() {
  return verifyRecurrentLayerOp(
      *this, getInput().getType(), getOutput().getType(),
      makeHiddenStateTypes(getH0().getType(), getHn().getType()),
      makeRecurrentLayerParameterTypes(getWIh().getType(), getWHh().getType(),
                                       getBIh(), getBHh()),
      getBatchFirst(), getHasBias(), getHiddenSizeAttr().getInt(),
      getLayerIndexAttr().getInt(), getNumLayersAttr().getInt(),
      RecurrentVerifierConfig{/*gateCount=*/1, /*hasCellState=*/false});
}

// Enforces the canonical one-layer GRU contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNGRULayerOp::verify() {
  return verifyRecurrentLayerOp(
      *this, getInput().getType(), getOutput().getType(),
      makeHiddenStateTypes(getH0().getType(), getHn().getType()),
      makeRecurrentLayerParameterTypes(getWIh().getType(), getWHh().getType(),
                                       getBIh(), getBHh()),
      getBatchFirst(), getHasBias(), getHiddenSizeAttr().getInt(),
      getLayerIndexAttr().getInt(), getNumLayersAttr().getInt(),
      RecurrentVerifierConfig{/*gateCount=*/3, /*hasCellState=*/false});
}

// Enforces the canonical one-layer LSTM contract while preserving
// dynamic-dimension permissiveness for shapes that cannot be compared
// statically.
mlir::LogicalResult mlir::sculptor::NNLSTMLayerOp::verify() {
  return verifyRecurrentLayerOp(
      *this, getInput().getType(), getOutput().getType(),
      makeHiddenCellStateTypes(getH0().getType(), getC0().getType(),
                               getHn().getType(), getCn().getType()),
      makeRecurrentLayerParameterTypes(getWIh().getType(), getWHh().getType(),
                                       getBIh(), getBHh()),
      getBatchFirst(), getHasBias(), getHiddenSizeAttr().getInt(),
      getLayerIndexAttr().getInt(), getNumLayersAttr().getInt(),
      RecurrentVerifierConfig{/*gateCount=*/4, /*hasCellState=*/true});
}
