#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_CONVOLUTIONLAYERPATTERNS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_CONVOLUTIONLAYERPATTERNS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/MatchedSubgraphUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <optional>

namespace mlir {
namespace sculptor {
namespace layer_patterns {

// Carries the ops and boundary values that make up one direct convolution
// layer slice.
struct DirectConvolutionMatch {
  Value input;
  Operation *weightConstant = nullptr;
  Operation *biasConstant = nullptr;
  Operation *outputEmpty = nullptr;
  Operation *outputFill = nullptr;
  Operation *outputFillConstant = nullptr;
  Operation *outputBroadcast = nullptr;
  Operation *convOp = nullptr;

  Operation *root = nullptr;
  llvm::SmallVector<Operation *> ops;
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> outputs;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> dilations;
};

// Carries the grouped Conv2D subgraph and boundary values.
struct Conv2DGroupedMatch {
  Value input;
  Operation *inputExpand = nullptr;
  Operation *groupedConvOp = nullptr;
  Operation *weightConstant = nullptr;
  Operation *weightExpand = nullptr;
  Operation *biasConstant = nullptr;
  Operation *outputEmpty = nullptr;
  Operation *outputFill = nullptr;
  Operation *outputFillConstant = nullptr;
  Operation *outputBroadcast = nullptr;
  Operation *outputExpand = nullptr;
  Operation *outputCollapse = nullptr;

  Operation *root = nullptr;
  llvm::SmallVector<Operation *> ops;
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> outputs;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> dilations;
  int64_t groups = 0;
};

// Identifies the non-weight convolution input after the constant weight side is
// known. The named linalg conv ops have exactly two inputs before the output
// init operand.
inline Value findDirectConvolutionInput(Operation *op,
                                        Operation *weightConstant) {
  if (!op || !weightConstant || op->getNumOperands() < 2)
    return {};

  if (layer_utils::operandProducer(op, 0) == weightConstant)
    return op->getOperand(1);

  return op->getOperand(0);
}

// Records the linalg convolution spatial attributes in the canonical sculptor.nn
// array-attribute form. Direct convolution fixtures currently have no padding.
inline bool populateDirectConvolutionSpatialAttrs(DirectConvolutionMatch &match,
                                                  Operation *op,
                                                  unsigned spatialRank) {
  if (!extractDenseI64ArrayAttr(op, "strides", spatialRank, match.strides))
    return false;

  if (!extractDenseI64ArrayAttr(op, "dilations", spatialRank, match.dilations))
    return false;

  match.padding.assign(spatialRank, 0);
  return true;
}

// Collects a matched direct-convolution slice in clone order for outlining.
inline void collectDirectConvolutionOps(DirectConvolutionMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.weightConstant);
  match_utils::appendUniqueOp(match.ops, match.biasConstant);
  match_utils::appendUniqueOp(match.ops, match.outputEmpty);
  match_utils::appendUniqueOp(match.ops, match.outputFillConstant);
  match_utils::appendUniqueOp(match.ops, match.outputFill);
  match_utils::appendUniqueOp(match.ops, match.outputBroadcast);
  match_utils::appendUniqueOp(match.ops, match.convOp);
}

// Computes external inputs and root outputs after all matched ops are known.
inline void finalizeDirectConvolutionMatch(DirectConvolutionMatch &match) {
  collectDirectConvolutionOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Recognizes a direct convolution whose output init broadcasts a constant bias.
template <typename WeightMatcher>
std::optional<DirectConvolutionMatch>
matchDirectConvolutionWithBias(Operation *op, llvm::StringRef opName,
                               unsigned spatialRank,
                               WeightMatcher matchWeight) {
  if (!op || op->getName().getStringRef() != opName)
    return std::nullopt;

  if (!layer_utils::hasDpsInputsAndOperands(op, 2, 3))
    return std::nullopt;

  auto outputInit = matchBroadcastOutputInit(op, 2);
  if (!outputInit)
    return std::nullopt;

  Operation *weightConstant = matchWeight(op);
  if (!weightConstant)
    return std::nullopt;

  DirectConvolutionMatch match;
  match.root = op;
  match.convOp = op;
  match.input = findDirectConvolutionInput(op, weightConstant);
  if (!match.input)
    return std::nullopt;

  match.biasConstant = outputInit->biasConstant;
  match.outputEmpty = outputInit->outputEmpty;
  match.outputBroadcast = outputInit->outputBroadcast;
  match.weightConstant = weightConstant;
  if (!populateDirectConvolutionSpatialAttrs(match, op, spatialRank))
    return std::nullopt;

  finalizeDirectConvolutionMatch(match);
  return match;
}

// Recognizes a direct convolution whose output init is a fill rather than bias.
template <typename WeightMatcher>
std::optional<DirectConvolutionMatch>
matchDirectConvolutionWithoutBias(Operation *op, llvm::StringRef opName,
                                  unsigned spatialRank,
                                  WeightMatcher matchWeight) {
  if (!op || op->getName().getStringRef() != opName)
    return std::nullopt;

  if (!layer_utils::hasDpsInputsAndOperands(op, 2, 3))
    return std::nullopt;

  auto outputInit = matchFillOutputInit(op, 2);
  if (!outputInit)
    return std::nullopt;

  Operation *weightConstant = matchWeight(op);
  if (!weightConstant)
    return std::nullopt;

  DirectConvolutionMatch match;
  match.root = op;
  match.convOp = op;
  match.input = findDirectConvolutionInput(op, weightConstant);
  if (!match.input)
    return std::nullopt;

  match.outputEmpty = outputInit->outputEmpty;
  match.outputFill = outputInit->outputFill;
  match.outputFillConstant = outputInit->outputFillConstant;
  match.weightConstant = weightConstant;
  if (!populateDirectConvolutionSpatialAttrs(match, op, spatialRank))
    return std::nullopt;

  finalizeDirectConvolutionMatch(match);
  return match;
}

// Requires exactly one convolution input to be the captured weight constant.
inline Operation *matchConv1DWeightInput(Operation *op) {
  auto weightInput = matchSingleConstantInputPair(op);
  return weightInput ? weightInput->constant : nullptr;
}

// Accepts the static weight from either convolution input position.
inline Operation *matchConv2DWeightInput(Operation *op) {
  return findConstantInput(op, 1, 0);
}

// Requires exactly one static weight among the two convolution inputs.
inline Operation *matchConv3DWeightInput(Operation *op) {
  auto weightInput = matchSingleConstantInputPair(op);
  return weightInput ? weightInput->constant : nullptr;
}

inline std::optional<int64_t> getStaticResultDim(Operation *op, unsigned dim) {
  if (!op || op->getNumResults() != 1)
    return std::nullopt;

  auto type = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || dim >= static_cast<unsigned>(type.getRank()))
    return std::nullopt;

  int64_t value = type.getDimSize(dim);
  if (ShapedType::isDynamic(value))
    return std::nullopt;

  return value;
}

inline bool updateGroupCount(int64_t candidate, int64_t &groups) {
  if (candidate <= 1)
    return false;

  if (groups == 0) {
    groups = candidate;
    return true;
  }

  return groups == candidate;
}

inline bool populateConv2DGroupedCanonicalAttrs(Conv2DGroupedMatch &match) {
  if (!match.inputExpand)
    return false;

  auto inputExpand = llvm::dyn_cast<tensor::ExpandShapeOp>(match.inputExpand);
  if (!inputExpand)
    return false;
  match.input = inputExpand.getSrc();

  if (!extractDenseI64ArrayAttr(match.groupedConvOp, "strides",
                                /*expectedSize=*/2, match.strides))
    return false;

  if (!extractDenseI64ArrayAttr(match.groupedConvOp, "dilations",
                                /*expectedSize=*/2, match.dilations))
    return false;

  match.padding.assign(2, 0);

  int64_t groups = 0;
  std::optional<int64_t> inputGroups = getStaticResultDim(match.inputExpand, 1);
  std::optional<int64_t> weightGroups =
      getStaticResultDim(match.weightExpand, 0);
  std::optional<int64_t> outputGroups =
      getStaticResultDim(match.outputExpand, 1);
  std::optional<int64_t> convGroups =
      getStaticResultDim(match.groupedConvOp, 1);

  if (!inputGroups || !weightGroups || !outputGroups || !convGroups)
    return false;

  if (!updateGroupCount(*inputGroups, groups) ||
      !updateGroupCount(*weightGroups, groups) ||
      !updateGroupCount(*outputGroups, groups) ||
      !updateGroupCount(*convGroups, groups))
    return false;

  match.groups = groups;
  return true;
}

// Collects the matched grouped Conv2D slice in clone order for outlining.
inline void collectConv2DGroupedOps(Conv2DGroupedMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.inputExpand);
  match_utils::appendUniqueOp(match.ops, match.weightConstant);
  match_utils::appendUniqueOp(match.ops, match.weightExpand);
  match_utils::appendUniqueOp(match.ops, match.biasConstant);
  match_utils::appendUniqueOp(match.ops, match.outputEmpty);
  match_utils::appendUniqueOp(match.ops, match.outputFillConstant);
  match_utils::appendUniqueOp(match.ops, match.outputBroadcast);
  match_utils::appendUniqueOp(match.ops, match.outputExpand);
  match_utils::appendUniqueOp(match.ops, match.outputFill);
  match_utils::appendUniqueOp(match.ops, match.groupedConvOp);
  match_utils::appendUniqueOp(match.ops, match.outputCollapse);
}

// Computes external inputs and root outputs after optional ops are ordered.
inline void finalizeConv2DGroupedMatch(Conv2DGroupedMatch &match) {
  if (!populateConv2DGroupedCanonicalAttrs(match))
    return;

  collectConv2DGroupedOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Recognizes grouped Conv2D layers whose output init broadcasts a bias.
inline std::optional<Conv2DGroupedMatch>
matchConv2DGroupedWithBias(Operation *op) {
  auto outputCollapse = llvm::dyn_cast_or_null<tensor::CollapseShapeOp>(op);
  if (!outputCollapse ||
      !layer_utils::hasOperands(outputCollapse.getOperation(), 1))
    return std::nullopt;

  Operation *groupedConvOp = layer_utils::producerOf(outputCollapse.getSrc());
  if (!groupedConvOp ||
      groupedConvOp->getName().getStringRef() != "linalg.conv_2d_ngchw_gfchw" ||
      !layer_utils::hasDpsInputsAndOperands(groupedConvOp, 2, 3))
    return std::nullopt;

  auto outputExpand = layer_utils::operandProducerOfType<tensor::ExpandShapeOp>(
      groupedConvOp, 2);
  if (!outputExpand ||
      !layer_utils::hasOperands(outputExpand.getOperation(), 1))
    return std::nullopt;

  auto outputInit = matchBroadcastOutputInit(outputExpand.getSrc());
  if (!outputInit)
    return std::nullopt;

  auto weightMatch = matchExpandedInputAndWeight(groupedConvOp);
  if (!weightMatch)
    return std::nullopt;

  Conv2DGroupedMatch match;
  match.root = outputCollapse.getOperation();
  match.outputCollapse = outputCollapse.getOperation();
  match.groupedConvOp = groupedConvOp;
  match.inputExpand = weightMatch->inputExpand;
  match.weightConstant = weightMatch->weightConstant;
  match.weightExpand = weightMatch->weightExpand;
  match.biasConstant = outputInit->biasConstant;
  match.outputEmpty = outputInit->outputEmpty;
  match.outputExpand = outputExpand.getOperation();
  match.outputBroadcast = outputInit->outputBroadcast;

  finalizeConv2DGroupedMatch(match);
  if (match.ops.empty())
    return std::nullopt;
  return match;
}

// Recognizes grouped Conv2D layers whose output init is a fill, not a bias.
inline std::optional<Conv2DGroupedMatch>
matchConv2DGroupedWithoutBias(Operation *op) {
  auto outputCollapse = llvm::dyn_cast_or_null<tensor::CollapseShapeOp>(op);
  if (!outputCollapse ||
      !layer_utils::hasOperands(outputCollapse.getOperation(), 1))
    return std::nullopt;

  Operation *groupedConvOp = layer_utils::producerOf(outputCollapse.getSrc());
  if (!groupedConvOp ||
      groupedConvOp->getName().getStringRef() != "linalg.conv_2d_ngchw_gfchw" ||
      !layer_utils::hasDpsInputsAndOperands(groupedConvOp, 2, 3))
    return std::nullopt;

  auto outputFill =
      layer_utils::operandProducerOfType<linalg::FillOp>(groupedConvOp, 2);
  if (!outputFill ||
      !layer_utils::hasDpsInputsAndOperands(outputFill.getOperation(), 1, 2))
    return std::nullopt;

  auto outputFillConstant =
      layer_utils::operandProducerOfType<arith::ConstantOp>(
          outputFill.getOperation(), 0);
  auto outputExpand = layer_utils::operandProducerOfType<tensor::ExpandShapeOp>(
      outputFill.getOperation(), 1);
  if (!outputFillConstant || !outputExpand ||
      !layer_utils::hasOperands(outputExpand.getOperation(), 1))
    return std::nullopt;

  auto outputEmpty =
      layer_utils::producerOfType<tensor::EmptyOp>(outputExpand.getSrc());
  if (!outputEmpty)
    return std::nullopt;

  auto weightMatch = matchExpandedInputAndWeight(groupedConvOp);
  if (!weightMatch)
    return std::nullopt;

  Conv2DGroupedMatch match;
  match.root = outputCollapse.getOperation();
  match.inputExpand = weightMatch->inputExpand;
  match.groupedConvOp = groupedConvOp;
  match.weightConstant = weightMatch->weightConstant;
  match.weightExpand = weightMatch->weightExpand;
  match.outputEmpty = outputEmpty.getOperation();
  match.outputFill = outputFill.getOperation();
  match.outputFillConstant = outputFillConstant.getOperation();
  match.outputExpand = outputExpand.getOperation();
  match.outputCollapse = outputCollapse.getOperation();

  finalizeConv2DGroupedMatch(match);
  if (match.ops.empty())
    return std::nullopt;
  return match;
}

} // namespace layer_patterns
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_CONVOLUTIONLAYERPATTERNS_H
