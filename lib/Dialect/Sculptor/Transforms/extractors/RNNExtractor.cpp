#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Extraction/RewriteUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <optional>
#include <string>

namespace layer_utils = mlir::sculptor::layer_utils;
namespace rewrite_utils = mlir::sculptor::rewrite_utils;

namespace {

using mlir::sculptor::NNRNNLayerOp;
using mlir::sculptor::NNRNNOp;

struct RNNMatch {
  bool hasBias = false;
};

struct SequenceAssembly {
  mlir::tensor::ConcatOp concat;
};

struct HiddenAssembly {
  mlir::tensor::ConcatOp concat;
};

struct RecurrentStepMatch {
  mlir::Value inputSliceSource;
  int64_t inputSliceIndex = -1;
  mlir::Value recurrentState;
  bool hasBias = false;
};

struct CanonicalRNNInfo {
  mlir::RankedTensorType inputType;
  mlir::RankedTensorType hiddenStateType;
  mlir::RankedTensorType sequenceResultType;
  mlir::RankedTensorType hiddenResultType;
  mlir::RankedTensorType hiddenSliceType;
  int64_t layerCount = 0;
  int64_t batchSize = 0;
  int64_t sequenceLength = 0;
  int64_t hiddenSize = 0;
  bool hasBias = false;
};

static std::optional<mlir::RankedTensorType>
getRankedTensorType(mlir::Type type) {
  auto rankedType = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!rankedType || !rankedType.hasStaticShape())
    return std::nullopt;
  return rankedType;
}

static bool isTensorConstantOfRank(mlir::Operation *op, unsigned rank) {
  auto constant = llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(op);
  if (!constant)
    return false;

  auto rankedType = getRankedTensorType(constant.getType());
  return rankedType && rankedType->getRank() == static_cast<int64_t>(rank);
}

static bool isSliceFromValueAtLayer(mlir::Operation *op, mlir::Value source,
                                    int64_t layer) {
  auto slice = llvm::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(op);
  if (!slice || slice.getSource() != source)
    return false;

  llvm::ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  llvm::ArrayRef<int64_t> sizes = slice.getStaticSizes();
  llvm::ArrayRef<int64_t> strides = slice.getStaticStrides();
  if (offsets.size() != 3 || sizes.size() != 3 || strides.size() != 3)
    return false;

  return offsets[0] == layer && offsets[1] == 0 && offsets[2] == 0 &&
         sizes[0] == 1 && strides[0] == 1 && strides[1] == 1 && strides[2] == 1;
}

static std::optional<int64_t> getLeadingSliceIndex(mlir::Operation *op) {
  auto slice = llvm::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(op);
  if (!slice)
    return std::nullopt;

  llvm::ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  llvm::ArrayRef<int64_t> sizes = slice.getStaticSizes();
  llvm::ArrayRef<int64_t> strides = slice.getStaticStrides();
  if (offsets.size() != 3 || sizes.size() != 3 || strides.size() != 3)
    return std::nullopt;

  if (offsets[1] != 0 || offsets[2] != 0 || sizes[0] != 1 || strides[0] != 1 ||
      strides[1] != 1 || strides[2] != 1)
    return std::nullopt;

  return offsets[0];
}

static bool hasShape(mlir::Value value, llvm::ArrayRef<int64_t> shape) {
  auto rankedType = getRankedTensorType(value.getType());
  return rankedType && rankedType->getShape() == shape;
}

static std::optional<SequenceAssembly>
matchFinalSequenceAssembly(mlir::Value value, int64_t sequenceLength,
                           int64_t batchSize, int64_t hiddenSize) {
  auto transpose =
      layer_utils::producerOfType<mlir::linalg::TransposeOp>(value);
  if (!transpose)
    return std::nullopt;

  auto sourceType = getRankedTensorType(transpose.getInput().getType());
  auto resultType = getRankedTensorType(transpose.getResult()[0].getType());
  if (!sourceType || !resultType ||
      sourceType->getShape() !=
          llvm::ArrayRef<int64_t>({sequenceLength, batchSize, hiddenSize}) ||
      resultType->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}))
    return std::nullopt;

  mlir::SmallVector<int64_t> permutation;
  for (int64_t index : transpose.getPermutation())
    permutation.push_back(index);

  if (permutation != llvm::ArrayRef<int64_t>({1, 0, 2}))
    return std::nullopt;

  auto concat =
      layer_utils::producerOfType<mlir::tensor::ConcatOp>(transpose.getInput());
  if (!concat || concat.getDim() != 0 ||
      concat.getInputs().size() != static_cast<size_t>(sequenceLength))
    return std::nullopt;

  return SequenceAssembly{concat};
}

static std::optional<HiddenAssembly>
matchFinalHiddenAssembly(mlir::Value value, int64_t layerCount,
                         int64_t batchSize, int64_t hiddenSize) {
  auto expand = layer_utils::producerOfType<mlir::tensor::ExpandShapeOp>(value);
  if (!expand)
    return std::nullopt;

  auto sourceType = getRankedTensorType(expand.getSrc().getType());
  auto resultType = getRankedTensorType(expand.getResult().getType());
  if (!sourceType || !resultType ||
      sourceType->getShape() !=
          llvm::ArrayRef<int64_t>({layerCount * batchSize, hiddenSize}) ||
      resultType->getShape() !=
          llvm::ArrayRef<int64_t>({layerCount, batchSize, hiddenSize}))
    return std::nullopt;

  auto concat =
      layer_utils::producerOfType<mlir::tensor::ConcatOp>(expand.getSrc());
  if (!concat || concat.getDim() != 0 ||
      concat.getInputs().size() != static_cast<size_t>(layerCount))
    return std::nullopt;

  return HiddenAssembly{concat};
}

static bool dependsOnValue(mlir::Value value, mlir::Value target,
                           llvm::SmallPtrSetImpl<mlir::Operation *> &seen) {
  if (value == target)
    return true;

  mlir::Operation *op = layer_utils::producerOf(value);
  if (!op || !seen.insert(op).second)
    return false;

  for (mlir::Value operand : op->getOperands()) {
    if (dependsOnValue(operand, target, seen))
      return true;
  }
  return false;
}

static bool dependsOnValue(mlir::Value value, mlir::Value target) {
  llvm::SmallPtrSet<mlir::Operation *, 32> seen;
  return dependsOnValue(value, target, seen);
}

static bool hasTensorConstantInput(mlir::linalg::GenericOp generic,
                                   unsigned rank) {
  for (mlir::Value input : generic.getInputs()) {
    if (isTensorConstantOfRank(layer_utils::producerOf(input), rank))
      return true;
  }
  return false;
}

static bool isStateMatmul(mlir::Operation *op, mlir::Value &state) {
  auto matmul = llvm::dyn_cast_or_null<mlir::linalg::MatmulOp>(op);
  if (!matmul || matmul.getInputs().size() != 2)
    return false;

  for (mlir::Value input : matmul.getInputs()) {
    auto collapse =
        layer_utils::producerOfType<mlir::tensor::CollapseShapeOp>(input);
    if (collapse) {
      state = collapse.getSrc();
      return true;
    }
  }
  return false;
}

static std::optional<RecurrentStepMatch> matchRecurrentStep(mlir::Value value) {
  auto tanh = layer_utils::producerOfType<mlir::linalg::GenericOp>(value);
  if (!tanh || !layer_utils::isTanhGeneric(tanh.getOperation()) ||
      tanh.getInputs().size() != 1)
    return std::nullopt;

  auto preActivationAdd =
      layer_utils::producerOfType<mlir::linalg::GenericOp>(tanh.getInputs()[0]);
  if (!preActivationAdd ||
      !layer_utils::isAddfGeneric(preActivationAdd.getOperation()) ||
      preActivationAdd.getInputs().size() != 2)
    return std::nullopt;

  mlir::Value inputSliceSource;
  mlir::Value recurrentState;
  bool foundRecurrentBranch = false;
  bool hasBias = false;
  for (mlir::Value input : preActivationAdd.getInputs()) {
    auto expand =
        layer_utils::producerOfType<mlir::tensor::ExpandShapeOp>(input);
    if (!expand)
      continue;

    mlir::Value state;
    if (isStateMatmul(layer_utils::producerOf(expand.getSrc()), state)) {
      recurrentState = state;
      foundRecurrentBranch = true;
      continue;
    }

    auto biasAdd =
        layer_utils::producerOfType<mlir::linalg::GenericOp>(expand.getSrc());
    if (biasAdd && layer_utils::isAddfGeneric(biasAdd.getOperation()) &&
        biasAdd.getInputs().size() == 2 && hasTensorConstantInput(biasAdd, 1)) {
      for (mlir::Value biasInput : biasAdd.getInputs()) {
        if (isStateMatmul(layer_utils::producerOf(biasInput), state)) {
          recurrentState = state;
          foundRecurrentBranch = true;
          hasBias = true;
          break;
        }
      }
    }
  }

  if (!foundRecurrentBranch)
    return std::nullopt;

  int64_t inputSliceIndex = -1;
  for (mlir::Value input : preActivationAdd.getInputs()) {
    auto collapse =
        layer_utils::producerOfType<mlir::tensor::CollapseShapeOp>(input);
    if (!collapse)
      continue;

    auto slice = layer_utils::producerOfType<mlir::tensor::ExtractSliceOp>(
        collapse.getSrc());
    if (!slice)
      continue;

    std::optional<int64_t> sliceIndex =
        getLeadingSliceIndex(slice.getOperation());
    if (!sliceIndex)
      continue;

    inputSliceSource = slice.getSource();
    inputSliceIndex = *sliceIndex;
    break;
  }

  if (!inputSliceSource)
    return std::nullopt;

  return RecurrentStepMatch{inputSliceSource, inputSliceIndex, recurrentState,
                            hasBias};
}

static mlir::Value getCollapsedTanhSource(mlir::Value value) {
  auto collapse =
      layer_utils::producerOfType<mlir::tensor::CollapseShapeOp>(value);
  if (!collapse ||
      !layer_utils::isTanhGeneric(layer_utils::producerOf(collapse.getSrc())))
    return {};

  return collapse.getSrc();
}

static mlir::Value getProjectionInputSequence(mlir::Value value) {
  auto expand = layer_utils::producerOfType<mlir::tensor::ExpandShapeOp>(value);
  if (!expand)
    return {};

  auto producer = layer_utils::producerOf(expand.getSrc());
  mlir::Value projectedInput;
  if (isStateMatmul(producer, projectedInput))
    return projectedInput;

  auto biasAdd = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(producer);
  if (!biasAdd || !layer_utils::isAddfGeneric(biasAdd.getOperation()) ||
      !hasTensorConstantInput(biasAdd, 1))
    return {};

  for (mlir::Value input : biasAdd.getInputs()) {
    if (isStateMatmul(layer_utils::producerOf(input), projectedInput))
      return projectedInput;
  }
  return {};
}

static bool validateInitialHiddenState(mlir::Value value,
                                       mlir::Value hiddenState, int64_t layer) {
  return isSliceFromValueAtLayer(layer_utils::producerOf(value), hiddenState,
                                 layer);
}

static bool validateLayerInputSource(mlir::Value inputSource, mlir::Value input,
                                     mlir::Value previousLayerConcat) {
  if (!previousLayerConcat)
    return dependsOnValue(inputSource, input);

  mlir::Value projectedInput = getProjectionInputSequence(inputSource);
  return projectedInput == previousLayerConcat;
}

// Validates one RNN layer's timestep chain and output concat.
static bool validateLayerDataflow(mlir::func::FuncOp func,
                                  llvm::ArrayRef<mlir::Value> outputs,
                                  mlir::Value hiddenResult, mlir::Value input,
                                  mlir::Value hiddenState, int64_t layer,
                                  int64_t batchSize, int64_t hiddenSize,
                                  mlir::Value previousLayerConcat,
                                  bool &hasBias, mlir::Value &layerConcat) {
  if (outputs.empty())
    return false;

  llvm::SmallVector<mlir::Value> tanhOutputs;
  llvm::SmallVector<RecurrentStepMatch> steps;
  std::optional<bool> expectedBias;
  for (mlir::Value output : outputs) {
    if (!hasShape(output, {1, batchSize, hiddenSize}))
      return false;

    std::optional<RecurrentStepMatch> step = matchRecurrentStep(output);
    if (!step)
      return false;

    tanhOutputs.push_back(output);
    steps.push_back(*step);
    if (!expectedBias)
      expectedBias = step->hasBias;
    else if (*expectedBias != step->hasBias)
      return false;
    hasBias = hasBias || step->hasBias;
  }

  mlir::Value commonInputSource = steps.front().inputSliceSource;
  for (auto [index, step] : llvm::enumerate(steps)) {
    if (step.inputSliceSource != commonInputSource)
      return false;
    if (step.inputSliceIndex != static_cast<int64_t>(index))
      return false;

    if (index == 0) {
      if (!validateInitialHiddenState(step.recurrentState, hiddenState, layer))
        return false;
    } else if (step.recurrentState != tanhOutputs[index - 1]) {
      return false;
    }
  }

  if (!validateLayerInputSource(commonInputSource, input, previousLayerConcat))
    return false;

  if (getCollapsedTanhSource(hiddenResult) != tanhOutputs.back())
    return false;

  layerConcat = {};
  func.walk([&](mlir::tensor::ConcatOp concat) {
    if (layerConcat || concat.getDim() != 0 ||
        concat.getInputs().size() != tanhOutputs.size())
      return;

    for (auto [actual, expected] : llvm::zip(concat.getInputs(), tanhOutputs)) {
      if (actual != expected)
        return;
    }
    layerConcat = concat.getResult();
  });
  return static_cast<bool>(layerConcat);
}

// Validates stacked RNN layers from input through final sequence output.
static bool validateRNNDataflow(mlir::func::FuncOp func,
                                SequenceAssembly sequenceAssembly,
                                HiddenAssembly hiddenAssembly,
                                int64_t layerCount, int64_t batchSize,
                                int64_t hiddenSize, bool &hasBias) {
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> layerOutputs(layerCount);
  layerOutputs.back().assign(sequenceAssembly.concat.getInputs().begin(),
                             sequenceAssembly.concat.getInputs().end());

  for (int64_t layer = layerCount - 2; layer >= 0; --layer) {
    mlir::Value inputSequence = {};
    for (mlir::Value currentLayerOutput : layerOutputs[layer + 1]) {
      std::optional<RecurrentStepMatch> step =
          matchRecurrentStep(currentLayerOutput);
      if (!step)
        return false;

      mlir::Value projectedInput =
          getProjectionInputSequence(step->inputSliceSource);
      if (!projectedInput)
        return false;

      if (!inputSequence)
        inputSequence = projectedInput;
      else if (inputSequence != projectedInput)
        return false;
    }

    auto concat =
        layer_utils::producerOfType<mlir::tensor::ConcatOp>(inputSequence);
    if (!concat || concat.getDim() != 0 ||
        concat.getInputs().size() != sequenceAssembly.concat.getInputs().size())
      return false;

    layerOutputs[layer].assign(concat.getInputs().begin(),
                               concat.getInputs().end());
  }

  mlir::Value previousLayerConcat;
  for (int64_t layer = 0; layer < layerCount; ++layer) {
    mlir::Value layerConcat;
    if (!validateLayerDataflow(
            func, layerOutputs[layer], hiddenAssembly.concat.getInputs()[layer],
            func.getArgument(0), func.getArgument(1), layer, batchSize,
            hiddenSize, previousLayerConcat, hasBias, layerConcat))
      return false;

    previousLayerConcat = layerConcat;
  }

  return previousLayerConcat == sequenceAssembly.concat.getResult();
}

static std::optional<RNNMatch> matchSupportedRNN(mlir::func::FuncOp func) {
  if (func.getNumArguments() != 2 || func.getNumResults() != 2 ||
      !func.getBody().hasOneBlock())
    return std::nullopt;

  auto inputType = getRankedTensorType(func.getArgument(0).getType());
  auto hiddenType = getRankedTensorType(func.getArgument(1).getType());
  auto sequenceResultType = getRankedTensorType(func.getResultTypes()[0]);
  auto hiddenResultType = getRankedTensorType(func.getResultTypes()[1]);
  if (!inputType || !hiddenType || !sequenceResultType || !hiddenResultType ||
      inputType->getRank() != 3 || hiddenType->getRank() != 3)
    return std::nullopt;

  int64_t batchSize = inputType->getDimSize(0);
  int64_t sequenceLength = inputType->getDimSize(1);
  int64_t inputSize = inputType->getDimSize(2);
  int64_t layerCount = hiddenType->getDimSize(0);
  int64_t hiddenBatchSize = hiddenType->getDimSize(1);
  int64_t hiddenSize = hiddenType->getDimSize(2);
  if (layerCount < 1 || sequenceLength < 1 || hiddenSize < 1 ||
      batchSize != hiddenBatchSize || inputSize < 1)
    return std::nullopt;

  if (sequenceResultType->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}) ||
      hiddenResultType->getShape() != hiddenType->getShape())
    return std::nullopt;

  auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(
      func.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 2)
    return std::nullopt;

  std::optional<SequenceAssembly> sequenceAssembly = matchFinalSequenceAssembly(
      returnOp.getOperand(0), sequenceLength, batchSize, hiddenSize);
  std::optional<HiddenAssembly> hiddenAssembly = matchFinalHiddenAssembly(
      returnOp.getOperand(1), layerCount, batchSize, hiddenSize);
  if (!sequenceAssembly || !hiddenAssembly)
    return std::nullopt;

  int64_t tanhCount = 0;
  func.walk([&](mlir::Operation *op) {
    if (layer_utils::isTanhGeneric(op))
      ++tanhCount;
  });

  if (tanhCount != layerCount * sequenceLength)
    return std::nullopt;

  bool hasBias = false;
  if (!validateRNNDataflow(func, *sequenceAssembly, *hiddenAssembly, layerCount,
                           batchSize, hiddenSize, hasBias))
    return std::nullopt;

  return RNNMatch{hasBias};
}

static void outlineUnmatchedForwardFunction(mlir::func::FuncOp func,
                                            mlir::StringRef layerType,
                                            mlir::RewriterBase &rewriter) {
  mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
  if (!module || !func.getBody().hasOneBlock())
    return;

  mlir::Block &forwardBlock = func.getBody().front();
  auto oldReturn =
      llvm::dyn_cast<mlir::func::ReturnOp>(forwardBlock.getTerminator());
  if (!oldReturn)
    return;

  llvm::SmallVector<mlir::Operation *> bodyOps;
  for (mlir::Operation &op : forwardBlock.without_terminator())
    bodyOps.push_back(&op);

  std::string functionName =
      rewrite_utils::makeUniqueFunctionName(module, layerType);
  auto functionType =
      rewriter.getFunctionType(func.getArgumentTypes(), func.getResultTypes());

  rewriter.setInsertionPointToEnd(module.getBody());
  auto outlinedFunc = rewriter.create<mlir::func::FuncOp>(
      func.getLoc(), functionName, functionType);
  outlinedFunc->setAttr("layer_type", rewriter.getStringAttr(layerType));

  mlir::Block *entryBlock = outlinedFunc.addEntryBlock();
  mlir::IRMapping mapping;
  for (auto [argument, outlinedArgument] :
       llvm::zip(func.getArguments(), entryBlock->getArguments()))
    mapping.map(argument, outlinedArgument);

  rewriter.setInsertionPointToStart(entryBlock);
  for (mlir::Operation *op : bodyOps)
    rewriter.clone(*op, mapping);

  llvm::SmallVector<mlir::Value> mappedReturns;
  for (mlir::Value value : oldReturn.getOperands())
    mappedReturns.push_back(mapping.lookupOrDefault(value));
  rewriter.create<mlir::func::ReturnOp>(oldReturn.getLoc(), mappedReturns);

  rewriter.setInsertionPoint(oldReturn);
  auto call = rewriter.create<mlir::func::CallOp>(
      oldReturn.getLoc(), functionName, func.getResultTypes(),
      func.getArguments());
  rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(oldReturn,
                                                    call.getResults());

  for (auto it = bodyOps.rbegin(); it != bodyOps.rend(); ++it) {
    if ((*it)->use_empty())
      rewriter.eraseOp(*it);
  }
}

static mlir::arith::ConstantOp getConstantDef(mlir::Value value) {
  return value ? layer_utils::producerOfType<mlir::arith::ConstantOp>(value)
               : mlir::arith::ConstantOp();
}

static std::optional<CanonicalRNNInfo> getCanonicalRNNInfo(NNRNNOp rnnOp) {
  if (!rnnOp || !rnnOp.getBatchFirst())
    return std::nullopt;

  std::optional<mlir::RankedTensorType> inputType =
      getRankedTensorType(rnnOp.getInput().getType());
  std::optional<mlir::RankedTensorType> hiddenStateType =
      getRankedTensorType(rnnOp.getH0().getType());
  std::optional<mlir::RankedTensorType> sequenceResultType =
      getRankedTensorType(rnnOp.getOutput().getType());
  std::optional<mlir::RankedTensorType> hiddenResultType =
      getRankedTensorType(rnnOp.getHn().getType());
  if (!inputType || !hiddenStateType || !sequenceResultType ||
      !hiddenResultType || inputType->getRank() != 3 ||
      hiddenStateType->getRank() != 3 || sequenceResultType->getRank() != 3 ||
      hiddenResultType->getRank() != 3)
    return std::nullopt;

  int64_t layerCount = rnnOp.getNumLayers();
  int64_t hiddenSize = rnnOp.getHiddenSize();
  int64_t batchSize = inputType->getDimSize(0);
  int64_t sequenceLength = inputType->getDimSize(1);
  if (layerCount < 1 || hiddenSize < 1 || batchSize < 1 || sequenceLength < 1)
    return std::nullopt;

  if (hiddenStateType->getShape() !=
          llvm::ArrayRef<int64_t>({layerCount, batchSize, hiddenSize}) ||
      hiddenResultType->getShape() != hiddenStateType->getShape() ||
      sequenceResultType->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}))
    return std::nullopt;

  int64_t operandsPerLayer = rnnOp.getHasBias() ? 4 : 2;
  if (static_cast<int64_t>(rnnOp.getRecurrentOperands().size()) !=
      layerCount * operandsPerLayer)
    return std::nullopt;

  auto hiddenSliceType = mlir::RankedTensorType::get(
      {1, batchSize, hiddenSize}, inputType->getElementType());

  return CanonicalRNNInfo{*inputType,          *hiddenStateType,
                          *sequenceResultType, *hiddenResultType,
                          hiddenSliceType,     layerCount,
                          batchSize,           sequenceLength,
                          hiddenSize,          rnnOp.getHasBias()};
}

static std::optional<NNRNNOp> matchCanonicalRNN(mlir::Operation *op) {
  auto rnnOp = llvm::dyn_cast_or_null<NNRNNOp>(op);
  if (!rnnOp || !getCanonicalRNNInfo(rnnOp))
    return std::nullopt;

  for (mlir::Value recurrentOperand : rnnOp.getRecurrentOperands()) {
    if (!getConstantDef(recurrentOperand))
      return std::nullopt;
  }

  return rnnOp;
}

static bool hasLayerFunctionNameConflict(mlir::ModuleOp module,
                                         mlir::StringRef baseName,
                                         int64_t layerCount) {
  if (module.lookupSymbol(baseName))
    return true;

  for (int64_t layer = 0; layer < layerCount; ++layer) {
    std::string functionName = (baseName + "_" + std::to_string(layer)).str();
    if (module.lookupSymbol(functionName))
      return true;
  }

  return false;
}

static std::string makeUniqueLayerFunctionBaseName(mlir::ModuleOp module,
                                                   mlir::StringRef layerType,
                                                   int64_t layerCount) {
  std::string stem = rewrite_utils::makeFunctionBaseName(layerType);
  unsigned functionIndex = 0;
  std::string baseName = stem + "_" + std::to_string(functionIndex);
  while (hasLayerFunctionNameConflict(module, baseName, layerCount)) {
    ++functionIndex;
    baseName = stem + "_" + std::to_string(functionIndex);
  }
  return baseName;
}

static mlir::RankedTensorType getLayerInputType(int64_t layer,
                                                const CanonicalRNNInfo &info) {
  return layer == 0 ? info.inputType : info.sequenceResultType;
}

// Creates one extracted RNN layer function with cloned parameters.
static mlir::func::FuncOp createRNNLayerFunction(
    NNRNNOp rnnOp, llvm::ArrayRef<mlir::arith::ConstantOp> recurrentConstants,
    const CanonicalRNNInfo &info, mlir::StringRef baseName, int64_t layer,
    mlir::RewriterBase &rewriter) {
  mlir::Operation *root = rnnOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};

  int64_t operandsPerLayer = info.hasBias ? 4 : 2;
  int64_t operandBase = layer * operandsPerLayer;
  mlir::RankedTensorType inputType = getLayerInputType(layer, info);
  llvm::SmallVector<mlir::Type, 2> inputTypes{inputType, info.hiddenStateType};
  llvm::SmallVector<mlir::Type, 2> outputTypes{info.sequenceResultType,
                                               info.hiddenSliceType};

  std::string functionName = (baseName + "_" + std::to_string(layer)).str();
  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto layerFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  layerFunc->setAttr("layer_type", rewriter.getStringAttr(
                                       info.hasBias ? "rnn_w_bias" : "rnn"));

  mlir::Block *entryBlock = layerFunc.addEntryBlock();
  mlir::IRMapping mapping;
  rewriter.setInsertionPointToStart(entryBlock);

  mlir::Value wIh =
      rewriter.clone(*recurrentConstants[operandBase], mapping)->getResult(0);
  mlir::Value wHh =
      rewriter.clone(*recurrentConstants[operandBase + 1], mapping)
          ->getResult(0);
  mlir::Value bIh;
  mlir::Value bHh;
  if (info.hasBias) {
    bIh = rewriter.clone(*recurrentConstants[operandBase + 2], mapping)
              ->getResult(0);
    bHh = rewriter.clone(*recurrentConstants[operandBase + 3], mapping)
              ->getResult(0);
  }

  auto layerOp = rewriter.create<NNRNNLayerOp>(
      root->getLoc(), mlir::TypeRange(outputTypes), entryBlock->getArgument(0),
      entryBlock->getArgument(1), wIh, wHh, bIh, bHh, rnnOp.getBatchFirstAttr(),
      rnnOp.getHasBiasAttr(), rnnOp.getHiddenSizeAttr(),
      rewriter.getI64IntegerAttr(layer), rnnOp.getNumLayersAttr());
  rewriter.create<mlir::func::ReturnOp>(
      root->getLoc(), mlir::ValueRange{layerOp.getOutput(), layerOp.getHn()});

  return layerFunc;
}

static mlir::Value
buildStackedHiddenStateResult(llvm::ArrayRef<mlir::Value> finalHiddenStates,
                              const CanonicalRNNInfo &info, mlir::Location loc,
                              mlir::RewriterBase &rewriter) {
  if (finalHiddenStates.size() == 1)
    return finalHiddenStates.front();

  auto concat = rewriter.create<mlir::tensor::ConcatOp>(
      loc, info.hiddenResultType, /*dim=*/0,
      mlir::ValueRange(finalHiddenStates));
  return concat.getResult();
}

// Rewrites a canonical RNN op into calls through extracted layer functions.
static void outlineRNNOpToLayerFunctions(NNRNNOp rnnOp,
                                         mlir::RewriterBase &rewriter) {
  if (!rnnOp || rnnOp->getNumResults() != 2)
    return;

  std::optional<CanonicalRNNInfo> info = getCanonicalRNNInfo(rnnOp);
  if (!info)
    return;

  llvm::SmallVector<mlir::arith::ConstantOp> recurrentConstants;
  for (mlir::Value recurrentOperand : rnnOp.getRecurrentOperands()) {
    mlir::arith::ConstantOp recurrentConstant =
        getConstantDef(recurrentOperand);
    if (!recurrentConstant)
      return;
    recurrentConstants.push_back(recurrentConstant);
  }

  mlir::Operation *root = rnnOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return;

  mlir::StringRef layerType =
      info->hasBias ? mlir::StringRef("rnn_w_bias") : mlir::StringRef("rnn");
  std::string baseName =
      makeUniqueLayerFunctionBaseName(module, layerType, info->layerCount);

  llvm::SmallVector<mlir::func::FuncOp> layerFuncs;
  layerFuncs.reserve(info->layerCount);
  for (int64_t layer = 0; layer < info->layerCount; ++layer) {
    mlir::func::FuncOp layerFunc = createRNNLayerFunction(
        rnnOp, recurrentConstants, *info, baseName, layer, rewriter);
    if (!layerFunc)
      return;
    layerFuncs.push_back(layerFunc);
  }

  mlir::Value currentSequence = rnnOp.getInput();
  llvm::SmallVector<mlir::Value> finalHiddenStates;
  finalHiddenStates.reserve(info->layerCount);

  rewriter.setInsertionPoint(root);
  for (mlir::func::FuncOp layerFunc : layerFuncs) {
    auto call = rewriter.create<mlir::func::CallOp>(
        root->getLoc(), layerFunc.getSymName(), layerFunc.getResultTypes(),
        mlir::ValueRange{currentSequence, rnnOp.getH0()});
    currentSequence = call.getResult(0);
    finalHiddenStates.push_back(call.getResult(1));
  }

  mlir::Value hiddenResult = buildStackedHiddenStateResult(
      finalHiddenStates, *info, root->getLoc(), rewriter);

  rnnOp.getOutput().replaceAllUsesWith(currentSequence);
  rnnOp.getHn().replaceAllUsesWith(hiddenResult);
  if (root->use_empty())
    rewriter.eraseOp(root);

  for (mlir::arith::ConstantOp recurrentConstant : recurrentConstants) {
    if (recurrentConstant->use_empty())
      rewriter.eraseOp(recurrentConstant);
  }
}

static bool extractCanonicalRNNs(mlir::func::FuncOp func) {
  llvm::SmallVector<NNRNNOp> matches;
  func.walk([&](mlir::Operation *op) {
    std::optional<NNRNNOp> match = matchCanonicalRNN(op);
    if (match)
      matches.push_back(*match);
  });

  if (matches.empty())
    return false;

  mlir::IRRewriter rewriter(func.getContext());
  for (NNRNNOp match : matches) {
    if (match && match->getBlock())
      outlineRNNOpToLayerFunctions(match, rewriter);
  }
  return true;
}

class RNNExtractor : public mlir::sculptor::LayerExtractor {
public:
  explicit RNNExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "rnn"; }

  void extract(mlir::func::FuncOp func) const override {
    if (extractCanonicalRNNs(func))
      return;

    std::optional<RNNMatch> match = matchSupportedRNN(func);
    if (!match)
      return;

    mlir::IRRewriter rewriter(func.getContext());
    outlineUnmatchedForwardFunction(func, match->hasBias ? "rnn_w_bias" : "rnn",
                                    rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerRNNExtractor(LayerExtractors &extractors, MLIRContext *context) {
  extractors.push_back(std::make_unique<RNNExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
