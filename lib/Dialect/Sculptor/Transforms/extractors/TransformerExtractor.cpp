#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <string>

namespace {

using mlir::sculptor::NNTransformerBlockOp;
using mlir::sculptor::NNTransformerDecoderOp;
using mlir::sculptor::NNTransformerEncoderOp;
using mlir::sculptor::NNTransformerOp;
using mlir::sculptor::TransformerBlockKind;
using mlir::sculptor::TransformerBlockKindAttr;

constexpr llvm::StringLiteral kEncoderLayerType = "transformer_encoder_block";
constexpr llvm::StringLiteral kDecoderLayerType = "transformer_decoder_block";

struct TransformerBlockParameters {
  mlir::Value qkvWeight;
  mlir::Value qkvBias;
  mlir::Value attnOutputWeight;
  mlir::Value attnOutputBias;
  mlir::Value attnNormWeight;
  mlir::Value attnNormBias;
  mlir::Value crossQueryWeight;
  mlir::Value crossQueryBias;
  mlir::Value crossKeyValueWeight;
  mlir::Value crossKeyValueBias;
  mlir::Value crossOutputWeight;
  mlir::Value crossOutputBias;
  mlir::Value crossNormWeight;
  mlir::Value crossNormBias;
  mlir::Value mlpUpWeight;
  mlir::Value mlpUpBias;
  mlir::Value mlpDownWeight;
  mlir::Value mlpDownBias;
  mlir::Value mlpNormWeight;
  mlir::Value mlpNormBias;
  mlir::Value finalNormWeight;
  mlir::Value finalNormBias;
  bool isDecoder = false;
  bool hasCrossAttention = false;
  bool hasFinalNorm = false;
  int64_t blockIndex = 0;
  int64_t numBlocks = 0;
};

static bool takeParameter(mlir::ValueRange parameters, int64_t &offset,
                          mlir::Value &value) {
  if (offset >= static_cast<int64_t>(parameters.size()))
    return false;
  value = parameters[offset++];
  return true;
}

static bool takeOptionalParameter(mlir::ValueRange parameters, int64_t &offset,
                                  bool present, mlir::Value &value) {
  if (!present)
    return true;
  return takeParameter(parameters, offset, value);
}

static bool parseEncoderBlockParameters(mlir::ValueRange parameters,
                                        int64_t &offset,
                                        bool hasProjectionBias,
                                        bool hasLayerNormAffine,
                                        int64_t blockIndex,
                                        int64_t numBlocks,
                                        TransformerBlockParameters &block) {
  block = TransformerBlockParameters{};
  block.blockIndex = blockIndex;
  block.numBlocks = numBlocks;

  return takeParameter(parameters, offset, block.qkvWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.qkvBias) &&
         takeParameter(parameters, offset, block.attnOutputWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.attnOutputBias) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.attnNormWeight) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.attnNormBias) &&
         takeParameter(parameters, offset, block.mlpUpWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.mlpUpBias) &&
         takeParameter(parameters, offset, block.mlpDownWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.mlpDownBias) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.mlpNormWeight) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.mlpNormBias);
}

static bool parseDecoderBlockParameters(mlir::ValueRange parameters,
                                        int64_t &offset,
                                        bool hasProjectionBias,
                                        bool hasLayerNormAffine,
                                        int64_t blockIndex,
                                        int64_t numBlocks,
                                        TransformerBlockParameters &block) {
  block = TransformerBlockParameters{};
  block.isDecoder = true;
  block.hasCrossAttention = true;
  block.blockIndex = blockIndex;
  block.numBlocks = numBlocks;

  return takeParameter(parameters, offset, block.qkvWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.qkvBias) &&
         takeParameter(parameters, offset, block.attnOutputWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.attnOutputBias) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.attnNormWeight) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.attnNormBias) &&
         takeParameter(parameters, offset, block.crossQueryWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.crossQueryBias) &&
         takeParameter(parameters, offset, block.crossKeyValueWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.crossKeyValueBias) &&
         takeParameter(parameters, offset, block.crossOutputWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.crossOutputBias) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.crossNormWeight) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.crossNormBias) &&
         takeParameter(parameters, offset, block.mlpUpWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.mlpUpBias) &&
         takeParameter(parameters, offset, block.mlpDownWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.mlpDownBias) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.mlpNormWeight) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.mlpNormBias);
}

static bool collectTransformerBlocks(
    NNTransformerOp transformerOp,
    llvm::SmallVectorImpl<TransformerBlockParameters> &encoderBlocks,
    llvm::SmallVectorImpl<TransformerBlockParameters> &decoderBlocks) {
  int64_t numEncoderBlocks = transformerOp.getNumEncoderBlocks();
  int64_t numDecoderBlocks = transformerOp.getNumDecoderBlocks();
  if (numEncoderBlocks < 1 || numDecoderBlocks < 1)
    return false;

  mlir::ValueRange parameters = transformerOp.getParameters();
  int64_t offset = 0;
  bool hasProjectionBias = transformerOp.getHasProjectionBias();
  bool hasLayerNormAffine = transformerOp.getHasLayerNormAffine();

  encoderBlocks.clear();
  decoderBlocks.clear();
  encoderBlocks.reserve(numEncoderBlocks);
  decoderBlocks.reserve(numDecoderBlocks);

  for (int64_t block = 0; block < numEncoderBlocks; ++block) {
    TransformerBlockParameters blockParameters;
    if (!parseEncoderBlockParameters(parameters, offset, hasProjectionBias,
                                     hasLayerNormAffine, block,
                                     numEncoderBlocks, blockParameters))
      return false;
    encoderBlocks.push_back(blockParameters);
  }

  for (int64_t block = 0; block < numDecoderBlocks; ++block) {
    TransformerBlockParameters blockParameters;
    if (!parseDecoderBlockParameters(parameters, offset, hasProjectionBias,
                                     hasLayerNormAffine, block,
                                     numDecoderBlocks, blockParameters))
      return false;
    decoderBlocks.push_back(blockParameters);
  }

  if (transformerOp.getHasFinalNorm()) {
    encoderBlocks.back().hasFinalNorm = true;
    decoderBlocks.back().hasFinalNorm = true;
    if (hasLayerNormAffine) {
      if (!takeParameter(parameters, offset,
                         encoderBlocks.back().finalNormWeight) ||
          !takeParameter(parameters, offset,
                         encoderBlocks.back().finalNormBias) ||
          !takeParameter(parameters, offset,
                         decoderBlocks.back().finalNormWeight) ||
          !takeParameter(parameters, offset,
                         decoderBlocks.back().finalNormBias))
        return false;
    }
  }

  return offset == static_cast<int64_t>(parameters.size());
}

static bool collectTransformerDecoderBlocks(
    NNTransformerDecoderOp transformerOp,
    llvm::SmallVectorImpl<TransformerBlockParameters> &decoderBlocks) {
  int64_t numBlocks = transformerOp.getNumBlocks();
  if (numBlocks < 1)
    return false;

  mlir::ValueRange parameters = transformerOp.getParameters();
  int64_t offset = 0;
  bool hasProjectionBias = transformerOp.getHasProjectionBias();
  bool hasLayerNormAffine = transformerOp.getHasLayerNormAffine();

  decoderBlocks.clear();
  decoderBlocks.reserve(numBlocks);

  for (int64_t block = 0; block < numBlocks; ++block) {
    TransformerBlockParameters blockParameters;
    if (!parseEncoderBlockParameters(parameters, offset, hasProjectionBias,
                                     hasLayerNormAffine, block, numBlocks,
                                     blockParameters))
      return false;
    blockParameters.isDecoder = true;
    blockParameters.hasCrossAttention = false;
    decoderBlocks.push_back(blockParameters);
  }

  if (transformerOp.getHasFinalNorm()) {
    decoderBlocks.back().hasFinalNorm = true;
    if (hasLayerNormAffine) {
      if (!takeParameter(parameters, offset,
                         decoderBlocks.back().finalNormWeight) ||
          !takeParameter(parameters, offset,
                         decoderBlocks.back().finalNormBias))
        return false;
    }
  }

  return offset == static_cast<int64_t>(parameters.size());
}

static bool collectTransformerEncoderBlocks(
    NNTransformerEncoderOp transformerOp,
    llvm::SmallVectorImpl<TransformerBlockParameters> &encoderBlocks) {
  int64_t numBlocks = transformerOp.getNumBlocks();
  if (numBlocks < 1)
    return false;

  mlir::ValueRange parameters = transformerOp.getParameters();
  int64_t offset = 0;
  bool hasProjectionBias = transformerOp.getHasProjectionBias();
  bool hasLayerNormAffine = transformerOp.getHasLayerNormAffine();

  encoderBlocks.clear();
  encoderBlocks.reserve(numBlocks);

  for (int64_t block = 0; block < numBlocks; ++block) {
    TransformerBlockParameters blockParameters;
    if (!parseEncoderBlockParameters(parameters, offset, hasProjectionBias,
                                     hasLayerNormAffine, block, numBlocks,
                                     blockParameters))
      return false;
    encoderBlocks.push_back(blockParameters);
  }

  if (transformerOp.getHasFinalNorm()) {
    encoderBlocks.back().hasFinalNorm = true;
    if (hasLayerNormAffine) {
      if (!takeParameter(parameters, offset,
                         encoderBlocks.back().finalNormWeight) ||
          !takeParameter(parameters, offset,
                         encoderBlocks.back().finalNormBias))
        return false;
    }
  }

  return offset == static_cast<int64_t>(parameters.size());
}

static std::string makeUniqueFunctionName(mlir::ModuleOp module,
                                          llvm::StringRef stem) {
  unsigned functionIndex = 0;
  std::string functionName = (stem + "_" + std::to_string(functionIndex)).str();
  while (module.lookupSymbol(functionName)) {
    ++functionIndex;
    functionName = (stem + "_" + std::to_string(functionIndex)).str();
  }
  return functionName;
}

static mlir::FailureOr<mlir::Value>
cloneProducerTree(mlir::Value value, mlir::IRMapping &mapping,
                  mlir::RewriterBase &rewriter) {
  if (!value)
    return mlir::failure();

  if (mlir::Value mapped = mapping.lookupOrNull(value))
    return mapped;

  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return mlir::failure();

  for (mlir::Value operand : definingOp->getOperands()) {
    if (mapping.lookupOrNull(operand))
      continue;
    mlir::FailureOr<mlir::Value> clonedOperand =
        cloneProducerTree(operand, mapping, rewriter);
    if (mlir::failed(clonedOperand))
      return mlir::failure();
    mapping.map(operand, *clonedOperand);
  }

  llvm::SetVector<mlir::Value> capturedValues;
  mlir::getUsedValuesDefinedAbove(definingOp->getRegions(), capturedValues);
  for (mlir::Value capturedValue : capturedValues) {
    if (mapping.lookupOrNull(capturedValue))
      continue;
    mlir::FailureOr<mlir::Value> clonedCapture =
        cloneProducerTree(capturedValue, mapping, rewriter);
    if (mlir::failed(clonedCapture))
      return mlir::failure();
    mapping.map(capturedValue, *clonedCapture);
  }

  mlir::Operation *clone = rewriter.clone(*definingOp, mapping);
  if (clone->getNumResults() != definingOp->getNumResults())
    return mlir::failure();

  for (unsigned i = 0, e = definingOp->getNumResults(); i < e; ++i)
    mapping.map(definingOp->getResult(i), clone->getResult(i));

  if (mlir::Value mapped = mapping.lookupOrNull(value))
    return mapped;
  return mlir::failure();
}

static mlir::FailureOr<mlir::Value>
cloneOptionalProducerTree(mlir::Value value, mlir::IRMapping &mapping,
                          mlir::RewriterBase &rewriter) {
  if (!value)
    return mlir::Value{};
  return cloneProducerTree(value, mapping, rewriter);
}

static mlir::FailureOr<TransformerBlockParameters>
cloneBlockParameters(const TransformerBlockParameters &block,
                     mlir::IRMapping &mapping,
                     mlir::RewriterBase &rewriter) {
  TransformerBlockParameters cloned = block;

  auto assignRequired = [&](mlir::Value source,
                            mlir::Value &target) -> mlir::LogicalResult {
    mlir::FailureOr<mlir::Value> clonedValue =
        cloneProducerTree(source, mapping, rewriter);
    if (mlir::failed(clonedValue))
      return mlir::failure();
    target = *clonedValue;
    return mlir::success();
  };
  auto assignOptional = [&](mlir::Value source,
                            mlir::Value &target) -> mlir::LogicalResult {
    mlir::FailureOr<mlir::Value> clonedValue =
        cloneOptionalProducerTree(source, mapping, rewriter);
    if (mlir::failed(clonedValue))
      return mlir::failure();
    target = *clonedValue;
    return mlir::success();
  };

  if (mlir::failed(assignRequired(block.qkvWeight, cloned.qkvWeight)) ||
      mlir::failed(assignOptional(block.qkvBias, cloned.qkvBias)) ||
      mlir::failed(assignRequired(block.attnOutputWeight,
                                  cloned.attnOutputWeight)) ||
      mlir::failed(assignOptional(block.attnOutputBias,
                                  cloned.attnOutputBias)) ||
      mlir::failed(assignOptional(block.attnNormWeight,
                                  cloned.attnNormWeight)) ||
      mlir::failed(assignOptional(block.attnNormBias, cloned.attnNormBias)) ||
      mlir::failed(assignOptional(block.crossQueryWeight,
                                  cloned.crossQueryWeight)) ||
      mlir::failed(assignOptional(block.crossQueryBias,
                                  cloned.crossQueryBias)) ||
      mlir::failed(assignOptional(block.crossKeyValueWeight,
                                  cloned.crossKeyValueWeight)) ||
      mlir::failed(assignOptional(block.crossKeyValueBias,
                                  cloned.crossKeyValueBias)) ||
      mlir::failed(assignOptional(block.crossOutputWeight,
                                  cloned.crossOutputWeight)) ||
      mlir::failed(assignOptional(block.crossOutputBias,
                                  cloned.crossOutputBias)) ||
      mlir::failed(assignOptional(block.crossNormWeight,
                                  cloned.crossNormWeight)) ||
      mlir::failed(assignOptional(block.crossNormBias, cloned.crossNormBias)) ||
      mlir::failed(assignRequired(block.mlpUpWeight, cloned.mlpUpWeight)) ||
      mlir::failed(assignOptional(block.mlpUpBias, cloned.mlpUpBias)) ||
      mlir::failed(assignRequired(block.mlpDownWeight, cloned.mlpDownWeight)) ||
      mlir::failed(assignOptional(block.mlpDownBias, cloned.mlpDownBias)) ||
      mlir::failed(assignOptional(block.mlpNormWeight,
                                  cloned.mlpNormWeight)) ||
      mlir::failed(assignOptional(block.mlpNormBias, cloned.mlpNormBias)) ||
      mlir::failed(assignOptional(block.finalNormWeight,
                                  cloned.finalNormWeight)) ||
      mlir::failed(assignOptional(block.finalNormBias, cloned.finalNormBias)))
    return mlir::failure();

  return cloned;
}

static mlir::func::FuncOp createTransformerBlockFunction(
    NNTransformerOp transformerOp, const TransformerBlockParameters &block,
    mlir::Type inputType, mlir::Type memoryType, mlir::Type outputType,
    mlir::RewriterBase &rewriter) {
  mlir::Operation *root = transformerOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};

  llvm::StringRef layerType =
      block.isDecoder ? kDecoderLayerType : kEncoderLayerType;
  std::string functionName = makeUniqueFunctionName(module, layerType);

  llvm::SmallVector<mlir::Type, 2> inputTypes{inputType};
  if (block.isDecoder)
    inputTypes.push_back(memoryType);

  auto functionType =
      rewriter.getFunctionType(inputTypes, mlir::TypeRange{outputType});
  rewriter.setInsertionPointToEnd(module.getBody());
  auto blockFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  blockFunc->setAttr("layer_type", rewriter.getStringAttr(layerType));

  mlir::Block *entryBlock = blockFunc.addEntryBlock();
  mlir::IRMapping mapping;
  rewriter.setInsertionPointToStart(entryBlock);

  mlir::FailureOr<TransformerBlockParameters> clonedBlock =
      cloneBlockParameters(block, mapping, rewriter);
  if (mlir::failed(clonedBlock)) {
    rewriter.eraseOp(blockFunc);
    return {};
  }

  mlir::Value memory =
      block.isDecoder ? entryBlock->getArgument(1) : mlir::Value{};
  TransformerBlockKind blockKind =
      block.isDecoder ? TransformerBlockKind::Decoder
                      : TransformerBlockKind::Encoder;
  auto blockOp = rewriter.create<NNTransformerBlockOp>(
      root->getLoc(), outputType, entryBlock->getArgument(0), memory,
      clonedBlock->qkvWeight, clonedBlock->qkvBias,
      clonedBlock->attnOutputWeight, clonedBlock->attnOutputBias,
      clonedBlock->attnNormWeight, clonedBlock->attnNormBias,
      clonedBlock->crossQueryWeight, clonedBlock->crossQueryBias,
      clonedBlock->crossKeyValueWeight, clonedBlock->crossKeyValueBias,
      clonedBlock->crossOutputWeight, clonedBlock->crossOutputBias,
      clonedBlock->crossNormWeight, clonedBlock->crossNormBias,
      clonedBlock->mlpUpWeight, clonedBlock->mlpUpBias,
      clonedBlock->mlpDownWeight, clonedBlock->mlpDownBias,
      clonedBlock->mlpNormWeight, clonedBlock->mlpNormBias,
      clonedBlock->finalNormWeight, clonedBlock->finalNormBias,
      transformerOp.getBatchFirstAttr(), transformerOp.getHasProjectionBiasAttr(),
      transformerOp.getHasLayerNormAffineAttr(),
      rewriter.getBoolAttr(clonedBlock->hasFinalNorm),
      transformerOp.getCausalAttr(),
      rewriter.getBoolAttr(clonedBlock->hasCrossAttention),
      transformerOp.getActivationAttr(),
      TransformerBlockKindAttr::get(rewriter.getContext(), blockKind),
      transformerOp.getNormModeAttr(), transformerOp.getHiddenSizeAttr(),
      transformerOp.getNumHeadsAttr(), transformerOp.getHeadDimAttr(),
      transformerOp.getMlpHiddenSizeAttr(),
      rewriter.getI64IntegerAttr(block.blockIndex),
      rewriter.getI64IntegerAttr(block.numBlocks),
      transformerOp.getLayerNormEpsAttr());

  rewriter.create<mlir::func::ReturnOp>(root->getLoc(),
                                        mlir::ValueRange{blockOp.getOutput()});
  return blockFunc;
}

static mlir::func::FuncOp createTransformerEncoderBlockFunction(
    NNTransformerEncoderOp transformerOp, const TransformerBlockParameters &block,
    mlir::Type inputType, mlir::Type outputType,
    mlir::RewriterBase &rewriter) {
  mlir::Operation *root = transformerOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};

  std::string functionName = makeUniqueFunctionName(module, kEncoderLayerType);

  auto functionType =
      rewriter.getFunctionType(mlir::TypeRange{inputType},
                               mlir::TypeRange{outputType});
  rewriter.setInsertionPointToEnd(module.getBody());
  auto blockFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  blockFunc->setAttr("layer_type", rewriter.getStringAttr(kEncoderLayerType));

  mlir::Block *entryBlock = blockFunc.addEntryBlock();
  mlir::IRMapping mapping;
  rewriter.setInsertionPointToStart(entryBlock);

  mlir::FailureOr<TransformerBlockParameters> clonedBlock =
      cloneBlockParameters(block, mapping, rewriter);
  if (mlir::failed(clonedBlock)) {
    rewriter.eraseOp(blockFunc);
    return {};
  }

  auto blockOp = rewriter.create<NNTransformerBlockOp>(
      root->getLoc(), outputType, entryBlock->getArgument(0), mlir::Value{},
      clonedBlock->qkvWeight, clonedBlock->qkvBias,
      clonedBlock->attnOutputWeight, clonedBlock->attnOutputBias,
      clonedBlock->attnNormWeight, clonedBlock->attnNormBias,
      clonedBlock->crossQueryWeight, clonedBlock->crossQueryBias,
      clonedBlock->crossKeyValueWeight, clonedBlock->crossKeyValueBias,
      clonedBlock->crossOutputWeight, clonedBlock->crossOutputBias,
      clonedBlock->crossNormWeight, clonedBlock->crossNormBias,
      clonedBlock->mlpUpWeight, clonedBlock->mlpUpBias,
      clonedBlock->mlpDownWeight, clonedBlock->mlpDownBias,
      clonedBlock->mlpNormWeight, clonedBlock->mlpNormBias,
      clonedBlock->finalNormWeight, clonedBlock->finalNormBias,
      transformerOp.getBatchFirstAttr(), transformerOp.getHasProjectionBiasAttr(),
      transformerOp.getHasLayerNormAffineAttr(),
      rewriter.getBoolAttr(clonedBlock->hasFinalNorm),
      transformerOp.getCausalAttr(), rewriter.getBoolAttr(false),
      transformerOp.getActivationAttr(),
      TransformerBlockKindAttr::get(rewriter.getContext(),
                                    TransformerBlockKind::Encoder),
      transformerOp.getNormModeAttr(), transformerOp.getHiddenSizeAttr(),
      transformerOp.getNumHeadsAttr(), transformerOp.getHeadDimAttr(),
      transformerOp.getMlpHiddenSizeAttr(),
      rewriter.getI64IntegerAttr(block.blockIndex),
      rewriter.getI64IntegerAttr(block.numBlocks),
      transformerOp.getLayerNormEpsAttr());

  rewriter.create<mlir::func::ReturnOp>(root->getLoc(),
                                        mlir::ValueRange{blockOp.getOutput()});
  return blockFunc;
}

static mlir::func::FuncOp createTransformerDecoderBlockFunction(
    NNTransformerDecoderOp transformerOp, const TransformerBlockParameters &block,
    mlir::Type inputType, mlir::Type outputType,
    mlir::RewriterBase &rewriter) {
  mlir::Operation *root = transformerOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};

  std::string functionName = makeUniqueFunctionName(module, kDecoderLayerType);

  auto functionType =
      rewriter.getFunctionType(mlir::TypeRange{inputType},
                               mlir::TypeRange{outputType});
  rewriter.setInsertionPointToEnd(module.getBody());
  auto blockFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  blockFunc->setAttr("layer_type", rewriter.getStringAttr(kDecoderLayerType));

  mlir::Block *entryBlock = blockFunc.addEntryBlock();
  mlir::IRMapping mapping;
  rewriter.setInsertionPointToStart(entryBlock);

  mlir::FailureOr<TransformerBlockParameters> clonedBlock =
      cloneBlockParameters(block, mapping, rewriter);
  if (mlir::failed(clonedBlock)) {
    rewriter.eraseOp(blockFunc);
    return {};
  }

  auto blockOp = rewriter.create<NNTransformerBlockOp>(
      root->getLoc(), outputType, entryBlock->getArgument(0), mlir::Value{},
      clonedBlock->qkvWeight, clonedBlock->qkvBias,
      clonedBlock->attnOutputWeight, clonedBlock->attnOutputBias,
      clonedBlock->attnNormWeight, clonedBlock->attnNormBias,
      clonedBlock->crossQueryWeight, clonedBlock->crossQueryBias,
      clonedBlock->crossKeyValueWeight, clonedBlock->crossKeyValueBias,
      clonedBlock->crossOutputWeight, clonedBlock->crossOutputBias,
      clonedBlock->crossNormWeight, clonedBlock->crossNormBias,
      clonedBlock->mlpUpWeight, clonedBlock->mlpUpBias,
      clonedBlock->mlpDownWeight, clonedBlock->mlpDownBias,
      clonedBlock->mlpNormWeight, clonedBlock->mlpNormBias,
      clonedBlock->finalNormWeight, clonedBlock->finalNormBias,
      transformerOp.getBatchFirstAttr(), transformerOp.getHasProjectionBiasAttr(),
      transformerOp.getHasLayerNormAffineAttr(),
      rewriter.getBoolAttr(clonedBlock->hasFinalNorm),
      transformerOp.getCausalAttr(), transformerOp.getHasCrossAttentionAttr(),
      transformerOp.getActivationAttr(),
      TransformerBlockKindAttr::get(rewriter.getContext(),
                                    TransformerBlockKind::Decoder),
      transformerOp.getNormModeAttr(), transformerOp.getHiddenSizeAttr(),
      transformerOp.getNumHeadsAttr(), transformerOp.getHeadDimAttr(),
      transformerOp.getMlpHiddenSizeAttr(),
      rewriter.getI64IntegerAttr(block.blockIndex),
      rewriter.getI64IntegerAttr(block.numBlocks),
      transformerOp.getLayerNormEpsAttr());

  rewriter.create<mlir::func::ReturnOp>(root->getLoc(),
                                        mlir::ValueRange{blockOp.getOutput()});
  return blockFunc;
}

static void collectProducerTreePostOrder(
    mlir::Value value, llvm::SmallPtrSetImpl<mlir::Operation *> &seen,
    llvm::SmallVectorImpl<mlir::Operation *> &ops) {
  if (!value)
    return;

  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp || !seen.insert(definingOp).second)
    return;

  for (mlir::Value operand : definingOp->getOperands())
    collectProducerTreePostOrder(operand, seen, ops);

  llvm::SetVector<mlir::Value> capturedValues;
  mlir::getUsedValuesDefinedAbove(definingOp->getRegions(), capturedValues);
  for (mlir::Value capturedValue : capturedValues)
    collectProducerTreePostOrder(capturedValue, seen, ops);

  ops.push_back(definingOp);
}

static void eraseDeadParameterProducers(llvm::ArrayRef<mlir::Value> parameters,
                                        mlir::RewriterBase &rewriter) {
  llvm::SmallPtrSet<mlir::Operation *, 32> seen;
  llvm::SmallVector<mlir::Operation *, 64> ops;
  for (mlir::Value parameter : parameters)
    collectProducerTreePostOrder(parameter, seen, ops);

  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    if ((*it)->use_empty())
      rewriter.eraseOp(*it);
  }
}

static void outlineTransformerOpToBlockFunctions(NNTransformerOp transformerOp,
                                                 mlir::RewriterBase &rewriter) {
  if (!transformerOp || transformerOp->getNumResults() != 1)
    return;

  llvm::SmallVector<TransformerBlockParameters> encoderBlocks;
  llvm::SmallVector<TransformerBlockParameters> decoderBlocks;
  if (!collectTransformerBlocks(transformerOp, encoderBlocks, decoderBlocks))
    return;

  mlir::Type encoderType = transformerOp.getSrc().getType();
  mlir::Type decoderType = transformerOp.getTgt().getType();
  mlir::Type outputType = transformerOp.getOutput().getType();

  llvm::SmallVector<mlir::func::FuncOp> encoderFuncs;
  llvm::SmallVector<mlir::func::FuncOp> decoderFuncs;
  encoderFuncs.reserve(encoderBlocks.size());
  decoderFuncs.reserve(decoderBlocks.size());

  for (const TransformerBlockParameters &block : encoderBlocks) {
    mlir::func::FuncOp blockFunc = createTransformerBlockFunction(
        transformerOp, block, encoderType, mlir::Type{}, encoderType, rewriter);
    if (!blockFunc)
      return;
    encoderFuncs.push_back(blockFunc);
  }

  for (const TransformerBlockParameters &block : decoderBlocks) {
    mlir::func::FuncOp blockFunc = createTransformerBlockFunction(
        transformerOp, block, decoderType, encoderType, outputType, rewriter);
    if (!blockFunc)
      return;
    decoderFuncs.push_back(blockFunc);
  }

  mlir::Operation *root = transformerOp.getOperation();
  mlir::Location loc = root->getLoc();
  llvm::SmallVector<mlir::Value> originalParameters(
      transformerOp.getParameters().begin(), transformerOp.getParameters().end());
  rewriter.setInsertionPoint(root);

  mlir::Value currentEncoder = transformerOp.getSrc();
  for (mlir::func::FuncOp blockFunc : encoderFuncs) {
    auto call = rewriter.create<mlir::func::CallOp>(
        loc, blockFunc.getSymName(), blockFunc.getResultTypes(),
        mlir::ValueRange{currentEncoder});
    currentEncoder = call.getResult(0);
  }

  mlir::Value currentDecoder = transformerOp.getTgt();
  for (mlir::func::FuncOp blockFunc : decoderFuncs) {
    auto call = rewriter.create<mlir::func::CallOp>(
        loc, blockFunc.getSymName(), blockFunc.getResultTypes(),
        mlir::ValueRange{currentDecoder, currentEncoder});
    currentDecoder = call.getResult(0);
  }

  transformerOp.getOutput().replaceAllUsesWith(currentDecoder);
  if (root->use_empty())
    rewriter.eraseOp(root);
  eraseDeadParameterProducers(originalParameters, rewriter);
}

static void outlineTransformerEncoderOpToBlockFunctions(
    NNTransformerEncoderOp transformerOp, mlir::RewriterBase &rewriter) {
  if (!transformerOp || transformerOp->getNumResults() != 1)
    return;

  llvm::SmallVector<TransformerBlockParameters> encoderBlocks;
  if (!collectTransformerEncoderBlocks(transformerOp, encoderBlocks))
    return;

  mlir::Type inputType = transformerOp.getInput().getType();
  mlir::Type outputType = transformerOp.getOutput().getType();

  llvm::SmallVector<mlir::func::FuncOp> encoderFuncs;
  encoderFuncs.reserve(encoderBlocks.size());
  for (const TransformerBlockParameters &block : encoderBlocks) {
    mlir::func::FuncOp blockFunc = createTransformerEncoderBlockFunction(
        transformerOp, block, inputType, outputType, rewriter);
    if (!blockFunc)
      return;
    encoderFuncs.push_back(blockFunc);
  }

  mlir::Operation *root = transformerOp.getOperation();
  mlir::Location loc = root->getLoc();
  llvm::SmallVector<mlir::Value> originalParameters(
      transformerOp.getParameters().begin(), transformerOp.getParameters().end());
  rewriter.setInsertionPoint(root);

  mlir::Value current = transformerOp.getInput();
  for (mlir::func::FuncOp blockFunc : encoderFuncs) {
    auto call = rewriter.create<mlir::func::CallOp>(
        loc, blockFunc.getSymName(), blockFunc.getResultTypes(),
        mlir::ValueRange{current});
    current = call.getResult(0);
  }

  transformerOp.getOutput().replaceAllUsesWith(current);
  if (root->use_empty())
    rewriter.eraseOp(root);
  eraseDeadParameterProducers(originalParameters, rewriter);
}

static void outlineTransformerDecoderOpToBlockFunctions(
    NNTransformerDecoderOp transformerOp, mlir::RewriterBase &rewriter) {
  if (!transformerOp || transformerOp->getNumResults() != 1)
    return;

  llvm::SmallVector<TransformerBlockParameters> decoderBlocks;
  if (!collectTransformerDecoderBlocks(transformerOp, decoderBlocks))
    return;

  mlir::Type inputType = transformerOp.getInput().getType();
  mlir::Type outputType = transformerOp.getOutput().getType();

  llvm::SmallVector<mlir::func::FuncOp> decoderFuncs;
  decoderFuncs.reserve(decoderBlocks.size());
  for (const TransformerBlockParameters &block : decoderBlocks) {
    mlir::func::FuncOp blockFunc = createTransformerDecoderBlockFunction(
        transformerOp, block, inputType, outputType, rewriter);
    if (!blockFunc)
      return;
    decoderFuncs.push_back(blockFunc);
  }

  mlir::Operation *root = transformerOp.getOperation();
  mlir::Location loc = root->getLoc();
  llvm::SmallVector<mlir::Value> originalParameters(
      transformerOp.getParameters().begin(), transformerOp.getParameters().end());
  rewriter.setInsertionPoint(root);

  mlir::Value current = transformerOp.getInput();
  for (mlir::func::FuncOp blockFunc : decoderFuncs) {
    auto call = rewriter.create<mlir::func::CallOp>(
        loc, blockFunc.getSymName(), blockFunc.getResultTypes(),
        mlir::ValueRange{current});
    current = call.getResult(0);
  }

  transformerOp.getOutput().replaceAllUsesWith(current);
  if (root->use_empty())
    rewriter.eraseOp(root);
  eraseDeadParameterProducers(originalParameters, rewriter);
}

static bool extractCanonicalTransformers(mlir::func::FuncOp func) {
  llvm::SmallVector<NNTransformerOp> matches;
  func.walk([&](NNTransformerOp transformerOp) {
    if (transformerOp)
      matches.push_back(transformerOp);
  });

  llvm::SmallVector<NNTransformerEncoderOp> encoderMatches;
  func.walk([&](NNTransformerEncoderOp transformerOp) {
    if (transformerOp)
      encoderMatches.push_back(transformerOp);
  });

  llvm::SmallVector<NNTransformerDecoderOp> decoderMatches;
  func.walk([&](NNTransformerDecoderOp transformerOp) {
    if (transformerOp)
      decoderMatches.push_back(transformerOp);
  });

  if (matches.empty() && encoderMatches.empty() && decoderMatches.empty())
    return false;

  mlir::IRRewriter rewriter(func.getContext());
  for (NNTransformerOp match : matches) {
    if (match && match->getBlock())
      outlineTransformerOpToBlockFunctions(match, rewriter);
  }

  for (NNTransformerEncoderOp match : encoderMatches) {
    if (match && match->getBlock())
      outlineTransformerEncoderOpToBlockFunctions(match, rewriter);
  }

  for (NNTransformerDecoderOp match : decoderMatches) {
    if (match && match->getBlock())
      outlineTransformerDecoderOpToBlockFunctions(match, rewriter);
  }

  return true;
}

class TransformerExtractor : public mlir::sculptor::LayerExtractor {
public:
  explicit TransformerExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "transformer"; }

  void extract(mlir::func::FuncOp func) const override {
    (void)extractCanonicalTransformers(func);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerTransformerExtractor(LayerExtractors &extractors,
                                  MLIRContext *context) {
  extractors.push_back(std::make_unique<TransformerExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
