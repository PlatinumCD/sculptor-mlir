#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/TransformerStackStructure.h"

namespace mlir {
namespace sculptor {
namespace transformer_structure {
namespace {

bool takeParameter(ValueRange parameters, int64_t &offset, Value &value) {
  if (offset >= static_cast<int64_t>(parameters.size()))
    return false;
  value = parameters[offset++];
  return true;
}

bool takeOptionalParameter(ValueRange parameters, int64_t &offset, bool present,
                           Value &value) {
  if (!present)
    return true;
  return takeParameter(parameters, offset, value);
}

bool parseEncoderBlockParameters(ValueRange parameters, int64_t &offset,
                                 bool hasProjectionBias,
                                 bool hasLayerNormAffine, bool hasLayerNormBias,
                                 int64_t blockIndex, int64_t numBlocks,
                                 TransformerBlockDescription &block) {
  block = TransformerBlockDescription{};
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
         takeOptionalParameter(parameters, offset, hasLayerNormBias,
                               block.attnNormBias) &&
         takeParameter(parameters, offset, block.mlpUpWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.mlpUpBias) &&
         takeParameter(parameters, offset, block.mlpDownWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.mlpDownBias) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.mlpNormWeight) &&
         takeOptionalParameter(parameters, offset, hasLayerNormBias,
                               block.mlpNormBias);
}

bool parseDecoderBlockParameters(ValueRange parameters, int64_t &offset,
                                 bool hasProjectionBias,
                                 bool hasLayerNormAffine, bool hasLayerNormBias,
                                 int64_t blockIndex, int64_t numBlocks,
                                 TransformerBlockDescription &block) {
  block = TransformerBlockDescription{};
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
         takeOptionalParameter(parameters, offset, hasLayerNormBias,
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
         takeOptionalParameter(parameters, offset, hasLayerNormBias,
                               block.crossNormBias) &&
         takeParameter(parameters, offset, block.mlpUpWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.mlpUpBias) &&
         takeParameter(parameters, offset, block.mlpDownWeight) &&
         takeOptionalParameter(parameters, offset, hasProjectionBias,
                               block.mlpDownBias) &&
         takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               block.mlpNormWeight) &&
         takeOptionalParameter(parameters, offset, hasLayerNormBias,
                               block.mlpNormBias);
}

} // namespace

FailureOr<TransformerStackDescription>
describeTransformerStack(NNTransformerOp transformerOp) {
  if (!transformerOp)
    return failure();

  int64_t numEncoderBlocks = transformerOp.getNumEncoderBlocks();
  int64_t numDecoderBlocks = transformerOp.getNumDecoderBlocks();
  if (numEncoderBlocks < 1 || numDecoderBlocks < 1)
    return failure();

  ValueRange parameters = transformerOp.getParameters();
  int64_t offset = 0;
  bool hasProjectionBias = transformerOp.getHasProjectionBias();
  bool hasLayerNormAffine = transformerOp.getHasLayerNormAffine();
  bool hasLayerNormBias = transformerOp.getHasLayerNormBias();

  TransformerStackDescription description;
  description.encoderBlocks.reserve(numEncoderBlocks);
  description.decoderBlocks.reserve(numDecoderBlocks);

  for (int64_t block = 0; block < numEncoderBlocks; ++block) {
    TransformerBlockDescription blockDescription;
    if (!parseEncoderBlockParameters(parameters, offset, hasProjectionBias,
                                     hasLayerNormAffine, hasLayerNormBias,
                                     block, numEncoderBlocks, blockDescription))
      return failure();
    description.encoderBlocks.push_back(blockDescription);
  }

  for (int64_t block = 0; block < numDecoderBlocks; ++block) {
    TransformerBlockDescription blockDescription;
    if (!parseDecoderBlockParameters(parameters, offset, hasProjectionBias,
                                     hasLayerNormAffine, hasLayerNormBias,
                                     block, numDecoderBlocks, blockDescription))
      return failure();
    description.decoderBlocks.push_back(blockDescription);
  }

  if (transformerOp.getHasFinalNorm()) {
    TransformerBlockDescription &encoder = description.encoderBlocks.back();
    TransformerBlockDescription &decoder = description.decoderBlocks.back();
    encoder.hasFinalNorm = true;
    decoder.hasFinalNorm = true;
    if ((!takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                                encoder.finalNormWeight) ||
         !takeOptionalParameter(parameters, offset, hasLayerNormBias,
                                encoder.finalNormBias) ||
         !takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                                decoder.finalNormWeight) ||
         !takeOptionalParameter(parameters, offset, hasLayerNormBias,
                                decoder.finalNormBias)))
      return failure();
  }

  if (offset != static_cast<int64_t>(parameters.size()))
    return failure();
  return description;
}

FailureOr<TransformerStackDescription>
describeTransformerStack(NNTransformerEncoderOp transformerOp) {
  if (!transformerOp)
    return failure();

  int64_t numBlocks = transformerOp.getNumBlocks();
  if (numBlocks < 1)
    return failure();

  ValueRange parameters = transformerOp.getParameters();
  int64_t offset = 0;
  bool hasProjectionBias = transformerOp.getHasProjectionBias();
  bool hasLayerNormAffine = transformerOp.getHasLayerNormAffine();
  bool hasLayerNormBias = transformerOp.getHasLayerNormBias();

  TransformerStackDescription description;
  description.encoderBlocks.reserve(numBlocks);
  for (int64_t block = 0; block < numBlocks; ++block) {
    TransformerBlockDescription blockDescription;
    if (!parseEncoderBlockParameters(parameters, offset, hasProjectionBias,
                                     hasLayerNormAffine, hasLayerNormBias,
                                     block, numBlocks, blockDescription))
      return failure();
    description.encoderBlocks.push_back(blockDescription);
  }

  if (transformerOp.getHasFinalNorm()) {
    TransformerBlockDescription &last = description.encoderBlocks.back();
    last.hasFinalNorm = true;
    if (!takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               last.finalNormWeight) ||
        !takeOptionalParameter(parameters, offset, hasLayerNormBias,
                               last.finalNormBias))
      return failure();
  }

  if (offset != static_cast<int64_t>(parameters.size()))
    return failure();
  return description;
}

FailureOr<TransformerStackDescription>
describeTransformerStack(NNTransformerDecoderOp transformerOp) {
  if (!transformerOp)
    return failure();

  int64_t numBlocks = transformerOp.getNumBlocks();
  if (numBlocks < 1)
    return failure();

  ValueRange parameters = transformerOp.getParameters();
  int64_t offset = 0;
  bool hasProjectionBias = transformerOp.getHasProjectionBias();
  bool hasLayerNormAffine = transformerOp.getHasLayerNormAffine();
  bool hasLayerNormBias = transformerOp.getHasLayerNormBias();

  TransformerStackDescription description;
  description.decoderBlocks.reserve(numBlocks);
  for (int64_t block = 0; block < numBlocks; ++block) {
    TransformerBlockDescription blockDescription;
    if (!parseEncoderBlockParameters(parameters, offset, hasProjectionBias,
                                     hasLayerNormAffine, hasLayerNormBias,
                                     block, numBlocks, blockDescription))
      return failure();
    blockDescription.isDecoder = true;
    blockDescription.hasCrossAttention = false;
    description.decoderBlocks.push_back(blockDescription);
  }

  if (transformerOp.getHasFinalNorm()) {
    TransformerBlockDescription &last = description.decoderBlocks.back();
    last.hasFinalNorm = true;
    if (!takeOptionalParameter(parameters, offset, hasLayerNormAffine,
                               last.finalNormWeight) ||
        !takeOptionalParameter(parameters, offset, hasLayerNormBias,
                               last.finalNormBias))
      return failure();
  }

  if (offset != static_cast<int64_t>(parameters.size()))
    return failure();
  return description;
}

} // namespace transformer_structure
} // namespace sculptor
} // namespace mlir
