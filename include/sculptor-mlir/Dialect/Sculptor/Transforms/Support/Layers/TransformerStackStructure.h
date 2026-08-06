#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_TRANSFORMERSTACKSTRUCTURE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_TRANSFORMERSTACKSTRUCTURE_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace transformer_structure {

// Describes one semantic Transformer block without assigning a function,
// task, backend, or placement boundary.
struct TransformerBlockDescription {
  Value qkvWeight;
  Value qkvBias;
  Value attnOutputWeight;
  Value attnOutputBias;
  Value attnNormWeight;
  Value attnNormBias;
  Value crossQueryWeight;
  Value crossQueryBias;
  Value crossKeyValueWeight;
  Value crossKeyValueBias;
  Value crossOutputWeight;
  Value crossOutputBias;
  Value crossNormWeight;
  Value crossNormBias;
  Value mlpUpWeight;
  Value mlpUpBias;
  Value mlpDownWeight;
  Value mlpDownBias;
  Value mlpNormWeight;
  Value mlpNormBias;
  Value finalNormWeight;
  Value finalNormBias;
  bool isDecoder = false;
  bool hasCrossAttention = false;
  bool hasFinalNorm = false;
  int64_t blockIndex = 0;
  int64_t numBlocks = 0;
};

// Preserves stack hierarchy and parameter ownership without mutating IR.
struct TransformerStackDescription {
  llvm::SmallVector<TransformerBlockDescription> encoderBlocks;
  llvm::SmallVector<TransformerBlockDescription> decoderBlocks;
};

FailureOr<TransformerStackDescription>
describeTransformerStack(NNTransformerOp transformerOp);

FailureOr<TransformerStackDescription>
describeTransformerStack(NNTransformerEncoderOp transformerOp);

FailureOr<TransformerStackDescription>
describeTransformerStack(NNTransformerDecoderOp transformerOp);

} // namespace transformer_structure
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_TRANSFORMERSTACKSTRUCTURE_H
