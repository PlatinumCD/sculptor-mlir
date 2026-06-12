#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_NNLAYERMATCHUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_NNLAYERMATCHUTILS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace sculptor {
namespace nn_layer_match {

// Reads the extracted layer kind used to select a converter.
inline StringAttr getLayerType(Operation *op) {
  if (!op)
    return {};
  return op->getAttrOfType<StringAttr>("layer_type");
}

// Compares a layer_type attribute while treating missing metadata as no match.
inline bool hasLayerType(Operation *op, StringRef layerType) {
  auto attr = getLayerType(op);
  return attr && attr.getValue() == layerType;
}

// Checks that a layer_type alias matches the sculptor.nn op bias form.
inline bool hasLayerTypeMatchingBias(Operation *op, StringRef noBiasLayerType,
                                     StringRef biasLayerType, bool hasBias) {
  bool isBiasFreeLayer = hasLayerType(op, noBiasLayerType);
  bool isBiasLayer = hasLayerType(op, biasLayerType);
  if (!isBiasFreeLayer && !isBiasLayer)
    return false;
  return hasBias == isBiasLayer;
}

// Identifies extracted sculptor.nn layer functions ready for NN-to-MVM lowering.
inline bool isSculptorLayer(Operation *op) {
  return static_cast<bool>(getLayerType(op));
}

// Describes an extracted sculptor.nn layer body once its shape has been validated.
struct MatchedNNLayerBody {
  Operation *op = nullptr;
  func::ReturnOp returnOp;
};

// Finds a single sculptor.nn op in a simple extracted analog layer body.
// Unsupported or ambiguous bodies are reported as failure silently.
FailureOr<MatchedNNLayerBody>
matchSingleNNLayerBody(func::FuncOp func, StringRef operationName);

template <typename OpT>
FailureOr<OpT> matchSingleNNLayerOp(func::FuncOp func) {
  auto body = matchSingleNNLayerBody(func, OpT::getOperationName());
  if (failed(body))
    return failure();

  auto op = llvm::dyn_cast<OpT>((*body).op);
  if (!op)
    return failure();

  return op;
}

} // namespace nn_layer_match
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_NNLAYERMATCHUTILS_H
