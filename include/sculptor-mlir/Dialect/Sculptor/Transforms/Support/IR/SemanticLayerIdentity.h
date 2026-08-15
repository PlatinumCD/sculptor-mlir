#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_SEMANTICLAYERIDENTITY_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_SEMANTICLAYERIDENTITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::sculptor {

// Stable parent-layer metadata shared by semantic decomposition, mapping, and
// routine formation. These attributes identify the canonical layer that owns
// an operation; they are deliberately independent of mapping-stage identity.
inline constexpr llvm::StringLiteral kSemanticLayerIdAttrName =
    "sculptor.semantic.layer_id";
inline constexpr llvm::StringLiteral kSemanticLayerKindAttrName =
    "sculptor.semantic.layer_kind";

inline void copySemanticLayerIdentity(Operation *source, Operation *target) {
  if (!source || !target)
    return;
  for (llvm::StringRef name : {kSemanticLayerIdAttrName,
                               kSemanticLayerKindAttrName}) {
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
  }
}

// Propagates the active canonical layer identity to every operation emitted by
// a rewriter. This is linear in emitted IR size and also covers operations
// created inside nested regions by builders that inherit the listener.
class SemanticLayerRewriteListener final : public OpBuilder::Listener {
public:
  void setSource(Operation *source) {
    attributes.clear();
    if (!source)
      return;
    for (llvm::StringRef name : {kSemanticLayerIdAttrName,
                                 kSemanticLayerKindAttrName}) {
      if (Attribute value = source->getAttr(name))
        attributes.set(name, value);
    }
  }

  void clearSource() { attributes.clear(); }

  void notifyOperationInserted(Operation *operation,
                               OpBuilder::InsertPoint previous) override {
    (void)previous;
    for (NamedAttribute attribute : attributes) {
      if (!operation->hasAttr(attribute.getName()))
        operation->setAttr(attribute.getName(), attribute.getValue());
    }
  }

private:
  NamedAttrList attributes;
};

// Prevents an early-return path from leaking one layer's identity into the
// next rewrite performed by a shared listener.
class SemanticLayerRewriteScope final {
public:
  SemanticLayerRewriteScope(SemanticLayerRewriteListener &listener,
                            Operation *source)
      : listener(listener) {
    listener.setSource(source);
  }

  ~SemanticLayerRewriteScope() { listener.clearSource(); }

  SemanticLayerRewriteScope(const SemanticLayerRewriteScope &) = delete;
  SemanticLayerRewriteScope &
  operator=(const SemanticLayerRewriteScope &) = delete;

private:
  SemanticLayerRewriteListener &listener;
};

} // namespace mlir::sculptor

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_SEMANTICLAYERIDENTITY_H
