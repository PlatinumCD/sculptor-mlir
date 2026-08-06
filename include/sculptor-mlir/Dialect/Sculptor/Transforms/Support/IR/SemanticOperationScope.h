#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_SEMANTICOPERATIONSCOPE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_SEMANTICOPERATIONSCOPE_H

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include <cassert>

namespace mlir::sculptor {

// Records semantic metadata on operations emitted at one insertion point.
// This keeps the decomposition IR flat while preserving the grouping signals
// consumed by mapping and planning passes.
class SemanticOperationScope {
public:
  SemanticOperationScope(OpBuilder &builder, llvm::StringRef section,
                         llvm::StringRef name = {})
      : block(builder.getInsertionBlock()) {
    assert(block && "expected a valid builder insertion block");

    Block::iterator insertionPoint = builder.getInsertionPoint();
    end = insertionPoint == block->end() ? nullptr : &*insertionPoint;
    before = end ? end->getPrevNode()
                 : (block->empty() ? nullptr : &block->back());

    attributes.set("sculptor.semantic.section",
                   builder.getStringAttr(section));
    if (!name.empty())
      attributes.set("sculptor.semantic.name", builder.getStringAttr(name));
  }

  void set(llvm::StringRef name, Attribute value) {
    if (value)
      attributes.set(name, value);
  }

  void copyIfPresent(Operation *source, llvm::StringRef name) {
    if (Attribute value = source->getAttr(name))
      attributes.set(name, value);
  }

  void annotate() {
    Operation *operation =
        before ? before->getNextNode()
               : (block->empty() ? nullptr : &block->front());
    while (operation && operation != end) {
      for (NamedAttribute attribute : attributes) {
        if (!operation->hasAttr(attribute.getName()))
          operation->setAttr(attribute.getName(), attribute.getValue());
      }
      operation = operation->getNextNode();
    }
  }

private:
  Block *block = nullptr;
  Operation *before = nullptr;
  Operation *end = nullptr;
  NamedAttrList attributes;
};

} // namespace mlir::sculptor

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_SEMANTICOPERATIONSCOPE_H
