#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_CANONICALIZELAYERS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_CANONICALIZELAYERS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;

namespace sculptor {

// Defines the extension point for rewriting one layer family in forward to
// inline canonical sculptor.nn operations.
class LayerCanonicalizer {
public:
  // Allows canonicalizer implementations to be owned through the base
  // interface.
  virtual ~LayerCanonicalizer() = default;

  // Returns the primary layer kind rewritten by this canonicalizer.
  virtual StringRef getName() const = 0;

  // Finds and rewrites every matching region inside the provided function.
  virtual void canonicalize(func::FuncOp func) const = 0;
};

// Owns canonicalizers in the order they should visit forward.
using LayerCanonicalizers = std::vector<std::unique_ptr<LayerCanonicalizer>>;

// Rewrites supported linalg layer bodies in forward to inline canonical
// sculptor.nn operations. Outlining remains the responsibility of
// sculptor-extract-layers.
struct CanonicalizeLayersPass
    : public mlir::PassWrapper<CanonicalizeLayersPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeLayersPass)

  mlir::StringRef getArgument() const final {
    return "sculptor-canonicalize-layers";
  }

  mlir::StringRef getDescription() const final {
    return "Canonicalize supported layers to sculptor.nn ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect>();
  }

  void runOnOperation() override;
};

// Installs the canonicalizer that rewrites linalg-based linear layers.
void registerLinearCanonicalizer(LayerCanonicalizers &canonicalizers,
                                 MLIRContext *context);

// Installs the canonicalizer that rewrites one-dimensional convolution layers.
void registerConv1DCanonicalizer(LayerCanonicalizers &canonicalizers,
                                 MLIRContext *context);

// Installs the canonicalizer that rewrites two-dimensional convolution layers.
void registerConv2DCanonicalizer(LayerCanonicalizers &canonicalizers,
                                 MLIRContext *context);

// Installs the canonicalizer that rewrites grouped two-dimensional
// convolutions.
void registerConv2DGroupedCanonicalizer(LayerCanonicalizers &canonicalizers,
                                        MLIRContext *context);

// Installs the canonicalizer that rewrites three-dimensional convolution
// layers.
void registerConv3DCanonicalizer(LayerCanonicalizers &canonicalizers,
                                 MLIRContext *context);

// Installs the canonicalizer that rewrites full RNNs.
void registerRNNCanonicalizer(LayerCanonicalizers &canonicalizers,
                              MLIRContext *context);

// Installs the canonicalizer that rewrites full LSTMs.
void registerLSTMCanonicalizer(LayerCanonicalizers &canonicalizers,
                               MLIRContext *context);

// Installs the canonicalizer that rewrites full GRUs.
void registerGRUCanonicalizer(LayerCanonicalizers &canonicalizers,
                              MLIRContext *context);

// Installs the canonicalizer that rewrites simple RNN cell layers.
void registerRNNCellCanonicalizer(LayerCanonicalizers &canonicalizers,
                                  MLIRContext *context);

// Installs the canonicalizer that rewrites LSTM cell layers.
void registerLSTMCellCanonicalizer(LayerCanonicalizers &canonicalizers,
                                   MLIRContext *context);

// Installs the canonicalizer that rewrites GRU cell layers.
void registerGRUCellCanonicalizer(LayerCanonicalizers &canonicalizers,
                                  MLIRContext *context);

// Registers the layer canonicalization pass with MLIR's global pass registry.
void registerCanonicalizeLayersPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_CANONICALIZELAYERS_H
