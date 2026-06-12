#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXTRACTLAYERS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXTRACTLAYERS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;

namespace sculptor {

// Defines the extension point for recognizing one layer family in forward and
// outlining each match into a layer function.
class LayerExtractor {
public:
  // Allows extractor implementations to be owned through the base interface.
  virtual ~LayerExtractor() = default;

  // Returns the primary layer kind recognized by this extractor.
  virtual StringRef getName() const = 0;

  // Finds and rewrites every matching region inside the provided function.
  virtual void extract(func::FuncOp func) const = 0;
};

// Owns extractors in the order they should visit forward.
using LayerExtractors = std::vector<std::unique_ptr<LayerExtractor>>;

// Scans the module for forward and delegates pattern outlining to the
// registered layer extractors.
struct ExtractLayersPass
    : public mlir::PassWrapper<ExtractLayersPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractLayersPass)

  // Provides the stable command-line name used to schedule this transform.
  mlir::StringRef getArgument() const final { return "sculptor-extract-layers"; }

  // Summarizes the pass behavior for MLIR pass registration and help text.
  mlir::StringRef getDescription() const final {
    return "Extract analog layers";
  }

  // Ensures canonical analog layer operations can be materialized by
  // extractors that replace matched linalg bodies.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::tensor::TensorDialect>();
  }

  // Builds the extractor pipeline and applies it to the module's forward.
  void runOnOperation() override;
};

// Installs the extractor that recognizes linalg-based linear layers.
void registerLinearExtractor(LayerExtractors &extractors, MLIRContext *context);

// Installs the extractor that recognizes one-dimensional convolution layers.
void registerConv1DExtractor(LayerExtractors &extractors, MLIRContext *context);

// Installs the extractor that recognizes three-dimensional convolution layers.
void registerConv3DExtractor(LayerExtractors &extractors, MLIRContext *context);

// Installs the extractor that recognizes two-dimensional convolution layers.
void registerConv2DExtractor(LayerExtractors &extractors, MLIRContext *context);

// Installs the extractor that recognizes grouped two-dimensional convolutions.
void registerConv2DGroupedExtractor(LayerExtractors &extractors,
                                    MLIRContext *context);

// Installs the extractor that recognizes simple recurrent cell computations.
void registerRNNCellExtractor(LayerExtractors &extractors,
                              MLIRContext *context);

// Installs the extractor that recognizes supported RNN computations.
void registerRNNExtractor(LayerExtractors &extractors, MLIRContext *context);

// Installs the extractor that recognizes supported LSTM computations.
void registerLSTMExtractor(LayerExtractors &extractors, MLIRContext *context);

// Installs the extractor that recognizes supported GRU computations.
void registerGRUExtractor(LayerExtractors &extractors, MLIRContext *context);

// Installs the extractor that recognizes LSTM cell computations.
void registerLSTMCellExtractor(LayerExtractors &extractors,
                               MLIRContext *context);

// Installs the extractor that recognizes GRU cell computations.
void registerGRUCellExtractor(LayerExtractors &extractors,
                              MLIRContext *context);

// Registers the layer extraction pass with MLIR's global pass registry.
void registerExtractLayersPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXTRACTLAYERS_H
