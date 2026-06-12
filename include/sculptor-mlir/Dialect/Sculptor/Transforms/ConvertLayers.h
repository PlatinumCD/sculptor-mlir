#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_CONVERTLAYERS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_CONVERTLAYERS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;

namespace sculptor {

// Defines the extension point for lowering one extracted sculptor.nn layer body
// into sculptor.mvm plus standard tensor/linalg/math/scf glue.
class LayerToMVMConverter {
public:
  // Allows converter implementations to be owned through the base interface.
  virtual ~LayerToMVMConverter() = default;

  // Identifies the layer_type this converter claims in dispatch maps.
  virtual StringRef getName() const = 0;

  // Lowers an extracted sculptor.nn layer body to sculptor.mvm in place.
  virtual void lowerToMVM(func::FuncOp func) const = 0;
};

// Owns converter instances while dispatch tables keep non-owning pointers.
using LayerToMVMConverters = std::vector<std::unique_ptr<LayerToMVMConverter>>;

// Maps layer_type strings to the converter that can lower them to sculptor.mvm.
using LayerToMVMConverterMap = llvm::StringMap<const LayerToMVMConverter *>;

// Scans forward's layer calls and delegates extracted sculptor.nn layer functions
// to converters that lower them to sculptor.mvm. The public pass name stays
// sculptor-convert-layers for compatibility even though the internal boundary is
// now sculptor.nn layer-to-MVM lowering.
struct ConvertLayersPass
    : public mlir::PassWrapper<ConvertLayersPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLayersPass)

  // Uses the default option values for command-line pass construction.
  ConvertLayersPass() = default;

  // Keeps MLIR pass cloning explicit even though this pass has no options.
  ConvertLayersPass(const ConvertLayersPass &pass) : PassWrapper(pass) {}

  // Provides the stable command-line name used to schedule this transform.
  mlir::StringRef getArgument() const final { return "sculptor-convert-layers"; }

  // Summarizes the pass behavior for MLIR pass registration and help text.
  mlir::StringRef getDescription() const final {
    return "Lower extracted sculptor.nn layers to sculptor.mvm";
  }

  // Ensures sculptor.mvm and standard glue operations introduced by converters
  // exist.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::sculptor::SculptorDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  // Lowers known extracted sculptor.nn layer functions called by forward.
  void runOnOperation() override;
};

// Installs the built-in linear layer-to-MVM converter and every layer_type
// alias it handles.
void registerLinearConverter(LayerToMVMConverters &converters,
                             LayerToMVMConverterMap &converterMap,
                             MLIRContext *context);

// Installs the built-in Conv1D layer-to-MVM converter and every layer_type
// alias it handles.
void registerConv1DConverter(LayerToMVMConverters &converters,
                             LayerToMVMConverterMap &converterMap,
                             MLIRContext *context);

// Installs the built-in Conv2D layer-to-MVM converter and every layer_type
// alias it handles.
void registerConv2DConverter(LayerToMVMConverters &converters,
                             LayerToMVMConverterMap &converterMap,
                             MLIRContext *context);

// Installs the built-in grouped Conv2D layer-to-MVM converter and every
// layer_type alias it handles.
void registerConv2DGroupedConverter(LayerToMVMConverters &converters,
                                    LayerToMVMConverterMap &converterMap,
                                    MLIRContext *context);

// Installs the built-in Conv3D layer-to-MVM converter and every layer_type
// alias it handles.
void registerConv3DConverter(LayerToMVMConverters &converters,
                             LayerToMVMConverterMap &converterMap,
                             MLIRContext *context);

// Installs the built-in RNN cell layer-to-MVM converter and every layer_type
// alias it handles.
void registerRNNCellConverter(LayerToMVMConverters &converters,
                              LayerToMVMConverterMap &converterMap,
                              MLIRContext *context);

// Installs the built-in LSTM cell layer-to-MVM converter and every layer_type
// alias it handles.
void registerLSTMCellConverter(LayerToMVMConverters &converters,
                               LayerToMVMConverterMap &converterMap,
                               MLIRContext *context);

// Installs the built-in LSTM layer-to-MVM converter and every layer_type alias
// it handles.
void registerLSTMConverter(LayerToMVMConverters &converters,
                           LayerToMVMConverterMap &converterMap,
                           MLIRContext *context);

// Installs the built-in GRU cell layer-to-MVM converter and every layer_type
// alias it handles.
void registerGRUCellConverter(LayerToMVMConverters &converters,
                              LayerToMVMConverterMap &converterMap,
                              MLIRContext *context);

// Installs the built-in GRU layer-to-MVM converter and every layer_type alias
// it handles.
void registerGRUConverter(LayerToMVMConverters &converters,
                          LayerToMVMConverterMap &converterMap,
                          MLIRContext *context);

// Installs the built-in RNN layer-to-MVM converter and every layer_type alias
// it handles.
void registerRNNConverter(LayerToMVMConverters &converters,
                          LayerToMVMConverterMap &converterMap,
                          MLIRContext *context);

// Registers the sculptor.nn-to-MVM lowering pass with MLIR's global pass registry.
void registerConvertLayersPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_CONVERTLAYERS_H
