#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_PASSES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_PASSES_H

namespace mlir {
namespace sculptor {

void registerFinalizeGolemIntrinsicsPass();
void registerEmitRuntimeGraphPass();

// Registers every Sculptor conversion pass exposed to textual pipelines and
// pass managers.
void registerSculptorConversionPasses();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_PASSES_H
