#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PASSES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PASSES_H

namespace mlir {
namespace sculptor {

// Registers every Sculptor transform pass exposed to textual pipelines and pass
// managers.
void registerSculptorPasses();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PASSES_H
