#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_OUTLINETILEROUTINES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_OUTLINETILEROUTINES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct OutlineTileRoutinesPass
    : public PassWrapper<OutlineTileRoutinesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineTileRoutinesPass)

  StringRef getArgument() const final {
    return "sculptor-outline-tile-routines";
  }
  StringRef getDescription() const final {
    return "Outline a locked logical-tile placement into per-tile routines";
  }

  Option<bool> fuseProducerConsumer{
      *this, "fuse-producer-consumer",
      llvm::cl::desc("Fuse safe linear same-tile digital routine chains"),
      llvm::cl::init(false)};

  Option<bool> consolidateLayerRegions{
      *this, "consolidate-layer-regions",
      llvm::cl::desc("Consolidate connected same-layer digital work on one "
                     "physical tile into one routine region"),
      llvm::cl::init(false)};

  Option<int64_t> sequenceWavesInFlight{
      *this, "sequence-waves-in-flight",
      llvm::cl::desc(
          "Maximum sequence-sharded MVM waves simultaneously in flight"),
      llvm::cl::init(1)};

  OutlineTileRoutinesPass() = default;
  OutlineTileRoutinesPass(const OutlineTileRoutinesPass &pass)
      : PassWrapper(pass) {
    fuseProducerConsumer = pass.fuseProducerConsumer;
    consolidateLayerRegions = pass.consolidateLayerRegions;
    sequenceWavesInFlight = pass.sequenceWavesInFlight;
  }

  void runOnOperation() override;
};

void registerOutlineTileRoutinesPass();

} // namespace sculptor
} // namespace mlir

#endif
