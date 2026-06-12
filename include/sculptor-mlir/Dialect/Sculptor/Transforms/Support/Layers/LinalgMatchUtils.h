#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_LINALGMATCHUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_LINALGMATCHUTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace linalg_match {

inline bool isDimProjectionMap(AffineMap map, unsigned loopRank,
                               unsigned firstProjectedDim) {
  if (!map || map.getNumDims() != loopRank || map.getNumSymbols() != 0 ||
      firstProjectedDim > loopRank ||
      map.getNumResults() != loopRank - firstProjectedDim)
    return false;

  for (auto [index, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() != firstProjectedDim + index)
      return false;
  }

  return true;
}

inline bool isIdentityMap(AffineMap map, unsigned rank) {
  return isDimProjectionMap(map, rank, /*firstProjectedDim=*/0);
}

inline bool hasParallelIterators(linalg::GenericOp genericOp,
                                 unsigned loopRank) {
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  if (iteratorTypes.size() != loopRank)
    return false;

  return llvm::all_of(iteratorTypes, [](utils::IteratorType iterator) {
    return iterator == utils::IteratorType::parallel;
  });
}

inline bool hasElementwiseIndexingMaps(linalg::GenericOp genericOp,
                                       unsigned rank) {
  if (!hasParallelIterators(genericOp, rank))
    return false;

  auto maps = genericOp.getIndexingMapsArray();
  return maps.size() == 2 && isIdentityMap(maps[0], rank) &&
         isIdentityMap(maps[1], rank);
}

inline bool hasBiasAddIndexingMaps(linalg::GenericOp genericOp, unsigned rank,
                                   unsigned biasInputIndex) {
  if (rank == 0 || biasInputIndex > 1 || !hasParallelIterators(genericOp, rank))
    return false;

  auto maps = genericOp.getIndexingMapsArray();
  if (maps.size() != 3 || !isIdentityMap(maps[2], rank))
    return false;

  for (unsigned inputIndex = 0; inputIndex < 2; ++inputIndex) {
    bool isExpectedMap =
        inputIndex == biasInputIndex
            ? isDimProjectionMap(maps[inputIndex], rank, rank - 1)
            : isIdentityMap(maps[inputIndex], rank);
    if (!isExpectedMap)
      return false;
  }

  return true;
}

inline bool hasPreActivationAddIndexingMaps(linalg::GenericOp genericOp,
                                            unsigned inputSliceIndex) {
  constexpr unsigned rank = 3;
  if (inputSliceIndex > 1 || !hasParallelIterators(genericOp, rank))
    return false;

  auto maps = genericOp.getIndexingMapsArray();
  if (maps.size() != 3 || !isIdentityMap(maps[2], rank))
    return false;

  unsigned recurrentIndex = inputSliceIndex == 0 ? 1 : 0;
  return isDimProjectionMap(maps[inputSliceIndex], rank,
                            /*firstProjectedDim=*/1) &&
         isIdentityMap(maps[recurrentIndex], rank);
}

inline bool hasWeightBroadcastIndexingMaps(linalg::GenericOp genericOp) {
  constexpr unsigned rank = 3;
  if (!hasParallelIterators(genericOp, rank))
    return false;

  auto maps = genericOp.getIndexingMapsArray();
  return maps.size() == 2 &&
         isDimProjectionMap(maps[0], rank, /*firstProjectedDim=*/1) &&
         isIdentityMap(maps[1], rank);
}

inline bool hasExpectedBodyShape(linalg::GenericOp genericOp,
                                 unsigned inputCount) {
  if (!genericOp || genericOp.getInputs().size() != inputCount ||
      genericOp.getOutputs().size() != 1 || genericOp.getNumResults() != 1 ||
      !genericOp.getRegion().hasOneBlock())
    return false;

  Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != inputCount + 1 ||
      body.getOperations().size() != 2)
    return false;

  return llvm::isa<linalg::YieldOp>(body.getOperations().back());
}

inline bool hasPermutation(linalg::TransposeOp transpose,
                           llvm::ArrayRef<int64_t> expected) {
  if (!transpose || transpose.getPermutation().size() != expected.size())
    return false;

  for (auto [actual, expectedValue] :
       llvm::zip(transpose.getPermutation(), expected)) {
    if (actual != expectedValue)
      return false;
  }
  return true;
}

} // namespace linalg_match
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_LINALGMATCHUTILS_H
