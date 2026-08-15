#include "sculptor-mlir/Dialect/Sculptor/Transforms/Passes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ApplyMappingPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/AuditTileBufferization.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/BindTileRoutineDestinations.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/BuildRATree.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExpandDigitalWork.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractTileModule.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/FoldAttentionLayouts.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/FoldInferenceParameters.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/FuseElementwiseRegions.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/FinalizeTileRuntimeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/DuplicateMatrices.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/ExpandMVMToGolem.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/MaterializeTileRuntimeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/OutlineTileRoutines.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlaceLogicalTiles.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlanMapping.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlanTileScratchpad.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ReportTileMemory.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/VectorizeDigitalKernels.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/VectorizeTileCopies.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlannerRegistry.h"

namespace mlir {
namespace sculptor {

// Registers the transform pass bundle exposed by this library entry point.
void registerSculptorPasses() {
  mapping::registerMappingPlanners();
  registerApplyMappingPlanPass();
  registerAuditTileBufferizationPass();
  registerBuildRATreePass();
  registerBindTileRoutineDestinationsPass();
  registerCanonicalizeLayersPass();
  registerConvertLayersPass();
  registerExpandDigitalWorkPass();
  registerDuplicateMatricesPass();
  registerExpandMVMToGolemPass();
  registerExtractTileModulePass();
  registerExtractLayersPass();
  registerFoldAttentionLayoutsPass();
  registerFoldInferenceParametersPass();
  registerFuseElementwiseRegionsPass();
  registerFinalizeTileRuntimeGraphPass();
  registerMaterializeTileRuntimeGraphPass();
  registerOutlineTileRoutinesPass();
  registerPlaceLogicalTilesPass();
  registerPlanTileScratchpadPass();
  registerPlanMappingPass();
  registerReportTileMemoryPass();
  registerVectorizeDigitalKernelsPass();
  registerVectorizeTileCopiesPass();
}

} // namespace sculptor
} // namespace mlir
