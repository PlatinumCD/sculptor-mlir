#include "sculptor-mlir/Dialect/Sculptor/Transforms/Passes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/AnalyzeTaskGraphTiming.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/AssembleTaskGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/BalanceTaskGraphReductions.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/BuildTaskGraphIslands.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphIslandMap.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphSimModel.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphVis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractCoreModule.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/FinalizeTaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/FuseTaskGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/ExpandMVMToGolem.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/LowerScheduledMVMToDigital.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/MaterializeTasks.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/OptimizeTaskGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/PartitionTaskGraphByCore.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ScheduleTaskGraph.h"

namespace mlir {
namespace sculptor {

// Registers the transform pass bundle exposed by this library entry point.
void registerSculptorPasses() {
  registerAnalyzeTaskGraphTimingPass();
  registerAssembleTaskGraphPass();
  registerBalanceTaskGraphReductionsPass();
  registerBuildTaskGraphIslandsPass();
  registerCanonicalizeLayersPass();
  registerConvertLayersPass();
  registerExportTaskGraphIslandMapPass();
  registerExportTaskGraphSimModelPass();
  registerExportTaskGraphVisPass();
  registerExpandMVMToGolemPass();
  registerExtractCoreModulePass();
  registerExtractLayersPass();
  registerFinalizeTaskGraphResourcesPass();
  registerFuseTaskGraphPass();
  registerLowerScheduledMVMToDigitalPass();
  registerMaterializeTasksPass();
  registerOptimizeTaskGraphPass();
  registerPartitionTaskGraphByCorePass();
  registerScheduleTaskGraphPass();
}

} // namespace sculptor
} // namespace mlir
