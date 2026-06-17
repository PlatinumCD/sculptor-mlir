#include "sculptor-mlir/Dialect/Sculptor/Transforms/Passes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/AssembleTaskGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphSimModel.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphVis.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/ExpandMVMToGolem.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/MaterializeTasks.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ScheduleTaskGraph.h"

namespace mlir {
namespace sculptor {

// Registers the transform pass bundle exposed by this library entry point.
void registerSculptorPasses() {
  registerAssembleTaskGraphPass();
  registerCanonicalizeLayersPass();
  registerConvertLayersPass();
  registerExportTaskGraphSimModelPass();
  registerExportTaskGraphVisPass();
  registerExpandMVMToGolemPass();
  registerExtractLayersPass();
  registerMaterializeTasksPass();
  registerScheduleTaskGraphPass();
}

} // namespace sculptor
} // namespace mlir
