#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"
#include "sculptor-mlir/Dialect/Sculptor/Conversion/Passes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char** argv) {
  DialectRegistry registry;

  registerAllDialects(registry);
  registry.insert<sculptor::SculptorDialect>();

  mlir::sculptor::registerSculptorPasses();
  mlir::sculptor::registerSculptorConversionPasses();
  registerAllPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Sculptor MLIR modular optimizer\n", registry));

}
