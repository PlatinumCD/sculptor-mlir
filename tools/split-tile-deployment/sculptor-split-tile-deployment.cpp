#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractTileModule.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::sculptor;

constexpr StringLiteral kActiveTileIdsAttr =
    "sculptor.deployment.active_tile_ids";

llvm::cl::OptionCategory splitCategory(
    "Sculptor tile deployment splitting options");

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<outlined deployment MLIR>"),
    llvm::cl::Required, llvm::cl::cat(splitCategory));

llvm::cl::opt<std::string> outputDirectory(
    "output-directory",
    llvm::cl::desc("Directory for core-<tile>-extracted.mlir files"),
    llvm::cl::value_desc("directory"), llvm::cl::Required,
    llvm::cl::cat(splitCategory));

llvm::cl::opt<std::string> manifestFilename(
    "manifest", llvm::cl::desc("Output active-tile manifest"),
    llvm::cl::value_desc("filename"), llvm::cl::Required,
    llvm::cl::cat(splitCategory));

LogicalResult writeModule(ModuleOp module, StringRef filename,
                          FallbackAsmResourceMap &resources) {
  std::error_code error;
  llvm::raw_fd_ostream stream(filename, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "failed to open '" << filename
                 << "' for writing: " << error.message() << '\n';
    return failure();
  }
  OpPrintingFlags flags;
  AsmState state(module, flags, /*locationMap=*/nullptr, &resources);
  module->print(stream, state);
  stream << '\n';
  return success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::HideUnrelatedOptions(splitCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Split a Sculptor deployment once\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<SculptorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  FallbackAsmResourceMap resources;
  ParserConfig parserConfig(&context, /*verifyAfterParse=*/true, &resources);
  OwningOpRef<ModuleOp> deployment =
      parseSourceFile<ModuleOp>(inputFilename, parserConfig);
  if (!deployment) {
    llvm::errs() << "failed to parse deployment '" << inputFilename << "'\n";
    return 1;
  }
  if (failed(verify(*deployment))) {
    llvm::errs() << "deployment failed verification\n";
    return 1;
  }

  auto activeIds = (*deployment)->getAttrOfType<ArrayAttr>(
      kActiveTileIdsAttr);
  if (!activeIds) {
    (*deployment).emitError("expected active tile ID metadata");
    return 1;
  }
  if (std::error_code error =
          llvm::sys::fs::create_directories(outputDirectory)) {
    llvm::errs() << "failed to create output directory '" << outputDirectory
                 << "': " << error.message() << '\n';
    return 1;
  }

  std::error_code manifestError;
  llvm::raw_fd_ostream manifest(manifestFilename, manifestError,
                                llvm::sys::fs::OF_Text);
  if (manifestError) {
    llvm::errs() << "failed to open manifest '" << manifestFilename
                 << "': " << manifestError.message() << '\n';
    return 1;
  }

  for (Attribute value : activeIds) {
    auto id = dyn_cast<IntegerAttr>(value);
    if (!id || id.getInt() < 0) {
      (*deployment).emitError(
          "active tile IDs must be non-negative integers");
      return 1;
    }
    int64_t tileId = id.getInt();
    FailureOr<OwningOpRef<ModuleOp>> extracted =
        cloneExtractedTileModule(*deployment, tileId);
    if (failed(extracted))
      return 1;
    llvm::SmallString<256> filename(outputDirectory);
    llvm::sys::path::append(filename,
                            "core-" + std::to_string(tileId) +
                                "-extracted.mlir");
    if (failed(writeModule(**extracted, filename, resources)))
      return 1;
    manifest << tileId << '\n';
  }
  return 0;
}
