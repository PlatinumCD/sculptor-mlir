#include "sculptor-mlir/Dialect/Sculptor/Conversion/InstrumentTileHeap.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallVector.h"

namespace {

using namespace mlir;

constexpr llvm::StringLiteral kMallocName = "malloc";
constexpr llvm::StringLiteral kFreeName = "free";
constexpr llvm::StringLiteral kProfiledMallocName =
    "golem_runtime_profiled_malloc";
constexpr llvm::StringLiteral kProfiledFreeName = "golem_runtime_profiled_free";

FailureOr<LLVM::LLVMFuncOp> getOrCreateWrapper(ModuleOp module,
                                               LLVM::LLVMFuncOp original,
                                               StringRef wrapperName) {
  if (auto wrapper = module.lookupSymbol<LLVM::LLVMFuncOp>(wrapperName)) {
    if (wrapper.getFunctionType() != original.getFunctionType()) {
      return wrapper.emitError()
             << "heap profiling wrapper '" << wrapperName
             << "' has a type that differs from @" << original.getName();
    }
    return wrapper;
  }

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<LLVM::LLVMFuncOp>(original.getLoc(), wrapperName,
                                          original.getFunctionType());
}

LogicalResult instrumentCalls(ModuleOp module, StringRef originalName,
                              StringRef wrapperName) {
  SmallVector<LLVM::CallOp> calls;
  module.walk([&](LLVM::CallOp call) {
    auto callee = call.getCalleeAttr();
    if (callee && callee.getValue() == originalName)
      calls.push_back(call);
  });
  if (calls.empty())
    return success();

  auto original = module.lookupSymbol<LLVM::LLVMFuncOp>(originalName);
  if (!original)
    return module.emitError() << "heap instrumentation found calls to @"
                              << originalName << " without a declaration";
  FailureOr<LLVM::LLVMFuncOp> wrapper =
      getOrCreateWrapper(module, original, wrapperName);
  if (failed(wrapper))
    return failure();

  FlatSymbolRefAttr wrapperRef =
      FlatSymbolRefAttr::get(module.getContext(), wrapperName);
  for (LLVM::CallOp call : calls)
    call.setCalleeAttr(wrapperRef);

  if (SymbolTable::symbolKnownUseEmpty(original, module))
    original.erase();
  return success();
}

} // namespace

namespace mlir::sculptor {

void InstrumentTileHeapPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (failed(instrumentCalls(module, kMallocName, kProfiledMallocName)) ||
      failed(instrumentCalls(module, kFreeName, kProfiledFreeName))) {
    signalPassFailure();
  }
}

void registerInstrumentTileHeapPass() {
  PassRegistration<InstrumentTileHeapPass>();
}

} // namespace mlir::sculptor
