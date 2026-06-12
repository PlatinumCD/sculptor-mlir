#include "sculptor-mlir/Dialect/Sculptor/Conversion/runtime/RuntimeGraphEmission.h"

namespace mlir {
namespace sculptor {

LogicalResult emitPublicWrappers(ModuleOp module, LLVM::LLVMFuncOp graphBuilder,
                                 const RuntimeDecls &decls) {
  MLIRContext *context = module.getContext();
  Builder builder(context);
  Location loc = module.getLoc();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = builder.getI32Type();

  if (module.lookupSymbol("runtime_init") ||
      module.lookupSymbol("runtime_execute") ||
      module.lookupSymbol("runtime_destroy")) {
    module.emitError("expected runtime_init/runtime_execute/runtime_destroy to "
                     "be absent before sculptor-emit-runtime-graph");
    return failure();
  }

  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());

  auto runtimeInitWrapper = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, "runtime_init",
      LLVM::LLVMFunctionType::get(ptrType, {}, /*isVarArg=*/false));
  auto *initBlock = runtimeInitWrapper.addEntryBlock(moduleBuilder);
  OpBuilder initBuilder = OpBuilder::atBlockBegin(initBlock);
  Value graph =
      initBuilder.create<LLVM::CallOp>(loc, graphBuilder, ValueRange{})
          .getResult();
  Value runtime =
      initBuilder
          .create<LLVM::CallOp>(loc, decls.runtimeInit, ValueRange{graph})
          .getResult();
  initBuilder.create<LLVM::ReturnOp>(loc, runtime);

  auto runtimeExecuteWrapper = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, "runtime_execute",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType, ptrType}, false));
  auto *executeBlock = runtimeExecuteWrapper.addEntryBlock(moduleBuilder);
  OpBuilder executeBuilder = OpBuilder::atBlockBegin(executeBlock);
  SmallVector<Value> executeOperands{executeBlock->getArgument(0),
                                     executeBlock->getArgument(1),
                                     executeBlock->getArgument(2)};
  auto executeCall = executeBuilder.create<LLVM::CallOp>(
      loc, decls.runtimeExecute, executeOperands);
  executeBuilder.create<LLVM::ReturnOp>(loc, executeCall.getResult());

  auto runtimeDestroyWrapper = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, "runtime_destroy",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), {ptrType},
                                  false));
  auto *destroyBlock = runtimeDestroyWrapper.addEntryBlock(moduleBuilder);
  OpBuilder destroyBuilder = OpBuilder::atBlockBegin(destroyBlock);
  SmallVector<Value> destroyOperands{destroyBlock->getArgument(0)};
  destroyBuilder.create<LLVM::CallOp>(loc, decls.runtimeDestroy,
                                      destroyOperands);
  destroyBuilder.create<LLVM::ReturnOp>(loc, ValueRange{});

  return success();
}

} // namespace sculptor
} // namespace mlir
