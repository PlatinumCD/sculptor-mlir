#include "sculptor-mlir/Dialect/Sculptor/Conversion/runtime/RuntimeGraphEmission.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace sculptor {

namespace {

Value emitGraphCreateCall(OpBuilder &builder, Location loc,
                          const RuntimeDecls &decls, const GraphModel &model) {
  return builder
      .create<LLVM::CallOp>(
          loc, decls.graphCreate,
          ValueRange{
              buildI32Constant(builder, loc,
                               static_cast<int32_t>(model.resources.size())),
              buildI32Constant(builder, loc,
                               static_cast<int32_t>(model.callables.size())),
              buildI32Constant(builder, loc,
                               static_cast<int32_t>(model.tasks.size())),
              buildI32Constant(builder, loc,
                               static_cast<int32_t>(model.bindings.size())),
              buildI32Constant(builder, loc,
                               static_cast<int32_t>(model.deps.size())),
              buildI32Constant(builder, loc, 0),
              buildI32Constant(builder, loc, 0),
              buildI64Constant(builder, loc,
                               static_cast<int64_t>(model.workspaceSize))})
      .getResult();
}

void emitResourceRecords(OpBuilder &bodyBuilder, Location loc,
                         const RuntimeDecls &decls, const GraphModel &model,
                         Value graph) {
  for (auto indexedResource : llvm::enumerate(model.resources)) {
    const ResourceModel &resource = indexedResource.value();
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetResource,
        ValueRange{
            graph,
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedResource.index())),
            buildI32Constant(bodyBuilder, loc, resource.kind),
            buildI32Constant(bodyBuilder, loc, resource.storage),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(resource.slot)),
            buildI64Constant(bodyBuilder, loc,
                             static_cast<int64_t>(resource.byteSize)),
            buildI64Constant(bodyBuilder, loc,
                             static_cast<int64_t>(resource.workspaceOffset))});
  }
}

void emitCallableRecords(OpBuilder &bodyBuilder, Location loc,
                         const RuntimeDecls &decls,
                         ArrayRef<LLVM::LLVMFuncOp> shims, Value graph) {
  for (auto indexedCallable : llvm::enumerate(shims)) {
    Value fnPtr =
        bodyBuilder.create<LLVM::AddressOfOp>(loc, indexedCallable.value());
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetCallable,
        ValueRange{
            graph,
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedCallable.index())),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedCallable.index())),
            fnPtr, buildI32Constant(bodyBuilder, loc, 0)});
  }
}

void emitTaskRecords(OpBuilder &bodyBuilder, Location loc,
                     const RuntimeDecls &decls, const GraphModel &model,
                     Value graph) {
  for (auto indexedTask : llvm::enumerate(model.tasks)) {
    const TaskModel &task = indexedTask.value();
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetTask,
        ValueRange{graph,
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(indexedTask.index())),
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(task.callableIndex)),
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(task.argBegin)),
                   buildI16Constant(bodyBuilder, loc, task.argCount),
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(task.depBegin)),
                   buildI16Constant(bodyBuilder, loc, task.depCount),
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(task.payloadOffset)),
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(task.payloadSize)),
                   buildI32Constant(bodyBuilder, loc, task.coreId)});
  }
}

void emitBindingRecords(OpBuilder &bodyBuilder, Location loc,
                        const RuntimeDecls &decls, const GraphModel &model,
                        Value graph) {
  for (auto indexedBinding : llvm::enumerate(model.bindings)) {
    const BindingModel &binding = indexedBinding.value();
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetBinding,
        ValueRange{
            graph,
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedBinding.index())),
            buildI32Constant(bodyBuilder, loc, binding.kind),
            buildI16Constant(bodyBuilder, loc, binding.flags),
            buildI32Constant(bodyBuilder, loc, binding.source),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(binding.sourceIndex)),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(binding.byteOffset)),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(binding.byteSize))});
  }
}

void emitDependencyRecords(OpBuilder &bodyBuilder, Location loc,
                           const RuntimeDecls &decls, const GraphModel &model,
                           Value graph) {
  for (auto indexedDep : llvm::enumerate(model.deps)) {
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetDep,
        ValueRange{graph,
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(indexedDep.index())),
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(indexedDep.value()))});
  }
}

} // namespace

LLVM::LLVMFuncOp emitGraphBuilder(ModuleOp module, const GraphModel &model,
                                  ArrayRef<LLVM::LLVMFuncOp> shims,
                                  const RuntimeDecls &decls) {
  MLIRContext *context = module.getContext();
  Builder builder(context);
  Location loc = module.getLoc();
  Type ptrType = LLVM::LLVMPointerType::get(context);

  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  std::string builderName =
      makeUniqueSymbolName(module, "__analog_rt_build_graph_image", "");
  auto builderFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, builderName, LLVM::LLVMFunctionType::get(ptrType, {}, false));
  setPrivateVisibility(builderFunc.getOperation(), builder);

  Block *entryBlock = builderFunc.addEntryBlock(moduleBuilder);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);

  Value graph = emitGraphCreateCall(bodyBuilder, loc, decls, model);
  emitResourceRecords(bodyBuilder, loc, decls, model, graph);
  emitCallableRecords(bodyBuilder, loc, decls, shims, graph);
  emitTaskRecords(bodyBuilder, loc, decls, model, graph);
  emitBindingRecords(bodyBuilder, loc, decls, model, graph);
  emitDependencyRecords(bodyBuilder, loc, decls, model, graph);

  bodyBuilder.create<LLVM::ReturnOp>(loc, graph);
  return builderFunc;
}

} // namespace sculptor
} // namespace mlir
