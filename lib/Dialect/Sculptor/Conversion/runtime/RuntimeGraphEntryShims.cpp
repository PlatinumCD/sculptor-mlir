#include "sculptor-mlir/Dialect/Sculptor/Conversion/runtime/RuntimeGraphEmission.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace sculptor {

namespace {

bool isMetadataOnlyResourceType(Type valueType) {
  return isa<LogicalArrayType>(valueType);
}

FailureOr<SmallVector<unsigned>> getTaskResultIndices(TaskCreateOp taskOp) {
  SmallVector<unsigned> resultIndices;
  resultIndices.reserve(taskOp.getOutputs().size());

  auto resultIndicesAttr =
      taskOp->getAttrOfType<ArrayAttr>(
          runtime_attrs::kTaskResultIndicesAttrName);
  if (!resultIndicesAttr) {
    for (unsigned outputIndex = 0, count = taskOp.getOutputs().size();
         outputIndex < count; ++outputIndex)
      resultIndices.push_back(outputIndex);
    return resultIndices;
  }

  if (resultIndicesAttr.size() != taskOp.getOutputs().size()) {
    taskOp.emitError("expected runtime attr '")
        << runtime_attrs::kTaskResultIndicesAttrName
        << "' to match the number of task outputs";
    return failure();
  }

  for (Attribute attr : resultIndicesAttr) {
    auto integerAttr = dyn_cast<IntegerAttr>(attr);
    if (!integerAttr || integerAttr.getInt() < 0) {
      taskOp.emitError("expected runtime attr '")
          << runtime_attrs::kTaskResultIndicesAttrName
          << "' to contain non-negative integer result indices";
      return failure();
    }

    resultIndices.push_back(static_cast<unsigned>(integerAttr.getInt()));
  }

  return resultIndices;
}

FailureOr<ShapedType> getSupportedBufferType(Operation *op, Type valueType) {
  auto shapedType = dyn_cast<ShapedType>(valueType);
  if (!shapedType || !shapedType.hasStaticShape() ||
      !shapedType.getElementType().isF32()) {
    op->emitError("expected runtime shims to lower only static shaped f32 "
                  "buffer resources");
    return failure();
  }

  return shapedType;
}

FailureOr<Value> selectCallResultDescriptor(OpBuilder &builder, Location loc,
                                            TaskCreateOp taskOp,
                                            LLVM::CallOp call,
                                            unsigned resultIndex,
                                            ShapedType shapedType) {
  if (call.getNumResults() != 1) {
    taskOp.emitError(
        "expected LLVM callable with buffer outputs to return one descriptor "
        "or one aggregate descriptor result");
    return failure();
  }

  Value callResult = call.getResult();
  Type expectedDescriptorType =
      getMemRefDescriptorType(builder.getContext(), shapedType);
  if (callResult.getType() == expectedDescriptorType) {
    if (resultIndex != 0) {
      taskOp.emitError("expected scalar LLVM callable result index to be 0");
      return failure();
    }
    return callResult;
  }

  auto aggregateType = dyn_cast<LLVM::LLVMStructType>(callResult.getType());
  if (!aggregateType || aggregateType.isOpaque()) {
    taskOp.emitError(
        "expected LLVM callable result to be a memref descriptor or a "
        "descriptor aggregate");
    return failure();
  }

  ArrayRef<Type> body = aggregateType.getBody();
  if (resultIndex >= body.size() ||
      body[resultIndex] != expectedDescriptorType) {
    taskOp.emitError("expected runtime result index ")
        << resultIndex << " to select a matching memref descriptor";
    return failure();
  }

  return builder
      .create<LLVM::ExtractValueOp>(
          loc, callResult,
          ArrayRef<int64_t>{static_cast<int64_t>(resultIndex)})
      .getResult();
}

} // namespace

FailureOr<SmallVector<LLVM::LLVMFuncOp>>
emitEntryShims(ModuleOp module, const GraphModel &model,
               const DenseMap<Value, unsigned> &resourceIndexByValue,
               const RuntimeDecls &decls) {
  MLIRContext *context = module.getContext();
  Builder builder(context);
  SmallVector<LLVM::LLVMFuncOp> shims;
  Type i32Type = builder.getI32Type();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Location loc = module.getLoc();

  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());

  for (const CallableModel &callable : model.callables) {
    TaskCreateOp representativeTask = callable.representativeTask;
    LLVM::LLVMFuncOp calleeFunc = callable.callee;
    std::string shimName =
        makeUniqueSymbolName(module, "__analog_rt_entry_", callable.symbol);
    auto shimType =
        LLVM::LLVMFunctionType::get(i32Type, {ptrType}, /*isVarArg=*/false);
    auto shim = moduleBuilder.create<LLVM::LLVMFuncOp>(loc, shimName, shimType);
    setPrivateVisibility(shim.getOperation(), builder);
    Block *entryBlock = shim.addEntryBlock(moduleBuilder);

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);
    Value opaque = entryBlock->getArgument(0);

    SmallVector<Value> callOperands;
    unsigned bindingIndex = 0;
    for (Value input : representativeTask.getInputs()) {
      auto resourceIt = resourceIndexByValue.find(input);
      if (resourceIt == resourceIndexByValue.end()) {
        representativeTask.emitError(
            "expected entry shim inputs to reference known resources");
        return failure();
      }

      const ResourceModel &resource = model.resources[resourceIt->second];
      if (isMetadataOnlyResourceType(resource.valueType))
        continue;

      if (resource.kind == RES_HANDLE) {
        ++bindingIndex;
        continue;
      }

      auto shapedType =
          getSupportedBufferType(representativeTask, resource.valueType);
      if (failed(shapedType))
        return failure();

      auto bufferCall = bodyBuilder.create<LLVM::CallOp>(
          loc, decls.taskArgBuffer,
          ValueRange{opaque,
                     buildI32Constant(bodyBuilder, loc, bindingIndex)});
      Value bufferView = bufferCall.getResult();
      Value dataPtr = bodyBuilder.create<LLVM::ExtractValueOp>(
          loc, bufferView, ArrayRef<int64_t>{0});
      Value descriptor =
          buildMemRefDescriptor(bodyBuilder, loc, *shapedType, dataPtr);
      flattenMemRefDescriptor(bodyBuilder, loc, descriptor, *shapedType,
                              callOperands);
      ++bindingIndex;
    }

    auto calleeType = calleeFunc.getFunctionType();
    if (calleeType.getNumParams() != callOperands.size()) {
      representativeTask.emitError(
          "runtime entry shim only supports callables whose LLVM signature "
          "matches the task's buffer inputs");
      return failure();
    }

    auto call = bodyBuilder.create<LLVM::CallOp>(loc, calleeFunc, callOperands);

    SmallVector<BufferOutputModel> bufferOutputs;
    SmallVector<unsigned> handleOutputs;
    for (auto indexedOutput :
         llvm::enumerate(representativeTask.getOutputs())) {
      auto resourceIt = resourceIndexByValue.find(indexedOutput.value());
      if (resourceIt == resourceIndexByValue.end()) {
        representativeTask.emitError(
            "expected entry shim outputs to reference known resources");
        return failure();
      }

      const ResourceModel &resource = model.resources[resourceIt->second];
      if (isMetadataOnlyResourceType(resource.valueType))
        continue;

      if (resource.kind == RES_HANDLE)
        handleOutputs.push_back(bindingIndex);
      else
        bufferOutputs.push_back(BufferOutputModel{
            bindingIndex, &resource, 0,
            static_cast<unsigned>(indexedOutput.index())});
      ++bindingIndex;
    }

    auto resultIndices = getTaskResultIndices(representativeTask);
    if (failed(resultIndices))
      return failure();
    for (BufferOutputModel &bufferOutput : bufferOutputs)
      bufferOutput.resultIndex = (*resultIndices)[bufferOutput.outputIndex];

    if (bufferOutputs.empty()) {
      if (call.getNumResults() != 0) {
        representativeTask.emitError(
            "expected void LLVM callable for tasks without buffer outputs");
        return failure();
      }
    } else {
      if (call.getNumResults() != 1) {
        representativeTask.emitError(
            "expected single-result LLVM callable for tasks with buffer "
            "output");
        return failure();
      }

      for (auto indexedOutput : llvm::enumerate(bufferOutputs)) {
        const BufferOutputModel &bufferOutput = indexedOutput.value();
        const ResourceModel &outputResource = *bufferOutput.resource;
        Value outputBufferCall =
            bodyBuilder
                .create<LLVM::CallOp>(
                    loc, decls.taskArgBuffer,
                    ValueRange{opaque,
                               buildI32Constant(bodyBuilder, loc,
                                                bufferOutput.bindingIndex)})
                .getResult();
        Value dstPtr = bodyBuilder.create<LLVM::ExtractValueOp>(
            loc, outputBufferCall, ArrayRef<int64_t>{0});

        auto outputShapedType =
            getSupportedBufferType(representativeTask, outputResource.valueType);
        if (failed(outputShapedType))
          return failure();

        auto selectedDescriptor = selectCallResultDescriptor(
            bodyBuilder, loc, representativeTask, call,
            bufferOutput.resultIndex, *outputShapedType);
        if (failed(selectedDescriptor))
          return failure();

        Value basePtr = bodyBuilder.create<LLVM::ExtractValueOp>(
            loc, *selectedDescriptor, ArrayRef<int64_t>{0});
        Value dataPtr = bodyBuilder.create<LLVM::ExtractValueOp>(
            loc, *selectedDescriptor, ArrayRef<int64_t>{1});
        bodyBuilder.create<LLVM::CallOp>(
            loc, decls.copyToBuffer,
            ValueRange{dstPtr, dataPtr,
                       buildI64Constant(
                           bodyBuilder, loc,
                           static_cast<int64_t>(outputResource.byteSize))});
        bodyBuilder.create<LLVM::CallOp>(loc, decls.freeResultBuffer,
                                         ValueRange{opaque, basePtr});
      }
    }

    for (unsigned bindingIndex : handleOutputs) {
      Value handle = bodyBuilder
                         .create<LLVM::CallOp>(
                             loc, decls.persistentHandleCreate,
                             ValueRange{buildI64Constant(bodyBuilder, loc, 0),
                                        buildZeroPointer(bodyBuilder, loc)})
                         .getResult();
      bodyBuilder.create<LLVM::CallOp>(
          loc, decls.taskSetArgHandle,
          ValueRange{opaque,
                     buildI32Constant(bodyBuilder, loc,
                                      static_cast<int32_t>(bindingIndex)),
                     handle});
    }

    bodyBuilder.create<LLVM::ReturnOp>(loc,
                                       buildI32Constant(bodyBuilder, loc, 0));
    shims.push_back(shim);
  }

  return shims;
}

} // namespace sculptor
} // namespace mlir
