#include "GolemTileABI.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/SymbolTable.h"

#include <limits>
#include <string>

namespace mlir {
namespace sculptor {
namespace golem_tile_abi {

namespace {

Value buildI32(OpBuilder &builder, Location loc, uint32_t value) {
  return builder.create<LLVM::ConstantOp>(
      loc, builder.getI32Type(),
      builder.getI32IntegerAttr(static_cast<int32_t>(value)));
}

Value buildI64(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                          builder.getI64IntegerAttr(value));
}

Value buildNullPointer(OpBuilder &builder, Location loc) {
  return builder.create<LLVM::ZeroOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()));
}

Value andValue(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return builder.create<LLVM::AndOp>(loc, lhs, rhs);
}

Value isEqual(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, lhs, rhs);
}

Value isNotEqual(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, lhs, rhs);
}

LLVM::LLVMFuncOp getOrCreateFree(ModuleOp module) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>("free"))
    return existing;

  MLIRContext *context = module.getContext();
  auto type =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                  {LLVM::LLVMPointerType::get(context)}, false);
  OpBuilder builder(context);
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "free", type);
}

SmallVector<int64_t> computeStaticStrides(ShapedType shapedType) {
  SmallVector<int64_t> strides(shapedType.getRank(), 1);
  int64_t stride = 1;
  for (int64_t dim = shapedType.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = stride;
    stride *= shapedType.getDimSize(dim);
  }
  return strides;
}

Value validateTensorHeader(OpBuilder &builder, Location loc, Value tensor,
                           ShapedType shapedType) {
  Value elementType =
      builder.create<LLVM::ExtractValueOp>(loc, tensor, ArrayRef<int64_t>{0});
  Value rank =
      builder.create<LLVM::ExtractValueOp>(loc, tensor, ArrayRef<int64_t>{1});
  Value descriptor =
      builder.create<LLVM::ExtractValueOp>(loc, tensor, ArrayRef<int64_t>{2});

  Value valid = isEqual(builder, loc, elementType,
                        buildI32(builder, loc, kFloat32ElementType));
  valid = andValue(builder, loc, valid,
                   isEqual(builder, loc, rank,
                           buildI64(builder, loc, shapedType.getRank())));
  valid = andValue(
      builder, loc, valid,
      isNotEqual(builder, loc, descriptor, buildNullPointer(builder, loc)));
  return valid;
}

Value validateDescriptor(OpBuilder &builder, Location loc, Value descriptor,
                         ShapedType shapedType) {
  Value aligned = builder.create<LLVM::ExtractValueOp>(loc, descriptor,
                                                       ArrayRef<int64_t>{1});
  Value valid =
      isNotEqual(builder, loc, aligned, buildNullPointer(builder, loc));

  SmallVector<int64_t> strides = computeStaticStrides(shapedType);
  for (auto indexedDim : llvm::enumerate(shapedType.getShape())) {
    int64_t dim = indexedDim.index();
    Value size = builder.create<LLVM::ExtractValueOp>(
        loc, descriptor, ArrayRef<int64_t>{3, dim});
    Value stride = builder.create<LLVM::ExtractValueOp>(
        loc, descriptor, ArrayRef<int64_t>{4, dim});
    valid = andValue(builder, loc, valid,
                     isEqual(builder, loc, size,
                             buildI64(builder, loc, indexedDim.value())));
    valid =
        andValue(builder, loc, valid,
                 isEqual(builder, loc, stride,
                         buildI64(builder, loc, strides[indexedDim.index()])));
  }
  return valid;
}

void flattenDescriptor(OpBuilder &builder, Location loc, Value descriptor,
                       ShapedType shapedType,
                       SmallVectorImpl<Value> &operands) {
  operands.push_back(builder.create<LLVM::ExtractValueOp>(
      loc, descriptor, ArrayRef<int64_t>{0}));
  operands.push_back(builder.create<LLVM::ExtractValueOp>(
      loc, descriptor, ArrayRef<int64_t>{1}));
  operands.push_back(builder.create<LLVM::ExtractValueOp>(
      loc, descriptor, ArrayRef<int64_t>{2}));
  for (int64_t dim = 0; dim < shapedType.getRank(); ++dim)
    operands.push_back(builder.create<LLVM::ExtractValueOp>(
        loc, descriptor, ArrayRef<int64_t>{3, dim}));
  for (int64_t dim = 0; dim < shapedType.getRank(); ++dim)
    operands.push_back(builder.create<LLVM::ExtractValueOp>(
        loc, descriptor, ArrayRef<int64_t>{4, dim}));
}

void copyDescriptorData(OpBuilder &builder, Location loc,
                        Value sourceDescriptor, Value destinationDescriptor,
                        ShapedType shapedType) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Type f32Type = Float32Type::get(builder.getContext());
  Value sourceAligned = builder.create<LLVM::ExtractValueOp>(
      loc, sourceDescriptor, ArrayRef<int64_t>{1});
  Value sourceOffset = builder.create<LLVM::ExtractValueOp>(
      loc, sourceDescriptor, ArrayRef<int64_t>{2});
  Value destinationAligned = builder.create<LLVM::ExtractValueOp>(
      loc, destinationDescriptor, ArrayRef<int64_t>{1});
  Value destinationOffset = builder.create<LLVM::ExtractValueOp>(
      loc, destinationDescriptor, ArrayRef<int64_t>{2});
  Value source = builder.create<LLVM::GEPOp>(loc, ptrType, f32Type,
                                             sourceAligned, sourceOffset);
  Value destination = builder.create<LLVM::GEPOp>(
      loc, ptrType, f32Type, destinationAligned, destinationOffset);
  int64_t byteSize = shapedType.getNumElements() * sizeof(float);
  builder.create<LLVM::MemcpyOp>(loc, destination, source,
                                 buildI64(builder, loc, byteSize),
                                 /*isVolatile=*/false);
}

FailureOr<Value> selectResultDescriptor(OpBuilder &builder, Location loc,
                                        const TaskModel &task,
                                        LLVM::CallOp call,
                                        unsigned outputIndex) {
  Value result = call.getResult();
  unsigned resultIndex = task.resultIndices[outputIndex];
  Type descriptorType = getMemRefDescriptorType(builder.getContext(),
                                                task.outputTypes[outputIndex]);
  if (result.getType() == descriptorType)
    return result;

  return builder
      .create<LLVM::ExtractValueOp>(
          loc, result, ArrayRef<int64_t>{static_cast<int64_t>(resultIndex)})
      .getResult();
}

LogicalResult emitTaskAdapter(ModuleOp module, TaskModel &task,
                              LLVM::LLVMFuncOp freeFunc) {
  MLIRContext *context = module.getContext();
  Location loc = task.op.getLoc();
  Type i32Type = IntegerType::get(context, 32);
  Type ptrType = LLVM::LLVMPointerType::get(context);
  auto adapterType = LLVM::LLVMFunctionType::get(
      i32Type, {ptrType, i32Type, ptrType, i32Type}, false);
  std::string adapterName =
      "__golem_tile_execute_task_" + std::to_string(task.globalId);
  if (module.lookupSymbol(adapterName)) {
    task.op.emitError("cannot emit duplicate Golem task adapter symbol @")
        << adapterName;
    return failure();
  }

  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  auto adapter = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, adapterName, adapterType, LLVM::Linkage::Internal);
  Block *entry = adapter.addEntryBlock(moduleBuilder);
  auto *tensorBlock = new Block();
  auto *descriptorBlock = new Block();
  auto *invokeBlock = new Block();
  bool returnsOutputDescriptors =
      !task.outputTypes.empty() && !task.usesOutputParameters;
  Block *copyBlock = returnsOutputDescriptors ? new Block() : nullptr;
  Block *resultFailureBlock = returnsOutputDescriptors ? new Block() : nullptr;
  auto *failureBlock = new Block();
  adapter.getBody().push_back(tensorBlock);
  adapter.getBody().push_back(descriptorBlock);
  adapter.getBody().push_back(invokeBlock);
  if (copyBlock)
    adapter.getBody().push_back(copyBlock);
  if (resultFailureBlock)
    adapter.getBody().push_back(resultFailureBlock);
  adapter.getBody().push_back(failureBlock);

  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entry);
  Value inputs = entry->getArgument(0);
  Value inputCount = entry->getArgument(1);
  Value outputs = entry->getArgument(2);
  Value outputCount = entry->getArgument(3);
  Value headerValid =
      isEqual(entryBuilder, loc, inputCount,
              buildI32(entryBuilder, loc, task.inputTypes.size()));
  headerValid =
      andValue(entryBuilder, loc, headerValid,
               isEqual(entryBuilder, loc, outputCount,
                       buildI32(entryBuilder, loc, task.outputTypes.size())));
  if (!task.inputTypes.empty()) {
    headerValid = andValue(entryBuilder, loc, headerValid,
                           isNotEqual(entryBuilder, loc, inputs,
                                      buildNullPointer(entryBuilder, loc)));
  }
  if (!task.outputTypes.empty()) {
    headerValid = andValue(entryBuilder, loc, headerValid,
                           isNotEqual(entryBuilder, loc, outputs,
                                      buildNullPointer(entryBuilder, loc)));
  }

  entryBuilder.create<LLVM::CondBrOp>(loc, headerValid, tensorBlock,
                                      failureBlock);

  OpBuilder tensorBuilder = OpBuilder::atBlockBegin(tensorBlock);
  LLVM::LLVMStructType tensorType = getTensorType(context);
  SmallVector<Value> inputTensors;
  SmallVector<Value> outputTensors;
  inputTensors.reserve(task.inputTypes.size());
  outputTensors.reserve(task.outputTypes.size());
  for (auto indexedType : llvm::enumerate(task.inputTypes)) {
    Value tensorPtr = tensorBuilder.create<LLVM::GEPOp>(
        loc, ptrType, tensorType, inputs,
        buildI64(tensorBuilder, loc, indexedType.index()));
    Value tensor =
        tensorBuilder.create<LLVM::LoadOp>(loc, tensorType, tensorPtr);
    inputTensors.push_back(tensor);
  }
  for (auto indexedType : llvm::enumerate(task.outputTypes)) {
    Value tensorPtr = tensorBuilder.create<LLVM::GEPOp>(
        loc, ptrType, tensorType, outputs,
        buildI64(tensorBuilder, loc, indexedType.index()));
    Value tensor =
        tensorBuilder.create<LLVM::LoadOp>(loc, tensorType, tensorPtr);
    outputTensors.push_back(tensor);
  }
  Value tensorsValid = tensorBuilder.create<LLVM::ConstantOp>(
      loc, tensorBuilder.getI1Type(), tensorBuilder.getBoolAttr(true));
  for (auto indexedTensor : llvm::enumerate(inputTensors))
    tensorsValid =
        andValue(tensorBuilder, loc, tensorsValid,
                 validateTensorHeader(tensorBuilder, loc, indexedTensor.value(),
                                      task.inputTypes[indexedTensor.index()]));
  for (auto indexedTensor : llvm::enumerate(outputTensors))
    tensorsValid =
        andValue(tensorBuilder, loc, tensorsValid,
                 validateTensorHeader(tensorBuilder, loc, indexedTensor.value(),
                                      task.outputTypes[indexedTensor.index()]));
  tensorBuilder.create<LLVM::CondBrOp>(loc, tensorsValid, descriptorBlock,
                                       failureBlock);

  OpBuilder descriptorBuilder = OpBuilder::atBlockBegin(descriptorBlock);
  SmallVector<Value> inputDescriptors;
  SmallVector<Value> outputDescriptors;
  Value descriptorsValid = descriptorBuilder.create<LLVM::ConstantOp>(
      loc, descriptorBuilder.getI1Type(), descriptorBuilder.getBoolAttr(true));
  for (auto indexedTensor : llvm::enumerate(inputTensors)) {
    Value descriptorPtr = descriptorBuilder.create<LLVM::ExtractValueOp>(
        loc, indexedTensor.value(), ArrayRef<int64_t>{2});
    auto descriptorType = getMemRefDescriptorType(
        context, task.inputTypes[indexedTensor.index()]);
    Value descriptor = descriptorBuilder.create<LLVM::LoadOp>(
        loc, descriptorType, descriptorPtr);
    descriptorsValid =
        andValue(descriptorBuilder, loc, descriptorsValid,
                 validateDescriptor(descriptorBuilder, loc, descriptor,
                                    task.inputTypes[indexedTensor.index()]));
    inputDescriptors.push_back(descriptor);
  }
  for (auto indexedTensor : llvm::enumerate(outputTensors)) {
    Value descriptorPtr = descriptorBuilder.create<LLVM::ExtractValueOp>(
        loc, indexedTensor.value(), ArrayRef<int64_t>{2});
    auto descriptorType = getMemRefDescriptorType(
        context, task.outputTypes[indexedTensor.index()]);
    Value descriptor = descriptorBuilder.create<LLVM::LoadOp>(
        loc, descriptorType, descriptorPtr);
    descriptorsValid =
        andValue(descriptorBuilder, loc, descriptorsValid,
                 validateDescriptor(descriptorBuilder, loc, descriptor,
                                    task.outputTypes[indexedTensor.index()]));
    outputDescriptors.push_back(descriptor);
  }
  descriptorBuilder.create<LLVM::CondBrOp>(loc, descriptorsValid, invokeBlock,
                                           failureBlock);

  OpBuilder invokeBuilder = OpBuilder::atBlockBegin(invokeBlock);
  SmallVector<Value> callOperands;
  for (auto indexedDescriptor : llvm::enumerate(inputDescriptors))
    flattenDescriptor(invokeBuilder, loc, indexedDescriptor.value(),
                      task.inputTypes[indexedDescriptor.index()], callOperands);
  if (task.usesOutputParameters) {
    for (unsigned outputIndex : task.canonicalOutputIndices)
      flattenDescriptor(invokeBuilder, loc, outputDescriptors[outputIndex],
                        task.outputTypes[outputIndex], callOperands);
  }
  auto call =
      invokeBuilder.create<LLVM::CallOp>(loc, task.callee, callOperands);
  if (task.usesOutputParameters) {
    for (auto indexedOutput : llvm::enumerate(task.outputTypes)) {
      unsigned outputIndex = indexedOutput.index();
      unsigned canonicalOutputIndex =
          task.canonicalOutputIndices[task.resultIndices[outputIndex]];
      if (outputIndex == canonicalOutputIndex)
        continue;
      copyDescriptorData(invokeBuilder, loc,
                         outputDescriptors[canonicalOutputIndex],
                         outputDescriptors[outputIndex], indexedOutput.value());
    }
    invokeBuilder.create<LLVM::ReturnOp>(
        loc, buildI32(invokeBuilder, loc, kTaskSuccess));
  } else if (task.outputTypes.empty()) {
    invokeBuilder.create<LLVM::ReturnOp>(
        loc, buildI32(invokeBuilder, loc, kTaskSuccess));
  } else {
    Value resultsValid = invokeBuilder.create<LLVM::ConstantOp>(
        loc, invokeBuilder.getI1Type(), invokeBuilder.getBoolAttr(true));
    for (auto indexedType : llvm::enumerate(task.outputTypes)) {
      auto resultDescriptor = selectResultDescriptor(invokeBuilder, loc, task,
                                                     call, indexedType.index());
      if (failed(resultDescriptor))
        return failure();
      resultsValid =
          andValue(invokeBuilder, loc, resultsValid,
                   validateDescriptor(invokeBuilder, loc, *resultDescriptor,
                                      indexedType.value()));
    }
    invokeBuilder.create<LLVM::CondBrOp>(loc, resultsValid, copyBlock,
                                         resultFailureBlock);
  }

  if (copyBlock) {
    OpBuilder copyBuilder = OpBuilder::atBlockBegin(copyBlock);
    for (auto indexedType : llvm::enumerate(task.outputTypes)) {
      auto resultDescriptor = selectResultDescriptor(copyBuilder, loc, task,
                                                     call, indexedType.index());
      if (failed(resultDescriptor))
        return failure();
      copyDescriptorData(copyBuilder, loc, *resultDescriptor,
                         outputDescriptors[indexedType.index()],
                         indexedType.value());
    }
    for (unsigned outputIndex : task.canonicalOutputIndices) {
      if (outputIndex == std::numeric_limits<unsigned>::max())
        continue;
      auto resultDescriptor =
          selectResultDescriptor(copyBuilder, loc, task, call, outputIndex);
      if (failed(resultDescriptor))
        return failure();
      Value sourceBase = copyBuilder.create<LLVM::ExtractValueOp>(
          loc, *resultDescriptor, ArrayRef<int64_t>{0});
      copyBuilder.create<LLVM::CallOp>(loc, freeFunc, ValueRange{sourceBase});
    }
    copyBuilder.create<LLVM::ReturnOp>(
        loc, buildI32(copyBuilder, loc, kTaskSuccess));

    OpBuilder resultFailureBuilder =
        OpBuilder::atBlockBegin(resultFailureBlock);
    for (unsigned outputIndex : task.canonicalOutputIndices) {
      if (outputIndex == std::numeric_limits<unsigned>::max())
        continue;
      auto resultDescriptor = selectResultDescriptor(resultFailureBuilder, loc,
                                                     task, call, outputIndex);
      if (failed(resultDescriptor))
        return failure();
      Value sourceBase = resultFailureBuilder.create<LLVM::ExtractValueOp>(
          loc, *resultDescriptor, ArrayRef<int64_t>{0});
      resultFailureBuilder.create<LLVM::CallOp>(loc, freeFunc,
                                                ValueRange{sourceBase});
    }
    resultFailureBuilder.create<LLVM::ReturnOp>(
        loc, buildI32(resultFailureBuilder, loc, kTaskFailure));
  }

  OpBuilder failureBuilder = OpBuilder::atBlockBegin(failureBlock);
  failureBuilder.create<LLVM::ReturnOp>(
      loc, buildI32(failureBuilder, loc, kTaskFailure));

  task.adapter = adapter;
  return success();
}

} // namespace

LogicalResult emitTaskAdapters(ModuleOp module, TileModel &model) {
  LLVM::LLVMFuncOp freeFunc = getOrCreateFree(module);
  auto expectedFreeType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(module.getContext()),
      {LLVM::LLVMPointerType::get(module.getContext())}, false);
  if (freeFunc.getFunctionType() != expectedFreeType) {
    freeFunc.emitError("existing @free declaration does not match void(ptr)");
    return failure();
  }

  for (TaskModel &task : model.tasks)
    if (failed(emitTaskAdapter(module, task, freeFunc)))
      return failure();
  return success();
}

} // namespace golem_tile_abi
} // namespace sculptor
} // namespace mlir
