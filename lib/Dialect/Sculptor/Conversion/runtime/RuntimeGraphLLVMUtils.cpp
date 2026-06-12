#include "sculptor-mlir/Dialect/Sculptor/Conversion/runtime/RuntimeGraphEmission.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

#include "mlir/IR/SymbolTable.h"

#include <cctype>

namespace mlir {
namespace sculptor {

namespace {

std::string sanitizeSymbolSuffix(llvm::StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value)
    result.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  return result;
}

SmallVector<int64_t> computeRowMajorStrides(ShapedType shapedType) {
  SmallVector<int64_t> strides(shapedType.getRank(), 1);
  int64_t stride = 1;
  for (int64_t index = shapedType.getRank() - 1; index >= 0; --index) {
    strides[index] = stride;
    stride *= shapedType.getDimSize(index);
  }
  return strides;
}

LLVM::LLVMStructType getBufferViewType(MLIRContext *context) {
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(context, {ptrType, i64Type});
}

LLVM::LLVMFuncOp getOrCreateExternFunc(ModuleOp module, StringRef name,
                                       LLVM::LLVMFunctionType type) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

} // namespace

std::string makeUniqueSymbolName(ModuleOp module, llvm::StringRef prefix,
                                 llvm::StringRef suffix) {
  std::string baseName = (prefix + sanitizeSymbolSuffix(suffix)).str();
  if (!module.lookupSymbol(baseName))
    return baseName;

  unsigned disambiguator = 0;
  std::string candidate = baseName + "_" + std::to_string(disambiguator);
  while (module.lookupSymbol(candidate)) {
    ++disambiguator;
    candidate = baseName + "_" + std::to_string(disambiguator);
  }
  return candidate;
}

void setPrivateVisibility(Operation *op, Builder &builder) {
  op->setAttr(SymbolTable::getVisibilityAttrName(),
              builder.getStringAttr("private"));
}

LLVM::LLVMStructType getMemRefDescriptorType(MLIRContext *context,
                                             ShapedType shapedType) {
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i64Type = IntegerType::get(context, 64);
  unsigned rank = shapedType.getRank();
  auto indexArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
  return LLVM::LLVMStructType::getLiteral(
      context, {ptrType, ptrType, i64Type, indexArrayType, indexArrayType});
}

Value buildI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(),
                                          builder.getI32IntegerAttr(value));
}

Value buildI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                          builder.getI64IntegerAttr(value));
}

Value buildI16Constant(OpBuilder &builder, Location loc, int16_t value) {
  return builder.create<LLVM::ConstantOp>(loc, builder.getI16Type(),
                                          builder.getI16IntegerAttr(value));
}

Value buildZeroPointer(OpBuilder &builder, Location loc) {
  return builder.create<LLVM::ZeroOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()));
}

Value buildMemRefDescriptor(OpBuilder &builder, Location loc,
                            ShapedType shapedType, Value dataPtr) {
  auto descriptorType =
      getMemRefDescriptorType(builder.getContext(), shapedType);
  Value descriptor = builder.create<LLVM::ZeroOp>(loc, descriptorType);
  descriptor = builder.create<LLVM::InsertValueOp>(loc, descriptor, dataPtr,
                                                   ArrayRef<int64_t>{0});
  descriptor = builder.create<LLVM::InsertValueOp>(loc, descriptor, dataPtr,
                                                   ArrayRef<int64_t>{1});
  descriptor = builder.create<LLVM::InsertValueOp>(
      loc, descriptor, buildI64Constant(builder, loc, 0), ArrayRef<int64_t>{2});

  SmallVector<int64_t> strides = computeRowMajorStrides(shapedType);
  for (auto indexedDim : llvm::enumerate(shapedType.getShape())) {
    descriptor = builder.create<LLVM::InsertValueOp>(
        loc, descriptor, buildI64Constant(builder, loc, indexedDim.value()),
        ArrayRef<int64_t>{3, static_cast<int64_t>(indexedDim.index())});
    descriptor = builder.create<LLVM::InsertValueOp>(
        loc, descriptor,
        buildI64Constant(builder, loc, strides[indexedDim.index()]),
        ArrayRef<int64_t>{4, static_cast<int64_t>(indexedDim.index())});
  }

  return descriptor;
}

void flattenMemRefDescriptor(OpBuilder &builder, Location loc, Value descriptor,
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

RuntimeDecls getRuntimeDecls(ModuleOp module) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type voidType = LLVM::LLVMVoidType::get(context);
  Type i16Type = IntegerType::get(context, 16);
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  auto bufferViewType = getBufferViewType(context);

  RuntimeDecls decls;
  decls.graphCreate = getOrCreateExternFunc(
      module, "sculptor_runtime_graph_create",
      LLVM::LLVMFunctionType::get(ptrType,
                                  {i32Type, i32Type, i32Type, i32Type, i32Type,
                                   i32Type, i32Type, i64Type},
                                  false));
  decls.graphSetResource = getOrCreateExternFunc(
      module, "sculptor_runtime_graph_set_resource",
      LLVM::LLVMFunctionType::get(
          voidType,
          {ptrType, i32Type, i32Type, i32Type, i32Type, i64Type, i64Type},
          false));
  decls.graphSetCallable = getOrCreateExternFunc(
      module, "sculptor_runtime_graph_set_callable",
      LLVM::LLVMFunctionType::get(
          voidType, {ptrType, i32Type, i32Type, ptrType, i32Type}, false));
  decls.graphSetTask = getOrCreateExternFunc(
      module, "sculptor_runtime_graph_set_task",
      LLVM::LLVMFunctionType::get(voidType,
                                  {ptrType, i32Type, i32Type, i32Type, i16Type,
                                   i32Type, i16Type, i32Type, i32Type, i32Type},
                                  false));
  decls.graphSetBinding = getOrCreateExternFunc(
      module, "sculptor_runtime_graph_set_binding",
      LLVM::LLVMFunctionType::get(voidType,
                                  {ptrType, i32Type, i32Type, i16Type, i32Type,
                                   i32Type, i32Type, i32Type},
                                  false));
  decls.graphSetDep =
      getOrCreateExternFunc(module, "sculptor_runtime_graph_set_dep",
                            LLVM::LLVMFunctionType::get(
                                voidType, {ptrType, i32Type, i32Type}, false));
  decls.runtimeInit = getOrCreateExternFunc(
      module, "sculptor_runtime_init",
      LLVM::LLVMFunctionType::get(ptrType, {ptrType}, false));
  decls.runtimeExecute = getOrCreateExternFunc(
      module, "sculptor_runtime_execute",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType, ptrType}, false));
  decls.runtimeDestroy = getOrCreateExternFunc(
      module, "sculptor_runtime_destroy",
      LLVM::LLVMFunctionType::get(voidType, {ptrType}, false));
  decls.taskArgBuffer = getOrCreateExternFunc(
      module, "sculptor_runtime_task_arg_buffer",
      LLVM::LLVMFunctionType::get(bufferViewType, {ptrType, i32Type}, false));
  decls.taskSetArgHandle = getOrCreateExternFunc(
      module, "sculptor_runtime_task_set_arg_handle",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, i32Type, ptrType}, false));
  decls.copyToBuffer =
      getOrCreateExternFunc(module, "sculptor_runtime_copy_to_buffer",
                            LLVM::LLVMFunctionType::get(
                                voidType, {ptrType, ptrType, i64Type}, false));
  decls.persistentHandleCreate = getOrCreateExternFunc(
      module, "sculptor_runtime_persistent_handle_create",
      LLVM::LLVMFunctionType::get(ptrType, {i64Type, ptrType}, false));
  decls.freeResultBuffer = getOrCreateExternFunc(
      module, "sculptor_runtime_free_result_buffer",
      LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType}, false));
  return decls;
}

} // namespace sculptor
} // namespace mlir
