#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_SUPPORT_RUNTIMEGRAPHEMISSION_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_SUPPORT_RUNTIMEGRAPHEMISSION_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "TaskGraphRuntime.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {

struct ResourceModel {
  Value value;
  Type valueType;
  int32_t kind = RES_BUFFER;
  int32_t storage = STORAGE_TEMP;
  uint32_t slot = 0;
  uint64_t byteSize = 0;
  uint64_t workspaceOffset = 0;
};

struct BindingModel {
  int32_t kind = ARG_BUFFER;
  uint16_t flags = 0;
  int32_t source = SRC_SLOT;
  uint32_t sourceIndex = 0;
  uint32_t byteOffset = 0;
  uint32_t byteSize = 0;
};

struct TaskModel {
  TaskCreateOp op;
  uint32_t callableIndex = 0;
  int32_t coreId = -1;
  uint32_t argBegin = 0;
  uint16_t argCount = 0;
  uint32_t depBegin = 0;
  uint16_t depCount = 0;
  uint32_t payloadOffset = 0;
  uint32_t payloadSize = 0;
};

struct CallableModel {
  std::string symbol;
  LLVM::LLVMFuncOp callee;
  TaskCreateOp representativeTask;
};

struct BufferOutputModel {
  unsigned bindingIndex = 0;
  const ResourceModel *resource = nullptr;
  unsigned resultIndex = 0;
  unsigned outputIndex = 0;
};

struct GraphModel {
  SmallVector<ResourceModel> resources;
  SmallVector<CallableModel> callables;
  SmallVector<BindingModel> bindings;
  SmallVector<uint32_t> deps;
  SmallVector<TaskModel> tasks;
  uint64_t workspaceSize = 0;
};

struct RuntimeDecls {
  LLVM::LLVMFuncOp graphCreate;
  LLVM::LLVMFuncOp graphSetResource;
  LLVM::LLVMFuncOp graphSetCallable;
  LLVM::LLVMFuncOp graphSetTask;
  LLVM::LLVMFuncOp graphSetBinding;
  LLVM::LLVMFuncOp graphSetDep;
  LLVM::LLVMFuncOp runtimeInit;
  LLVM::LLVMFuncOp runtimeExecute;
  LLVM::LLVMFuncOp runtimeDestroy;
  LLVM::LLVMFuncOp taskArgBuffer;
  LLVM::LLVMFuncOp taskSetArgHandle;
  LLVM::LLVMFuncOp copyToBuffer;
  LLVM::LLVMFuncOp persistentHandleCreate;
  LLVM::LLVMFuncOp freeResultBuffer;
};

std::string makeUniqueSymbolName(ModuleOp module, llvm::StringRef prefix,
                                 llvm::StringRef suffix);
void setPrivateVisibility(Operation *op, Builder &builder);

Value buildI32Constant(OpBuilder &builder, Location loc, int32_t value);
Value buildI64Constant(OpBuilder &builder, Location loc, int64_t value);
Value buildI16Constant(OpBuilder &builder, Location loc, int16_t value);
Value buildZeroPointer(OpBuilder &builder, Location loc);
Value buildMemRefDescriptor(OpBuilder &builder, Location loc,
                            ShapedType shapedType, Value dataPtr);
void flattenMemRefDescriptor(OpBuilder &builder, Location loc, Value descriptor,
                             ShapedType shapedType,
                             SmallVectorImpl<Value> &operands);
LLVM::LLVMStructType getMemRefDescriptorType(MLIRContext *context,
                                             ShapedType shapedType);
RuntimeDecls getRuntimeDecls(ModuleOp module);

FailureOr<func::FuncOp> findTaskGraphFunc(ModuleOp module);
LogicalResult
collectGraphModel(ModuleOp module, func::FuncOp taskGraphFunc,
                  GraphModel &model,
                  llvm::DenseMap<Value, unsigned> &resourceIndexByValue);

FailureOr<SmallVector<LLVM::LLVMFuncOp>>
emitEntryShims(ModuleOp module, const GraphModel &model,
               const llvm::DenseMap<Value, unsigned> &resourceIndexByValue,
               const RuntimeDecls &decls);
LLVM::LLVMFuncOp emitGraphBuilder(ModuleOp module, const GraphModel &model,
                                  ArrayRef<LLVM::LLVMFuncOp> shims,
                                  const RuntimeDecls &decls);
LogicalResult emitPublicWrappers(ModuleOp module, LLVM::LLVMFuncOp graphBuilder,
                                 const RuntimeDecls &decls);

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_SUPPORT_RUNTIMEGRAPHEMISSION_H
