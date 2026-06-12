#include "sculptor-mlir/Dialect/Sculptor/Conversion/FinalizeGolemIntrinsics.h"

#include "sculptor-mlir/Dialect/Sculptor/Conversion/golem/GolemUtils.h"

// Finalizes LLVM Golem shim calls into target Golem ISA intrinsics. This is the
// last backend stage after Sculptor Golem ops have already lowered to LLVM calls.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir {
namespace sculptor {

namespace {

constexpr llvm::StringLiteral kRISCVSetIntrinsic =
    "llvm.riscv.golem.analog.mvm.set";
constexpr llvm::StringLiteral kRISCVLoadIntrinsic =
    "llvm.riscv.golem.analog.mvm.load";
constexpr llvm::StringLiteral kRISCVMvmIntrinsic =
    "llvm.riscv.golem.analog.mvm";
constexpr llvm::StringLiteral kRISCVStoreIntrinsic =
    "llvm.riscv.golem.analog.mvm.store";

enum class ShimKind { Set, Load, Compute, Store };

std::optional<ShimKind> classifyShimName(StringRef name) {
  return llvm::StringSwitch<std::optional<ShimKind>>(name)
      .Case(golem::kSetShimName, ShimKind::Set)
      .Case(golem::kLoadShimName, ShimKind::Load)
      .Case(golem::kComputeShimName, ShimKind::Compute)
      .Case(golem::kStoreShimName, ShimKind::Store)
      .Default(std::nullopt);
}

StringRef getTargetIntrinsicName(ShimKind kind) {
  switch (kind) {
  case ShimKind::Set:
    return kRISCVSetIntrinsic;
  case ShimKind::Load:
    return kRISCVLoadIntrinsic;
  case ShimKind::Compute:
    return kRISCVMvmIntrinsic;
  case ShimKind::Store:
    return kRISCVStoreIntrinsic;
  }

  llvm_unreachable("unknown shim kind");
}

bool isLLVMOpaquePointer(Type type) {
  return llvm::isa<LLVM::LLVMPointerType>(type);
}

bool isI64(Type type) { return type.isSignlessInteger(64); }

bool isI32(Type type) { return type.isSignlessInteger(32); }

LogicalResult verifyOperandType(Operation *op, Type type,
                                function_ref<bool(Type)> predicate,
                                StringRef expectation, unsigned operandIndex) {
  if (predicate(type))
    return success();

  return op->emitError() << "expected operand #" << operandIndex << " to be "
                         << expectation << ", got " << type;
}

LogicalResult verifyShimCall(LLVM::CallOp call, StringRef calleeName,
                             ShimKind kind) {
  if (call.getNumResults() != 0)
    return call.emitError()
           << "expected void shim call for '" << calleeName << "'";

  auto operandTypes = call.getOperandTypes();
  switch (kind) {
  case ShimKind::Compute:
    if (operandTypes.size() != 1) {
      return call.emitError() << "expected one operand for '" << calleeName
                              << "', got " << operandTypes.size();
    }
    return verifyOperandType(call, operandTypes.front(), isI32, "i32", 0);

  case ShimKind::Set:
  case ShimKind::Load:
    if (operandTypes.size() != 8) {
      return call.emitError() << "expected 8 operands for '" << calleeName
                              << "', got " << operandTypes.size();
    }
    if (failed(verifyOperandType(call, operandTypes[0], isLLVMOpaquePointer,
                                 "!llvm.ptr", 0)) ||
        failed(verifyOperandType(call, operandTypes[1], isLLVMOpaquePointer,
                                 "!llvm.ptr", 1))) {
      return failure();
    }
    for (unsigned index = 2; index < 7; ++index) {
      if (failed(verifyOperandType(call, operandTypes[index], isI64, "i64",
                                   index)))
        return failure();
    }
    return verifyOperandType(call, operandTypes[7], isI32, "i32", 7);

  case ShimKind::Store:
    if (operandTypes.size() != 10) {
      return call.emitError() << "expected 10 operands for '" << calleeName
                              << "', got " << operandTypes.size();
    }
    if (failed(verifyOperandType(call, operandTypes[0], isLLVMOpaquePointer,
                                 "!llvm.ptr", 0)) ||
        failed(verifyOperandType(call, operandTypes[1], isLLVMOpaquePointer,
                                 "!llvm.ptr", 1))) {
      return failure();
    }
    for (unsigned index = 2; index < 9; ++index) {
      if (failed(verifyOperandType(call, operandTypes[index], isI64, "i64",
                                   index)))
        return failure();
    }
    return verifyOperandType(call, operandTypes[9], isI32, "i32", 9);
  }

  llvm_unreachable("unknown shim kind");
}

Value buildEffectiveDataPointer(IRRewriter &rewriter, Location loc,
                                Value dataPtr, Value offset) {
  return rewriter.create<LLVM::GEPOp>(loc, dataPtr.getType(),
                                      rewriter.getF32Type(), dataPtr,
                                      ValueRange{offset});
}

LogicalResult rewriteShimCall(LLVM::CallOp call, StringRef calleeName,
                              ShimKind kind, IRRewriter &rewriter) {
  if (failed(verifyShimCall(call, calleeName, kind)))
    return failure();

  rewriter.setInsertionPoint(call);

  if (kind == ShimKind::Compute) {
    rewriter.create<LLVM::CallIntrinsicOp>(
        call.getLoc(), Type{},
        rewriter.getStringAttr(getTargetIntrinsicName(kind)),
        call.getOperands());
    rewriter.eraseOp(call);
    return success();
  }

  Value dataPtr = call.getOperand(1);
  Value offset = call.getOperand(2);
  Value arrayId = call.getOperand(call.getNumOperands() - 1);
  Value effectiveDataPtr =
      buildEffectiveDataPointer(rewriter, call.getLoc(), dataPtr, offset);

  rewriter.create<LLVM::CallIntrinsicOp>(
      call.getLoc(), Type{},
      rewriter.getStringAttr(getTargetIntrinsicName(kind)),
      ValueRange{effectiveDataPtr, arrayId});
  rewriter.eraseOp(call);
  return success();
}

void eraseDeadShimDecls(ModuleOp module) {
  SymbolTable symbolTable(module);
  for (StringRef name :
       {StringRef(golem::kSetShimName), StringRef(golem::kLoadShimName),
        StringRef(golem::kComputeShimName), StringRef(golem::kStoreShimName)}) {
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (func && SymbolTable::symbolKnownUseEmpty(func, module))
      symbolTable.erase(func);
  }
}

} // namespace

// Rewrites each recognized shim call to its target intrinsic and removes unused
// shim declarations.
void FinalizeGolemIntrinsicsPass::runOnOperation() {
  ModuleOp module = getOperation();

  SmallVector<std::pair<LLVM::CallOp, ShimKind>> shimCalls;
  module.walk([&](LLVM::CallOp call) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      return;

    if (auto kind = classifyShimName(calleeAttr.getValue()))
      shimCalls.emplace_back(call, *kind);
  });

  IRRewriter rewriter(&getContext());
  for (auto [call, kind] : shimCalls) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      continue;

    if (failed(rewriteShimCall(call, calleeAttr.getValue(), kind, rewriter))) {
      signalPassFailure();
      return;
    }
  }

  eraseDeadShimDecls(module);
}

void registerFinalizeGolemIntrinsicsPass() {
  PassRegistration<FinalizeGolemIntrinsicsPass>();
}

} // namespace sculptor
} // namespace mlir
