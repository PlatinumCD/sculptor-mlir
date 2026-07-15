#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "analog-base"

using namespace mlir;
using namespace mlir::sculptor;

//===- Generated implementation -------------------------------------------===//

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.cpp.inc"

namespace {

// Parses the resource wrapper while leaving payload validation to the type.
Type parseTaskResourceType(DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  Type valueType;
  if (parser.parseType(valueType) || parser.parseGreater())
    return {};

  return TaskResourceType::get(parser.getContext(), valueType);
}

// Prints analog types whose payload printers already emit the inner syntax.
template <typename TypeT>
bool printParameterizedType(Type type, DialectAsmPrinter &printer) {
  auto analogType = llvm::dyn_cast<TypeT>(type);
  if (!analogType)
    return false;

  printer << TypeT::getMnemonic() << "<";
  analogType.print(printer);
  printer << ">";
  return true;
}

} // namespace

// Installs the analog types and operations when the dialect is loaded.
void SculptorDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOps();
}

// Dispatches each analog type mnemonic to the parser that owns its payload.
Type SculptorDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return {};

  // Handle task graph handles and resources before container-specific parsing.
  if (mnemonic == TaskGraphType::getMnemonic())
    return TaskGraphType::get(parser.getContext());

  if (mnemonic == TaskType::getMnemonic())
    return TaskType::get(parser.getContext());

  if (mnemonic == RuntimeHandleType::getMnemonic())
    return RuntimeHandleType::get(parser.getContext());

  if (mnemonic == LogicalArrayType::getMnemonic())
    return LogicalArrayType::get(parser.getContext());

  if (mnemonic == ArrayResultType::getMnemonic())
    return ArrayResultType::get(parser.getContext());

  if (mnemonic == TaskResourceType::getMnemonic())
    return parseTaskResourceType(parser);

  if (parser.parseLess())
    return {};

  // Delegate parameter parsing to the matched container or view type.
  Type result;
  if (mnemonic == MatrixType::getMnemonic()) {
    result = MatrixType::parse(parser);
  } else if (mnemonic == VectorType::getMnemonic()) {
    result = VectorType::parse(parser);
  } else if (mnemonic == MatrixGridType::getMnemonic()) {
    result = MatrixGridType::parse(parser);
  } else if (mnemonic == VectorSliceType::getMnemonic()) {
    result = VectorSliceType::parse(parser);
  } else {
    parser.emitError(parser.getNameLoc())
        << "unknown analog type: " << mnemonic;
    return {};
  }

  if (parser.parseGreater())
    return {};

  return result;
}

// Emits the assembly spelling for every analog type owned by the dialect.
void SculptorDialect::printType(Type type,
                              DialectAsmPrinter &printer) const {
  // Share the mnemonic<...> wrapper across container and view types.
  if (printParameterizedType<MatrixType>(type, printer) ||
      printParameterizedType<VectorType>(type, printer) ||
      printParameterizedType<MatrixGridType>(type, printer) ||
      printParameterizedType<VectorSliceType>(type, printer)) {
    return;
  }

  // Task graph handles either print as bare mnemonics or a wrapped payload type.
  if (llvm::isa<TaskGraphType>(type)) {
    printer << TaskGraphType::getMnemonic();
    return;
  }

  if (llvm::isa<TaskType>(type)) {
    printer << TaskType::getMnemonic();
    return;
  }

  if (llvm::isa<RuntimeHandleType>(type)) {
    printer << RuntimeHandleType::getMnemonic();
    return;
  }

  if (llvm::isa<LogicalArrayType>(type)) {
    printer << LogicalArrayType::getMnemonic();
    return;
  }

  if (llvm::isa<ArrayResultType>(type)) {
    printer << ArrayResultType::getMnemonic();
    return;
  }

  if (auto resourceType = llvm::dyn_cast<TaskResourceType>(type)) {
    printer << TaskResourceType::getMnemonic() << "<";
    printer.printType(resourceType.getValueType());
    printer << ">";
    return;
  }

  llvm_unreachable("unknown analog type");
}
