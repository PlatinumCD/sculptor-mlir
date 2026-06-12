#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpAsmSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

#define DEBUG_TYPE "analog-types"

using namespace mlir;
using namespace mlir::sculptor;

#define GET_TYPEDEF_CLASSES
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.cpp.inc"

// Registers the TableGen-generated type classes with the analog dialect.
void SculptorDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.cpp.inc"
      >();
}

namespace {

// Checks the rank and element-type contract for matrix/vector containers.
mlir::LogicalResult verifyRank2FloatContainerType(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  if (shape.empty())
    return emitError() << "shape must have at least 1 dimension";

  if (!elementType || !mlir::isa<mlir::FloatType>(elementType))
    return emitError() << "elementType must be a float type";

  if (shape.size() != 2)
    return emitError() << "expected rank-2 matrix, got rank " << shape.size();

  return mlir::success();
}

// Checks the concrete payload carried by task resources across tensors/memrefs.
mlir::LogicalResult verifyPayloadRankAndElementType(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
    llvm::StringRef kind) {
  if (shape.empty() || shape.size() > 5)
    return emitError() << "expected " << kind
                       << " payload rank between 1 and 5";

  if (llvm::any_of(shape, [](int64_t dim) { return dim <= 0; }))
    return emitError() << "expected static positive " << kind
                       << " payload shape";

  if (!llvm::isa<mlir::Float32Type>(elementType))
    return emitError() << "expected f32 " << kind << " payload type";

  return mlir::success();
}

// Restricts graph resource slots to handles, float scalars, or static f32
// tensor/memrefs.
mlir::LogicalResult verifyTaskResourcePayloadType(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type valueType) {
  if (llvm::isa<mlir::sculptor::RuntimeHandleType,
                mlir::sculptor::LogicalArrayType>(valueType))
    return mlir::success();

  if (llvm::isa<mlir::FloatType>(valueType))
    return mlir::success();

  if (auto rankedTensorType = llvm::dyn_cast<mlir::RankedTensorType>(valueType))
    return verifyPayloadRankAndElementType(
        emitError, rankedTensorType.getShape(),
        rankedTensorType.getElementType(), "tensor");

  if (auto memRefType = llvm::dyn_cast<mlir::MemRefType>(valueType)) {
    if (!memRefType.hasStaticShape())
      return emitError() << "expected static memref payload type";
    return verifyPayloadRankAndElementType(
        emitError, memRefType.getShape(), memRefType.getElementType(), "memref");
  }

  return emitError() << "expected runtime handle, logical array, float scalar, "
                        "ranked tensor, or memref payload type";
}

// Parses the compact shape-and-element spelling shared by matrix and vector.
mlir::ParseResult parseShapeAndEltImpl(mlir::AsmParser &parser,
                                       llvm::SmallVector<int64_t> &shape,
                                       mlir::Type &elementType) {
  shape.clear();

  while (true) {
    int64_t dim;
    mlir::OptionalParseResult maybeInt = parser.parseOptionalInteger(dim);
    if (!maybeInt.has_value())
      break;

    shape.push_back(dim);

    if (parser.parseXInDimensionList())
      return mlir::failure();
  }

  if (parser.parseType(elementType))
    return mlir::failure();

  return mlir::success();
}

// Emits the compact shape-and-element spelling shared by matrix and vector.
void printShapeAndEltImpl(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
                          mlir::Type elementType) {
  for (int64_t dim : shape)
    printer << dim << 'x';

  printer.printType(elementType);
}

// Prints tiled container views with a common grid/array-shape prefix.
void printTiledContainerType(mlir::AsmPrinter &printer,
                             llvm::ArrayRef<int64_t> gridShape,
                             llvm::ArrayRef<int64_t> arrayShape,
                             mlir::Type containerType) {
  printer << "grid_shape=[";
  llvm::interleaveComma(gridShape, printer);
  printer << "], array_shape=[";
  llvm::interleaveComma(arrayShape, printer);
  printer << "], ";
  printer.printType(containerType);
}

} // namespace

//===----------------------------------------------------------------------===//
// MatrixType - verify
//===----------------------------------------------------------------------===//

// Enforces the analog matrix contract before uniquing the type storage.
mlir::LogicalResult mlir::sculptor::MatrixType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  return verifyRank2FloatContainerType(emitError, shape, elementType);
}

//===----------------------------------------------------------------------===//
// MatrixType - parseShapeAndElt
//===----------------------------------------------------------------------===//

// Parses the matrix payload shape and element type from custom assembly.
mlir::ParseResult mlir::sculptor::MatrixType::parseShapeAndElt(
    mlir::AsmParser &parser,
    llvm::SmallVector<int64_t> &shape,
    mlir::Type &elementType) {
  return parseShapeAndEltImpl(parser, shape, elementType);
}

//===----------------------------------------------------------------------===//
// MatrixType - printShapeAndElt
//===----------------------------------------------------------------------===//

// Prints the matrix payload shape and element type for custom assembly.
void mlir::sculptor::MatrixType::printShapeAndElt(
    mlir::AsmPrinter &printer,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType) {
  printShapeAndEltImpl(printer, shape, elementType);
}

//===----------------------------------------------------------------------===//
// MatrixType - parse
//===----------------------------------------------------------------------===//

// Builds a matrix type from the payload inside the dialect mnemonic wrapper.
mlir::Type mlir::sculptor::MatrixType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;

  if (parseShapeAndElt(parser, shape, elementType))
    return {};

  return get(parser.getContext(), shape, elementType);
}

//===----------------------------------------------------------------------===//
// MatrixType - print
//===----------------------------------------------------------------------===//

// Emits the matrix payload expected inside the dialect mnemonic wrapper.
void mlir::sculptor::MatrixType::print(mlir::AsmPrinter &printer) const {
  printShapeAndElt(printer, getShape(), getElementType());
}

//===----------------------------------------------------------------------===//
// VectorType - verify
//===----------------------------------------------------------------------===//

// Enforces the analog vector contract before uniquing the type storage.
mlir::LogicalResult mlir::sculptor::VectorType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  return verifyRank2FloatContainerType(emitError, shape, elementType);
}

//===----------------------------------------------------------------------===//
// VectorType - parseShapeAndElt
//===----------------------------------------------------------------------===//

// Parses the vector payload shape and element type from custom assembly.
mlir::ParseResult mlir::sculptor::VectorType::parseShapeAndElt(
    mlir::AsmParser &parser,
    llvm::SmallVector<int64_t> &shape,
    mlir::Type &elementType) {
  return parseShapeAndEltImpl(parser, shape, elementType);
}

//===----------------------------------------------------------------------===//
// VectorType - printShapeAndElt
//===----------------------------------------------------------------------===//

// Prints the vector payload shape and element type for custom assembly.
void mlir::sculptor::VectorType::printShapeAndElt(
    mlir::AsmPrinter &printer,
    llvm::ArrayRef<int64_t> shape,
    mlir::Type elementType) {
  printShapeAndEltImpl(printer, shape, elementType);
}

//===----------------------------------------------------------------------===//
// VectorType - parse
//===----------------------------------------------------------------------===//

// Builds a vector type from the payload inside the dialect mnemonic wrapper.
mlir::Type mlir::sculptor::VectorType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;

  if (parseShapeAndElt(parser, shape, elementType))
    return {};

  return get(parser.getContext(), shape, elementType);
}

//===----------------------------------------------------------------------===//
// VectorType - print
//===----------------------------------------------------------------------===//

// Emits the vector payload expected inside the dialect mnemonic wrapper.
void mlir::sculptor::VectorType::print(mlir::AsmPrinter &printer) const {
  printShapeAndElt(printer, getShape(), getElementType());
}

//===----------------------------------------------------------------------===//
// TaskResourceType - verify
//===----------------------------------------------------------------------===//

// Accepts only payload types that task graph resources can safely carry.
mlir::LogicalResult mlir::sculptor::TaskResourceType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type valueType) {
  return verifyTaskResourcePayloadType(emitError, valueType);
}

//===----------------------------------------------------------------------===//
// TaskResourceType - parse
//===----------------------------------------------------------------------===//

// Builds a resource handle type around its parsed payload type.
mlir::Type mlir::sculptor::TaskResourceType::parse(mlir::AsmParser &parser) {
  mlir::Type valueType;
  if (parser.parseType(valueType))
    return {};

  return get(parser.getContext(), valueType);
}

//===----------------------------------------------------------------------===//
// TaskResourceType - print
//===----------------------------------------------------------------------===//

// Emits only the payload because the dialect printer owns the resource wrapper.
void mlir::sculptor::TaskResourceType::print(mlir::AsmPrinter &printer) const {
  printer.printType(getValueType());
}

//===----------------------------------------------------------------------===//
// MatrixGridType - verify
//===----------------------------------------------------------------------===//

// Validates the tiled matrix view metadata and its underlying matrix container.
mlir::LogicalResult mlir::sculptor::MatrixGridType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> gridShape,
    llvm::ArrayRef<int64_t> arrayShape,
    mlir::sculptor::MatrixType matrix) {
  if (arrayShape.size() != 2) {
    return emitError() << "array_shape must have exactly 2 dimensions";
  }

  if (arrayShape[0] <= 0 || arrayShape[1] <= 0) {
    return emitError() << "array_shape dimensions must be positive";
  }

  auto matrixShape = matrix.getShape();
  if (matrixShape.size() != 2) {
    return emitError() << "matrix must be rank-2";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MatrixGridType - parse
//===----------------------------------------------------------------------===//

// Builds a tiled matrix-grid view from its custom assembly payload.
mlir::Type mlir::sculptor::MatrixGridType::parse(mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> gridShape;
  llvm::SmallVector<int64_t> arrayShape;
  mlir::Type matrixType;

  // Parse the wrapper and grid-shape segment before the array metadata.
  if (parser.parseLess()) {
    return {};
  }

  if (parser.parseKeyword("grid_shape") ||
      parser.parseEqual() ||
      parser.parseLSquare()) {
    return {};
  }

  int64_t gridDim;
  if (parser.parseInteger(gridDim)) {
    return {};
  }
  gridShape.push_back(gridDim);

  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(gridDim)) {
      return {};
    }
    gridShape.push_back(gridDim);
  }

  if (parser.parseRSquare()) {
    return {};
  }

  if (parser.parseComma()) {
    return {};
  }

  // Parse the array-shape segment using the same bracketed form.
  if (parser.parseKeyword("array_shape") ||
      parser.parseEqual() ||
      parser.parseLSquare()) {
    return {};
  }

  int64_t arrayDim;
  if (parser.parseInteger(arrayDim)) {
    return {};
  }
  arrayShape.push_back(arrayDim);

  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(arrayDim)) {
      return {};
    }
    arrayShape.push_back(arrayDim);
  }

  if (parser.parseRSquare()) {
    return {};
  }

  if (parser.parseComma()) {
    return {};
  }

  // Finish with the nested matrix type and reject other payload kinds.
  if (parser.parseType(matrixType)) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  auto matrix = llvm::dyn_cast<mlir::sculptor::MatrixType>(matrixType);
  if (!matrix) {
    parser.emitError(parser.getCurrentLocation(), "expected sculptor.matrix type");
    return {};
  }

  return get(parser.getContext(), gridShape, arrayShape, matrix);
}

//===----------------------------------------------------------------------===//
// MatrixGridType - print
//===----------------------------------------------------------------------===//

// Emits the tiled matrix-grid payload expected by the custom parser.
void mlir::sculptor::MatrixGridType::print(mlir::AsmPrinter &printer) const {
  printTiledContainerType(printer, getGridShape(), getArrayShape(), getMatrix());
}

//===----------------------------------------------------------------------===//
// VectorSliceType - verify
//===----------------------------------------------------------------------===//

// Validates vector-slice metadata and its underlying vector container.
mlir::LogicalResult mlir::sculptor::VectorSliceType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> gridShape,
    llvm::ArrayRef<int64_t> arrayShape,
    mlir::sculptor::VectorType vector) {
  if (gridShape.size() != 2) {
    return emitError() << "grid_shape must be 2-dimensional";
  }

  if (gridShape[0] <= 0 || gridShape[1] <= 0) {
    return emitError() << "grid_shape dimension must be positive";
  }

  if (arrayShape.size() != 2) {
    return emitError() << "array_shape must be 2-dimensional";
  }

  if (arrayShape[0] <= 0 || arrayShape[1] <= 0) {
    return emitError() << "array_shape dimension must be positive";
  }

  auto vecShape = vector.getShape();
  if (vecShape.size() != 2) {
    return emitError() << "vector must be rank-2 (1xN)";
  }

  if (vecShape[0] != 1) {
    return emitError() << "vector must have leading dimension 1";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// VectorSliceType - parse
//===----------------------------------------------------------------------===//

// Builds a tiled vector-slice view from its custom assembly payload.
mlir::Type mlir::sculptor::VectorSliceType::parse(mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> gridShape;
  llvm::SmallVector<int64_t> arrayShape;
  mlir::Type vectorType;

  // Parse the wrapper and grid-shape segment before the array metadata.
  if (parser.parseLess()) {
    return {};
  }

  if (parser.parseKeyword("grid_shape") ||
      parser.parseEqual() ||
      parser.parseLSquare()) {
    return {};
  }

  int64_t gridDim;
  if (parser.parseInteger(gridDim)) {
    return {};
  }
  gridShape.push_back(gridDim);

  if (parser.parseRSquare()) {
    return {};
  }

  if (parser.parseComma()) {
    return {};
  }

  // Parse the array-shape segment before the nested vector type.
  if (parser.parseKeyword("array_shape") ||
      parser.parseEqual() ||
      parser.parseLSquare()) {
    return {};
  }

  int64_t arrayDim;
  if (parser.parseInteger(arrayDim)) {
    return {};
  }
  arrayShape.push_back(arrayDim);

  if (parser.parseRSquare()) {
    return {};
  }

  if (parser.parseComma()) {
    return {};
  }

  // Finish with the nested vector type and reject other payload kinds.
  if (parser.parseType(vectorType)) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  auto vector = llvm::dyn_cast<mlir::sculptor::VectorType>(vectorType);
  if (!vector) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected sculptor.vector type");
    return {};
  }

  return get(parser.getContext(), gridShape, arrayShape, vector);
}

//===----------------------------------------------------------------------===//
// VectorSliceType - print
//===----------------------------------------------------------------------===//

// Emits the tiled vector-slice payload expected by the custom parser.
void mlir::sculptor::VectorSliceType::print(mlir::AsmPrinter &printer) const {
  printTiledContainerType(printer, getGridShape(), getArrayShape(), getVector());
}
