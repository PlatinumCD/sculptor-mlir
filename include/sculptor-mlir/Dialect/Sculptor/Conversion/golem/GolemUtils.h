#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_GOLEMUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_GOLEMUTILS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

// Helpers for the Sculptor Golem -> LLVM shim lowering stage. These utilities do
// not expand sculptor.mvm and do not emit target ISA intrinsics; they build the
// stable shim calls consumed by FinalizeGolemIntrinsics.

namespace mlir {
namespace sculptor {
namespace golem {

// Runtime shim names emitted by the Sculptor Golem -> LLVM shim lowering. The
// final intrinsic pass maps these stable shim calls to target Golem ISA names.
inline constexpr llvm::StringLiteral kSetShimName = "golem_analog_mvm_set";
inline constexpr llvm::StringLiteral kLoadShimName = "golem_analog_mvm_load";
inline constexpr llvm::StringLiteral kComputeShimName =
    "golem_analog_mvm_compute";
inline constexpr llvm::StringLiteral kStoreShimName = "golem_analog_mvm_store";

// Carries static tile shape and dynamic bounds for placing one matrix tile.
struct MatrixPlacementPlan {
  // Static hardware shape used for scratch allocation and array id math.
  int64_t arrayRows;
  int64_t arrayCols;
  int64_t gridCols;

  // Reusable index constants keep generated loop bounds consistent.
  Value c0;
  Value c1;
  Value cArrayRows;
  Value cArrayCols;

  // Source offsets and clamped extents describe the active edge tile region.
  Value rowOffset;
  Value colOffset;
  Value copyRows;
  Value copyCols;
};

// Carries static slice shape and dynamic bounds for placing one vector slice.
struct VectorPlacementPlan {
  // Static hardware shape used for scratch allocation and array id math.
  int64_t arrayCols;
  int64_t gridCols;

  // Reusable index constants keep generated loop bounds consistent.
  Value c0;
  Value c1;
  Value cArrayCols;

  // Source offset and clamped extent describe the active edge slice region.
  Value colOffset;
  Value copyCols;
};

// Returns a private shim declaration, creating it in the module if needed.
func::FuncOp getOrCreateShimDecl(ModuleOp module, llvm::StringRef name,
                                 FunctionType type);

// Centralizes the opaque LLVM pointer type used by runtime ABI values.
Type getOpaquePointerType(MLIRContext *context);

// Deduplicates an LLVM string global by contents before creating a new constant.
LLVM::GlobalOp getOrCreateStringConstant(OpBuilder &builder, Location loc,
                                         ModuleOp module, llvm::StringRef prefix,
                                         llvm::StringRef value);

// Emits an LLVM pointer to a deduplicated global string in the given module.
Value getOrCreateGlobalStringPtr(OpBuilder &builder, Location loc,
                                 ModuleOp module, llvm::StringRef prefix,
                                 llvm::StringRef value);

// Finds the enclosing module from the rewrite point before emitting the pointer.
Value getOrCreateGlobalStringPtr(PatternRewriter &rewriter, Location loc,
                                 llvm::StringRef prefix,
                                 llvm::StringRef value);

// Normalizes values to the i32 Golem ABI, using zero for unsupported types.
Value castToI32(PatternRewriter &rewriter, Location loc, Value value);

// Computes the row-major hardware array id for grid coordinates.
Value buildLinearArrayId(PatternRewriter &rewriter, Location loc, Value row,
                         Value col, int64_t gridCols);

// Declares the runtime/backend shim if needed and emits a void call.
void emitShimCall(PatternRewriter &rewriter, Location loc,
                  llvm::StringRef shimName, ValueRange operands);

// Returns the scheduled local array id attached to an analog task function.
FailureOr<int64_t> getRequiredLocalArrayId(Operation *op);

// Materializes the scheduled local array id as the i32 shim ABI value.
FailureOr<Value> buildLocalArrayId(PatternRewriter &rewriter, Operation *op);

// Buffers an already-ranked tensor as a same-shaped memref for scratch copies.
Value materializeTensorMemref(PatternRewriter &rewriter, Location loc,
                              Value tensor);

// Returns a zero attribute or reports a match failure for unsupported elements.
FailureOr<TypedAttr> getZeroAttrForElementType(PatternRewriter &rewriter,
                                               Operation *op,
                                               Type elementType);

// Emits nested loops that zero-initialize the active rectangle of a 2D memref.
void zeroFill2DMemref(PatternRewriter &rewriter, Location loc,
                      Value memrefValue, Value rowUpper, Value colUpper,
                      Value zero, Value c0, Value c1);

// Copies a clamped matrix tile from the full buffer into scratch memory.
void copyMatrixTileIntoScratch(PatternRewriter &rewriter, Location loc,
                               Value fullMemref, Value arrayMemref,
                               Value rowOffset, Value colOffset, Value copyRows,
                               Value copyCols, Value c0, Value c1);

// Copies a clamped vector slice from the full buffer into scratch memory.
void copyVectorSliceIntoScratch(PatternRewriter &rewriter, Location loc,
                                Value fullMemref, Value arrayMemref,
                                Value colOffset, Value copyCols, Value c0,
                                Value c1);

// Narrows a store destination memref to the selected one-array result slice.
Value buildStoreDestinationSubview(PatternRewriter &rewriter, Location loc,
                                   Value destination, Value arrayRow,
                                   Value arrayCol, int64_t arrayRows, Value c0);

// Allocates the aligned ABI-shaped scratch buffer populated by store runtime.
Value allocateStoreScratchBuffer(PatternRewriter &rewriter, Location loc,
                                 int64_t arrayRows, Type elementType);

// Copies runtime-populated store lanes back into the selected destination slice.
void copyStoreScratchToDestination(PatternRewriter &rewriter, Location loc,
                                   Value scratch, Value destinationSlice,
                                   int64_t arrayRows, Value c0);

// Computes min(sourceUpper - offset, tileUpper) for partial edge tiles.
Value buildClampedCopyUpperBound(PatternRewriter &rewriter, Location loc,
                                 Value sourceUpper, Value offset,
                                 Value tileUpper);

// Builds the constants, offsets, and copy bounds shared by matrix placement.
MatrixPlacementPlan buildMatrixPlacementPlan(
    PatternRewriter &rewriter, sculptor::ArrayMatrixPlaceOp op, Value rowIndex,
    Value colIndex, Value fullMemref, sculptor::MatrixGridType gridTy);

// Builds the constants, offsets, and copy bounds shared by vector placement.
VectorPlacementPlan buildVectorPlacementPlan(
    PatternRewriter &rewriter, sculptor::ArrayVectorPlaceOp op, Value sliceIndex,
    Value fullMemref, sculptor::VectorSliceType sliceTy);

// Allocates scratch memory and fails if the element type cannot be zero-filled.
FailureOr<Value> allocateZeroedScratchTile(PatternRewriter &rewriter,
                                           Operation *op,
                                           llvm::ArrayRef<int64_t> shape,
                                           Type elementType, Value rowUpper,
                                           Value colUpper, Value c0, Value c1);

// Maps Sculptor aggregate types to ranked tensors while preserving other types.
void populateSculptorTypeConversions(TypeConverter &typeConverter);

// Adds matrix lowering patterns that erase wrappers and emit Golem set shims.
void populateLowerMatrixPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext *ctx);

// Adds vector lowering patterns that erase wrappers and emit Golem load shims.
void populateLowerVectorPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext *ctx);

// Adds execute lowering patterns that emit Golem compute shims.
void populateLowerExecutePatterns(RewritePatternSet &patterns,
                                  TypeConverter &typeConverter,
                                  MLIRContext *ctx);

// Adds store lowering patterns that emit Golem store shims into destinations.
void populateLowerStorePatterns(RewritePatternSet &patterns,
                                TypeConverter &typeConverter,
                                MLIRContext *ctx);

} // namespace golem
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_GOLEMUTILS_H
