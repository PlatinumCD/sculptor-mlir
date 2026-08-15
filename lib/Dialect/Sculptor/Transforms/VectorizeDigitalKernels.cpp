#include "sculptor-mlir/Dialect/Sculptor/Transforms/VectorizeDigitalKernels.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <functional>
#include <optional>

namespace {

using namespace mlir;
using namespace mlir::sculptor;

constexpr StringLiteral kSummaryAttrName =
    "sculptor.digital.kernel_vectorization";

struct GeluKernel {
  linalg::GenericOp operation;
  Value source;
  Value target;
  MemRefType sourceType;
  MemRefType targetType;
  int64_t vectorLanes = 0;
  bool maskedTail = false;
};

struct BinaryKernel {
  Operation *operation = nullptr;
  Value left;
  Value right;
  Value target;
  MemRefType leftType;
  MemRefType rightType;
  MemRefType targetType;
  int64_t vectorLanes = 0;
  bool maskedTail = false;
  StringRef kind;
};

FailureOr<SmallVector<int64_t>> getStaticStrides(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset) ||
      llvm::is_contained(strides, ShapedType::kDynamic))
    return failure();
  return strides;
}

bool hasGeluRegion(linalg::GenericOp operation) {
  if (!operation.getRegion().hasOneBlock())
    return false;
  int64_t erfCount = 0;
  operation.getRegion().walk([&](math::ErfOp) { ++erfCount; });
  return erfCount == 1;
}

FailureOr<GeluKernel> classifyGelu(linalg::GenericOp operation,
                                  int64_t vectorBits) {
  auto activation = operation->getAttrOfType<StringAttr>("activation");
  if (!activation || activation.getValue() != "gelu")
    return failure();
  if (!operation.hasPureBufferSemantics())
    return operation.emitError("vector GELU requires buffer semantics");
  if (operation.getNumDpsInputs() != 1 ||
      operation.getNumDpsInits() != 1)
    return operation.emitError(
        "vector GELU requires one input and one output");
  if (!hasGeluRegion(operation))
    return operation.emitError(
        "GELU semantic marker does not contain one math.erf operation");

  Value source = operation.getDpsInputs()[0];
  Value target = operation.getDpsInits()[0];
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto targetType = dyn_cast<MemRefType>(target.getType());
  if (!sourceType || !targetType ||
      sourceType.getShape() != targetType.getShape() ||
      sourceType.getElementType() != targetType.getElementType())
    return operation.emitError(
        "vector activation requires equal input and output shapes and "
        "element types");
  if (!sourceType.hasStaticShape() || sourceType.getRank() == 0 ||
      !sourceType.getElementType().isF32())
    return operation.emitError(
        "vector GELU requires a static ranked f32 memref");
  if (sourceType.getDimSize(sourceType.getRank() - 1) <= 0)
    return operation.emitError(
        "vector GELU requires a nonempty innermost dimension");
  if (operation.getNumLoops() != sourceType.getRank() ||
      operation.getNumParallelLoops() != sourceType.getRank())
    return operation.emitError(
        "vector GELU requires parallel loops for all dimensions");
  for (AffineMap map : operation.getIndexingMapsArray()) {
    if (!map.isIdentity())
      return operation.emitError(
          "vector GELU requires identity indexing maps");
  }

  auto sourceStrides = getStaticStrides(sourceType);
  auto targetStrides = getStaticStrides(targetType);
  if (failed(sourceStrides) || failed(targetStrides) ||
      sourceStrides->back() != 1 || targetStrides->back() != 1)
    return operation.emitError(
        "vector GELU requires a unit-stride innermost dimension");
  if (vectorBits <= 0 || vectorBits % 32 != 0 || vectorBits / 32 < 2)
    return operation.emitError(
        "vector GELU width must contain at least two f32 lanes");

  int64_t vectorLanes = vectorBits / 32;
  return GeluKernel{operation,
                    source,
                    target,
                    sourceType,
                    targetType,
                    vectorLanes,
                    sourceType.getShape().back() % vectorLanes != 0};
}

FailureOr<BinaryKernel> classifyElementwiseAdd(linalg::AddOp operation,
                                               int64_t vectorBits) {
  auto section =
      operation->getAttrOfType<StringAttr>("sculptor.semantic.section");
  if (!section || !section.getValue().starts_with("digital."))
    return failure();
  if (!operation.hasPureBufferSemantics() ||
      operation.getNumDpsInputs() != 2 || operation.getNumDpsInits() != 1)
    return operation.emitError(
        "vector elementwise addition requires two inputs and one output");

  Value left = operation.getDpsInputs()[0];
  Value right = operation.getDpsInputs()[1];
  Value target = operation.getDpsInits()[0];
  auto leftType = dyn_cast<MemRefType>(left.getType());
  auto rightType = dyn_cast<MemRefType>(right.getType());
  auto targetType = dyn_cast<MemRefType>(target.getType());
  if (!leftType || !rightType || !targetType ||
      leftType.getShape() != rightType.getShape() ||
      leftType.getShape() != targetType.getShape() ||
      leftType.getElementType() != rightType.getElementType() ||
      leftType.getElementType() != targetType.getElementType())
    return operation.emitError(
        "vector elementwise addition requires equal input and output shapes");
  if (!leftType.hasStaticShape() || leftType.getRank() == 0 ||
      !leftType.getElementType().isF32())
    return operation.emitError(
        "vector elementwise addition requires a static ranked f32 memref");
  if (leftType.getDimSize(leftType.getRank() - 1) <= 0)
    return operation.emitError(
        "vector elementwise addition requires a nonempty inner dimension");

  for (MemRefType type : {leftType, rightType, targetType}) {
    auto strides = getStaticStrides(type);
    if (failed(strides) || strides->back() != 1)
      return operation.emitError(
          "vector elementwise addition requires unit inner strides");
  }
  if (vectorBits <= 0 || vectorBits % 32 != 0 || vectorBits / 32 < 2)
    return operation.emitError(
        "vector elementwise width must contain at least two f32 lanes");

  int64_t vectorLanes = vectorBits / 32;
  return BinaryKernel{operation,
                      left,
                      right,
                      target,
                      leftType,
                      rightType,
                      targetType,
                      vectorLanes,
                      leftType.getShape().back() % vectorLanes != 0,
                      "add"};
}

Value indexConstant(OpBuilder &builder, Location location, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(location, value);
}

Value f32Splat(OpBuilder &builder, Location location, VectorType type,
               double value) {
  auto scalar = builder.getF32FloatAttr(static_cast<float>(value));
  auto splat = DenseElementsAttr::get(type, scalar);
  return builder.create<arith::ConstantOp>(location, type, splat);
}

Value i32Splat(OpBuilder &builder, Location location, VectorType type,
               int32_t value) {
  auto scalar = builder.getI32IntegerAttr(value);
  auto splat = DenseElementsAttr::get(type, scalar);
  return builder.create<arith::ConstantOp>(location, type, splat);
}

Value emitVectorExpApproximation(OpBuilder &builder, Location location,
                                 Value value, VectorType floatType) {
  VectorType intType =
      VectorType::get(floatType.getShape(), builder.getI32Type());
  Value zero = f32Splat(builder, location, floatType, 0.0);
  Value one = f32Splat(builder, location, floatType, 1.0);
  Value maximum =
      f32Splat(builder, location, floatType, 3.402823466e38);
  Value positiveLimit = f32Splat(builder, location, floatType, 88.0);
  Value negativeLimit = f32Splat(builder, location, floatType, -88.0);
  Value inverseLn2 =
      f32Splat(builder, location, floatType, 1.44269504089);
  Value ln2 = f32Splat(builder, location, floatType, 0.69314718056);

  Value scaled = builder.create<arith::MulFOp>(location, value, inverseLn2);
  Value exponent =
      builder.create<arith::FPToSIOp>(location, intType, scaled);
  Value exponentFloat =
      builder.create<arith::SIToFPOp>(location, floatType, exponent);
  Value fractionProduct =
      builder.create<arith::MulFOp>(location, exponentFloat, ln2);
  Value fraction =
      builder.create<arith::SubFOp>(location, value, fractionProduct);

  Value cHalf = f32Splat(builder, location, floatType, 0.5);
  Value cSixth = f32Splat(builder, location, floatType, 0.16666667);
  Value cTwentyFourth = f32Splat(builder, location, floatType, 0.04166667);
  Value cOneTwenty = f32Splat(builder, location, floatType, 0.00833333);
  Value polynomial = cOneTwenty;
  polynomial = builder.create<arith::AddFOp>(
      location, cTwentyFourth,
      builder.create<arith::MulFOp>(location, fraction, polynomial));
  polynomial = builder.create<arith::AddFOp>(
      location, cSixth,
      builder.create<arith::MulFOp>(location, fraction, polynomial));
  polynomial = builder.create<arith::AddFOp>(
      location, cHalf,
      builder.create<arith::MulFOp>(location, fraction, polynomial));
  polynomial = builder.create<arith::AddFOp>(
      location, one,
      builder.create<arith::MulFOp>(location, fraction, polynomial));
  polynomial = builder.create<arith::AddFOp>(
      location, one,
      builder.create<arith::MulFOp>(location, fraction, polynomial));

  Value bias = i32Splat(builder, location, intType, 127);
  Value shift = i32Splat(builder, location, intType, 23);
  Value biasedExponent =
      builder.create<arith::AddIOp>(location, exponent, bias);
  Value scaleBits =
      builder.create<arith::ShLIOp>(location, biasedExponent, shift);
  Value scale =
      builder.create<arith::BitcastOp>(location, floatType, scaleBits);
  Value result = builder.create<arith::MulFOp>(location, polynomial, scale);

  Value zeroInt = i32Splat(builder, location, intType, 0);
  Value maximumInt = i32Splat(builder, location, intType, 255);
  Value underflow = builder.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::sle, biasedExponent, zeroInt);
  Value overflow = builder.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::sge, biasedExponent, maximumInt);
  result = builder.create<arith::SelectOp>(location, underflow, zero, result);
  result =
      builder.create<arith::SelectOp>(location, overflow, maximum, result);

  Value belowLimit = builder.create<arith::CmpFOp>(
      location, arith::CmpFPredicate::OLT, value, negativeLimit);
  Value aboveLimit = builder.create<arith::CmpFOp>(
      location, arith::CmpFPredicate::OGT, value, positiveLimit);
  result = builder.create<arith::SelectOp>(location, belowLimit, zero, result);
  return builder.create<arith::SelectOp>(location, aboveLimit, maximum,
                                         result);
}

Value emitVectorGelu(OpBuilder &builder, Location location, Value input,
                     VectorType vectorType) {
  Value zero = f32Splat(builder, location, vectorType, 0.0);
  Value one = f32Splat(builder, location, vectorType, 1.0);
  Value negativeOne = f32Splat(builder, location, vectorType, -1.0);
  Value half = f32Splat(builder, location, vectorType, 0.5);
  Value inverseSqrt2 =
      f32Splat(builder, location, vectorType, 0.7071067811865475);

  Value scaled =
      builder.create<arith::MulFOp>(location, input, inverseSqrt2);
  Value negative = builder.create<arith::CmpFOp>(
      location, arith::CmpFPredicate::OLT, scaled, zero);
  Value sign = builder.create<arith::SelectOp>(location, negative,
                                               negativeOne, one);
  Value negated = builder.create<arith::NegFOp>(location, scaled);
  Value absolute =
      builder.create<arith::SelectOp>(location, negative, negated, scaled);

  Value p = f32Splat(builder, location, vectorType, 0.3275911);
  Value denominator = builder.create<arith::AddFOp>(
      location, one,
      builder.create<arith::MulFOp>(location, p, absolute));
  Value t = builder.create<arith::DivFOp>(location, one, denominator);

  Value polynomial = f32Splat(builder, location, vectorType, 1.061405429);
  Value c1 = f32Splat(builder, location, vectorType, -1.453152027);
  Value c2 = f32Splat(builder, location, vectorType, 1.421413741);
  Value c3 = f32Splat(builder, location, vectorType, -0.284496736);
  Value c4 = f32Splat(builder, location, vectorType, 0.254829592);
  polynomial = builder.create<arith::AddFOp>(
      location,
      builder.create<arith::MulFOp>(location, polynomial, t), c1);
  polynomial = builder.create<arith::AddFOp>(
      location,
      builder.create<arith::MulFOp>(location, polynomial, t), c2);
  polynomial = builder.create<arith::AddFOp>(
      location,
      builder.create<arith::MulFOp>(location, polynomial, t), c3);
  polynomial = builder.create<arith::AddFOp>(
      location,
      builder.create<arith::MulFOp>(location, polynomial, t), c4);
  polynomial = builder.create<arith::MulFOp>(location, polynomial, t);

  Value square =
      builder.create<arith::MulFOp>(location, absolute, absolute);
  Value negativeSquare = builder.create<arith::NegFOp>(location, square);
  Value exponential = emitVectorExpApproximation(
      builder, location, negativeSquare, vectorType);
  Value correction =
      builder.create<arith::MulFOp>(location, polynomial, exponential);
  Value erfMagnitude =
      builder.create<arith::SubFOp>(location, one, correction);
  Value erf = builder.create<arith::MulFOp>(location, sign, erfMagnitude);
  Value factor = builder.create<arith::AddFOp>(location, one, erf);
  Value halfInput = builder.create<arith::MulFOp>(location, half, input);
  return builder.create<arith::MulFOp>(location, halfInput, factor);
}

Operation *emitInnermostGelu(OpBuilder &builder, Location location,
                             const GeluKernel &kernel,
                             SmallVector<Value> indices) {
  VectorType vectorType = VectorType::get(
      {kernel.vectorLanes}, kernel.sourceType.getElementType());
  int64_t innerSize = kernel.sourceType.getShape().back();
  int64_t fullEnd = innerSize - innerSize % kernel.vectorLanes;
  unsigned innerDimension = indices.size() - 1;
  Operation *root = nullptr;

  if (fullEnd > 0) {
    Value lower = indexConstant(builder, location, 0);
    Value upper = indexConstant(builder, location, fullEnd);
    Value step = indexConstant(builder, location, kernel.vectorLanes);
    scf::ForOp loop = builder.create<scf::ForOp>(location, lower, upper, step);
    root = loop;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    indices[innerDimension] = loop.getInductionVar();
    Value loaded = builder.create<vector::LoadOp>(
        location, vectorType, kernel.source, indices);
    Value result = emitVectorGelu(builder, location, loaded, vectorType);
    builder.create<vector::StoreOp>(location, result, kernel.target, indices);
  }

  int64_t tail = innerSize - fullEnd;
  if (tail == 0)
    return root;
  indices[innerDimension] = indexConstant(builder, location, fullEnd);
  Value tailSize = indexConstant(builder, location, tail);
  VectorType maskType =
      VectorType::get({kernel.vectorLanes}, builder.getI1Type());
  Value mask =
      builder.create<vector::CreateMaskOp>(location, maskType, tailSize);
  Value passThrough = f32Splat(builder, location, vectorType, 0.0);
  auto load = builder.create<vector::MaskedLoadOp>(
      location, vectorType, kernel.source, indices, mask, passThrough);
  if (!root)
    root = load;
  Value result = emitVectorGelu(builder, location, load, vectorType);
  builder.create<vector::MaskedStoreOp>(location, kernel.target, indices, mask,
                                        result);
  return root;
}

Operation *emitInnermostBinary(OpBuilder &builder, Location location,
                               const BinaryKernel &kernel,
                               SmallVector<Value> indices) {
  VectorType vectorType =
      VectorType::get({kernel.vectorLanes}, builder.getF32Type());
  int64_t innerSize = kernel.leftType.getShape().back();
  int64_t fullEnd = innerSize - innerSize % kernel.vectorLanes;
  unsigned innerDimension = indices.size() - 1;
  Operation *root = nullptr;

  auto calculate = [&](Value left, Value right) -> Value {
    if (kernel.kind == "add")
      return builder.create<arith::AddFOp>(location, left, right);
    llvm_unreachable("unregistered binary vector kernel");
  };

  if (fullEnd > 0) {
    Value lower = indexConstant(builder, location, 0);
    Value upper = indexConstant(builder, location, fullEnd);
    Value step = indexConstant(builder, location, kernel.vectorLanes);
    scf::ForOp loop = builder.create<scf::ForOp>(location, lower, upper, step);
    root = loop;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    indices[innerDimension] = loop.getInductionVar();
    Value left = builder.create<vector::LoadOp>(location, vectorType,
                                                kernel.left, indices);
    Value right = builder.create<vector::LoadOp>(location, vectorType,
                                                 kernel.right, indices);
    Value result = calculate(left, right);
    builder.create<vector::StoreOp>(location, result, kernel.target, indices);
  }

  int64_t tail = innerSize - fullEnd;
  if (tail == 0)
    return root;
  indices[innerDimension] = indexConstant(builder, location, fullEnd);
  Value tailSize = indexConstant(builder, location, tail);
  VectorType maskType =
      VectorType::get({kernel.vectorLanes}, builder.getI1Type());
  Value mask =
      builder.create<vector::CreateMaskOp>(location, maskType, tailSize);
  Value passThrough = f32Splat(builder, location, vectorType, 0.0);
  Value left = builder.create<vector::MaskedLoadOp>(
      location, vectorType, kernel.left, indices, mask, passThrough);
  if (!root)
    root = left.getDefiningOp();
  Value right = builder.create<vector::MaskedLoadOp>(
      location, vectorType, kernel.right, indices, mask, passThrough);
  Value result = calculate(left, right);
  builder.create<vector::MaskedStoreOp>(location, kernel.target, indices, mask,
                                        result);
  return root;
}

void copySemanticAttributes(Operation *source, Operation *target,
                            Builder &builder, int64_t vectorBits,
                            StringRef kernelFamily, StringRef kernelKind) {
  for (NamedAttribute attribute : source->getAttrs()) {
    StringRef name = attribute.getName().getValue();
    if (name == "activation" || name.starts_with("sculptor.semantic.") ||
        name.starts_with("sculptor.mapping.") ||
        name == "sculptor.deployment.physical_tile_id")
      target->setAttr(attribute.getName(), attribute.getValue());
  }
  target->setAttr("sculptor.digital.vectorized", builder.getUnitAttr());
  target->setAttr("sculptor.digital.kernel_family",
                  builder.getStringAttr(kernelFamily));
  target->setAttr("sculptor.digital.kernel_kind",
                  builder.getStringAttr(kernelKind));
  target->setAttr("sculptor.digital.vector_bits",
                  builder.getI64IntegerAttr(vectorBits));
}

LogicalResult emitGeluKernel(const GeluKernel &kernel, int64_t vectorBits) {
  linalg::GenericOp operation = kernel.operation;
  OpBuilder builder(operation);
  Location location = operation.getLoc();
  SmallVector<Value> indices(kernel.sourceType.getRank());
  Operation *root = nullptr;

  std::function<void(unsigned)> emitDimension = [&](unsigned dimension) {
    if (dimension + 1 ==
        static_cast<unsigned>(kernel.sourceType.getRank())) {
      Operation *inner =
          emitInnermostGelu(builder, location, kernel, indices);
      if (!root)
        root = inner;
      return;
    }
    Value lower = indexConstant(builder, location, 0);
    Value upper = indexConstant(builder, location,
                                kernel.sourceType.getDimSize(dimension));
    Value step = indexConstant(builder, location, 1);
    scf::ForOp loop = builder.create<scf::ForOp>(location, lower, upper, step);
    if (!root)
      root = loop;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    indices[dimension] = loop.getInductionVar();
    emitDimension(dimension + 1);
  };

  emitDimension(0);
  if (!root)
    return operation.emitError("failed to create vector GELU loop");
  copySemanticAttributes(operation, root, builder, vectorBits, "activation",
                         "gelu");
  operation.erase();
  return success();
}

LogicalResult emitBinaryKernel(const BinaryKernel &kernel,
                               int64_t vectorBits) {
  OpBuilder builder(kernel.operation);
  Location location = kernel.operation->getLoc();
  SmallVector<Value> indices(kernel.leftType.getRank());
  Operation *root = nullptr;

  std::function<void(unsigned)> emitDimension = [&](unsigned dimension) {
    if (dimension + 1 ==
        static_cast<unsigned>(kernel.leftType.getRank())) {
      Operation *inner =
          emitInnermostBinary(builder, location, kernel, indices);
      if (!root)
        root = inner;
      return;
    }
    Value lower = indexConstant(builder, location, 0);
    Value upper =
        indexConstant(builder, location, kernel.leftType.getDimSize(dimension));
    Value step = indexConstant(builder, location, 1);
    scf::ForOp loop = builder.create<scf::ForOp>(location, lower, upper, step);
    if (!root)
      root = loop;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    indices[dimension] = loop.getInductionVar();
    emitDimension(dimension + 1);
  };

  emitDimension(0);
  if (!root)
    return kernel.operation->emitError(
        "failed to create elementwise vector loop");
  copySemanticAttributes(kernel.operation, root, builder, vectorBits,
                         "elementwise", kernel.kind);
  kernel.operation->erase();
  return success();
}

} // namespace

void VectorizeDigitalKernelsPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (vectorBits <= 0 || vectorBits % 32 != 0) {
    module.emitError("digital kernel vector width must be a positive "
                     "multiple of 32 bits");
    return signalPassFailure();
  }
  SmallVector<linalg::GenericOp> candidates;
  module.walk([&](linalg::GenericOp operation) {
    auto activation = operation->getAttrOfType<StringAttr>("activation");
    if (activation && activation.getValue() == "gelu")
      candidates.push_back(operation);
  });
  SmallVector<linalg::AddOp> addCandidates;
  module.walk([&](linalg::AddOp operation) {
    auto section =
        operation->getAttrOfType<StringAttr>("sculptor.semantic.section");
    if (section && section.getValue().starts_with("digital."))
      addCandidates.push_back(operation);
  });

  int64_t vectorizedCount = 0;
  int64_t elementwiseCount = 0;
  int64_t maskedTailCount = 0;
  for (linalg::GenericOp operation : candidates) {
    FailureOr<GeluKernel> kernel = classifyGelu(operation, vectorBits);
    if (failed(kernel))
      return signalPassFailure();
    maskedTailCount += kernel->maskedTail;
    if (failed(emitGeluKernel(*kernel, vectorBits)))
      return signalPassFailure();
    ++vectorizedCount;
  }
  for (linalg::AddOp operation : addCandidates) {
    FailureOr<BinaryKernel> kernel =
        classifyElementwiseAdd(operation, vectorBits);
    if (failed(kernel))
      return signalPassFailure();
    maskedTailCount += kernel->maskedTail;
    if (failed(emitBinaryKernel(*kernel, vectorBits)))
      return signalPassFailure();
    ++elementwiseCount;
    ++vectorizedCount;
  }

  Builder builder(module.getContext());
  module->setAttr(
      kSummaryAttrName,
      builder.getDictionaryAttr({
          builder.getNamedAttr("version", builder.getI64IntegerAttr(1)),
          builder.getNamedAttr("vector_bits",
                               builder.getI64IntegerAttr(vectorBits)),
          builder.getNamedAttr("activation_kernel_count",
                               builder.getI64IntegerAttr(
                                   vectorizedCount - elementwiseCount)),
          builder.getNamedAttr("elementwise_kernel_count",
                               builder.getI64IntegerAttr(elementwiseCount)),
          builder.getNamedAttr("vectorized_kernel_count",
                               builder.getI64IntegerAttr(vectorizedCount)),
          builder.getNamedAttr("masked_tail_count",
                               builder.getI64IntegerAttr(maskedTailCount)),
      }));

  if (failed(verify(module))) {
    module.emitError("digital kernel vectorization produced invalid IR");
    signalPassFailure();
  }
}

void mlir::sculptor::registerVectorizeDigitalKernelsPass() {
  PassRegistration<VectorizeDigitalKernelsPass>();
}
