#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTELEMENTWISEUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTELEMENTWISEUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace sculptor {
namespace converter_recurrent_elementwise {

namespace detail {

// Builds unary linalg.generic gate math with a new tensor init.
template <typename BodyBuilder>
inline Value buildUnaryElementwise(Location loc, RankedTensorType resultTy,
                                   Value input, OpBuilder &builder,
                                   BodyBuilder bodyBuilder) {
  Value init =
      builder.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                      resultTy.getElementType());
  SmallVector<AffineMap, 2> indexingMaps(
      2, builder.getMultiDimIdentityMap(resultTy.getRank()));
  SmallVector<utils::IteratorType, 2> iteratorTypes(
      resultTy.getRank(), utils::IteratorType::parallel);

  return builder
      .create<linalg::GenericOp>(loc, resultTy, ValueRange{input},
                                 ValueRange{init}, indexingMaps, iteratorTypes,
                                 bodyBuilder)
      .getResult(0);
}

// Builds binary linalg.generic gate math with a new tensor init.
template <typename BodyBuilder>
inline Value buildBinaryElementwise(Location loc, RankedTensorType resultTy,
                                    Value lhs, Value rhs, OpBuilder &builder,
                                    BodyBuilder bodyBuilder) {
  Value init =
      builder.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                      resultTy.getElementType());
  SmallVector<AffineMap, 3> indexingMaps(
      3, builder.getMultiDimIdentityMap(resultTy.getRank()));
  SmallVector<utils::IteratorType, 2> iteratorTypes(
      resultTy.getRank(), utils::IteratorType::parallel);

  return builder
      .create<linalg::GenericOp>(loc, resultTy, ValueRange{lhs, rhs},
                                 ValueRange{init}, indexingMaps, iteratorTypes,
                                 bodyBuilder)
      .getResult(0);
}

} // namespace detail

inline Value buildSigmoid(Location loc, RankedTensorType resultTy, Value input,
                          OpBuilder &builder) {
  return detail::buildUnaryElementwise(
      loc, resultTy, input, builder,
      [](OpBuilder &builder, Location nestedLoc, ValueRange args) {
        auto elementType = llvm::cast<FloatType>(args[0].getType());
        Value one = builder.create<arith::ConstantOp>(
            nestedLoc, elementType, builder.getFloatAttr(elementType, 1.0));
        Value neg = builder.create<arith::NegFOp>(nestedLoc, args[0]);
        Value exp = builder.create<math::ExpOp>(nestedLoc, neg);
        Value denom = builder.create<arith::AddFOp>(nestedLoc, exp, one);
        Value sigmoid = builder.create<arith::DivFOp>(nestedLoc, one, denom);
        builder.create<linalg::YieldOp>(nestedLoc, sigmoid);
      });
}

inline Value buildTanh(Location loc, RankedTensorType resultTy, Value input,
                       OpBuilder &builder) {
  return detail::buildUnaryElementwise(
      loc, resultTy, input, builder,
      [](OpBuilder &builder, Location nestedLoc, ValueRange args) {
        Value tanh = builder.create<math::TanhOp>(nestedLoc, args[0]);
        builder.create<linalg::YieldOp>(nestedLoc, tanh);
      });
}

// Keeps the loop-carried hidden tensor as the tanh destination.
inline Value buildTanh(Location loc, Value input, Value init,
                       RankedTensorType resultTy, OpBuilder &builder) {
  AffineMap valueMap = builder.getMultiDimIdentityMap(resultTy.getRank());
  SmallVector<AffineMap, 2> indexingMaps = {valueMap, valueMap};
  SmallVector<utils::IteratorType, 2> iteratorTypes(
      resultTy.getRank(), utils::IteratorType::parallel);

  return builder
      .create<linalg::GenericOp>(
          loc, resultTy, ValueRange{input}, ValueRange{init}, indexingMaps,
          iteratorTypes,
          [](OpBuilder &builder, Location nestedLoc, ValueRange args) {
            Value tanh = builder.create<math::TanhOp>(nestedLoc, args[0]);
            builder.create<linalg::YieldOp>(nestedLoc, tanh);
          })
      .getResult(0);
}

inline Value buildAdd(Location loc, RankedTensorType resultTy, Value lhs,
                      Value rhs, OpBuilder &builder) {
  return detail::buildBinaryElementwise(
      loc, resultTy, lhs, rhs, builder,
      [](OpBuilder &builder, Location nestedLoc, ValueRange args) {
        Value add = builder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
        builder.create<linalg::YieldOp>(nestedLoc, add);
      });
}

inline Value buildSub(Location loc, RankedTensorType resultTy, Value lhs,
                      Value rhs, OpBuilder &builder) {
  return detail::buildBinaryElementwise(
      loc, resultTy, lhs, rhs, builder,
      [](OpBuilder &builder, Location nestedLoc, ValueRange args) {
        Value sub = builder.create<arith::SubFOp>(nestedLoc, args[0], args[1]);
        builder.create<linalg::YieldOp>(nestedLoc, sub);
      });
}

inline Value buildMul(Location loc, RankedTensorType resultTy, Value lhs,
                      Value rhs, OpBuilder &builder) {
  return detail::buildBinaryElementwise(
      loc, resultTy, lhs, rhs, builder,
      [](OpBuilder &builder, Location nestedLoc, ValueRange args) {
        Value mul = builder.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
        builder.create<linalg::YieldOp>(nestedLoc, mul);
      });
}

} // namespace converter_recurrent_elementwise
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTELEMENTWISEUTILS_H
