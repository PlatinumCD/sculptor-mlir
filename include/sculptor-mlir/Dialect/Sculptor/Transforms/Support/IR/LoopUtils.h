#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_LOOPUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_LOOPUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace sculptor {
namespace loop_utils {

// Builds a row-major 2D index loop nest and invokes the body callback inside
// the inner loop with the row and column induction values.
template <typename BodyBuilderFn>
inline mlir::scf::ForOp build2DIndexLoopNest(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             int64_t numRows, int64_t numCols,
                                             BodyBuilderFn &&bodyBuilder) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto upperRows =
      builder.create<mlir::arith::ConstantIndexOp>(loc, numRows);
  auto upperCols =
      builder.create<mlir::arith::ConstantIndexOp>(loc, numCols);

  auto outerLoop = builder.create<mlir::scf::ForOp>(loc, zero, upperRows, one);

  mlir::OpBuilder::InsertionGuard outerGuard(builder);
  builder.setInsertionPointToStart(outerLoop.getBody());

  auto innerLoop = builder.create<mlir::scf::ForOp>(loc, zero, upperCols, one);

  mlir::OpBuilder::InsertionGuard innerGuard(builder);
  builder.setInsertionPointToStart(innerLoop.getBody());
  bodyBuilder(builder, loc, outerLoop.getInductionVar(),
              innerLoop.getInductionVar());
  return outerLoop;
}

// Builds a row-major 3D index loop nest and invokes the body callback inside
// the innermost loop with row, column, and depth induction values.
template <typename BodyBuilderFn>
inline mlir::scf::ForOp build3DIndexLoopNest(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             int64_t numRows, int64_t numCols,
                                             int64_t depth,
                                             BodyBuilderFn &&bodyBuilder) {
  auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto upperRows =
      builder.create<mlir::arith::ConstantIndexOp>(loc, numRows);
  auto upperCols =
      builder.create<mlir::arith::ConstantIndexOp>(loc, numCols);
  auto upperDepth =
      builder.create<mlir::arith::ConstantIndexOp>(loc, depth);

  auto outerLoop = builder.create<mlir::scf::ForOp>(loc, zero, upperRows, one);

  mlir::OpBuilder::InsertionGuard outerGuard(builder);
  builder.setInsertionPointToStart(outerLoop.getBody());

  auto middleLoop =
      builder.create<mlir::scf::ForOp>(loc, zero, upperCols, one);

  mlir::OpBuilder::InsertionGuard middleGuard(builder);
  builder.setInsertionPointToStart(middleLoop.getBody());

  auto innerLoop =
      builder.create<mlir::scf::ForOp>(loc, zero, upperDepth, one);

  mlir::OpBuilder::InsertionGuard innerGuard(builder);
  builder.setInsertionPointToStart(innerLoop.getBody());
  bodyBuilder(builder, loc, outerLoop.getInductionVar(),
              middleLoop.getInductionVar(), innerLoop.getInductionVar());
  return outerLoop;
}

} // namespace loop_utils
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_IR_LOOPUTILS_H
