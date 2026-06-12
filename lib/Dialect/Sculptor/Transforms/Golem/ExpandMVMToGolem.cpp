#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/ExpandMVMToGolem.h"

// ExpandMVMToGolem is the sculptor.mvm -> Golem execution IR boundary. It
// prepares fixed-size resource tiles for MVM matrix constants inside
// layer/helper functions while leaving forward alone.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace mlir {
namespace sculptor {

namespace {

constexpr llvm::StringLiteral kForwardFunctionName = "forward";
constexpr llvm::StringLiteral kSourceResourceAttr = "sculptor.source_resource";
constexpr llvm::StringLiteral kTileAttr = "sculptor.tile";
constexpr llvm::StringLiteral kTileGridAttr = "sculptor.tile_grid";
constexpr llvm::StringLiteral kVectorTileAttr = "sculptor.vector_tile";
constexpr llvm::StringLiteral kVectorTileGridAttr = "sculptor.vector_tile_grid";

namespace task_graph_names = mlir::sculptor::task_graph_names;

struct MatrixPartitionSpec {
  mlir::arith::ConstantOp constant;
  mlir::RankedTensorType type;
  std::string sourceResource;
  std::string taskPrefix;
  llvm::SmallVector<float> values;
  int64_t gridRows = 0;
  int64_t gridCols = 0;
};

static int64_t ceilDiv(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

static llvm::SmallVector<mlir::OpFoldResult>
buildIndexAttrs(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::OpFoldResult> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getIndexAttr(value));
  return attrs;
}

// Keeps the new expansion scoped to layer/helper bodies. Forward is the caller
// graph and should be rewritten by outlining/task graph passes instead.
static bool shouldProcessFunction(mlir::func::FuncOp func) {
  return func.getSymName() != kForwardFunctionName;
}

static std::string getTaskNamePrefix(mlir::Operation *op) {
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return "mvm";

  return func.getSymName().str();
}

static mlir::StringAttr buildMatrixSetupName(mlir::OpBuilder &builder,
                                             const MatrixPartitionSpec &spec,
                                             int64_t tileRow, int64_t tileCol) {
  std::string name = spec.taskPrefix;
  name += "_matrix_tile_";
  name += std::to_string(tileRow);
  name += "_";
  name += std::to_string(tileCol);
  return builder.getStringAttr(name);
}

static mlir::StringAttr buildVectorTileName(mlir::OpBuilder &builder,
                                            mlir::sculptor::MVMOp mvmOp,
                                            int64_t vectorTile) {
  std::string name = getTaskNamePrefix(mvmOp.getOperation());
  name += "_vector_tile_";
  name += std::to_string(vectorTile);
  return builder.getStringAttr(name);
}

static mlir::StringAttr buildMVMName(mlir::OpBuilder &builder,
                                     mlir::sculptor::MVMOp mvmOp, int64_t tileRow,
                                     int64_t tileCol) {
  std::string name = getTaskNamePrefix(mvmOp.getOperation());
  name += "_mvm_";
  name += std::to_string(tileRow);
  name += "_";
  name += std::to_string(tileCol);
  return builder.getStringAttr(name);
}

static mlir::StringAttr buildTileRecombineName(mlir::OpBuilder &builder,
                                               mlir::sculptor::MVMOp mvmOp) {
  std::string name = getTaskNamePrefix(mvmOp.getOperation());
  name += "_tile_recombine";
  return builder.getStringAttr(name);
}

static mlir::FailureOr<mlir::RankedTensorType>
getStaticRank2F32Tensor(mlir::Type type) {
  auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorTy || !tensorTy.hasStaticShape() || tensorTy.getRank() != 2)
    return mlir::failure();

  if (!tensorTy.getElementType().isF32())
    return mlir::failure();

  if (llvm::any_of(tensorTy.getShape(), [](int64_t dim) { return dim <= 0; }))
    return mlir::failure();

  return tensorTy;
}

static mlir::FailureOr<MatrixPartitionSpec>
buildMatrixPartitionSpec(mlir::sculptor::MVMOp mvmOp, int64_t arrayRows,
                         int64_t arrayCols) {
  mlir::Value matrix = mvmOp.getMatrix();
  auto matrixConst = matrix.getDefiningOp<mlir::arith::ConstantOp>();
  if (!matrixConst)
    return mvmOp.emitError("expected sculptor.mvm matrix operand to be an "
                           "arith.constant"),
           mlir::failure();

  auto matrixType = getStaticRank2F32Tensor(matrix.getType());
  if (failed(matrixType))
    return mvmOp.emitError(
               "expected sculptor.mvm matrix operand to be static rank-2 f32"),
           mlir::failure();

  auto denseResourceAttr = llvm::dyn_cast<mlir::DenseF32ResourceElementsAttr>(
      matrixConst.getValue());
  if (!denseResourceAttr)
    return mvmOp.emitError("expected sculptor.mvm matrix constant to use a "
                           "dense f32 resource"),
           mlir::failure();

  auto values =
      mlir::sculptor::converter_constant::getF32ConstantValues(matrixConst);
  if (failed(values))
    return mvmOp.emitError("failed to read dense f32 matrix resource"),
           mlir::failure();

  if (static_cast<int64_t>(values->size()) != (*matrixType).getNumElements())
    return mvmOp.emitError("dense f32 matrix resource element count does not "
                           "match the tensor type"),
           mlir::failure();

  auto shape = (*matrixType).getShape();
  MatrixPartitionSpec spec;
  spec.constant = matrixConst;
  spec.type = *matrixType;
  spec.sourceResource = denseResourceAttr.getRawHandle().getKey().str();
  spec.taskPrefix = getTaskNamePrefix(mvmOp.getOperation());
  spec.values = std::move(*values);
  spec.gridRows = ceilDiv(shape[0], arrayRows);
  spec.gridCols = ceilDiv(shape[1], arrayCols);
  return spec;
}

static int64_t getTileIndex(const MatrixPartitionSpec &spec, int64_t tileRow,
                            int64_t tileCol) {
  return tileRow * spec.gridCols + tileCol;
}

static bool hasMatchingTileGrid(mlir::Operation *op, int64_t gridRows,
                                int64_t gridCols) {
  auto tileGrid = op->getAttrOfType<mlir::ArrayAttr>(kTileGridAttr);
  if (!tileGrid || tileGrid.size() != 2)
    return false;

  auto rowAttr = llvm::dyn_cast<mlir::IntegerAttr>(tileGrid[0]);
  auto colAttr = llvm::dyn_cast<mlir::IntegerAttr>(tileGrid[1]);
  return rowAttr && colAttr && rowAttr.getInt() == gridRows &&
         colAttr.getInt() == gridCols;
}

static bool hasMatchingTileCoordinate(mlir::Operation *op, int64_t tileRow,
                                      int64_t tileCol) {
  auto tile = op->getAttrOfType<mlir::ArrayAttr>(kTileAttr);
  if (!tile || tile.size() != 2)
    return false;

  auto rowAttr = llvm::dyn_cast<mlir::IntegerAttr>(tile[0]);
  auto colAttr = llvm::dyn_cast<mlir::IntegerAttr>(tile[1]);
  return rowAttr && colAttr && rowAttr.getInt() == tileRow &&
         colAttr.getInt() == tileCol;
}

static bool hasMatchingSourceResource(mlir::Operation *op,
                                      llvm::StringRef sourceResource) {
  auto sourceAttr = op->getAttrOfType<mlir::StringAttr>(kSourceResourceAttr);
  return sourceAttr && sourceAttr.getValue() == sourceResource;
}

static bool hasMatchingMatrixTileAttrs(mlir::Operation *op,
                                       llvm::StringRef sourceResource,
                                       int64_t tileRow, int64_t tileCol,
                                       int64_t gridRows, int64_t gridCols) {
  return hasMatchingSourceResource(op, sourceResource) &&
         hasMatchingTileCoordinate(op, tileRow, tileCol) &&
         hasMatchingTileGrid(op, gridRows, gridCols);
}

static mlir::Value findExistingLogicalArray(mlir::func::FuncOp func,
                                            llvm::StringRef sourceResource,
                                            int64_t tileRow, int64_t tileCol,
                                            int64_t gridRows,
                                            int64_t gridCols) {
  mlir::Value found;
  func.walk([&](mlir::sculptor::TaskRegionOp region) {
    if (found)
      return;
    if (region.getKind() != task_graph_names::kMatrixSetupTaskKind ||
        region.getNumResults() != 1)
      return;
    if (!llvm::isa<mlir::sculptor::LogicalArrayType>(
            region.getResult(0).getType()))
      return;

    if (hasMatchingMatrixTileAttrs(region.getOperation(), sourceResource,
                                   tileRow, tileCol, gridRows, gridCols))
      found = region.getResult(0);
  });
  if (found)
    return found;

  func.walk([&](mlir::sculptor::ArraySetOp arraySet) {
    if (found)
      return;
    auto sourceAttr =
        arraySet->getAttrOfType<mlir::StringAttr>(kSourceResourceAttr);
    if (!sourceAttr)
      return;

    if (sourceAttr.getValue() == sourceResource &&
        hasMatchingTileCoordinate(arraySet.getOperation(), tileRow, tileCol) &&
        hasMatchingTileGrid(arraySet.getOperation(), gridRows, gridCols)) {
      found = arraySet.getArray();
    }
  });

  return found;
}

static llvm::SmallVector<float>
buildZeroPaddedTileValues(const MatrixPartitionSpec &spec, int64_t tileRow,
                          int64_t tileCol, int64_t arrayRows,
                          int64_t arrayCols) {
  llvm::SmallVector<float> tileValues(arrayRows * arrayCols, 0.0f);
  auto matrixShape = spec.type.getShape();
  int64_t matrixRows = matrixShape[0];
  int64_t matrixCols = matrixShape[1];

  for (int64_t r = 0; r < arrayRows; ++r) {
    int64_t sourceRow = tileRow * arrayRows + r;
    if (sourceRow >= matrixRows)
      continue;

    for (int64_t c = 0; c < arrayCols; ++c) {
      int64_t sourceCol = tileCol * arrayCols + c;
      if (sourceCol >= matrixCols)
        continue;

      tileValues[r * arrayCols + c] =
          spec.values[sourceRow * matrixCols + sourceCol];
    }
  }

  return tileValues;
}

static std::string buildTileResourceName(llvm::StringRef sourceResource,
                                         int64_t tileRow, int64_t tileCol) {
  return (llvm::Twine(sourceResource) + "__tile_" + llvm::Twine(tileRow) + "_" +
          llvm::Twine(tileCol))
      .str();
}

static mlir::arith::ConstantOp
createTileConstant(const MatrixPartitionSpec &spec, int64_t tileRow,
                   int64_t tileCol, int64_t arrayRows, int64_t arrayCols,
                   mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType tileType = mlir::RankedTensorType::get(
      {arrayRows, arrayCols}, spec.type.getElementType());
  llvm::SmallVector<float> tileValues =
      buildZeroPaddedTileValues(spec, tileRow, tileCol, arrayRows, arrayCols);

  std::string tileResourceName =
      buildTileResourceName(spec.sourceResource, tileRow, tileCol);
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
      llvm::ArrayRef<float>(tileValues), /*dataIsMutable=*/false);
  auto tileAttr =
      llvm::cast<mlir::TypedAttr>(mlir::DenseF32ResourceElementsAttr::get(
          tileType, tileResourceName, std::move(blob)));

  auto tileConstant = rewriter.create<mlir::arith::ConstantOp>(
      spec.constant->getLoc(), tileType, tileAttr);
  return tileConstant;
}

static void attachMatrixTileAttrs(mlir::Operation *op,
                                  const MatrixPartitionSpec &spec,
                                  int64_t tileRow, int64_t tileCol,
                                  mlir::RewriterBase &rewriter) {
  op->setAttr(kSourceResourceAttr, rewriter.getStringAttr(spec.sourceResource));
  op->setAttr(kTileAttr, rewriter.getI64ArrayAttr({tileRow, tileCol}));
  op->setAttr(kTileGridAttr,
              rewriter.getI64ArrayAttr({spec.gridRows, spec.gridCols}));
}

static mlir::sculptor::ArraySetOp
createArraySet(mlir::arith::ConstantOp tileConstant,
               mlir::RewriterBase &rewriter) {
  auto logicalArrayType =
      mlir::sculptor::LogicalArrayType::get(rewriter.getContext());
  auto arraySet = rewriter.create<mlir::sculptor::ArraySetOp>(
      tileConstant.getLoc(), logicalArrayType, tileConstant.getResult());
  return arraySet;
}

static mlir::Value createMatrixSetupRegion(MatrixPartitionSpec &spec,
                                           int64_t tileRow, int64_t tileCol,
                                           int64_t arrayRows, int64_t arrayCols,
                                           mlir::RewriterBase &rewriter) {
  auto logicalArrayType =
      mlir::sculptor::LogicalArrayType::get(rewriter.getContext());
  mlir::Location loc = spec.constant.getOperation()->getLoc();
  auto setupRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{logicalArrayType}, mlir::ValueRange{},
      task_graph_names::kMatrixSetupTaskKind,
      buildMatrixSetupName(rewriter, spec, tileRow, tileCol));
  attachMatrixTileAttrs(setupRegion.getOperation(), spec, tileRow, tileCol,
                        rewriter);

  mlir::Block *body = new mlir::Block();
  setupRegion.getBody().push_back(body);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);

  mlir::arith::ConstantOp tileConstant = createTileConstant(
      spec, tileRow, tileCol, arrayRows, arrayCols, rewriter);
  mlir::sculptor::ArraySetOp arraySet = createArraySet(tileConstant, rewriter);
  rewriter.create<mlir::sculptor::YieldOp>(loc, arraySet.getArray());
  return setupRegion.getResult(0);
}

static mlir::FailureOr<mlir::RankedTensorType>
getMVMVectorType(mlir::sculptor::MVMOp mvmOp,
                 const MatrixPartitionSpec &matrixSpec) {
  auto vectorType = getStaticRank2F32Tensor(mvmOp.getVector().getType());
  if (failed(vectorType))
    return mvmOp.emitError(
               "expected sculptor.mvm vector operand to be static rank-2 f32"),
           mlir::failure();

  if ((*vectorType).getDimSize(0) != 1)
    return mvmOp.emitError("expected sculptor.mvm vector operand to have "
                           "leading dimension 1"),
           mlir::failure();

  if ((*vectorType).getDimSize(1) != matrixSpec.type.getDimSize(1))
    return mvmOp.emitError("expected sculptor.mvm vector width to match matrix "
                           "input dimension"),
           mlir::failure();

  return *vectorType;
}

static void attachVectorTileAttrs(mlir::Operation *op, int64_t vectorTile,
                                  int64_t vectorTileGrid,
                                  mlir::RewriterBase &rewriter) {
  op->setAttr(kVectorTileAttr, rewriter.getI64IntegerAttr(vectorTile));
  op->setAttr(kVectorTileGridAttr, rewriter.getI64IntegerAttr(vectorTileGrid));
}

static mlir::sculptor::TaskRegionOp createDigitalPreparationRegion(
    mlir::sculptor::MVMOp mvmOp, mlir::RankedTensorType resultType,
    int64_t vectorTile, int64_t vectorTileGrid, mlir::RewriterBase &rewriter) {
  auto prepRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      mvmOp.getLoc(), mlir::TypeRange{resultType},
      mlir::ValueRange{mvmOp.getVector()}, "digital.vector_tile",
      buildVectorTileName(rewriter, mvmOp, vectorTile));
  attachVectorTileAttrs(prepRegion.getOperation(), vectorTile, vectorTileGrid,
                        rewriter);

  mlir::Block *body = new mlir::Block();
  prepRegion.getBody().push_back(body);
  body->addArgument(mvmOp.getVector().getType(), mvmOp.getVector().getLoc());
  return prepRegion;
}

static mlir::Value createFullVectorTile(mlir::sculptor::MVMOp mvmOp,
                                        int64_t vectorTile,
                                        int64_t vectorTileGrid,
                                        int64_t arrayCols,
                                        mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType tileType =
      mlir::RankedTensorType::get({1, arrayCols}, rewriter.getF32Type());
  auto prepRegion = createDigitalPreparationRegion(mvmOp, tileType, vectorTile,
                                                   vectorTileGrid, rewriter);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&prepRegion.getBody().front());
  mlir::Value vectorArg = prepRegion.getBody().front().getArgument(0);
  auto slice = rewriter.create<mlir::tensor::ExtractSliceOp>(
      mvmOp.getLoc(), tileType, vectorArg,
      buildIndexAttrs(rewriter, {0, vectorTile * arrayCols}),
      buildIndexAttrs(rewriter, {1, arrayCols}),
      buildIndexAttrs(rewriter, {1, 1}));
  rewriter.create<mlir::sculptor::YieldOp>(mvmOp.getLoc(), slice.getResult());
  return prepRegion.getResult(0);
}

static mlir::Value
createPaddedVectorTile(mlir::sculptor::MVMOp mvmOp, int64_t vectorTile,
                       int64_t vectorTileGrid, int64_t remainingCols,
                       int64_t arrayCols, mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType fullTileType =
      mlir::RankedTensorType::get({1, arrayCols}, rewriter.getF32Type());
  mlir::RankedTensorType sourceTileType =
      mlir::RankedTensorType::get({1, remainingCols}, rewriter.getF32Type());
  auto prepRegion = createDigitalPreparationRegion(
      mvmOp, fullTileType, vectorTile, vectorTileGrid, rewriter);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&prepRegion.getBody().front());
  mlir::Value vectorArg = prepRegion.getBody().front().getArgument(0);
  auto zeroAttr =
      llvm::cast<mlir::TypedAttr>(rewriter.getZeroAttr(fullTileType));
  auto zeroTile = rewriter.create<mlir::arith::ConstantOp>(
      mvmOp.getLoc(), fullTileType, zeroAttr);
  auto sourceSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
      mvmOp.getLoc(), sourceTileType, vectorArg,
      buildIndexAttrs(rewriter, {0, vectorTile * arrayCols}),
      buildIndexAttrs(rewriter, {1, remainingCols}),
      buildIndexAttrs(rewriter, {1, 1}));
  auto paddedTile = rewriter.create<mlir::tensor::InsertSliceOp>(
      mvmOp.getLoc(), sourceSlice.getResult(), zeroTile.getResult(),
      buildIndexAttrs(rewriter, {0, 0}),
      buildIndexAttrs(rewriter, {1, remainingCols}),
      buildIndexAttrs(rewriter, {1, 1}));
  rewriter.create<mlir::sculptor::YieldOp>(mvmOp.getLoc(),
                                         paddedTile.getResult());
  return prepRegion.getResult(0);
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
createVectorTiles(mlir::sculptor::MVMOp mvmOp, const MatrixPartitionSpec &spec,
                  int64_t arrayCols, mlir::RewriterBase &rewriter) {
  auto vectorType = getMVMVectorType(mvmOp, spec);
  if (failed(vectorType))
    return mlir::failure();

  llvm::SmallVector<mlir::Value> vectorTiles;
  vectorTiles.reserve(spec.gridCols);
  int64_t vectorWidth = (*vectorType).getDimSize(1);
  for (int64_t vectorTile = 0; vectorTile < spec.gridCols; ++vectorTile) {
    int64_t colOffset = vectorTile * arrayCols;
    int64_t remainingCols = std::min(arrayCols, vectorWidth - colOffset);
    if (remainingCols == arrayCols) {
      vectorTiles.push_back(createFullVectorTile(
          mvmOp, vectorTile, spec.gridCols, arrayCols, rewriter));
      continue;
    }

    vectorTiles.push_back(createPaddedVectorTile(
        mvmOp, vectorTile, spec.gridCols, remainingCols, arrayCols, rewriter));
  }

  return vectorTiles;
}

static void attachArrayExecutionAttrs(mlir::Operation *op,
                                      const MatrixPartitionSpec &spec,
                                      int64_t tileRow, int64_t tileCol,
                                      int64_t vectorTile,
                                      mlir::RewriterBase &rewriter) {
  attachMatrixTileAttrs(op, spec, tileRow, tileCol, rewriter);
  attachVectorTileAttrs(op, vectorTile, spec.gridCols, rewriter);
}

static mlir::Value createArrayExecutionRegion(
    mlir::sculptor::MVMOp mvmOp, const MatrixPartitionSpec &spec,
    mlir::Value vectorTileValue, mlir::Value logicalArray, int64_t tileRow,
    int64_t vectorTile, int64_t arrayRows, mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType storeType =
      mlir::RankedTensorType::get({1, arrayRows}, rewriter.getF32Type());

  llvm::SmallVector<mlir::Value> inputs = {vectorTileValue, logicalArray};
  auto executionRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      mvmOp.getLoc(), mlir::TypeRange{storeType}, mlir::ValueRange(inputs),
      task_graph_names::kMVMTaskKind,
      buildMVMName(rewriter, mvmOp, tileRow, vectorTile));
  attachArrayExecutionAttrs(executionRegion.getOperation(), spec, tileRow,
                            vectorTile, vectorTile, rewriter);

  mlir::Block *body = new mlir::Block();
  executionRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {vectorTileValue.getType(),
                                              logicalArray.getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {vectorTileValue.getLoc(),
                                                 logicalArray.getLoc()};
  body->addArguments(inputTypes, inputLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);

  mlir::Value vectorArg = body->getArgument(0);
  mlir::Value arrayArg = body->getArgument(1);
  rewriter.create<mlir::sculptor::ArrayLoadOp>(mvmOp.getLoc(), vectorArg,
                                             arrayArg);

  auto resultType = mlir::sculptor::ArrayResultType::get(rewriter.getContext());
  auto arrayExecute = rewriter.create<mlir::sculptor::ArrayExecuteOp>(
      mvmOp.getLoc(), resultType, arrayArg);

  auto arrayStore = rewriter.create<mlir::sculptor::ArrayStoreOp>(
      mvmOp.getLoc(), storeType, arrayExecute.getResult());

  rewriter.create<mlir::sculptor::YieldOp>(mvmOp.getLoc(),
                                         arrayStore.getOutput());
  return executionRegion.getResult(0);
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
getOrCreateLogicalArrays(mlir::func::FuncOp func, MatrixPartitionSpec &spec,
                         int64_t arrayRows, int64_t arrayCols,
                         mlir::RewriterBase &rewriter) {
  llvm::SmallVector<mlir::Value> logicalArrays(spec.gridRows * spec.gridCols);
  mlir::Operation *insertAfter = spec.constant.getOperation();

  for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
    for (int64_t tileCol = 0; tileCol < spec.gridCols; ++tileCol) {
      int64_t index = getTileIndex(spec, tileRow, tileCol);
      mlir::Value existing =
          findExistingLogicalArray(func, spec.sourceResource, tileRow, tileCol,
                                   spec.gridRows, spec.gridCols);
      if (existing) {
        logicalArrays[index] = existing;
        insertAfter = existing.getDefiningOp();
        continue;
      }

      rewriter.setInsertionPointAfter(insertAfter);
      mlir::Value setupResult = createMatrixSetupRegion(
          spec, tileRow, tileCol, arrayRows, arrayCols, rewriter);
      logicalArrays[index] = setupResult;
      insertAfter = setupResult.getDefiningOp();
    }
  }

  return logicalArrays;
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
createArrayLoads(mlir::sculptor::MVMOp mvmOp, const MatrixPartitionSpec &spec,
                 llvm::ArrayRef<mlir::Value> logicalArrays,
                 llvm::ArrayRef<mlir::Value> vectorTiles, int64_t arrayRows,
                 mlir::RewriterBase &rewriter) {
  if (static_cast<int64_t>(logicalArrays.size()) !=
          spec.gridRows * spec.gridCols ||
      static_cast<int64_t>(vectorTiles.size()) != spec.gridCols)
    return mvmOp.emitError("internal error: mismatched logical array or vector "
                           "tile count for MVM expansion");

  llvm::SmallVector<mlir::Value> partialTiles(spec.gridRows * spec.gridCols);
  for (int64_t vectorTile = 0; vectorTile < spec.gridCols; ++vectorTile) {
    for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
      int64_t index = getTileIndex(spec, tileRow, vectorTile);
      partialTiles[index] = createArrayExecutionRegion(
          mvmOp, spec, vectorTiles[vectorTile], logicalArrays[index], tileRow,
          vectorTile, arrayRows, rewriter);
    }
  }

  return partialTiles;
}

static mlir::FailureOr<mlir::RankedTensorType>
getStaticMVMResultType(mlir::sculptor::MVMOp mvmOp,
                       const MatrixPartitionSpec &spec) {
  auto resultType = getStaticRank2F32Tensor(mvmOp.getResult().getType());
  if (failed(resultType))
    return mvmOp.emitError(
               "expected sculptor.mvm result to be static rank-2 f32"),
           mlir::failure();

  if ((*resultType).getDimSize(0) != 1)
    return mvmOp.emitError("expected sculptor.mvm result to have leading "
                           "dimension 1"),
           mlir::failure();

  if ((*resultType).getDimSize(1) != spec.type.getDimSize(0))
    return mvmOp.emitError("expected sculptor.mvm result width to match matrix "
                           "output dimension"),
           mlir::failure();

  return *resultType;
}

static mlir::Value createEmptyTensor(mlir::Location loc,
                                     mlir::RankedTensorType type,
                                     mlir::RewriterBase &rewriter) {
  return rewriter
      .create<mlir::tensor::EmptyOp>(loc, type.getShape(),
                                     type.getElementType())
      .getResult();
}

static mlir::Value sumRowPartials(mlir::sculptor::MVMOp mvmOp,
                                  const MatrixPartitionSpec &spec,
                                  llvm::ArrayRef<mlir::Value> partialTiles,
                                  int64_t tileRow,
                                  mlir::RankedTensorType rowTileType,
                                  mlir::RewriterBase &rewriter) {
  mlir::Value row = partialTiles[getTileIndex(spec, tileRow, 0)];
  for (int64_t tileCol = 1; tileCol < spec.gridCols; ++tileCol) {
    mlir::Value rhs = partialTiles[getTileIndex(spec, tileRow, tileCol)];
    mlir::Value init = createEmptyTensor(mvmOp.getLoc(), rowTileType, rewriter);
    row = rewriter
              .create<mlir::linalg::AddOp>(mvmOp.getLoc(),
                                           mlir::ValueRange{row, rhs},
                                           mlir::ValueRange{init})
              .getResult(0);
  }

  return row;
}

static mlir::FailureOr<mlir::Value>
createRecombinedMVMResult(mlir::sculptor::MVMOp mvmOp,
                          const MatrixPartitionSpec &spec,
                          llvm::ArrayRef<mlir::Value> partialTiles,
                          int64_t arrayRows, mlir::RewriterBase &rewriter) {
  if (static_cast<int64_t>(partialTiles.size()) !=
      spec.gridRows * spec.gridCols)
    return mvmOp.emitError("internal error: mismatched partial tile count for "
                           "MVM recombine"),
           mlir::failure();

  auto resultType = getStaticMVMResultType(mvmOp, spec);
  if (failed(resultType))
    return mlir::failure();

  int64_t wideWidth = spec.gridRows * arrayRows;
  int64_t resultWidth = (*resultType).getDimSize(1);
  if (resultWidth > wideWidth)
    return mvmOp.emitError("internal error: recombined tensor is narrower than "
                           "the original MVM result"),
           mlir::failure();

  auto recombineRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      mvmOp.getLoc(), mlir::TypeRange{*resultType},
      mlir::ValueRange(partialTiles), "digital.tile_recombine",
      buildTileRecombineName(rewriter, mvmOp));

  mlir::Block *body = new mlir::Block();
  recombineRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> partialTypes;
  llvm::SmallVector<mlir::Location> partialLocs;
  partialTypes.reserve(partialTiles.size());
  partialLocs.reserve(partialTiles.size());
  for (mlir::Value partialTile : partialTiles) {
    partialTypes.push_back(partialTile.getType());
    partialLocs.push_back(partialTile.getLoc());
  }
  body->addArguments(partialTypes, partialLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);

  llvm::SmallVector<mlir::Value> regionPartialTiles;
  regionPartialTiles.reserve(body->getNumArguments());
  for (mlir::BlockArgument arg : body->getArguments())
    regionPartialTiles.push_back(arg);

  mlir::RankedTensorType rowTileType =
      mlir::RankedTensorType::get({1, arrayRows}, rewriter.getF32Type());
  llvm::SmallVector<mlir::Value> rowResults;
  rowResults.reserve(spec.gridRows);
  for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
    rowResults.push_back(sumRowPartials(mvmOp, spec, regionPartialTiles,
                                        tileRow, rowTileType, rewriter));
  }

  mlir::Value wideResult = rowResults.front();
  if (spec.gridRows > 1) {
    mlir::RankedTensorType wideType =
        mlir::RankedTensorType::get({1, wideWidth}, rewriter.getF32Type());
    wideResult =
        rewriter
            .create<mlir::tensor::ConcatOp>(mvmOp.getLoc(), wideType, /*dim=*/1,
                                            mlir::ValueRange(rowResults))
            .getResult();
  }

  mlir::Value recombined = wideResult;
  if (resultWidth != wideWidth) {
    auto slice = rewriter.create<mlir::tensor::ExtractSliceOp>(
        mvmOp.getLoc(), *resultType, wideResult,
        buildIndexAttrs(rewriter, {0, 0}),
        buildIndexAttrs(rewriter, {1, resultWidth}),
        buildIndexAttrs(rewriter, {1, 1}));
    recombined = slice.getResult();
  }

  rewriter.create<mlir::sculptor::YieldOp>(mvmOp.getLoc(), recombined);
  return recombineRegion.getResult(0);
}

struct MVMExpansionWalker {
  MVMExpansionWalker(int64_t arrayRows, int64_t arrayCols)
      : arrayRows(arrayRows), arrayCols(arrayCols) {}

  mlir::LogicalResult run(mlir::ModuleOp module) {
    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      if (!shouldProcessFunction(func))
        continue;

      if (failed(walkFunction(func)))
        return mlir::failure();
    }

    return mlir::success();
  }

private:
  int64_t arrayRows;
  int64_t arrayCols;

  mlir::LogicalResult walkFunction(mlir::func::FuncOp func) {
    llvm::SmallVector<mlir::sculptor::MVMOp> mvmOps;
    func.walk([&](mlir::sculptor::MVMOp mvmOp) { mvmOps.push_back(mvmOp); });

    for (mlir::sculptor::MVMOp mvmOp : mvmOps) {
      if (!mvmOp || !mvmOp->getBlock())
        continue;

      if (failed(handleMVMOp(func, mvmOp)))
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult handleMVMOp(mlir::func::FuncOp func,
                                  mlir::sculptor::MVMOp op) {
    auto spec = buildMatrixPartitionSpec(op, arrayRows, arrayCols);
    if (failed(spec))
      return mlir::failure();

    mlir::IRRewriter rewriter(op.getContext());
    auto logicalArrays =
        getOrCreateLogicalArrays(func, *spec, arrayRows, arrayCols, rewriter);
    if (failed(logicalArrays))
      return mlir::failure();

    rewriter.setInsertionPoint(op);
    auto vectorTiles = createVectorTiles(op, *spec, arrayCols, rewriter);
    if (failed(vectorTiles))
      return mlir::failure();

    auto partialTiles = createArrayLoads(op, *spec, *logicalArrays,
                                         *vectorTiles, arrayRows, rewriter);
    if (failed(partialTiles))
      return mlir::failure();

    auto recombined = createRecombinedMVMResult(op, *spec, *partialTiles,
                                                arrayRows, rewriter);
    if (failed(recombined))
      return mlir::failure();

    op.getResult().replaceAllUsesWith(*recombined);
    rewriter.eraseOp(op);
    if (spec->constant->use_empty())
      rewriter.eraseOp(spec->constant);
    return mlir::success();
  }
};

} // namespace

void ExpandMVMToGolemPass::runOnOperation() {
  if (arrayRows <= 0 || arrayCols <= 0) {
    getOperation().emitError(
        "expected positive array-rows and array-cols for MVM expansion");
    signalPassFailure();
    return;
  }

  MVMExpansionWalker walker(arrayRows, arrayCols);
  if (failed(walker.run(getOperation())))
    signalPassFailure();
}

void registerExpandMVMToGolemPass() {
  mlir::PassRegistration<ExpandMVMToGolemPass>();
}

} // namespace sculptor
} // namespace mlir
