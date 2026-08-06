#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/ExpandMVMToGolem.h"

// ExpandMVMToGolem materializes fixed-size logical arrays and inline Golem
// execution operations. Mapping consumes this physical realization rather
// than the logical sculptor.mvm operations that preceded it.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/SemanticOperationScope.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/SemanticOperationNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/GolemTilingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/GolemMVMPlanning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace sculptor {

namespace {

namespace semantic_operation_names = mlir::sculptor::semantic_operation_names;
namespace tile_runtime_attrs = mlir::sculptor::tile_runtime_attrs;
namespace golem_tiling_attrs = mlir::sculptor::golem_tiling_attrs;
namespace mapping = mlir::sculptor::mapping;

struct MatrixPartitionSpec {
  mlir::arith::ConstantOp constant;
  mlir::RankedTensorType type;
  std::string sourceResource;
  std::string taskPrefix;
  llvm::SmallVector<float> values;
  int64_t gridRows = 0;
  int64_t gridCols = 0;
  mapping::GolemMVMPlan physicalPlan;
  mlir::IntegerAttr mappingOperationId;
  mlir::IntegerAttr raLeafId;
};

struct MatrixOperand {
  mlir::arith::ConstantOp constant;
  mlir::RankedTensorType type;
  mlir::DenseF32ResourceElementsAttr resource;
};

struct MatrixTileExtent {
  int64_t physicalRows = 0;
  int64_t physicalCols = 0;
  int64_t validRows = 0;
  int64_t validCols = 0;
};

struct LogicalArrayGrid {
  int64_t gridRows = 0;
  int64_t gridCols = 0;
  llvm::SmallVector<mlir::Value> values;
};

struct FunctionExpansionState {
  llvm::StringMap<MatrixPartitionSpec> matrixSpecs;
  llvm::StringMap<LogicalArrayGrid> logicalArrays;
  llvm::SmallVector<mlir::arith::ConstantOp> matrixConstants;
};

struct MVMSequenceMatch {
  mlir::Operation *anchor = nullptr;
  llvm::SmallVector<mlir::Operation *> members;
  mlir::sculptor::MVMOp mvm;
  mlir::Value vectors;
  mlir::RankedTensorType vectorSequenceType;
  mlir::RankedTensorType resultSequenceType;
  int64_t sequenceLength = 0;

  mlir::Location getLoc() const { return anchor->getLoc(); }

  mlir::InFlightDiagnostic emitError(llvm::StringRef message) const {
    return anchor->emitError(message);
  }

  mlir::Operation *getInsertionAnchor() const {
    return members.empty() ? anchor : members.front();
  }
};

static MatrixTileExtent getMatrixTileExtent(const MatrixPartitionSpec &spec,
                                            int64_t tileRow, int64_t tileCol,
                                            int64_t arrayRows,
                                            int64_t arrayCols) {
  (void)arrayRows;
  (void)arrayCols;
  const mapping::GolemMVMTile &tile =
      spec.physicalPlan.tiles[tileRow * spec.gridCols + tileCol];
  return MatrixTileExtent{tile.physicalRows, tile.physicalColumns,
                          tile.validRows, tile.validColumns};
}

static llvm::SmallVector<mlir::OpFoldResult>
buildIndexAttrs(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::OpFoldResult> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getIndexAttr(value));
  return attrs;
}

static llvm::SmallVector<mlir::OpFoldResult>
buildRowOffsets(mlir::OpBuilder &builder, mlir::Value row,
                int64_t columnOffset) {
  return {row, builder.getIndexAttr(columnOffset)};
}

static std::string getTaskNamePrefix(mlir::Operation *op) {
  if (auto semanticName =
          op->getAttrOfType<mlir::StringAttr>("sculptor.semantic.name"))
    return semanticName.getValue().str();

  auto func = op->getParentOfType<mlir::func::FuncOp>();
  if (!func)
    return "mvm";

  return func.getSymName().str();
}

static void copyMappingIdentity(mlir::Operation *source,
                                SemanticOperationScope &scope) {
  scope.copyIfPresent(source, mapping::kMappingOperationIdAttrName);
  scope.copyIfPresent(source, mapping::kRALeafIdAttrName);
  for (mlir::NamedAttribute attribute : source->getAttrs()) {
    llvm::StringRef name = attribute.getName().strref();
    if (name.starts_with("sculptor.semantic.") &&
        name != "sculptor.semantic.name" &&
        name != "sculptor.semantic.section")
      scope.set(attribute.getName().strref(), attribute.getValue());
  }
}

static llvm::StringRef getStageKind(llvm::StringRef kind) {
  if (kind == semantic_operation_names::kMatrixSetupTaskKind)
    return mapping::kMatrixSetupStageKind;
  if (kind == "digital.vector_tile")
    return mapping::kVectorTileStageKind;
  if (kind == semantic_operation_names::kMVMTaskKind ||
      kind == semantic_operation_names::kConvTileMVMTaskKind)
    return mapping::kPhysicalMVMStageKind;
  if (kind == semantic_operation_names::kTileRecombineTaskKind ||
      kind == "digital.tile_recombine")
    return mapping::kTileRecombineStageKind;
  return mapping::kDigitalStageKind;
}

static void assignStageMetadata(mlir::func::FuncOp func) {
  mlir::Builder builder(func.getContext());
  mlir::StringAttr previousSection;
  mlir::StringAttr previousName;
  int64_t stageId = -1;

  for (mlir::Operation &operation : func.front().without_terminator()) {
    auto section = operation.getAttrOfType<mlir::StringAttr>(
        "sculptor.semantic.section");
    if (!section)
      continue;
    auto name = operation.getAttrOfType<mlir::StringAttr>(
        "sculptor.semantic.name");
    if (!name)
      name = section;

    if (section != previousSection || name != previousName) {
      ++stageId;
      previousSection = section;
      previousName = name;
    }

    operation.setAttr(mapping::kStageIdAttrName,
                      builder.getI64IntegerAttr(stageId));
    operation.setAttr(mapping::kStageKindAttrName,
                      builder.getStringAttr(getStageKind(section.getValue())));
    operation.setAttr(mapping::kStageNameAttrName, name);
  }
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
                                     mlir::sculptor::MVMOp mvmOp,
                                     int64_t tileRow, int64_t tileCol) {
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

static mlir::FailureOr<MatrixOperand>
matchMatrixOperand(mlir::sculptor::MVMOp mvmOp) {
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

  return MatrixOperand{matrixConst, *matrixType, denseResourceAttr};
}

static bool belongsToSemanticGroup(mlir::Operation *operation,
                                   mlir::StringAttr section,
                                   mlir::StringAttr name) {
  return operation->getAttrOfType<mlir::StringAttr>(
             "sculptor.semantic.section") == section &&
         operation->getAttrOfType<mlir::StringAttr>(
             "sculptor.semantic.name") == name;
}

static mlir::FailureOr<MVMSequenceMatch>
matchMVMSequence(mlir::scf::ForOp loop) {
  auto section = loop->getAttrOfType<mlir::StringAttr>(
      "sculptor.semantic.section");
  if (!section || section.getValue() != semantic_operation_names::kMVMSequenceTaskKind)
    return mlir::failure();

  if (loop.getNumResults() != 1)
    return loop.emitOpError("expected MVM sequence loop to have one result"),
           mlir::failure();

  auto resultSequenceType =
      getStaticRank2F32Tensor(loop.getResult(0).getType());
  if (failed(resultSequenceType))
    return loop.emitOpError(
               "expected MVM sequence result to be a static rank-2 f32 tensor"),
           mlir::failure();

  llvm::SmallVector<mlir::sculptor::MVMOp> mvmOps;
  loop.walk([&](mlir::sculptor::MVMOp mvmOp) { mvmOps.push_back(mvmOp); });
  if (mvmOps.size() != 1)
    return loop.emitOpError(
               "expected MVM sequence loop to contain exactly one sculptor.mvm"),
           mlir::failure();

  mlir::sculptor::MVMOp mvmOp = mvmOps.front();
  auto vectorSlice =
      mvmOp.getVector().getDefiningOp<mlir::tensor::ExtractSliceOp>();
  if (!vectorSlice)
    return mvmOp.emitOpError(
               "expected MVM sequence vector to come from tensor.extract_slice"),
           mlir::failure();

  mlir::Value vectors = vectorSlice.getSource();
  auto vectorSequenceType = getStaticRank2F32Tensor(vectors.getType());
  auto vectorType = getStaticRank2F32Tensor(mvmOp.getVector().getType());
  auto resultType = getStaticRank2F32Tensor(mvmOp.getResult().getType());
  if (failed(vectorSequenceType) || failed(vectorType) || failed(resultType))
    return mvmOp.emitOpError(
               "expected static rank-2 f32 MVM sequence tensor types"),
           mlir::failure();

  int64_t sequenceLength = (*vectorSequenceType).getDimSize(0);
  if ((*resultSequenceType).getDimSize(0) != sequenceLength ||
      (*vectorType).getDimSize(0) != 1 || (*resultType).getDimSize(0) != 1 ||
      (*vectorType).getDimSize(1) != (*vectorSequenceType).getDimSize(1) ||
      (*resultType).getDimSize(1) != (*resultSequenceType).getDimSize(1)) {
    return mvmOp.emitOpError(
               "expected loop-carried MVM row types to match the sequence "
               "tensor widths"),
           mlir::failure();
  }

  auto name = loop->getAttrOfType<mlir::StringAttr>("sculptor.semantic.name");
  if (!name)
    name = section;
  llvm::SmallVector<mlir::Operation *> members;
  mlir::Operation *first = loop.getOperation();
  while (mlir::Operation *previous = first->getPrevNode()) {
    if (!belongsToSemanticGroup(previous, section, name))
      break;
    first = previous;
  }
  for (mlir::Operation *operation = first;
       operation && belongsToSemanticGroup(operation, section, name);
       operation = operation->getNextNode())
    members.push_back(operation);

  return MVMSequenceMatch{loop.getOperation(),
                          std::move(members),
                          mvmOp,
                          vectors,
                          *vectorSequenceType,
                          *resultSequenceType,
                          sequenceLength};
}

static mlir::FailureOr<MatrixPartitionSpec>
buildMatrixPartitionSpec(mlir::sculptor::MVMOp mvmOp,
                         const MatrixOperand &matrixOperand, int64_t arrayRows,
                         int64_t arrayCols) {
  auto matrixConst = matrixOperand.constant;

  auto values =
      mlir::sculptor::converter_constant::getF32ConstantValues(matrixConst);
  if (failed(values))
    return mvmOp.emitError("failed to read dense f32 matrix resource"),
           mlir::failure();

  if (static_cast<int64_t>(values->size()) !=
      matrixOperand.type.getNumElements())
    return mvmOp.emitError("dense f32 matrix resource element count does not "
                           "match the tensor type"),
           mlir::failure();

  auto shape = matrixOperand.type.getShape();
  MatrixPartitionSpec spec;
  spec.constant = matrixConst;
  spec.type = matrixOperand.type;
  spec.sourceResource = matrixOperand.resource.getRawHandle().getKey().str();
  spec.taskPrefix = getTaskNamePrefix(mvmOp.getOperation());
  spec.values = std::move(*values);
  auto physicalPlan =
      mapping::planGolemMVM(mvmOp, shape[0], shape[1], arrayRows, arrayCols);
  if (failed(physicalPlan))
    return mlir::failure();
  spec.gridRows = physicalPlan->gridRows;
  spec.gridCols = physicalPlan->gridColumns;
  spec.physicalPlan = std::move(*physicalPlan);
  spec.mappingOperationId = mvmOp->getAttrOfType<mlir::IntegerAttr>(
      mapping::kMappingOperationIdAttrName);
  spec.raLeafId =
      mvmOp->getAttrOfType<mlir::IntegerAttr>(mapping::kRALeafIdAttrName);
  return spec;
}

static int64_t getTileIndex(const MatrixPartitionSpec &spec, int64_t tileRow,
                            int64_t tileCol) {
  return tileRow * spec.gridCols + tileCol;
}

static std::optional<std::pair<int64_t, int64_t>>
getI64PairAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto values = op->getAttrOfType<mlir::ArrayAttr>(attrName);
  if (!values || values.size() != 2)
    return std::nullopt;

  auto first = llvm::dyn_cast<mlir::IntegerAttr>(values[0]);
  auto second = llvm::dyn_cast<mlir::IntegerAttr>(values[1]);
  if (!first || !second)
    return std::nullopt;
  return std::make_pair(first.getInt(), second.getInt());
}

static mlir::LogicalResult
recordLogicalArray(mlir::Operation *op, mlir::Value logicalArray,
                   llvm::StringMap<LogicalArrayGrid> &logicalArrays) {
  auto sourceAttr = op->getAttrOfType<mlir::StringAttr>(
      golem_tiling_attrs::kSourceResourceAttrName);
  auto tile = getI64PairAttr(op, golem_tiling_attrs::kTileAttrName);
  auto grid = getI64PairAttr(op, golem_tiling_attrs::kTileGridAttrName);
  if (!sourceAttr || !tile || !grid)
    return mlir::success();

  int64_t gridRows = grid->first;
  int64_t gridCols = grid->second;
  int64_t tileRow = tile->first;
  int64_t tileCol = tile->second;
  if (gridRows <= 0 || gridCols <= 0 || tileRow < 0 || tileCol < 0 ||
      tileRow >= gridRows || tileCol >= gridCols)
    return op->emitError("invalid cached logical-array tile metadata");

  auto [it, inserted] = logicalArrays.try_emplace(sourceAttr.getValue());
  LogicalArrayGrid &cached = it->second;
  if (inserted) {
    cached.gridRows = gridRows;
    cached.gridCols = gridCols;
    cached.values.resize(gridRows * gridCols);
  } else if (cached.gridRows != gridRows || cached.gridCols != gridCols) {
    return op->emitError(
        "inconsistent logical-array tile grids for one matrix resource");
  }

  int64_t index = tileRow * gridCols + tileCol;
  if (!cached.values[index])
    cached.values[index] = logicalArray;
  return mlir::success();
}

static mlir::LogicalResult
indexExistingLogicalArrays(mlir::func::FuncOp func,
                           llvm::StringMap<LogicalArrayGrid> &logicalArrays) {
  mlir::WalkResult result = func.walk([&](mlir::sculptor::ArraySetOp arraySet) {
    if (failed(recordLogicalArray(arraySet.getOperation(), arraySet.getArray(),
                                  logicalArrays)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted())
    return mlir::failure();
  return mlir::success();
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

static void setMatrixTileAttrs(SemanticOperationScope &scope,
                               const MatrixPartitionSpec &spec,
                               int64_t tileRow, int64_t tileCol,
                               int64_t arrayRows, int64_t arrayCols,
                               mlir::Builder &builder) {
  MatrixTileExtent extent =
      getMatrixTileExtent(spec, tileRow, tileCol, arrayRows, arrayCols);
  scope.set(golem_tiling_attrs::kSourceResourceAttrName,
            builder.getStringAttr(spec.sourceResource));
  scope.set(golem_tiling_attrs::kTileAttrName,
            builder.getI64ArrayAttr({tileRow, tileCol}));
  scope.set(golem_tiling_attrs::kTileGridAttrName,
            builder.getI64ArrayAttr({spec.gridRows, spec.gridCols}));
  scope.set(golem_tiling_attrs::kTilePhysicalShapeAttrName,
            builder.getI64ArrayAttr(
                {extent.physicalRows, extent.physicalCols}));
  scope.set(golem_tiling_attrs::kTileValidShapeAttrName,
            builder.getI64ArrayAttr({extent.validRows, extent.validCols}));
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

static mlir::Value createMatrixSetup(MatrixPartitionSpec &spec, int64_t tileRow,
                                     int64_t tileCol, int64_t arrayRows,
                                     int64_t arrayCols,
                                     mlir::RewriterBase &rewriter) {
  SemanticOperationScope scope(
      rewriter, semantic_operation_names::kMatrixSetupTaskKind,
      buildMatrixSetupName(rewriter, spec, tileRow, tileCol).getValue());
  if (spec.mappingOperationId)
    scope.set(mapping::kMappingOperationIdAttrName, spec.mappingOperationId);
  if (spec.raLeafId)
    scope.set(mapping::kRALeafIdAttrName, spec.raLeafId);
  setMatrixTileAttrs(scope, spec, tileRow, tileCol, arrayRows, arrayCols,
                     rewriter);

  mlir::arith::ConstantOp tileConstant = createTileConstant(
      spec, tileRow, tileCol, arrayRows, arrayCols, rewriter);
  mlir::sculptor::ArraySetOp arraySet = createArraySet(tileConstant, rewriter);
  scope.annotate();
  return arraySet.getArray();
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

static void setVectorTileAttrs(SemanticOperationScope &scope,
                               int64_t vectorTile, int64_t vectorTileGrid,
                               int64_t physicalCols, int64_t validCols,
                               mlir::Builder &builder) {
  scope.set(golem_tiling_attrs::kVectorTileAttrName,
            builder.getI64IntegerAttr(vectorTile));
  scope.set(golem_tiling_attrs::kVectorTileGridAttrName,
            builder.getI64IntegerAttr(vectorTileGrid));
  scope.set(golem_tiling_attrs::kVectorTilePhysicalColsAttrName,
            builder.getI64IntegerAttr(physicalCols));
  scope.set(golem_tiling_attrs::kVectorTileValidColsAttrName,
            builder.getI64IntegerAttr(validCols));
}

static mlir::Value createFullVectorTile(mlir::sculptor::MVMOp mvmOp,
                                        int64_t vectorTile,
                                        int64_t vectorTileGrid,
                                        int64_t arrayCols,
                                        mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType tileType =
      mlir::RankedTensorType::get({1, arrayCols}, rewriter.getF32Type());
  SemanticOperationScope scope(
      rewriter, "digital.vector_tile",
      buildVectorTileName(rewriter, mvmOp, vectorTile).getValue());
  copyMappingIdentity(mvmOp, scope);
  setVectorTileAttrs(scope, vectorTile, vectorTileGrid, arrayCols, arrayCols,
                     rewriter);
  auto slice = rewriter.create<mlir::tensor::ExtractSliceOp>(
      mvmOp.getLoc(), tileType, mvmOp.getVector(),
      buildIndexAttrs(rewriter, {0, vectorTile * arrayCols}),
      buildIndexAttrs(rewriter, {1, arrayCols}),
      buildIndexAttrs(rewriter, {1, 1}));
  scope.annotate();
  return slice.getResult();
}

static mlir::Value
createPaddedVectorTile(mlir::sculptor::MVMOp mvmOp, int64_t vectorTile,
                       int64_t vectorTileGrid, int64_t remainingCols,
                       int64_t arrayCols, mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType fullTileType =
      mlir::RankedTensorType::get({1, arrayCols}, rewriter.getF32Type());
  mlir::RankedTensorType sourceTileType =
      mlir::RankedTensorType::get({1, remainingCols}, rewriter.getF32Type());
  SemanticOperationScope scope(
      rewriter, "digital.vector_tile",
      buildVectorTileName(rewriter, mvmOp, vectorTile).getValue());
  copyMappingIdentity(mvmOp, scope);
  setVectorTileAttrs(scope, vectorTile, vectorTileGrid, arrayCols, remainingCols,
                     rewriter);
  auto zeroAttr =
      llvm::cast<mlir::TypedAttr>(rewriter.getZeroAttr(fullTileType));
  auto zeroTile = rewriter.create<mlir::arith::ConstantOp>(
      mvmOp.getLoc(), fullTileType, zeroAttr);
  auto sourceSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
      mvmOp.getLoc(), sourceTileType, mvmOp.getVector(),
      buildIndexAttrs(rewriter, {0, vectorTile * arrayCols}),
      buildIndexAttrs(rewriter, {1, remainingCols}),
      buildIndexAttrs(rewriter, {1, 1}));
  auto paddedTile = rewriter.create<mlir::tensor::InsertSliceOp>(
      mvmOp.getLoc(), sourceSlice.getResult(), zeroTile.getResult(),
      buildIndexAttrs(rewriter, {0, 0}),
      buildIndexAttrs(rewriter, {1, remainingCols}),
      buildIndexAttrs(rewriter, {1, 1}));
  scope.annotate();
  return paddedTile.getResult();
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

static void setArrayExecutionAttrs(SemanticOperationScope &scope,
                                   const MatrixPartitionSpec &spec,
                                   int64_t tileRow, int64_t tileCol,
                                   int64_t vectorTile, int64_t arrayRows,
                                   int64_t arrayCols, mlir::Builder &builder) {
  MatrixTileExtent extent =
      getMatrixTileExtent(spec, tileRow, tileCol, arrayRows, arrayCols);
  setMatrixTileAttrs(scope, spec, tileRow, tileCol, arrayRows, arrayCols,
                     builder);
  setVectorTileAttrs(scope, vectorTile, spec.gridCols, extent.physicalCols,
                     extent.validCols, builder);
}

static void attachArrayStoreShapeAttrs(mlir::Operation *op,
                                       const MatrixTileExtent &extent,
                                       mlir::Builder &builder) {
  op->setAttr(
      golem_tiling_attrs::kTilePhysicalShapeAttrName,
      builder.getI64ArrayAttr({extent.physicalRows, extent.physicalCols}));
  op->setAttr(golem_tiling_attrs::kTileValidShapeAttrName,
              builder.getI64ArrayAttr({extent.validRows, extent.validCols}));
}

static mlir::Value createArrayExecution(
    mlir::sculptor::MVMOp mvmOp, const MatrixPartitionSpec &spec,
    mlir::Value vectorTileValue, mlir::Value logicalArray, int64_t tileRow,
    int64_t vectorTile, int64_t arrayRows, int64_t arrayCols,
    mlir::RewriterBase &rewriter) {
  MatrixTileExtent extent =
      getMatrixTileExtent(spec, tileRow, vectorTile, arrayRows, arrayCols);
  mlir::RankedTensorType storeType =
      mlir::RankedTensorType::get({1, extent.validRows}, rewriter.getF32Type());

  SemanticOperationScope scope(
      rewriter, semantic_operation_names::kMVMTaskKind,
      buildMVMName(rewriter, mvmOp, tileRow, vectorTile).getValue());
  copyMappingIdentity(mvmOp, scope);
  setArrayExecutionAttrs(scope, spec, tileRow, vectorTile, vectorTile,
                         arrayRows, arrayCols, rewriter);

  auto loadedType = mlir::sculptor::ArrayLoadedType::get(rewriter.getContext());
  auto arrayLoad = rewriter.create<mlir::sculptor::ArrayLoadOp>(
      mvmOp.getLoc(), loadedType, vectorTileValue, logicalArray);

  auto resultType = mlir::sculptor::ArrayResultType::get(rewriter.getContext());
  auto arrayExecute = rewriter.create<mlir::sculptor::ArrayExecuteOp>(
      mvmOp.getLoc(), resultType, arrayLoad.getLoaded(), logicalArray);

  auto arrayStore = rewriter.create<mlir::sculptor::ArrayStoreOp>(
      mvmOp.getLoc(), storeType, arrayExecute.getResult(), logicalArray);
  attachArrayStoreShapeAttrs(arrayStore, extent, rewriter);
  scope.annotate();
  return arrayStore.getOutput();
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
getOrCreateLogicalArrays(MatrixPartitionSpec &spec,
                         llvm::StringMap<LogicalArrayGrid> &logicalArrayCache,
                         int64_t arrayRows, int64_t arrayCols,
                         mlir::RewriterBase &rewriter) {
  auto [it, inserted] = logicalArrayCache.try_emplace(spec.sourceResource);
  LogicalArrayGrid &cached = it->second;
  if (inserted) {
    cached.gridRows = spec.gridRows;
    cached.gridCols = spec.gridCols;
    cached.values.resize(spec.gridRows * spec.gridCols);
  } else if (cached.gridRows != spec.gridRows ||
             cached.gridCols != spec.gridCols) {
    return spec.constant.emitError(
               "inconsistent logical-array cache grid for matrix resource"),
           mlir::failure();
  }

  mlir::Operation *insertAfter = spec.constant.getOperation();

  for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
    for (int64_t tileCol = 0; tileCol < spec.gridCols; ++tileCol) {
      int64_t index = getTileIndex(spec, tileRow, tileCol);
      if (cached.values[index]) {
        insertAfter = cached.values[index].getDefiningOp();
        continue;
      }

      rewriter.setInsertionPointAfter(insertAfter);
      mlir::Value setupResult = createMatrixSetup(
          spec, tileRow, tileCol, arrayRows, arrayCols, rewriter);
      cached.values[index] = setupResult;
      insertAfter = setupResult.getDefiningOp();
    }
  }

  return cached.values;
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
createArrayLoads(mlir::sculptor::MVMOp mvmOp, const MatrixPartitionSpec &spec,
                 llvm::ArrayRef<mlir::Value> logicalArrays,
                 llvm::ArrayRef<mlir::Value> vectorTiles, int64_t arrayRows,
                 int64_t arrayCols, mlir::RewriterBase &rewriter) {
  if (static_cast<int64_t>(logicalArrays.size()) !=
          spec.gridRows * spec.gridCols ||
      static_cast<int64_t>(vectorTiles.size()) != spec.gridCols)
    return mvmOp.emitError("internal error: mismatched logical array or vector "
                           "tile count for MVM expansion");

  llvm::SmallVector<mlir::Value> partialTiles(spec.gridRows * spec.gridCols);
  for (int64_t vectorTile = 0; vectorTile < spec.gridCols; ++vectorTile) {
    for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
      int64_t index = getTileIndex(spec, tileRow, vectorTile);
      partialTiles[index] = createArrayExecution(
          mvmOp, spec, vectorTiles[vectorTile], logicalArrays[index], tileRow,
          vectorTile, arrayRows, arrayCols, rewriter);
    }
  }

  return partialTiles;
}

static mlir::Value createEmptyTensor(mlir::Location loc,
                                     mlir::RankedTensorType type,
                                     mlir::RewriterBase &rewriter);

static mlir::Value createSequenceArrayExecution(
    MVMSequenceMatch &sequence, const MatrixPartitionSpec &spec,
    mlir::Value logicalArray, int64_t tileRow, int64_t vectorTile,
    int64_t arrayRows, int64_t arrayCols, mlir::RewriterBase &rewriter) {
  MatrixTileExtent extent =
      getMatrixTileExtent(spec, tileRow, vectorTile, arrayRows, arrayCols);
  mlir::RankedTensorType partialType = mlir::RankedTensorType::get(
      {sequence.sequenceLength, extent.validRows}, rewriter.getF32Type());
  mlir::RankedTensorType sourceVectorType =
      mlir::RankedTensorType::get({1, extent.validCols}, rewriter.getF32Type());
  mlir::RankedTensorType physicalVectorType =
      mlir::RankedTensorType::get({1, arrayCols}, rewriter.getF32Type());
  mlir::RankedTensorType storedRowType =
      mlir::RankedTensorType::get({1, extent.validRows}, rewriter.getF32Type());

  SemanticOperationScope scope(
      rewriter, semantic_operation_names::kConvTileMVMTaskKind,
      buildMVMName(rewriter, sequence.mvm, tileRow, vectorTile).getValue());
  copyMappingIdentity(sequence.mvm, scope);
  setArrayExecutionAttrs(scope, spec, tileRow, vectorTile, vectorTile,
                         arrayRows, arrayCols, rewriter);
  mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(
      sequence.getLoc(), 0);
  mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(
      sequence.getLoc(), 1);
  mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
      sequence.getLoc(), sequence.sequenceLength);
  mlir::Value partialInit = rewriter.create<mlir::tensor::EmptyOp>(
      sequence.getLoc(), partialType.getShape(),
      partialType.getElementType());

  mlir::Value zeroVector;
  if (extent.validCols != arrayCols) {
    auto zeroAttr =
        llvm::cast<mlir::TypedAttr>(rewriter.getZeroAttr(physicalVectorType));
    zeroVector = rewriter
                     .create<mlir::arith::ConstantOp>(
                         sequence.getLoc(), physicalVectorType, zeroAttr)
                     .getResult();
  }

  int64_t columnOffset = vectorTile * arrayCols;
  auto rowLoop = rewriter.create<mlir::scf::ForOp>(
      sequence.getLoc(), zero, upper, one, mlir::ValueRange{partialInit},
      [&](mlir::OpBuilder &loopBuilder, mlir::Location loopLoc, mlir::Value row,
          mlir::ValueRange iterArgs) {
        auto sourceSlice = loopBuilder.create<mlir::tensor::ExtractSliceOp>(
            loopLoc, sourceVectorType, sequence.vectors,
            buildRowOffsets(loopBuilder, row, columnOffset),
            buildIndexAttrs(loopBuilder, {1, extent.validCols}),
            buildIndexAttrs(loopBuilder, {1, 1}));

        mlir::Value vector = sourceSlice.getResult();
        if (extent.validCols != arrayCols) {
          vector = loopBuilder
                       .create<mlir::tensor::InsertSliceOp>(
                           loopLoc, vector, zeroVector,
                           buildIndexAttrs(loopBuilder, {0, 0}),
                           buildIndexAttrs(loopBuilder, {1, extent.validCols}),
                           buildIndexAttrs(loopBuilder, {1, 1}))
                       .getResult();
        }

        mlir::Value array = logicalArray;
        auto loadedType =
            mlir::sculptor::ArrayLoadedType::get(rewriter.getContext());
        auto loaded = loopBuilder.create<mlir::sculptor::ArrayLoadOp>(
            loopLoc, loadedType, vector, array);
        auto resultType =
            mlir::sculptor::ArrayResultType::get(rewriter.getContext());
        auto executed = loopBuilder.create<mlir::sculptor::ArrayExecuteOp>(
            loopLoc, resultType, loaded.getLoaded(), array);
        auto store = loopBuilder.create<mlir::sculptor::ArrayStoreOp>(
            loopLoc, storedRowType, executed.getResult(), array);
        attachArrayStoreShapeAttrs(store, extent, loopBuilder);
        mlir::Value stored = store.getOutput();
        mlir::Value updated =
            loopBuilder
                .create<mlir::tensor::InsertSliceOp>(
                    loopLoc, stored, iterArgs[0],
                    buildRowOffsets(loopBuilder, row, 0),
                    buildIndexAttrs(loopBuilder, {1, extent.validRows}),
                    buildIndexAttrs(loopBuilder, {1, 1}))
                .getResult();
        loopBuilder.create<mlir::scf::YieldOp>(loopLoc, updated);
      });

  rewriter.setInsertionPointAfter(rowLoop);
  scope.annotate();
  return rowLoop.getResult(0);
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
createSequenceArrayExecutions(MVMSequenceMatch &sequence,
                              const MatrixPartitionSpec &spec,
                              llvm::ArrayRef<mlir::Value> logicalArrays,
                              int64_t arrayRows, int64_t arrayCols,
                              mlir::RewriterBase &rewriter) {
  if (static_cast<int64_t>(logicalArrays.size()) !=
      spec.gridRows * spec.gridCols) {
    return sequence.emitError(
               "internal error: mismatched logical-array tile count for "
               "MVM sequence expansion"),
           mlir::failure();
  }

  llvm::SmallVector<mlir::Value> partials(spec.gridRows * spec.gridCols);
  for (int64_t vectorTile = 0; vectorTile < spec.gridCols; ++vectorTile) {
    for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
      int64_t index = getTileIndex(spec, tileRow, vectorTile);
      partials[index] = createSequenceArrayExecution(
          sequence, spec, logicalArrays[index], tileRow, vectorTile, arrayRows,
          arrayCols, rewriter);
    }
  }
  return partials;
}

static mlir::Value sumSequenceRowPartials(MVMSequenceMatch &sequence,
                                          const MatrixPartitionSpec &spec,
                                          llvm::ArrayRef<mlir::Value> partials,
                                          int64_t tileRow,
                                          mlir::RankedTensorType rowType,
                                          mlir::RewriterBase &rewriter) {
  mlir::Value row = partials[getTileIndex(spec, tileRow, 0)];
  for (int64_t tileCol = 1; tileCol < spec.gridCols; ++tileCol) {
    mlir::Value rhs = partials[getTileIndex(spec, tileRow, tileCol)];
    mlir::Value init =
        createEmptyTensor(sequence.getLoc(), rowType, rewriter);
    row = rewriter
              .create<mlir::linalg::AddOp>(sequence.getLoc(),
                                           mlir::ValueRange{row, rhs},
                                           mlir::ValueRange{init})
              .getResult(0);
  }
  return row;
}

static mlir::FailureOr<mlir::Value> createRecombinedMVMSequenceResult(
    MVMSequenceMatch &sequence, const MatrixPartitionSpec &spec,
    llvm::ArrayRef<mlir::Value> partials, int64_t arrayRows, int64_t arrayCols,
    mlir::RewriterBase &rewriter) {
  if (static_cast<int64_t>(partials.size()) != spec.gridRows * spec.gridCols) {
    return sequence.emitError(
               "internal error: mismatched partial tile count for MVM "
               "sequence recombination"),
           mlir::failure();
  }

  SemanticOperationScope scope(
      rewriter, semantic_operation_names::kTileRecombineTaskKind,
      buildTileRecombineName(rewriter, sequence.mvm).getValue());
  copyMappingIdentity(sequence.mvm, scope);
  if (spec.gridRows == 1 && spec.gridCols > 1) {
    scope.set(
        mlir::sculptor::task_graph_attrs::kTaskReductionAttrName,
        mlir::sculptor::TaskReductionAttr::get(
            rewriter.getContext(), mlir::sculptor::TaskReductionKind::Add,
            rewriter.getBoolAttr(true)));
  }

  llvm::SmallVector<mlir::Value> rowResults;
  rowResults.reserve(spec.gridRows);
  int64_t recombinedWidth = 0;
  for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
    MatrixTileExtent extent =
        getMatrixTileExtent(spec, tileRow, 0, arrayRows, arrayCols);
    mlir::RankedTensorType rowType = mlir::RankedTensorType::get(
        {sequence.sequenceLength, extent.validRows}, rewriter.getF32Type());
    rowResults.push_back(sumSequenceRowPartials(sequence, spec, partials,
                                                tileRow, rowType, rewriter));
    recombinedWidth += extent.validRows;
  }

  if (recombinedWidth != sequence.resultSequenceType.getDimSize(1)) {
    return sequence.emitError(
               "internal error: valid row tile extents do not match MVM "
               "sequence result width"),
           mlir::failure();
  }

  mlir::Value recombined = rowResults.front();
  if (spec.gridRows > 1) {
    recombined = rewriter
                     .create<mlir::tensor::ConcatOp>(
                         sequence.getLoc(), sequence.resultSequenceType,
                         /*dim=*/1, mlir::ValueRange(rowResults))
                     .getResult();
  }

  scope.annotate();
  return recombined;
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

static mlir::FailureOr<mlir::Value> createRecombinedMVMResult(
    mlir::sculptor::MVMOp mvmOp, const MatrixPartitionSpec &spec,
    llvm::ArrayRef<mlir::Value> partialTiles, int64_t arrayRows,
    int64_t arrayCols, mlir::RewriterBase &rewriter) {
  if (static_cast<int64_t>(partialTiles.size()) !=
      spec.gridRows * spec.gridCols)
    return mvmOp.emitError("internal error: mismatched partial tile count for "
                           "MVM recombine"),
           mlir::failure();

  auto resultType = getStaticMVMResultType(mvmOp, spec);
  if (failed(resultType))
    return mlir::failure();

  int64_t resultWidth = (*resultType).getDimSize(1);

  SemanticOperationScope scope(
      rewriter, "digital.tile_recombine",
      buildTileRecombineName(rewriter, mvmOp).getValue());
  copyMappingIdentity(mvmOp, scope);
  // One-row recombination is an associative sum of equal-shaped partials.
  // Multi-row recombination also concatenates rows and is not one reduction.
  if (spec.gridRows == 1 && spec.gridCols > 1) {
    scope.set(
        mlir::sculptor::task_graph_attrs::kTaskReductionAttrName,
        mlir::sculptor::TaskReductionAttr::get(
            rewriter.getContext(), mlir::sculptor::TaskReductionKind::Add,
            rewriter.getBoolAttr(true)));
  }

  llvm::SmallVector<mlir::Value> rowResults;
  rowResults.reserve(spec.gridRows);
  int64_t recombinedWidth = 0;
  for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
    MatrixTileExtent extent =
        getMatrixTileExtent(spec, tileRow, 0, arrayRows, arrayCols);
    mlir::RankedTensorType rowTileType = mlir::RankedTensorType::get(
        {1, extent.validRows}, rewriter.getF32Type());
    rowResults.push_back(sumRowPartials(mvmOp, spec, partialTiles,
                                        tileRow, rowTileType, rewriter));
    recombinedWidth += extent.validRows;
  }

  if (recombinedWidth != resultWidth)
    return mvmOp.emitError("internal error: valid row tile extents do not "
                           "match the original MVM result width"),
           mlir::failure();

  mlir::Value recombined = rowResults.front();
  if (spec.gridRows > 1) {
    recombined = rewriter
                     .create<mlir::tensor::ConcatOp>(
                         mvmOp.getLoc(), *resultType,
                         /*dim=*/1, mlir::ValueRange(rowResults))
                     .getResult();
  }

  scope.annotate();
  return recombined;
}

struct MVMExpansionWalker {
  MVMExpansionWalker(int64_t arrayRows, int64_t arrayCols)
      : arrayRows(arrayRows), arrayCols(arrayCols) {}

  mlir::LogicalResult run(mlir::ModuleOp module) {
    for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
      if (func.isExternal())
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
    FunctionExpansionState state;
    if (failed(indexExistingLogicalArrays(func, state.logicalArrays)))
      return mlir::failure();

    llvm::SmallVector<MVMSequenceMatch> sequences;
    for (mlir::Operation &op : func.front().without_terminator()) {
      auto loop = llvm::dyn_cast<mlir::scf::ForOp>(&op);
      if (!loop)
        continue;
      auto section = loop->getAttrOfType<mlir::StringAttr>(
          "sculptor.semantic.section");
      if (!section ||
          section.getValue() != semantic_operation_names::kMVMSequenceTaskKind)
        continue;
      auto sequence = matchMVMSequence(loop);
      if (failed(sequence))
        return mlir::failure();
      sequences.push_back(std::move(*sequence));
    }

    for (MVMSequenceMatch &sequence : sequences) {
      if (failed(handleMVMSequence(state, sequence)))
        return mlir::failure();
    }

    llvm::SmallVector<mlir::sculptor::MVMOp> mvmOps;
    func.walk([&](mlir::sculptor::MVMOp mvmOp) { mvmOps.push_back(mvmOp); });

    for (mlir::sculptor::MVMOp mvmOp : mvmOps) {
      if (!mvmOp || !mvmOp->getBlock())
        continue;

      if (failed(handleMVMOp(state, mvmOp)))
        return mlir::failure();
    }

    assignStageMetadata(func);

    llvm::SmallPtrSet<mlir::Operation *, 16> seenConstants;
    for (mlir::arith::ConstantOp constant : state.matrixConstants) {
      if (constant && seenConstants.insert(constant.getOperation()).second &&
          constant->use_empty())
        constant.erase();
    }

    return mlir::success();
  }

  mlir::LogicalResult handleMVMSequence(FunctionExpansionState &state,
                                        MVMSequenceMatch &sequence) {
    auto spec = getOrCreateMatrixSpec(state, sequence.mvm);
    if (failed(spec))
      return mlir::failure();
    if (sequence.vectorSequenceType.getDimSize(1) !=
        (*spec)->type.getDimSize(1)) {
      return sequence.emitError(
          "expected MVM sequence vector width to match matrix width");
    }

    mlir::IRRewriter rewriter(sequence.anchor->getContext());
    auto logicalArrays = getOrCreateLogicalArrays(
        **spec, state.logicalArrays, arrayRows, arrayCols, rewriter);
    if (failed(logicalArrays))
      return mlir::failure();

    rewriter.setInsertionPoint(sequence.getInsertionAnchor());
    auto partials = createSequenceArrayExecutions(
        sequence, **spec, *logicalArrays, arrayRows, arrayCols, rewriter);
    if (failed(partials))
      return mlir::failure();

    auto recombined = createRecombinedMVMSequenceResult(
        sequence, **spec, *partials, arrayRows, arrayCols, rewriter);
    if (failed(recombined))
      return mlir::failure();

    sequence.anchor->getResult(0).replaceAllUsesWith(*recombined);
    for (mlir::Operation *member : llvm::reverse(sequence.members))
      rewriter.eraseOp(member);
    return mlir::success();
  }

  mlir::FailureOr<MatrixPartitionSpec *>
  getOrCreateMatrixSpec(FunctionExpansionState &state,
                        mlir::sculptor::MVMOp op) {
    auto matrixOperand = matchMatrixOperand(op);
    if (failed(matrixOperand))
      return mlir::failure();
    state.matrixConstants.push_back(matrixOperand->constant);

    llvm::StringRef sourceResource =
        matrixOperand->resource.getRawHandle().getKey();
    auto existing = state.matrixSpecs.find(sourceResource);
    if (existing != state.matrixSpecs.end()) {
      if (existing->second.type != matrixOperand->type)
        return op.emitError(
                   "one matrix resource has inconsistent tensor types"),
               mlir::failure();
      return &existing->second;
    }

    auto spec =
        buildMatrixPartitionSpec(op, *matrixOperand, arrayRows, arrayCols);
    if (failed(spec))
      return mlir::failure();
    auto [it, inserted] =
        state.matrixSpecs.try_emplace(sourceResource, std::move(*spec));
    (void)inserted;
    return &it->second;
  }

  mlir::LogicalResult handleMVMOp(FunctionExpansionState &state,
                                  mlir::sculptor::MVMOp op) {
    auto spec = getOrCreateMatrixSpec(state, op);
    if (failed(spec))
      return mlir::failure();

    mlir::IRRewriter rewriter(op.getContext());
    auto logicalArrays = getOrCreateLogicalArrays(
        **spec, state.logicalArrays, arrayRows, arrayCols, rewriter);
    if (failed(logicalArrays))
      return mlir::failure();

    rewriter.setInsertionPoint(op);
    auto vectorTiles = createVectorTiles(op, **spec, arrayCols, rewriter);
    if (failed(vectorTiles))
      return mlir::failure();

    auto partialTiles =
        createArrayLoads(op, **spec, *logicalArrays, *vectorTiles, arrayRows,
                         arrayCols, rewriter);
    if (failed(partialTiles))
      return mlir::failure();

    auto recombined = createRecombinedMVMResult(op, **spec, *partialTiles,
                                                arrayRows, arrayCols, rewriter);
    if (failed(recombined))
      return mlir::failure();

    op.getResult().replaceAllUsesWith(*recombined);
    rewriter.eraseOp(op);
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
