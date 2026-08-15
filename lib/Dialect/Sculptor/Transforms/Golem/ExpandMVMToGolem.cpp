#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/ExpandMVMToGolem.h"

// ExpandMVMToGolem materializes fixed-size logical arrays and inline Golem
// execution operations. Mapping consumes this physical realization rather
// than the logical sculptor.mvm operations that preceded it.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/GolemTilingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/SemanticOperationNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/SemanticOperationScope.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/GolemMVMPlanning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cassert>
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
  int64_t sourceMVMId = -1;
  llvm::SmallVector<float> values;
  std::optional<float> splatValue;
  int64_t gridRows = 0;
  int64_t gridCols = 0;
  mapping::GolemMVMPlan physicalPlan;
  mlir::IntegerAttr mappingOperationId;
  mlir::IntegerAttr raLeafId;
  mlir::IntegerAttr semanticLayerId;
  mlir::StringAttr semanticLayerKind;
};

struct MatrixOperand {
  mlir::arith::ConstantOp constant;
  mlir::RankedTensorType type;
  std::string sourceResource;
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
  llvm::DenseMap<mlir::Operation *, int64_t> sourceMVMIds;
  llvm::DenseMap<mlir::Operation *, std::string> inlineMatrixResources;
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

static llvm::SmallVector<mlir::OpFoldResult>
buildIndexAttrs(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> values);

static mlir::Value createEmptyTensor(mlir::Location loc,
                                     mlir::RankedTensorType type,
                                     mlir::RewriterBase &rewriter);

struct PointwiseEpilogueStage {
  mlir::linalg::GenericOp generic;
  unsigned primaryInput = 0;
};

struct PointwiseEpilogueMatch {
  mlir::tensor::ExpandShapeOp view;
  mlir::RankedTensorType logicalType;
  llvm::SmallVector<PointwiseEpilogueStage> stages;

  mlir::Value getResult() { return stages.back().generic.getResult(0); }
};

static bool hasCompatibleLayerIdentity(mlir::Operation *source,
                                       mlir::Operation *candidate) {
  for (llvm::StringRef name :
       {kSemanticLayerIdAttrName, kSemanticLayerKindAttrName}) {
    mlir::Attribute sourceValue = source->getAttr(name);
    mlir::Attribute candidateValue = candidate->getAttr(name);
    if (sourceValue && candidateValue && sourceValue != candidateValue)
      return false;
  }
  return true;
}

static bool isChannelOnlyMap(mlir::AffineMap map, unsigned logicalRank) {
  if (map.getNumDims() != logicalRank || map.getNumSymbols() != 0)
    return false;
  for (mlir::AffineExpr expression : map.getResults()) {
    if (auto dimension = llvm::dyn_cast<mlir::AffineDimExpr>(expression)) {
      if (dimension.getPosition() != logicalRank - 1)
        return false;
      continue;
    }
    if (!llvm::isa<mlir::AffineConstantExpr>(expression))
      return false;
  }
  return true;
}

static std::optional<PointwiseEpilogueStage>
matchPointwiseStage(mlir::Value current, mlir::Operation *source,
                    mlir::linalg::GenericOp generic,
                    mlir::RankedTensorType logicalType) {
  if (!generic || !generic.hasPureTensorSemantics() ||
      generic.getNumResults() != 1 || generic.getInputs().empty() ||
      generic.getOutputs().size() != 1 ||
      !mlir::linalg::isElementwise(generic) ||
      !hasCompatibleLayerIdentity(source, generic))
    return std::nullopt;

  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(generic.getResult(0).getType());
  if (!resultType || resultType != logicalType)
    return std::nullopt;

  llvm::SmallVector<mlir::AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != generic.getInputs().size() + 1 ||
      maps.back() != mlir::AffineMap::getMultiDimIdentityMap(
                         logicalType.getRank(), generic.getContext()))
    return std::nullopt;

  unsigned primaryInput = 0;
  unsigned primaryCount = 0;
  for (auto [index, input] : llvm::enumerate(generic.getInputs())) {
    if (input == current) {
      primaryInput = index;
      ++primaryCount;
      if (maps[index] != mlir::AffineMap::getMultiDimIdentityMap(
                             logicalType.getRank(), generic.getContext()))
        return std::nullopt;
      continue;
    }
    if (!isChannelOnlyMap(maps[index], logicalType.getRank()))
      return std::nullopt;
    auto inputType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape() ||
        inputType.getRank() !=
            static_cast<int64_t>(maps[index].getNumResults()))
      return std::nullopt;
    for (auto [dimension, expression] :
         llvm::enumerate(maps[index].getResults())) {
      int64_t size = inputType.getDimSize(dimension);
      if (auto channel = llvm::dyn_cast<mlir::AffineDimExpr>(expression)) {
        (void)channel;
        if (size != logicalType.getShape().back())
          return std::nullopt;
      } else if (auto constant =
                     llvm::dyn_cast<mlir::AffineConstantExpr>(expression)) {
        if (constant.getValue() < 0 || constant.getValue() >= size)
          return std::nullopt;
      }
    }
  }
  if (primaryCount != 1)
    return std::nullopt;

  mlir::OpOperand *outputOperand = generic.getDpsInitOperand(0);
  auto empty = outputOperand->get().getDefiningOp<mlir::tensor::EmptyOp>();
  if (!empty || !generic.getRegionOutputArgs().front().use_empty())
    return std::nullopt;

  bool hasIndex = false;
  generic.getRegion().walk([&](mlir::linalg::IndexOp) { hasIndex = true; });
  if (hasIndex)
    return std::nullopt;
  return PointwiseEpilogueStage{generic, primaryInput};
}

static std::optional<PointwiseEpilogueMatch>
matchPointwiseEpilogue(mlir::Value mvmResult, mlir::Operation *source) {
  if (!mvmResult.hasOneUse())
    return std::nullopt;
  auto rank2Type = llvm::dyn_cast<mlir::RankedTensorType>(mvmResult.getType());
  if (!rank2Type || !rank2Type.hasStaticShape() || rank2Type.getRank() != 2)
    return std::nullopt;

  PointwiseEpilogueMatch match;
  mlir::Value current = mvmResult;
  mlir::Operation *user = *current.getUsers().begin();
  if (auto view = llvm::dyn_cast<mlir::tensor::ExpandShapeOp>(user)) {
    if (!view.getResult().hasOneUse() ||
        !hasCompatibleLayerIdentity(source, view))
      return std::nullopt;
    auto viewType = llvm::dyn_cast<mlir::RankedTensorType>(view.getType());
    if (!viewType || !viewType.hasStaticShape() || viewType.getRank() < 2 ||
        viewType.getShape().back() != rank2Type.getShape().back())
      return std::nullopt;
    int64_t leadingElements = 1;
    for (int64_t size : viewType.getShape().drop_back()) {
      std::optional<int64_t> product = llvm::checkedMul(leadingElements, size);
      if (!product)
        return std::nullopt;
      leadingElements = *product;
    }
    if (leadingElements != rank2Type.getDimSize(0))
      return std::nullopt;
    match.view = view;
    current = view.getResult();
    user = *current.getUsers().begin();
  }
  match.logicalType = llvm::cast<mlir::RankedTensorType>(current.getType());

  while (auto generic = llvm::dyn_cast<mlir::linalg::GenericOp>(user)) {
    std::optional<PointwiseEpilogueStage> stage =
        matchPointwiseStage(current, source, generic, match.logicalType);
    if (!stage)
      break;
    match.stages.push_back(*stage);
    current = generic.getResult(0);
    if (!current.hasOneUse())
      break;
    user = *current.getUsers().begin();
  }
  if (match.stages.empty())
    return std::nullopt;
  return match;
}

static mlir::Value createAuxiliaryChannelShard(
    mlir::Location loc, mlir::Value input, mlir::AffineMap originalMap,
    int64_t featureOffset, int64_t featureWidth, mlir::RewriterBase &rewriter) {
  auto inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> offsets(inputType.getRank(), 0);
  llvm::SmallVector<int64_t> sizes(inputType.getShape());
  bool needsSlice = false;
  for (auto [dimension, expression] :
       llvm::enumerate(originalMap.getResults())) {
    if (llvm::isa<mlir::AffineDimExpr>(expression)) {
      offsets[dimension] = featureOffset;
      sizes[dimension] = featureWidth;
      needsSlice =
          featureOffset != 0 || featureWidth != inputType.getDimSize(dimension);
    }
  }
  if (!needsSlice)
    return input;
  mlir::RankedTensorType shardType = mlir::RankedTensorType::get(
      sizes, inputType.getElementType(), inputType.getEncoding());
  llvm::SmallVector<int64_t> strides(inputType.getRank(), 1);
  return rewriter
      .create<mlir::tensor::ExtractSliceOp>(
          loc, shardType, input, buildIndexAttrs(rewriter, offsets),
          buildIndexAttrs(rewriter, sizes), buildIndexAttrs(rewriter, strides))
      .getResult();
}

static mlir::Value applyPointwiseStageToShard(
    const PointwiseEpilogueStage &stage, mlir::Value shard,
    int64_t featureOffset, int64_t featureWidth,
    llvm::SmallVectorImpl<mlir::Operation *> &fusedOperations,
    mlir::RewriterBase &rewriter) {
  mlir::linalg::GenericOp generic = stage.generic;
  auto shardType = llvm::cast<mlir::RankedTensorType>(shard.getType());
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::AffineMap> maps;
  inputs.reserve(generic.getInputs().size());
  maps.reserve(generic.getInputs().size() + 1);
  mlir::AffineMap identity = rewriter.getMultiDimIdentityMap(2);
  llvm::SmallVector<mlir::AffineMap> originalMaps =
      generic.getIndexingMapsArray();
  for (auto [index, input] : llvm::enumerate(generic.getInputs())) {
    if (index == stage.primaryInput) {
      inputs.push_back(shard);
      maps.push_back(identity);
      continue;
    }
    mlir::Value auxiliaryShard = createAuxiliaryChannelShard(
        generic.getLoc(), input, originalMaps[index], featureOffset,
        featureWidth, rewriter);
    inputs.push_back(auxiliaryShard);
    if (auxiliaryShard != input)
      fusedOperations.push_back(auxiliaryShard.getDefiningOp());
    llvm::SmallVector<mlir::AffineExpr> results;
    for (mlir::AffineExpr expression : originalMaps[index].getResults()) {
      if (llvm::isa<mlir::AffineDimExpr>(expression))
        results.push_back(rewriter.getAffineDimExpr(1));
      else
        results.push_back(expression);
    }
    maps.push_back(mlir::AffineMap::get(2, 0, results, rewriter.getContext()));
  }
  maps.push_back(identity);
  mlir::Value init = createEmptyTensor(generic.getLoc(), shardType, rewriter);
  fusedOperations.push_back(init.getDefiningOp());
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      2, mlir::utils::IteratorType::parallel);
  auto fused = rewriter.create<mlir::linalg::GenericOp>(
      generic.getLoc(), shardType, inputs, mlir::ValueRange{init}, maps,
      iterators,
      [&](mlir::OpBuilder &builder, mlir::Location,
          mlir::ValueRange arguments) {
        mlir::IRMapping mapping;
        mlir::Block &sourceBlock = generic.getRegion().front();
        for (auto [oldArgument, newArgument] :
             llvm::zip_equal(sourceBlock.getArguments(), arguments))
          mapping.map(oldArgument, newArgument);
        for (mlir::Operation &operation : sourceBlock.without_terminator())
          builder.clone(operation, mapping);
        auto oldYield =
            llvm::cast<mlir::linalg::YieldOp>(sourceBlock.getTerminator());
        builder.create<mlir::linalg::YieldOp>(
            oldYield.getLoc(),
            mapping.lookupOrDefault(oldYield.getValues()[0]));
      });
  fused->setAttr("sculptor.golem.fused_pointwise_epilogue",
                 rewriter.getUnitAttr());
  fusedOperations.push_back(fused);
  return fused.getResult(0);
}

static mlir::Value applyPointwiseEpilogueToShard(
    const PointwiseEpilogueMatch &epilogue, mlir::Value shard,
    int64_t featureOffset, int64_t featureWidth,
    llvm::SmallVectorImpl<mlir::Operation *> &fusedOperations,
    mlir::RewriterBase &rewriter) {
  for (const PointwiseEpilogueStage &stage : epilogue.stages) {
    shard = applyPointwiseStageToShard(stage, shard, featureOffset,
                                       featureWidth, fusedOperations, rewriter);
  }
  return shard;
}

static mlir::Value restoreEpilogueResultShape(PointwiseEpilogueMatch &epilogue,
                                              mlir::Value rank2Result,
                                              mlir::RewriterBase &rewriter) {
  if (!epilogue.view)
    return rank2Result;
  auto restored = rewriter.create<mlir::tensor::ExpandShapeOp>(
      epilogue.view.getLoc(), epilogue.logicalType, rank2Result,
      epilogue.view.getReassociationIndices());
  for (mlir::NamedAttribute attribute : epilogue.view->getAttrs()) {
    if (attribute.getName().strref().starts_with("sculptor."))
      restored->setAttr(attribute.getName(), attribute.getValue());
  }
  return restored.getResult();
}

static void erasePointwiseEpilogue(PointwiseEpilogueMatch &epilogue,
                                   mlir::RewriterBase &rewriter) {
  for (PointwiseEpilogueStage &stage : llvm::reverse(epilogue.stages)) {
    mlir::Value init = stage.generic.getDpsInits().front();
    rewriter.eraseOp(stage.generic);
    if (auto empty = init.getDefiningOp<mlir::tensor::EmptyOp>();
        empty && empty->use_empty())
      rewriter.eraseOp(empty);
  }
  if (epilogue.view && epilogue.view->use_empty())
    rewriter.eraseOp(epilogue.view);
}

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

static std::string buildSourceMVMPrefix(mlir::sculptor::MVMOp mvmOp,
                                        int64_t sourceMVMId) {
  return (llvm::Twine(getTaskNamePrefix(mvmOp.getOperation())) + "_mvm_" +
          llvm::Twine(sourceMVMId))
      .str();
}

static int64_t getSourceMVMId(mlir::sculptor::MVMOp mvmOp) {
  auto sourceMVMId = mvmOp->getAttrOfType<mlir::IntegerAttr>(
      golem_tiling_attrs::kSourceMVMIdAttrName);
  assert(sourceMVMId &&
         "source MVM identity must be assigned before expansion");
  return sourceMVMId.getInt();
}

static void copyMappingIdentity(mlir::Operation *source,
                                SemanticOperationScope &scope) {
  scope.copyIfPresent(source, mapping::kMappingOperationIdAttrName);
  scope.copyIfPresent(source, mapping::kRALeafIdAttrName);
  scope.copyIfPresent(source, golem_tiling_attrs::kSourceMVMIdAttrName);
  for (mlir::NamedAttribute attribute : source->getAttrs()) {
    llvm::StringRef name = attribute.getName().strref();
    if (name.starts_with("sculptor.semantic.") &&
        name != "sculptor.semantic.name" && name != "sculptor.semantic.section")
      scope.set(attribute.getName().strref(), attribute.getValue());
  }
}

static llvm::StringRef getStageKind(llvm::StringRef kind) {
  if (kind == semantic_operation_names::kMatrixSetupTaskKind)
    return mapping::kMatrixSetupStageKind;
  // Patch shards are the vector-preparation member of a physical MVM wave.
  // Keeping them as anonymous digital work lets the generic digital sharder
  // split an already memory-bounded patch and lets placement scatter those
  // pieces away from their analog consumer, duplicating activation traffic.
  if (kind == "digital.vector_tile" ||
      kind == semantic_operation_names::kConvPatchTaskKind)
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
    auto section =
        operation.getAttrOfType<mlir::StringAttr>("sculptor.semantic.section");
    if (!section)
      continue;
    auto name =
        operation.getAttrOfType<mlir::StringAttr>("sculptor.semantic.name");
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
  std::string name = buildSourceMVMPrefix(mvmOp, getSourceMVMId(mvmOp));
  name += "_vector_tile_";
  name += std::to_string(vectorTile);
  return builder.getStringAttr(name);
}

static mlir::StringAttr buildMVMName(mlir::OpBuilder &builder,
                                     mlir::sculptor::MVMOp mvmOp,
                                     int64_t tileRow, int64_t tileCol) {
  std::string name = buildSourceMVMPrefix(mvmOp, getSourceMVMId(mvmOp));
  name += "_array_";
  name += std::to_string(tileRow);
  name += "_";
  name += std::to_string(tileCol);
  return builder.getStringAttr(name);
}

static mlir::StringAttr buildSequenceShardName(mlir::OpBuilder &builder,
                                               mlir::StringAttr base,
                                               int64_t shardIndex) {
  return builder.getStringAttr((llvm::Twine(base.getValue()) +
                                "_sequence_shard_" + llvm::Twine(shardIndex))
                                   .str());
}

static mlir::StringAttr buildTileRecombineName(mlir::OpBuilder &builder,
                                               mlir::sculptor::MVMOp mvmOp) {
  std::string name = buildSourceMVMPrefix(mvmOp, getSourceMVMId(mvmOp));
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

  auto values =
      mlir::sculptor::converter_constant::getF32ConstantValues(matrixConst);
  if (failed(values))
    return mvmOp.emitError("expected sculptor.mvm matrix constant to contain "
                           "dense f32 elements"),
           mlir::failure();

  std::string sourceResource;
  if (auto resource = llvm::dyn_cast<mlir::DenseF32ResourceElementsAttr>(
          matrixConst.getValue()))
    sourceResource = resource.getRawHandle().getKey().str();
  return MatrixOperand{matrixConst, *matrixType, std::move(sourceResource)};
}

static bool belongsToSemanticGroup(mlir::Operation *operation,
                                   mlir::StringAttr section,
                                   mlir::StringAttr name) {
  return operation->getAttrOfType<mlir::StringAttr>(
             "sculptor.semantic.section") == section &&
         operation->getAttrOfType<mlir::StringAttr>("sculptor.semantic.name") ==
             name;
}

static mlir::FailureOr<MVMSequenceMatch>
matchMVMSequence(mlir::scf::ForOp loop) {
  auto section =
      loop->getAttrOfType<mlir::StringAttr>("sculptor.semantic.section");
  if (!section ||
      section.getValue() != semantic_operation_names::kMVMSequenceTaskKind)
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
    return loop.emitOpError("expected MVM sequence loop to contain exactly one "
                            "sculptor.mvm"),
           mlir::failure();

  mlir::sculptor::MVMOp mvmOp = mvmOps.front();
  auto vectorSlice =
      mvmOp.getVector().getDefiningOp<mlir::tensor::ExtractSliceOp>();
  if (!vectorSlice)
    return mvmOp.emitOpError("expected MVM sequence vector to come from "
                             "tensor.extract_slice"),
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

  return MVMSequenceMatch{
      loop.getOperation(), std::move(members),  mvmOp,         vectors,
      *vectorSequenceType, *resultSequenceType, sequenceLength};
}

static mlir::FailureOr<MatrixPartitionSpec>
buildMatrixPartitionSpec(mlir::sculptor::MVMOp mvmOp,
                         const MatrixOperand &matrixOperand, int64_t arrayRows,
                         int64_t arrayCols, int64_t sourceMVMId,
                         llvm::StringRef sourceResource) {
  auto matrixConst = matrixOperand.constant;

  auto shape = matrixOperand.type.getShape();
  MatrixPartitionSpec spec;
  spec.constant = matrixConst;
  spec.type = matrixOperand.type;
  spec.sourceResource = sourceResource.str();
  spec.sourceMVMId = sourceMVMId;
  spec.taskPrefix = buildSourceMVMPrefix(mvmOp, spec.sourceMVMId);
  if (auto dense =
          llvm::dyn_cast<mlir::DenseElementsAttr>(matrixConst.getValue());
      dense && dense.isSplat()) {
    // Keep splats symbolic. Expanding a zero-initialized model into one float
    // per physical array cell turns a compact model into gigabytes of resource
    // blobs even though every cell has the same value.
    spec.splatValue = dense.getSplatValue<float>();
  } else {
    auto values =
        mlir::sculptor::converter_constant::getF32ConstantValues(matrixConst);
    if (failed(values))
      return mvmOp.emitError("failed to read dense f32 matrix resource"),
             mlir::failure();
    if (static_cast<int64_t>(values->size()) !=
        matrixOperand.type.getNumElements())
      return mvmOp.emitError(
                 "dense f32 matrix resource element count does not match the "
                 "tensor type"),
             mlir::failure();
    spec.values = std::move(*values);
  }
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
  spec.semanticLayerId =
      mvmOp->getAttrOfType<mlir::IntegerAttr>(kSemanticLayerIdAttrName);
  spec.semanticLayerKind =
      mvmOp->getAttrOfType<mlir::StringAttr>(kSemanticLayerKindAttrName);
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
buildValidTileValues(const MatrixPartitionSpec &spec, int64_t tileRow,
                     int64_t tileCol, int64_t validRows, int64_t validCols,
                     int64_t arrayRows, int64_t arrayCols) {
  llvm::SmallVector<float> tileValues(validRows * validCols);
  auto matrixShape = spec.type.getShape();
  int64_t matrixRows = matrixShape[0];
  int64_t matrixCols = matrixShape[1];

  for (int64_t r = 0; r < validRows; ++r) {
    int64_t sourceRow = tileRow * arrayRows + r;
    if (sourceRow >= matrixRows)
      continue;

    for (int64_t c = 0; c < validCols; ++c) {
      int64_t sourceCol = tileCol * arrayCols + c;
      if (sourceCol >= matrixCols)
        continue;

      tileValues[r * validCols + c] =
          spec.splatValue ? *spec.splatValue
                          : spec.values[sourceRow * matrixCols + sourceCol];
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
  MatrixTileExtent extent =
      getMatrixTileExtent(spec, tileRow, tileCol, arrayRows, arrayCols);
  // Persist only live matrix cells. Edge-tile zero padding is a physical-array
  // concern and is materialized by ArraySetLowering into boot scratch memory.
  // Keeping it out of the constant avoids a full resource blob for every
  // partial tile and preserves splat constants compactly.
  mlir::RankedTensorType tileType = mlir::RankedTensorType::get(
      {extent.validRows, extent.validCols}, spec.type.getElementType());
  mlir::TypedAttr tileAttr;
  if (spec.splatValue) {
    tileAttr = mlir::DenseElementsAttr::get(
        tileType, rewriter.getF32FloatAttr(*spec.splatValue));
  } else {
    llvm::SmallVector<float> tileValues =
        buildValidTileValues(spec, tileRow, tileCol, extent.validRows,
                             extent.validCols, arrayRows, arrayCols);
    std::string tileResourceName =
        buildTileResourceName(spec.sourceResource, tileRow, tileCol);
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
        llvm::ArrayRef<float>(tileValues), /*dataIsMutable=*/false);
    tileAttr =
        llvm::cast<mlir::TypedAttr>(mlir::DenseF32ResourceElementsAttr::get(
            tileType, tileResourceName, std::move(blob)));
  }

  auto tileConstant = rewriter.create<mlir::arith::ConstantOp>(
      spec.constant->getLoc(), tileType, tileAttr);
  return tileConstant;
}

static void setMatrixTileAttrs(SemanticOperationScope &scope,
                               const MatrixPartitionSpec &spec, int64_t tileRow,
                               int64_t tileCol, int64_t arrayRows,
                               int64_t arrayCols, mlir::Builder &builder) {
  MatrixTileExtent extent =
      getMatrixTileExtent(spec, tileRow, tileCol, arrayRows, arrayCols);
  scope.set(golem_tiling_attrs::kSourceResourceAttrName,
            builder.getStringAttr(spec.sourceResource));
  scope.set(golem_tiling_attrs::kSourceMVMIdAttrName,
            builder.getI64IntegerAttr(spec.sourceMVMId));
  scope.set(golem_tiling_attrs::kTileAttrName,
            builder.getI64ArrayAttr({tileRow, tileCol}));
  scope.set(golem_tiling_attrs::kTileGridAttrName,
            builder.getI64ArrayAttr({spec.gridRows, spec.gridCols}));
  scope.set(
      golem_tiling_attrs::kTilePhysicalShapeAttrName,
      builder.getI64ArrayAttr({extent.physicalRows, extent.physicalCols}));
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
  if (spec.semanticLayerId)
    scope.set(kSemanticLayerIdAttrName, spec.semanticLayerId);
  if (spec.semanticLayerKind)
    scope.set(kSemanticLayerKindAttrName, spec.semanticLayerKind);
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

static mlir::Value createPaddedPhysicalVector(mlir::Location loc,
                                              mlir::Value source,
                                              mlir::RankedTensorType resultType,
                                              int64_t validCols,
                                              mlir::OpBuilder &builder) {
  int64_t paddingCols = resultType.getDimSize(1) - validCols;
  assert(paddingCols > 0 && "padded vector requires an invalid tail");
  mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());
  mlir::RankedTensorType tailType = mlir::RankedTensorType::get(
      {1, paddingCols}, resultType.getElementType());
  mlir::Value tail = builder.create<mlir::tensor::ExtractSliceOp>(
      loc, tailType, empty, buildIndexAttrs(builder, {0, validCols}),
      buildIndexAttrs(builder, {1, paddingCols}),
      buildIndexAttrs(builder, {1, 1}));
  mlir::Value zero = builder.create<mlir::arith::ConstantOp>(
      loc, resultType.getElementType(),
      builder.getFloatAttr(resultType.getElementType(), 0.0));
  mlir::Value filledTail =
      builder.create<mlir::linalg::FillOp>(loc, zero, tail).getResult(0);
  mlir::Value withTail = builder.create<mlir::tensor::InsertSliceOp>(
      loc, filledTail, empty, buildIndexAttrs(builder, {0, validCols}),
      buildIndexAttrs(builder, {1, paddingCols}),
      buildIndexAttrs(builder, {1, 1}));
  auto padded = builder.create<mlir::tensor::InsertSliceOp>(
      loc, source, withTail, buildIndexAttrs(builder, {0, 0}),
      buildIndexAttrs(builder, {1, validCols}),
      buildIndexAttrs(builder, {1, 1}));
  padded->setAttr("sculptor.memory.physical_vector_padding_generated",
                  builder.getUnitAttr());
  padded->setAttr("sculptor.memory.physical_vector_valid_cols",
                  builder.getI64IntegerAttr(validCols));
  return padded.getResult();
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
  setVectorTileAttrs(scope, vectorTile, vectorTileGrid, arrayCols,
                     remainingCols, rewriter);
  auto sourceSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
      mvmOp.getLoc(), sourceTileType, mvmOp.getVector(),
      buildIndexAttrs(rewriter, {0, vectorTile * arrayCols}),
      buildIndexAttrs(rewriter, {1, remainingCols}),
      buildIndexAttrs(rewriter, {1, 1}));
  mlir::Value paddedTile = createPaddedPhysicalVector(
      mvmOp.getLoc(), sourceSlice, fullTileType, remainingCols, rewriter);
  scope.annotate();
  return paddedTile;
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

static mlir::Value createArrayExecution(mlir::sculptor::MVMOp mvmOp,
                                        const MatrixPartitionSpec &spec,
                                        mlir::Value vectorTileValue,
                                        mlir::Value logicalArray,
                                        int64_t tileRow, int64_t vectorTile,
                                        int64_t arrayRows, int64_t arrayCols,
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

static mlir::Value sumEqualShapePartials(mlir::Location loc,
                                         mlir::RankedTensorType resultType,
                                         llvm::ArrayRef<mlir::Value> partials,
                                         mlir::RewriterBase &rewriter) {
  assert(!partials.empty() && "a partial sum requires at least one input");
  if (partials.size() == 1)
    return partials.front();

  mlir::Value init = createEmptyTensor(loc, resultType, rewriter);
  mlir::AffineMap identity =
      rewriter.getMultiDimIdentityMap(resultType.getRank());
  llvm::SmallVector<mlir::AffineMap> indexingMaps(partials.size() + 1,
                                                  identity);
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      resultType.getRank(), mlir::utils::IteratorType::parallel);
  const size_t inputCount = partials.size();
  return rewriter
      .create<mlir::linalg::GenericOp>(
          loc, resultType, mlir::ValueRange(partials), mlir::ValueRange{init},
          indexingMaps, iteratorTypes,
          [inputCount](mlir::OpBuilder &builder, mlir::Location nestedLoc,
                       mlir::ValueRange arguments) {
            mlir::Value total = arguments.front();
            for (size_t index = 1; index < inputCount; ++index)
              total = builder.create<mlir::arith::AddFOp>(nestedLoc, total,
                                                          arguments[index]);
            builder.create<mlir::linalg::YieldOp>(nestedLoc, total);
          })
      .getResult(0);
}

static mlir::Value createSequenceArrayExecution(
    MVMSequenceMatch &sequence, const MatrixPartitionSpec &spec,
    mlir::Value vectorShard, mlir::Value logicalArray, int64_t tileRow,
    int64_t vectorTile, int64_t sequenceOffset, int64_t sequenceRows,
    int64_t sequenceShardIndex, int64_t sequenceShardCount, int64_t arrayRows,
    int64_t arrayCols, mlir::RewriterBase &rewriter) {
  MatrixTileExtent extent =
      getMatrixTileExtent(spec, tileRow, vectorTile, arrayRows, arrayCols);
  mlir::RankedTensorType partialType = mlir::RankedTensorType::get(
      {sequenceRows, extent.validRows}, rewriter.getF32Type());
  mlir::RankedTensorType sourceVectorType =
      mlir::RankedTensorType::get({1, extent.validCols}, rewriter.getF32Type());
  mlir::RankedTensorType physicalVectorType =
      mlir::RankedTensorType::get({1, arrayCols}, rewriter.getF32Type());
  mlir::RankedTensorType storedRowType =
      mlir::RankedTensorType::get({1, extent.validRows}, rewriter.getF32Type());

  SemanticOperationScope scope(
      rewriter, semantic_operation_names::kConvTileMVMTaskKind,
      buildSequenceShardName(
          rewriter, buildMVMName(rewriter, sequence.mvm, tileRow, vectorTile),
          sequenceShardIndex)
          .getValue());
  copyMappingIdentity(sequence.mvm, scope);
  scope.set("sculptor.sequence_shard_offset",
            rewriter.getI64IntegerAttr(sequenceOffset));
  scope.set("sculptor.sequence_shard_rows",
            rewriter.getI64IntegerAttr(sequenceRows));
  scope.set("sculptor.sequence_shard_index",
            rewriter.getI64IntegerAttr(sequenceShardIndex));
  scope.set("sculptor.sequence_shard_count",
            rewriter.getI64IntegerAttr(sequenceShardCount));
  setArrayExecutionAttrs(scope, spec, tileRow, vectorTile, vectorTile,
                         arrayRows, arrayCols, rewriter);
  mlir::Value zero =
      rewriter.create<mlir::arith::ConstantIndexOp>(sequence.getLoc(), 0);
  mlir::Value one =
      rewriter.create<mlir::arith::ConstantIndexOp>(sequence.getLoc(), 1);
  mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
      sequence.getLoc(), sequenceRows);
  mlir::Value partialInit = rewriter.create<mlir::tensor::EmptyOp>(
      sequence.getLoc(), partialType.getShape(), partialType.getElementType());

  auto rowLoop = rewriter.create<mlir::scf::ForOp>(
      sequence.getLoc(), zero, upper, one, mlir::ValueRange{partialInit},
      [&](mlir::OpBuilder &loopBuilder, mlir::Location loopLoc, mlir::Value row,
          mlir::ValueRange iterArgs) {
        auto sourceSlice = loopBuilder.create<mlir::tensor::ExtractSliceOp>(
            loopLoc, sourceVectorType, vectorShard,
            buildRowOffsets(loopBuilder, row, 0),
            buildIndexAttrs(loopBuilder, {1, extent.validCols}),
            buildIndexAttrs(loopBuilder, {1, 1}));

        mlir::Value vector = sourceSlice.getResult();
        if (extent.validCols != arrayCols)
          vector =
              createPaddedPhysicalVector(loopLoc, vector, physicalVectorType,
                                         extent.validCols, loopBuilder);

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

static mlir::FailureOr<mlir::Value>
createSequenceVectorShard(MVMSequenceMatch &sequence, int64_t sequenceOffset,
                          int64_t sequenceRows, int64_t columnOffset,
                          int64_t columns, int64_t sequenceShardIndex,
                          int64_t vectorTile, mlir::RewriterBase &rewriter) {
  auto shardType = mlir::RankedTensorType::get({sequenceRows, columns},
                                               rewriter.getF32Type());
  mlir::Operation *producer = sequence.vectors.getDefiningOp();
  auto section = producer ? producer->getAttrOfType<mlir::StringAttr>(
                                "sculptor.semantic.section")
                          : mlir::StringAttr{};
  auto tiling = producer ? llvm::dyn_cast<mlir::TilingInterface>(producer)
                         : mlir::TilingInterface{};
  if (producer && tiling && section &&
      section.getValue() == semantic_operation_names::kConvPatchTaskKind &&
      producer->getNumResults() == 1 &&
      producer->getResult(0) == sequence.vectors) {
    llvm::SmallVector<mlir::OpFoldResult> offsets =
        buildIndexAttrs(rewriter, {sequenceOffset, columnOffset});
    llvm::SmallVector<mlir::OpFoldResult> sizes =
        buildIndexAttrs(rewriter, {sequenceRows, columns});
    auto tiled = tiling.getTiledImplementation(rewriter, offsets, sizes);
    if (mlir::failed(tiled) || tiled->tiledValues.size() != 1 ||
        tiled->tiledOps.empty() ||
        tiled->tiledValues.front().getType() != shardType) {
      return producer->emitError(
                 "failed to tile convolution patch producer to the exact "
                 "physical-MVM sequence shard"),
             mlir::failure();
    }

    // The patch generic overwrites every element of its destination.  The
    // generic TilingInterface nevertheless forms its tiled destination as a
    // slice of the original tensor.empty.  Keeping that false read creates an
    // uninitialized runtime resource and an otherwise empty producer task.
    // Rebase the tiled op onto an exact fresh destination before mapping.
    auto tiledGeneric =
        llvm::dyn_cast<mlir::linalg::GenericOp>(tiled->tiledOps.front());
    if (!tiledGeneric || tiledGeneric.getOutputs().size() != 1) {
      return producer->emitError(
                 "expected tiled convolution patch producer to remain one "
                 "linalg.generic"),
             mlir::failure();
    }
    mlir::Value oldDestination = tiledGeneric.getOutputs().front();
    rewriter.setInsertionPoint(tiledGeneric);
    mlir::Value freshDestination = rewriter.create<mlir::tensor::EmptyOp>(
        sequence.getLoc(), shardType.getShape(), shardType.getElementType());
    tiledGeneric.getOutputsMutable().assign(freshDestination);
    if (oldDestination.use_empty()) {
      auto slice = oldDestination.getDefiningOp<mlir::tensor::ExtractSliceOp>();
      if (slice)
        rewriter.eraseOp(slice);
    }
    rewriter.setInsertionPointAfter(tiledGeneric);

    mlir::StringAttr originalName =
        producer->getAttrOfType<mlir::StringAttr>("sculptor.semantic.name");
    llvm::StringRef baseName =
        originalName ? originalName.getValue() : section.getValue();
    mlir::StringAttr shardName =
        rewriter.getStringAttr((llvm::Twine(baseName) + "_sequence_shard_" +
                                llvm::Twine(sequenceShardIndex) +
                                "_vector_tile_" + llvm::Twine(vectorTile))
                                   .str());
    for (mlir::Operation *operation : tiled->tiledOps) {
      operation->setAttr("sculptor.semantic.section", section);
      operation->setAttr("sculptor.semantic.name", shardName);
      operation->setAttr(
          golem_tiling_attrs::kSourceMVMIdAttrName,
          rewriter.getI64IntegerAttr(getSourceMVMId(sequence.mvm)));
      operation->setAttr("sculptor.sequence_shard_offset",
                         rewriter.getI64IntegerAttr(sequenceOffset));
      operation->setAttr("sculptor.sequence_shard_rows",
                         rewriter.getI64IntegerAttr(sequenceRows));
      operation->setAttr("sculptor.sequence_shard_index",
                         rewriter.getI64IntegerAttr(sequenceShardIndex));
      operation->setAttr("sculptor.vector_tile",
                         rewriter.getI64IntegerAttr(vectorTile));
    }
    return tiled->tiledValues.front();
  }

  return rewriter
      .create<mlir::tensor::ExtractSliceOp>(
          sequence.getLoc(), shardType, sequence.vectors,
          buildIndexAttrs(rewriter, {sequenceOffset, columnOffset}),
          buildIndexAttrs(rewriter, {sequenceRows, columns}),
          buildIndexAttrs(rewriter, {1, 1}))
      .getResult();
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
createSequenceArrayExecutions(MVMSequenceMatch &sequence,
                              const MatrixPartitionSpec &spec,
                              llvm::ArrayRef<mlir::Value> logicalArrays,
                              int64_t sequenceOffset, int64_t sequenceRows,
                              int64_t sequenceShardIndex,
                              int64_t sequenceShardCount, int64_t arrayRows,
                              int64_t arrayCols, mlir::RewriterBase &rewriter) {
  if (static_cast<int64_t>(logicalArrays.size()) !=
      spec.gridRows * spec.gridCols) {
    return sequence.emitError(
               "internal error: mismatched logical-array tile count for "
               "MVM sequence expansion"),
           mlir::failure();
  }

  llvm::SmallVector<mlir::Value> partials(spec.gridRows * spec.gridCols);
  for (int64_t vectorTile = 0; vectorTile < spec.gridCols; ++vectorTile) {
    MatrixTileExtent vectorExtent = getMatrixTileExtent(
        spec, /*tileRow=*/0, vectorTile, arrayRows, arrayCols);
    auto vectorShard = createSequenceVectorShard(
        sequence, sequenceOffset, sequenceRows, vectorTile * arrayCols,
        vectorExtent.validCols, sequenceShardIndex, vectorTile, rewriter);
    if (mlir::failed(vectorShard))
      return mlir::failure();
    for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
      int64_t index = getTileIndex(spec, tileRow, vectorTile);
      partials[index] = createSequenceArrayExecution(
          sequence, spec, *vectorShard, logicalArrays[index], tileRow,
          vectorTile, sequenceOffset, sequenceRows, sequenceShardIndex,
          sequenceShardCount, arrayRows, arrayCols, rewriter);
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
  llvm::SmallVector<mlir::Value> rowPartials;
  rowPartials.reserve(spec.gridCols);
  for (int64_t tileCol = 0; tileCol < spec.gridCols; ++tileCol)
    rowPartials.push_back(partials[getTileIndex(spec, tileRow, tileCol)]);
  return sumEqualShapePartials(sequence.getLoc(), rowType, rowPartials,
                               rewriter);
}

static mlir::FailureOr<mlir::Value> createRecombinedMVMSequenceResult(
    MVMSequenceMatch &sequence, const MatrixPartitionSpec &spec,
    llvm::ArrayRef<mlir::Value> partials, int64_t sequenceOffset,
    int64_t sequenceRows, int64_t sequenceShardIndex,
    int64_t sequenceShardCount, int64_t arrayRows, int64_t arrayCols,
    const PointwiseEpilogueMatch *epilogue, mlir::RewriterBase &rewriter) {
  if (static_cast<int64_t>(partials.size()) != spec.gridRows * spec.gridCols) {
    return sequence.emitError(
               "internal error: mismatched partial tile count for MVM "
               "sequence recombination"),
           mlir::failure();
  }

  SemanticOperationScope scope(
      rewriter, semantic_operation_names::kTileRecombineTaskKind,
      buildSequenceShardName(rewriter,
                             buildTileRecombineName(rewriter, sequence.mvm),
                             sequenceShardIndex)
          .getValue());
  copyMappingIdentity(sequence.mvm, scope);
  scope.set("sculptor.sequence_shard_offset",
            rewriter.getI64IntegerAttr(sequenceOffset));
  scope.set("sculptor.sequence_shard_rows",
            rewriter.getI64IntegerAttr(sequenceRows));
  scope.set("sculptor.sequence_shard_index",
            rewriter.getI64IntegerAttr(sequenceShardIndex));
  scope.set("sculptor.sequence_shard_count",
            rewriter.getI64IntegerAttr(sequenceShardCount));
  if (spec.gridRows == 1 && spec.gridCols > 1) {
    scope.set(mlir::sculptor::task_graph_attrs::kTaskReductionAttrName,
              mlir::sculptor::TaskReductionAttr::get(
                  rewriter.getContext(), mlir::sculptor::TaskReductionKind::Add,
                  rewriter.getBoolAttr(true)));
  }

  llvm::SmallVector<mlir::Value> rowResults;
  llvm::SmallVector<mlir::Operation *> fusedOperations;
  rowResults.reserve(spec.gridRows);
  int64_t recombinedWidth = 0;
  for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
    MatrixTileExtent extent =
        getMatrixTileExtent(spec, tileRow, 0, arrayRows, arrayCols);
    mlir::RankedTensorType rowType = mlir::RankedTensorType::get(
        {sequenceRows, extent.validRows}, rewriter.getF32Type());
    mlir::Value rowResult = sumSequenceRowPartials(sequence, spec, partials,
                                                   tileRow, rowType, rewriter);
    if (epilogue)
      rowResult = applyPointwiseEpilogueToShard(
          *epilogue, rowResult, recombinedWidth, extent.validRows,
          fusedOperations, rewriter);
    rowResults.push_back(rowResult);
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
    mlir::RankedTensorType shardResultType = mlir::RankedTensorType::get(
        {sequenceRows, sequence.resultSequenceType.getDimSize(1)},
        rewriter.getF32Type());
    recombined = rewriter
                     .create<mlir::tensor::ConcatOp>(
                         sequence.getLoc(), shardResultType,
                         /*dim=*/1, mlir::ValueRange(rowResults))
                     .getResult();
  }

  scope.annotate();
  for (mlir::Operation *operation : fusedOperations)
    operation->removeAttr(
        mlir::sculptor::task_graph_attrs::kTaskReductionAttrName);
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
  llvm::SmallVector<mlir::Value> rowPartials;
  rowPartials.reserve(spec.gridCols);
  for (int64_t tileCol = 0; tileCol < spec.gridCols; ++tileCol)
    rowPartials.push_back(partialTiles[getTileIndex(spec, tileRow, tileCol)]);
  return sumEqualShapePartials(mvmOp.getLoc(), rowTileType, rowPartials,
                               rewriter);
}

static mlir::FailureOr<mlir::Value> createRecombinedMVMResult(
    mlir::sculptor::MVMOp mvmOp, const MatrixPartitionSpec &spec,
    llvm::ArrayRef<mlir::Value> partialTiles, int64_t arrayRows,
    int64_t arrayCols, const PointwiseEpilogueMatch *epilogue,
    mlir::RewriterBase &rewriter) {
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
    scope.set(mlir::sculptor::task_graph_attrs::kTaskReductionAttrName,
              mlir::sculptor::TaskReductionAttr::get(
                  rewriter.getContext(), mlir::sculptor::TaskReductionKind::Add,
                  rewriter.getBoolAttr(true)));
  }

  llvm::SmallVector<mlir::Value> rowResults;
  llvm::SmallVector<mlir::Operation *> fusedOperations;
  rowResults.reserve(spec.gridRows);
  int64_t recombinedWidth = 0;
  for (int64_t tileRow = 0; tileRow < spec.gridRows; ++tileRow) {
    MatrixTileExtent extent =
        getMatrixTileExtent(spec, tileRow, 0, arrayRows, arrayCols);
    mlir::RankedTensorType rowTileType = mlir::RankedTensorType::get(
        {1, extent.validRows}, rewriter.getF32Type());
    mlir::Value rowResult = sumRowPartials(mvmOp, spec, partialTiles, tileRow,
                                           rowTileType, rewriter);
    if (epilogue)
      rowResult = applyPointwiseEpilogueToShard(
          *epilogue, rowResult, recombinedWidth, extent.validRows,
          fusedOperations, rewriter);
    rowResults.push_back(rowResult);
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
  for (mlir::Operation *operation : fusedOperations)
    operation->removeAttr(
        mlir::sculptor::task_graph_attrs::kTaskReductionAttrName);
  return recombined;
}

struct MVMExpansionWalker {
  MVMExpansionWalker(int64_t arrayRows, int64_t arrayCols,
                     int64_t sequenceShardRows, int64_t sequenceShardBytes,
                     bool fusePointwiseEpilogues)
      : arrayRows(arrayRows), arrayCols(arrayCols),
        sequenceShardRows(sequenceShardRows),
        sequenceShardBytes(sequenceShardBytes),
        fusePointwiseEpilogues(fusePointwiseEpilogues) {}

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
  int64_t sequenceShardRows;
  int64_t sequenceShardBytes;
  bool fusePointwiseEpilogues;

  mlir::LogicalResult walkFunction(mlir::func::FuncOp func) {
    FunctionExpansionState state;
    llvm::SmallVector<mlir::sculptor::MVMOp> sourceMVMOps;
    func.walk(
        [&](mlir::sculptor::MVMOp mvmOp) { sourceMVMOps.push_back(mvmOp); });
    mlir::Builder builder(func.getContext());
    for (auto [sourceMVMIndex, mvmOp] : llvm::enumerate(sourceMVMOps)) {
      int64_t sourceMVMId = static_cast<int64_t>(sourceMVMIndex);
      state.sourceMVMIds[mvmOp.getOperation()] = sourceMVMId;
      mvmOp->setAttr(golem_tiling_attrs::kSourceMVMIdAttrName,
                     builder.getI64IntegerAttr(sourceMVMId));
    }

    if (failed(indexExistingLogicalArrays(func, state.logicalArrays)))
      return mlir::failure();

    llvm::SmallVector<MVMSequenceMatch> sequences;
    for (mlir::Operation &op : func.front().without_terminator()) {
      auto loop = llvm::dyn_cast<mlir::scf::ForOp>(&op);
      if (!loop)
        continue;
      auto section =
          loop->getAttrOfType<mlir::StringAttr>("sculptor.semantic.section");
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
    std::optional<PointwiseEpilogueMatch> epilogue;
    if (fusePointwiseEpilogues)
      epilogue = matchPointwiseEpilogue(sequence.anchor->getResult(0),
                                        sequence.mvm.getOperation());
    auto logicalArrays = getOrCreateLogicalArrays(
        **spec, state.logicalArrays, arrayRows, arrayCols, rewriter);
    if (failed(logicalArrays))
      return mlir::failure();

    int64_t rowsPerShard = sequence.sequenceLength;
    if (sequenceShardRows > 0) {
      rowsPerShard = std::min(sequenceShardRows, sequence.sequenceLength);
    } else if (sequenceShardBytes > 0) {
      int64_t partialColumns = 0;
      for (const mapping::GolemMVMTile &tile : (*spec)->physicalPlan.tiles) {
        std::optional<int64_t> next =
            llvm::checkedAdd(partialColumns, tile.validRows);
        if (!next)
          return sequence.emitError(
              "MVM sequence per-row partial width overflows int64");
        partialColumns = *next;
      }
      // The shard budget bounds every cross-routine tensor that one sequence
      // wave can keep live: its source vector/patch, all physical-MVM
      // partials, and the recombined result.  Omitting the source width made
      // the option appear to cap memory while large convolution patches alone
      // could exceed the requested budget.
      std::optional<int64_t> liveColumns = llvm::checkedAdd(
          sequence.vectorSequenceType.getDimSize(1), partialColumns);
      if (liveColumns)
        liveColumns = llvm::checkedAdd(
            *liveColumns, sequence.resultSequenceType.getDimSize(1));
      std::optional<int64_t> bytesPerRow =
          liveColumns ? llvm::checkedMul(*liveColumns, int64_t{4})
                      : std::nullopt;
      if (!bytesPerRow || *bytesPerRow <= 0)
        return sequence.emitError(
            "MVM sequence per-row live-byte estimate overflows int64");
      rowsPerShard =
          std::max<int64_t>(1, std::min(sequence.sequenceLength,
                                        sequenceShardBytes / *bytesPerRow));
    }
    int64_t shardCount =
        (sequence.sequenceLength + rowsPerShard - 1) / rowsPerShard;
    mlir::Operation *vectorProducer = sequence.vectors.getDefiningOp();
    llvm::SmallVector<mlir::Value> shardResults;
    shardResults.reserve(shardCount);
    rewriter.setInsertionPoint(sequence.getInsertionAnchor());
    for (int64_t shardIndex = 0; shardIndex < shardCount; ++shardIndex) {
      int64_t shardOffset = shardIndex * rowsPerShard;
      int64_t shardRows =
          std::min(rowsPerShard, sequence.sequenceLength - shardOffset);
      auto partials = createSequenceArrayExecutions(
          sequence, **spec, *logicalArrays, shardOffset, shardRows, shardIndex,
          shardCount, arrayRows, arrayCols, rewriter);
      if (failed(partials))
        return mlir::failure();

      auto recombined = createRecombinedMVMSequenceResult(
          sequence, **spec, *partials, shardOffset, shardRows, shardIndex,
          shardCount, arrayRows, arrayCols, epilogue ? &*epilogue : nullptr,
          rewriter);
      if (failed(recombined))
        return mlir::failure();
      shardResults.push_back(*recombined);
      rewriter.setInsertionPointAfter(recombined->getDefiningOp());
    }

    mlir::Value result = shardResults.front();
    if (shardResults.size() > 1) {
      SemanticOperationScope scope(rewriter, "digital.sequence_assembly",
                                   "mvm_sequence_shard_assembly");
      copyMappingIdentity(sequence.mvm, scope);
      result = rewriter
                   .create<mlir::tensor::ConcatOp>(
                       sequence.getLoc(), sequence.resultSequenceType,
                       /*dim=*/0, mlir::ValueRange(shardResults))
                   .getResult();
      scope.annotate();
    }

    if (epilogue) {
      result = restoreEpilogueResultShape(*epilogue, result, rewriter);
      epilogue->getResult().replaceAllUsesWith(result);
      erasePointwiseEpilogue(*epilogue, rewriter);
    } else {
      sequence.anchor->getResult(0).replaceAllUsesWith(result);
    }
    for (mlir::Operation *member : llvm::reverse(sequence.members))
      rewriter.eraseOp(member);
    if (vectorProducer && vectorProducer->use_empty()) {
      auto producerSection = vectorProducer->getAttrOfType<mlir::StringAttr>(
          "sculptor.semantic.section");
      if (producerSection &&
          producerSection.getValue() ==
              semantic_operation_names::kConvPatchTaskKind &&
          llvm::isa<mlir::linalg::LinalgOp>(vectorProducer)) {
        llvm::SmallVector<mlir::Value> producerOperands(
            vectorProducer->getOperands());
        rewriter.eraseOp(vectorProducer);
        for (mlir::Value operand : producerOperands) {
          auto empty = operand.getDefiningOp<mlir::tensor::EmptyOp>();
          if (empty && empty->use_empty())
            rewriter.eraseOp(empty);
        }
      }
    }
    return mlir::success();
  }

  mlir::FailureOr<MatrixPartitionSpec *>
  getOrCreateMatrixSpec(FunctionExpansionState &state,
                        mlir::sculptor::MVMOp op) {
    auto matrixOperand = matchMatrixOperand(op);
    if (failed(matrixOperand))
      return mlir::failure();
    state.matrixConstants.push_back(matrixOperand->constant);

    auto sourceMVMId = state.sourceMVMIds.find(op.getOperation());
    if (sourceMVMId == state.sourceMVMIds.end())
      return op.emitError("source MVM has no expansion identity"),
             mlir::failure();

    llvm::StringRef sourceResource = matrixOperand->sourceResource;
    if (sourceResource.empty()) {
      auto [resource, inserted] = state.inlineMatrixResources.try_emplace(
          matrixOperand->constant.getOperation());
      if (inserted)
        resource->second =
            "inline_dense_matrix_" +
            std::to_string(state.inlineMatrixResources.size() - 1);
      sourceResource = resource->second;
    }

    auto existing = state.matrixSpecs.find(sourceResource);
    if (existing != state.matrixSpecs.end()) {
      if (existing->second.type != matrixOperand->type)
        return op.emitError(
                   "one matrix resource has inconsistent tensor types"),
               mlir::failure();
      return &existing->second;
    }

    auto spec =
        buildMatrixPartitionSpec(op, *matrixOperand, arrayRows, arrayCols,
                                 sourceMVMId->second, sourceResource);
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
    std::optional<PointwiseEpilogueMatch> epilogue;
    if (fusePointwiseEpilogues)
      epilogue = matchPointwiseEpilogue(op.getResult(), op.getOperation());
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

    auto recombined = createRecombinedMVMResult(
        op, **spec, *partialTiles, arrayRows, arrayCols,
        epilogue ? &*epilogue : nullptr, rewriter);
    if (failed(recombined))
      return mlir::failure();

    mlir::Value result = *recombined;
    if (epilogue) {
      result = restoreEpilogueResultShape(*epilogue, result, rewriter);
      epilogue->getResult().replaceAllUsesWith(result);
      erasePointwiseEpilogue(*epilogue, rewriter);
    } else {
      op.getResult().replaceAllUsesWith(result);
    }
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
  if (sequenceShardRows < 0 || sequenceShardBytes < 0) {
    getOperation().emitError(
        "sequence-shard-rows and sequence-shard-bytes must be nonnegative");
    signalPassFailure();
    return;
  }
  if (sequenceShardRows > 0 && sequenceShardBytes > 0) {
    getOperation().emitError(
        "sequence-shard-rows and sequence-shard-bytes are mutually exclusive");
    signalPassFailure();
    return;
  }

  MVMExpansionWalker walker(arrayRows, arrayCols, sequenceShardRows,
                            sequenceShardBytes, fusePointwiseEpilogues);
  if (failed(walker.run(getOperation())))
    signalPassFailure();
}

void registerExpandMVMToGolemPass() {
  mlir::PassRegistration<ExpandMVMToGolemPass>();
}

} // namespace sculptor
} // namespace mlir
