#include "sculptor-mlir/Dialect/Sculptor/Transforms/VectorizeTileCopies.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <functional>
#include <map>
#include <optional>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
namespace tile_memory = mlir::sculptor::tile_memory;

constexpr StringLiteral kAuditAttrName = "sculptor.memory.bufferization_audit";
constexpr StringLiteral kSummaryAttrName =
    "sculptor.memory.vectorized_copy_summary";

struct CopyClassification {
  memref::CopyOp copy;
  int64_t ordinal = -1;
  std::string routine;
  std::string auditClass = "unclassified";
  std::string materializationKind = "bufferization_copy";
  std::string geometry = "unknown";
  std::string result = "fallback";
  std::string reason;
  int64_t staticBytes = -1;
  int64_t vectorLanes = 0;
  bool maskedTail = false;
  bool rankZeroScalar = false;
};

std::optional<int64_t> getStaticByteSize(MemRefType type) {
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat())
    return std::nullopt;
  unsigned bitWidth = type.getElementTypeBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0 || type.getNumElements() < 0)
    return std::nullopt;
  return llvm::checkedMul(type.getNumElements(),
                          static_cast<int64_t>(bitWidth / 8));
}

bool isAliasOperation(Operation *operation) {
  if (!operation)
    return false;
  StringRef name = operation->getName().getStringRef();
  return name == "memref.cast" || name == "memref.subview" ||
         name == "memref.reinterpret_cast" || name == "memref.view" ||
         name == "memref.expand_shape" || name == "memref.collapse_shape";
}

Value getRootValue(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    if (!isAliasOperation(definition) || definition->getNumOperands() == 0)
      break;
    value = definition->getOperand(0);
  }
  return value;
}

IntegerAttr getArgumentOwnerId(BlockArgument argument) {
  auto function =
      dyn_cast<FunctionOpInterface>(argument.getOwner()->getParentOp());
  if (!function)
    return {};
  return function.getArgAttrOfType<IntegerAttr>(argument.getArgNumber(),
                                                tile_memory::kOwnerIdAttrName);
}

bool isFreshAllocation(Value value) {
  Operation *definition = value.getDefiningOp();
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(definition);
}

bool canProveNoOverlap(Value source, Value target) {
  Value sourceRoot = getRootValue(source);
  Value targetRoot = getRootValue(target);
  if (sourceRoot == targetRoot)
    return false;
  if (isFreshAllocation(sourceRoot) || isFreshAllocation(targetRoot))
    return true;

  auto sourceArgument = dyn_cast<BlockArgument>(sourceRoot);
  auto targetArgument = dyn_cast<BlockArgument>(targetRoot);
  if (sourceArgument && targetArgument) {
    IntegerAttr sourceOwner = getArgumentOwnerId(sourceArgument);
    IntegerAttr targetOwner = getArgumentOwnerId(targetArgument);
    return sourceOwner && targetOwner && sourceOwner != targetOwner;
  }

  auto sourceGlobal = sourceRoot.getDefiningOp<memref::GetGlobalOp>();
  auto targetGlobal = targetRoot.getDefiningOp<memref::GetGlobalOp>();
  return sourceGlobal && targetGlobal &&
         sourceGlobal.getName() != targetGlobal.getName();
}

FailureOr<SmallVector<int64_t>> getStaticStrides(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset) ||
      llvm::is_contained(strides, ShapedType::kDynamic))
    return failure();
  return strides;
}

bool isFullyContiguous(MemRefType type, ArrayRef<int64_t> strides) {
  int64_t expected = 1;
  for (int64_t dimension = type.getRank() - 1; dimension >= 0; --dimension) {
    if (strides[dimension] != expected)
      return false;
    std::optional<int64_t> next =
        llvm::checkedMul(expected, type.getDimSize(dimension));
    if (!next)
      return false;
    expected = *next;
  }
  return true;
}

std::map<std::pair<std::string, int64_t>, std::string>
readAuditClasses(ModuleOp module) {
  std::map<std::pair<std::string, int64_t>, std::string> result;
  auto audit = module->getAttrOfType<DictionaryAttr>(kAuditAttrName);
  auto copies = audit ? audit.getAs<ArrayAttr>("copies") : ArrayAttr();
  if (!copies)
    return result;
  for (Attribute value : copies) {
    auto record = dyn_cast<DictionaryAttr>(value);
    auto routine = record ? record.getAs<StringAttr>("routine") : StringAttr();
    auto ordinal =
        record ? record.getAs<IntegerAttr>("ordinal") : IntegerAttr();
    auto copyClass = record ? record.getAs<StringAttr>("class") : StringAttr();
    if (routine && ordinal && copyClass)
      result[{routine.getValue().str(), ordinal.getInt()}] =
          copyClass.getValue().str();
  }
  return result;
}

DenseMap<int64_t, MemoryOwnerKind> readOwnerKinds(ModuleOp module) {
  DenseMap<int64_t, MemoryOwnerKind> result;
  auto owners = module->getAttrOfType<ArrayAttr>(tile_memory::kOwnersAttrName);
  if (!owners)
    return result;
  for (Attribute value : owners)
    if (auto owner = dyn_cast<TileMemoryOwnerAttr>(value))
      result[owner.getId().getInt()] = owner.getKind();
  return result;
}

std::optional<MemoryOwnerKind>
getRootOwnerKind(Value value,
                 const DenseMap<int64_t, MemoryOwnerKind> &ownerKinds) {
  auto argument = dyn_cast<BlockArgument>(getRootValue(value));
  if (!argument)
    return std::nullopt;
  IntegerAttr owner = getArgumentOwnerId(argument);
  if (!owner)
    return std::nullopt;
  auto found = ownerKinds.find(owner.getInt());
  if (found == ownerKinds.end())
    return std::nullopt;
  return found->second;
}

bool isRouteOwner(std::optional<MemoryOwnerKind> kind) {
  return kind && (*kind == MemoryOwnerKind::RouteInput ||
                  *kind == MemoryOwnerKind::RouteOutput);
}

std::string
classifyMaterialization(const CopyClassification &classification,
                        const DenseMap<int64_t, MemoryOwnerKind> &ownerKinds) {
  if (classification.auditClass == "planned_assembly")
    return "assembly_pack";
  if (classification.auditClass == "planned_boot_staging")
    return "boot_staging";
  if (isRouteOwner(
          getRootOwnerKind(classification.copy->getOperand(0), ownerKinds)) ||
      isRouteOwner(
          getRootOwnerKind(classification.copy->getOperand(1), ownerKinds)))
    return "route_pack";
  return "bufferization_copy";
}

CopyClassification classifyCopy(
    memref::CopyOp copy, int64_t ordinal, int64_t vectorBits,
    const std::map<std::pair<std::string, int64_t>, std::string> &auditClasses,
    const DenseMap<int64_t, MemoryOwnerKind> &ownerKinds) {
  CopyClassification classification;
  classification.copy = copy;
  classification.ordinal = ordinal;
  if (func::FuncOp function = copy->getParentOfType<func::FuncOp>())
    classification.routine = function.getName().str();
  auto auditClass =
      auditClasses.find({classification.routine, classification.ordinal});
  if (auditClass != auditClasses.end())
    classification.auditClass = auditClass->second;
  classification.materializationKind =
      classifyMaterialization(classification, ownerKinds);

  auto sourceType = dyn_cast<MemRefType>(copy.getSource().getType());
  auto targetType = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!sourceType || !targetType) {
    classification.reason = "unranked_memref";
    classification.geometry = "unsupported_layout";
    return classification;
  }
  if (!sourceType.hasStaticShape() || !targetType.hasStaticShape()) {
    classification.reason = "dynamic_shape";
    classification.geometry = "dynamic";
    return classification;
  }
  if (sourceType.getShape() != targetType.getShape()) {
    classification.reason = "incompatible_shape";
    classification.geometry = "incompatible_shard_geometry";
    return classification;
  }
  if (sourceType.getElementType() != targetType.getElementType()) {
    classification.reason = "incompatible_element_type";
    classification.geometry = "incompatible_shard_geometry";
    return classification;
  }
  classification.staticBytes = getStaticByteSize(sourceType).value_or(-1);
  if (!sourceType.getElementType().isIntOrFloat()) {
    classification.reason = "unsupported_element_type";
    classification.geometry = "unsupported_layout";
    return classification;
  }
  if (llvm::any_of(sourceType.getShape(),
                   [](int64_t size) { return size <= 0; })) {
    classification.reason = "zero_extent";
    classification.geometry = "static_empty";
    return classification;
  }

  auto sourceStrides = getStaticStrides(sourceType);
  auto targetStrides = getStaticStrides(targetType);
  if (failed(sourceStrides) || failed(targetStrides)) {
    classification.reason = "dynamic_or_unsupported_strides";
    classification.geometry = "unsupported_layout";
    return classification;
  }
  if (sourceType.getRank() > 0 &&
      (sourceStrides->back() != 1 || targetStrides->back() != 1)) {
    classification.reason = "non_unit_innermost_stride";
    classification.geometry = "non_contiguous";
    return classification;
  }
  classification.geometry =
      isFullyContiguous(sourceType, *sourceStrides) &&
              isFullyContiguous(targetType, *targetStrides)
          ? "contiguous"
          : "row_contiguous";

  if (!canProveNoOverlap(copy.getSource(), copy.getTarget())) {
    classification.reason = "possible_alias_or_overlap";
    return classification;
  }
  if (vectorBits <= 0) {
    classification.reason = "nonpositive_vector_width";
    return classification;
  }
  unsigned elementBits = sourceType.getElementTypeBitWidth();
  if (elementBits == 0 || vectorBits % elementBits != 0) {
    classification.reason = "incompatible_vector_width";
    return classification;
  }
  classification.vectorLanes = vectorBits / elementBits;
  if (sourceType.getRank() == 0) {
    classification.result = "scalar_rank_zero";
    classification.reason = "rank_zero";
    classification.rankZeroScalar = true;
    return classification;
  }
  if (classification.vectorLanes < 2) {
    classification.reason = "single_lane_vector";
    return classification;
  }
  classification.maskedTail =
      sourceType.getShape().back() % classification.vectorLanes != 0;
  classification.result = "vectorized";
  classification.reason.clear();
  return classification;
}

Value createIndexConstant(OpBuilder &builder, Location location,
                          int64_t value) {
  return builder.create<arith::ConstantIndexOp>(location, value);
}

void emitInnermostVectorCopy(OpBuilder &builder, Location location,
                             Value source, Value target,
                             SmallVector<Value> indices, int64_t innerSize,
                             int64_t vectorLanes, Type elementType) {
  VectorType vectorType = VectorType::get({vectorLanes}, elementType);
  int64_t fullEnd = innerSize - innerSize % vectorLanes;
  unsigned innerDimension = indices.size() - 1;

  if (fullEnd > 0) {
    Value lower = createIndexConstant(builder, location, 0);
    Value upper = createIndexConstant(builder, location, fullEnd);
    Value step = createIndexConstant(builder, location, vectorLanes);
    scf::ForOp loop = builder.create<scf::ForOp>(location, lower, upper, step);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    indices[innerDimension] = loop.getInductionVar();
    Value loaded =
        builder.create<vector::LoadOp>(location, vectorType, source, indices);
    builder.create<vector::StoreOp>(location, loaded, target, indices);
  }

  int64_t tail = innerSize - fullEnd;
  if (tail == 0)
    return;
  indices[innerDimension] = createIndexConstant(builder, location, fullEnd);
  Value tailSize = createIndexConstant(builder, location, tail);
  VectorType maskType = VectorType::get({vectorLanes}, builder.getI1Type());
  Value mask =
      builder.create<vector::CreateMaskOp>(location, maskType, tailSize);
  Value passThrough = builder.create<arith::ConstantOp>(
      location, vectorType, builder.getZeroAttr(vectorType));
  Value loaded = builder.create<vector::MaskedLoadOp>(
      location, vectorType, source, indices, mask, passThrough);
  builder.create<vector::MaskedStoreOp>(location, target, indices, mask,
                                        loaded);
}

void emitVectorCopy(memref::CopyOp copy, int64_t vectorLanes) {
  auto sourceType = cast<MemRefType>(copy.getSource().getType());
  OpBuilder builder(copy);
  Location location = copy.getLoc();
  SmallVector<Value> indices(sourceType.getRank());

  std::function<void(unsigned)> emitDimension = [&](unsigned dimension) {
    if (dimension + 1 == static_cast<unsigned>(sourceType.getRank())) {
      emitInnermostVectorCopy(builder, location, copy.getSource(),
                              copy.getTarget(), indices,
                              sourceType.getDimSize(dimension), vectorLanes,
                              sourceType.getElementType());
      return;
    }
    Value lower = createIndexConstant(builder, location, 0);
    Value upper = createIndexConstant(builder, location,
                                      sourceType.getDimSize(dimension));
    Value step = createIndexConstant(builder, location, 1);
    scf::ForOp loop = builder.create<scf::ForOp>(location, lower, upper, step);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    indices[dimension] = loop.getInductionVar();
    emitDimension(dimension + 1);
  };

  emitDimension(0);
  copy.erase();
}

void emitRankZeroCopy(memref::CopyOp copy) {
  OpBuilder builder(copy);
  Value value = builder.create<memref::LoadOp>(copy.getLoc(), copy.getSource());
  builder.create<memref::StoreOp>(copy.getLoc(), value, copy.getTarget());
  copy.erase();
}

DictionaryAttr buildCopyRecord(Builder &builder,
                               const CopyClassification &classification) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr("ordinal",
                           builder.getI64IntegerAttr(classification.ordinal)),
      builder.getNamedAttr("routine",
                           builder.getStringAttr(classification.routine)),
      builder.getNamedAttr("audit_class",
                           builder.getStringAttr(classification.auditClass)),
      builder.getNamedAttr(
          "materialization_kind",
          builder.getStringAttr(classification.materializationKind)),
      builder.getNamedAttr("geometry",
                           builder.getStringAttr(classification.geometry)),
      builder.getNamedAttr("result",
                           builder.getStringAttr(classification.result)),
      builder.getNamedAttr("reason",
                           builder.getStringAttr(classification.reason)),
      builder.getNamedAttr("static_bytes", builder.getI64IntegerAttr(
                                               classification.staticBytes)),
      builder.getNamedAttr("vector_lanes", builder.getI64IntegerAttr(
                                               classification.vectorLanes)),
      builder.getNamedAttr("masked_tail",
                           builder.getBoolAttr(classification.maskedTail)),
  });
}

LogicalResult attachSummary(ModuleOp module,
                            ArrayRef<CopyClassification> classifications,
                            int64_t vectorBits) {
  int64_t vectorizedCount = 0;
  int64_t vectorizedBytes = 0;
  int64_t maskedTailCount = 0;
  int64_t scalarRankZeroCount = 0;
  int64_t fallbackCount = 0;
  int64_t fallbackBytes = 0;
  int64_t unknownFallbackBytesCount = 0;
  int64_t assemblyPackCount = 0;
  int64_t routePackCount = 0;
  int64_t nonContiguousCount = 0;
  SmallVector<Attribute> records;
  Builder builder(module.getContext());

  for (const CopyClassification &classification : classifications) {
    records.push_back(buildCopyRecord(builder, classification));
    assemblyPackCount += classification.materializationKind == "assembly_pack";
    routePackCount += classification.materializationKind == "route_pack";
    nonContiguousCount += classification.geometry == "non_contiguous" ||
                          classification.geometry == "row_contiguous";
    if (classification.result == "vectorized") {
      ++vectorizedCount;
      maskedTailCount += classification.maskedTail;
      if (classification.staticBytes >= 0 &&
          !llvm::checkedAdd(vectorizedBytes, classification.staticBytes))
        return module.emitError("vectorized copy-byte accounting overflow");
      vectorizedBytes += classification.staticBytes;
    } else if (classification.rankZeroScalar) {
      ++scalarRankZeroCount;
    } else {
      ++fallbackCount;
      if (classification.staticBytes < 0)
        ++unknownFallbackBytesCount;
      else {
        std::optional<int64_t> sum =
            llvm::checkedAdd(fallbackBytes, classification.staticBytes);
        if (!sum)
          return module.emitError("fallback copy-byte accounting overflow");
        fallbackBytes = *sum;
      }
    }
  }

  auto i64 = [&](int64_t value) { return builder.getI64IntegerAttr(value); };
  module->setAttr(
      kSummaryAttrName,
      builder.getDictionaryAttr({
          builder.getNamedAttr("schema_version", i64(1)),
          builder.getNamedAttr("vector_bits", i64(vectorBits)),
          builder.getNamedAttr("copy_count", i64(classifications.size())),
          builder.getNamedAttr("vectorized_copy_count", i64(vectorizedCount)),
          builder.getNamedAttr("vectorized_copy_bytes", i64(vectorizedBytes)),
          builder.getNamedAttr("masked_tail_copy_count", i64(maskedTailCount)),
          builder.getNamedAttr("scalar_rank_zero_copy_count",
                               i64(scalarRankZeroCount)),
          builder.getNamedAttr("fallback_copy_count", i64(fallbackCount)),
          builder.getNamedAttr("fallback_copy_bytes", i64(fallbackBytes)),
          builder.getNamedAttr("unknown_fallback_bytes_count",
                               i64(unknownFallbackBytesCount)),
          builder.getNamedAttr("assembly_pack_count", i64(assemblyPackCount)),
          builder.getNamedAttr("route_pack_count", i64(routePackCount)),
          builder.getNamedAttr("non_contiguous_copy_count",
                               i64(nonContiguousCount)),
          builder.getNamedAttr("copies", builder.getArrayAttr(records)),
      }));
  return success();
}

} // namespace

namespace mlir::sculptor {

void VectorizeTileCopiesPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto auditClasses = readAuditClasses(module);
  auto ownerKinds = readOwnerKinds(module);
  SmallVector<memref::CopyOp> copies;
  module.walk([&](memref::CopyOp copy) { copies.push_back(copy); });

  SmallVector<CopyClassification> classifications;
  classifications.reserve(copies.size());
  for (auto [ordinal, copy] : llvm::enumerate(copies))
    classifications.push_back(
        classifyCopy(copy, ordinal, vectorBits, auditClasses, ownerKinds));

  if (failed(attachSummary(module, classifications, vectorBits))) {
    signalPassFailure();
    return;
  }
  for (CopyClassification &classification : classifications) {
    if (classification.result == "vectorized")
      emitVectorCopy(classification.copy, classification.vectorLanes);
    else if (classification.rankZeroScalar)
      emitRankZeroCopy(classification.copy);
  }
  if (failed(verify(module)))
    signalPassFailure();
}

void registerVectorizeTileCopiesPass() {
  PassRegistration<VectorizeTileCopiesPass>();
}

} // namespace mlir::sculptor
