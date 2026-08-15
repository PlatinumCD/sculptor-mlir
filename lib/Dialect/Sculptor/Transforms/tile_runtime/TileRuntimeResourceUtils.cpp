#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeResourceUtils.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <climits>
#include <limits>

namespace {

mlir::FailureOr<int64_t> getStaticPayloadByteSize(mlir::Type payloadType) {
  if (llvm::isa<mlir::sculptor::RuntimeHandleType,
                mlir::sculptor::LogicalArrayType>(payloadType))
    return int64_t{0};

  auto getElementByteSize = [](mlir::Type type) -> std::optional<int64_t> {
    if (!type.isIntOrFloat())
      return std::nullopt;
    unsigned bitWidth = type.getIntOrFloatBitWidth();
    if (bitWidth == 0 || bitWidth % CHAR_BIT != 0)
      return std::nullopt;
    return static_cast<int64_t>(bitWidth / CHAR_BIT);
  };

  if (std::optional<int64_t> bytes = getElementByteSize(payloadType))
    return *bytes;

  auto shapedType = llvm::dyn_cast<mlir::ShapedType>(payloadType);
  if (!shapedType || !shapedType.hasStaticShape())
    return mlir::failure();

  int64_t elementCount = shapedType.getNumElements();
  std::optional<int64_t> elementBytes =
      getElementByteSize(shapedType.getElementType());
  if (elementCount < 0 || !elementBytes)
    return mlir::failure();

  std::optional<int64_t> bytes = llvm::checkedMul(elementCount, *elementBytes);
  if (!bytes)
    return mlir::failure();
  return *bytes;
}

} // namespace

namespace mlir {
namespace sculptor {

FailureOr<int64_t> getTaskResourceByteSize(Value resource) {
  auto resourceType = llvm::dyn_cast<TaskResourceType>(resource.getType());
  if (!resourceType)
    return failure();

  return getStaticPayloadByteSize(resourceType.getValueType());
}

} // namespace sculptor
} // namespace mlir
