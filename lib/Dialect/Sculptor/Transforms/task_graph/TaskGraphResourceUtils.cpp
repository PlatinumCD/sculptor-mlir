#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResourceUtils.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include <climits>
#include <limits>

namespace {

mlir::FailureOr<int64_t> getStaticPayloadByteSize(mlir::Type payloadType) {
  if (llvm::isa<mlir::sculptor::RuntimeHandleType,
                mlir::sculptor::LogicalArrayType>(payloadType))
    return int64_t{0};

  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(payloadType)) {
    unsigned bitWidth = floatType.getWidth();
    if (bitWidth == 0)
      return mlir::failure();
    return static_cast<int64_t>(llvm::divideCeil(bitWidth, CHAR_BIT));
  }

  auto shapedType = llvm::dyn_cast<mlir::ShapedType>(payloadType);
  if (!shapedType || !shapedType.hasStaticShape() ||
      !shapedType.getElementType().isF32())
    return mlir::failure();

  int64_t elementCount = shapedType.getNumElements();
  constexpr int64_t kF32ByteSize = sizeof(float);
  if (elementCount < 0 ||
      elementCount > std::numeric_limits<int64_t>::max() / kF32ByteSize)
    return mlir::failure();

  return elementCount * kF32ByteSize;
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
