#include "sculptor-mlir/Dialect/Sculptor/Transforms/DistributeDigitalMatmul.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphMatmulDistributor.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "mlir/Pass/PassRegistry.h"

namespace {

static bool returnsTaskGraph(mlir::func::FuncOp func) {
  mlir::FunctionType type = func.getFunctionType();
  return type.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(type.getResult(0));
}

static mlir::FailureOr<
    std::optional<mlir::sculptor::MatmulDistributionStrategy>>
parseStrategy(mlir::Operation *anchor, llvm::StringRef value) {
  value = value.trim();
  if (value == "auto")
    return std::optional<mlir::sculptor::MatmulDistributionStrategy>();
  if (value == "output-columns")
    return std::optional<mlir::sculptor::MatmulDistributionStrategy>(
        mlir::sculptor::MatmulDistributionStrategy::OutputColumns);
  if (value == "output-rows")
    return std::optional<mlir::sculptor::MatmulDistributionStrategy>(
        mlir::sculptor::MatmulDistributionStrategy::OutputRows);
  if (value == "two-dimensional")
    return std::optional<mlir::sculptor::MatmulDistributionStrategy>(
        mlir::sculptor::MatmulDistributionStrategy::TwoDimensional);
  if (value == "attention-heads")
    return std::optional<mlir::sculptor::MatmulDistributionStrategy>(
        mlir::sculptor::MatmulDistributionStrategy::AttentionHeads);
  anchor->emitError("unknown digital matmul distribution strategy '")
      << value
      << "'; expected auto, output-columns, output-rows, two-dimensional, or "
         "attention-heads";
  return mlir::failure();
}

static mlir::FailureOr<mlir::sculptor::DistributionPlacementPolicy>
parsePlacementPolicy(mlir::Operation *anchor, llvm::StringRef value) {
  value = value.trim();
  if (value == "unconstrained")
    return mlir::sculptor::DistributionPlacementPolicy::Unconstrained;
  if (value == "prefer-distinct")
    return mlir::sculptor::DistributionPlacementPolicy::PreferDistinct;
  if (value == "require-distinct")
    return mlir::sculptor::DistributionPlacementPolicy::RequireDistinct;
  anchor->emitError("unknown digital matmul placement policy '")
      << value
      << "'; expected unconstrained, prefer-distinct, or require-distinct";
  return mlir::failure();
}

} // namespace

namespace mlir {
namespace sculptor {

DistributeDigitalMatmulPass::DistributeDigitalMatmulPass(
    const DistributeDigitalMatmulPass &pass)
    : PassWrapper(pass),
      maxShards(*this, "max-shards",
                llvm::cl::desc("Maximum number of digital matmul shards"),
                llvm::cl::init(8)),
      minOpsPerShard(
          *this, "min-ops-per-shard",
          llvm::cl::desc("Minimum scalar matmul operations in every shard"),
          llvm::cl::init(65536)),
      strategy(*this, "strategy",
               llvm::cl::desc(
                   "auto, output-columns, output-rows, two-dimensional, or "
                   "attention-heads"),
               llvm::cl::init("auto")),
      placementPolicy(
          *this, "placement-policy",
          llvm::cl::desc("unconstrained, prefer-distinct, or require-distinct"),
          llvm::cl::init("prefer-distinct")),
      requireChange(
          *this, "require-change",
          llvm::cl::desc("Fail when no eligible digital matmul is distributed"),
          llvm::cl::init(false)) {
  maxShards = pass.maxShards;
  minOpsPerShard = pass.minOpsPerShard;
  strategy = pass.strategy;
  placementPolicy = pass.placementPolicy;
  requireChange = pass.requireChange;
}

void DistributeDigitalMatmulPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (maxShards < 2) {
    module.emitError("expected max-shards to be at least two");
    signalPassFailure();
    return;
  }
  if (minOpsPerShard <= 0) {
    module.emitError("expected min-ops-per-shard to be positive");
    signalPassFailure();
    return;
  }

  auto parsedStrategy = parseStrategy(module, strategy);
  auto parsedPlacement = parsePlacementPolicy(module, placementPolicy);
  if (failed(parsedStrategy) || failed(parsedPlacement)) {
    signalPassFailure();
    return;
  }

  task_graph::DigitalMatmulDistributionOptions options;
  options.maxShards = maxShards;
  options.minOpsPerShard = minOpsPerShard;
  options.strategy = *parsedStrategy;
  options.placement = *parsedPlacement;

  bool foundTaskGraph = false;
  unsigned distributedCount = 0;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    foundTaskGraph = true;
    auto distributed =
        task_graph::distributeDigitalMatmuls(module, func, options);
    if (failed(distributed)) {
      signalPassFailure();
      return;
    }
    if (*distributed == 0)
      continue;
    distributedCount += *distributed;
    task_timing::invalidateTaskGraphStructure(func);
  }

  if (!foundTaskGraph) {
    module.emitError(
        "expected a task graph function when distributing digital matmuls");
    signalPassFailure();
    return;
  }
  if (requireChange && distributedCount == 0) {
    module.emitError("expected at least one eligible digital matmul");
    signalPassFailure();
  }
}

void registerDistributeDigitalMatmulPass() {
  PassRegistration<DistributeDigitalMatmulPass>();
}

} // namespace sculptor
} // namespace mlir
