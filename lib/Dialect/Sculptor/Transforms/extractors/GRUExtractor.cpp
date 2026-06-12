#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Extraction/RewriteUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <optional>
#include <string>

namespace rewrite_utils = mlir::sculptor::rewrite_utils;
namespace layer_utils = mlir::sculptor::layer_utils;

namespace {

using mlir::sculptor::NNGRULayerOp;
using mlir::sculptor::NNGRUOp;

struct CanonicalGRUInfo {
  mlir::RankedTensorType inputType;
  mlir::RankedTensorType hiddenStateType;
  mlir::RankedTensorType sequenceResultType;
  mlir::RankedTensorType hiddenResultType;
  mlir::RankedTensorType hiddenSliceType;
  int64_t layerCount = 0;
  int64_t batchSize = 0;
  int64_t sequenceLength = 0;
  int64_t hiddenSize = 0;
  bool hasBias = false;
};

static std::optional<mlir::RankedTensorType>
getRankedTensorType(mlir::Type type) {
  auto rankedType = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!rankedType || !rankedType.hasStaticShape())
    return std::nullopt;
  return rankedType;
}

static mlir::arith::ConstantOp getConstantDef(mlir::Value value) {
  return value ? layer_utils::producerOfType<mlir::arith::ConstantOp>(value)
               : mlir::arith::ConstantOp();
}

static std::optional<CanonicalGRUInfo> getCanonicalGRUInfo(NNGRUOp gruOp) {
  if (!gruOp || !gruOp.getBatchFirst())
    return std::nullopt;

  std::optional<mlir::RankedTensorType> inputType =
      getRankedTensorType(gruOp.getInput().getType());
  std::optional<mlir::RankedTensorType> hiddenStateType =
      getRankedTensorType(gruOp.getH0().getType());
  std::optional<mlir::RankedTensorType> sequenceResultType =
      getRankedTensorType(gruOp.getOutput().getType());
  std::optional<mlir::RankedTensorType> hiddenResultType =
      getRankedTensorType(gruOp.getHn().getType());
  if (!inputType || !hiddenStateType || !sequenceResultType ||
      !hiddenResultType || inputType->getRank() != 3 ||
      hiddenStateType->getRank() != 3 || sequenceResultType->getRank() != 3 ||
      hiddenResultType->getRank() != 3)
    return std::nullopt;

  int64_t layerCount = gruOp.getNumLayers();
  int64_t hiddenSize = gruOp.getHiddenSize();
  int64_t batchSize = inputType->getDimSize(0);
  int64_t sequenceLength = inputType->getDimSize(1);
  if (layerCount < 1 || hiddenSize < 1 || batchSize < 1 || sequenceLength < 1)
    return std::nullopt;

  if (hiddenStateType->getShape() !=
          llvm::ArrayRef<int64_t>({layerCount, batchSize, hiddenSize}) ||
      hiddenResultType->getShape() != hiddenStateType->getShape() ||
      sequenceResultType->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}))
    return std::nullopt;

  int64_t operandsPerLayer = gruOp.getHasBias() ? 4 : 2;
  if (static_cast<int64_t>(gruOp.getRecurrentOperands().size()) !=
      layerCount * operandsPerLayer)
    return std::nullopt;

  auto hiddenSliceType = mlir::RankedTensorType::get(
      {1, batchSize, hiddenSize}, inputType->getElementType());

  return CanonicalGRUInfo{*inputType,          *hiddenStateType,
                          *sequenceResultType, *hiddenResultType,
                          hiddenSliceType,     layerCount,
                          batchSize,           sequenceLength,
                          hiddenSize,          gruOp.getHasBias()};
}

static std::optional<NNGRUOp> matchCanonicalGRU(mlir::Operation *op) {
  auto gruOp = llvm::dyn_cast_or_null<NNGRUOp>(op);
  if (!gruOp || !getCanonicalGRUInfo(gruOp))
    return std::nullopt;

  for (mlir::Value recurrentOperand : gruOp.getRecurrentOperands()) {
    if (!getConstantDef(recurrentOperand))
      return std::nullopt;
  }

  return gruOp;
}

static bool hasLayerFunctionNameConflict(mlir::ModuleOp module,
                                         mlir::StringRef baseName,
                                         int64_t layerCount) {
  if (module.lookupSymbol(baseName))
    return true;

  for (int64_t layer = 0; layer < layerCount; ++layer) {
    std::string functionName = (baseName + "_" + std::to_string(layer)).str();
    if (module.lookupSymbol(functionName))
      return true;
  }

  return false;
}

static std::string makeUniqueLayerFunctionBaseName(mlir::ModuleOp module,
                                                   mlir::StringRef layerType,
                                                   int64_t layerCount) {
  std::string stem = rewrite_utils::makeFunctionBaseName(layerType);
  unsigned functionIndex = 0;
  std::string baseName = stem + "_" + std::to_string(functionIndex);
  while (hasLayerFunctionNameConflict(module, baseName, layerCount)) {
    ++functionIndex;
    baseName = stem + "_" + std::to_string(functionIndex);
  }
  return baseName;
}

static mlir::RankedTensorType getLayerInputType(int64_t layer,
                                                const CanonicalGRUInfo &info) {
  return layer == 0 ? info.inputType : info.sequenceResultType;
}

// Creates one extracted GRU layer function with cloned parameters.
static mlir::func::FuncOp createGRULayerFunction(
    NNGRUOp gruOp, llvm::ArrayRef<mlir::arith::ConstantOp> recurrentConstants,
    const CanonicalGRUInfo &info, mlir::StringRef baseName, int64_t layer,
    mlir::RewriterBase &rewriter) {
  mlir::Operation *root = gruOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};

  int64_t operandsPerLayer = info.hasBias ? 4 : 2;
  int64_t operandBase = layer * operandsPerLayer;
  mlir::RankedTensorType inputType = getLayerInputType(layer, info);
  llvm::SmallVector<mlir::Type, 2> inputTypes{inputType, info.hiddenStateType};
  llvm::SmallVector<mlir::Type, 2> outputTypes{info.sequenceResultType,
                                               info.hiddenSliceType};

  std::string functionName = (baseName + "_" + std::to_string(layer)).str();
  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto layerFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  layerFunc->setAttr("layer_type", rewriter.getStringAttr(
                                       info.hasBias ? "gru_w_bias" : "gru"));

  mlir::Block *entryBlock = layerFunc.addEntryBlock();
  mlir::IRMapping mapping;
  rewriter.setInsertionPointToStart(entryBlock);

  mlir::Value wIh =
      rewriter.clone(*recurrentConstants[operandBase], mapping)->getResult(0);
  mlir::Value wHh =
      rewriter.clone(*recurrentConstants[operandBase + 1], mapping)
          ->getResult(0);
  mlir::Value bIh;
  mlir::Value bHh;
  if (info.hasBias) {
    bIh = rewriter.clone(*recurrentConstants[operandBase + 2], mapping)
              ->getResult(0);
    bHh = rewriter.clone(*recurrentConstants[operandBase + 3], mapping)
              ->getResult(0);
  }

  auto layerOp = rewriter.create<NNGRULayerOp>(
      root->getLoc(), mlir::TypeRange(outputTypes), entryBlock->getArgument(0),
      entryBlock->getArgument(1), wIh, wHh, bIh, bHh, gruOp.getBatchFirstAttr(),
      gruOp.getHasBiasAttr(), gruOp.getHiddenSizeAttr(),
      rewriter.getI64IntegerAttr(layer), gruOp.getNumLayersAttr());
  rewriter.create<mlir::func::ReturnOp>(
      root->getLoc(), mlir::ValueRange{layerOp.getOutput(), layerOp.getHn()});

  return layerFunc;
}

static mlir::Value
buildStackedHiddenStateResult(llvm::ArrayRef<mlir::Value> finalHiddenStates,
                              const CanonicalGRUInfo &info, mlir::Location loc,
                              mlir::RewriterBase &rewriter) {
  if (finalHiddenStates.size() == 1)
    return finalHiddenStates.front();

  auto concat = rewriter.create<mlir::tensor::ConcatOp>(
      loc, info.hiddenResultType, /*dim=*/0,
      mlir::ValueRange(finalHiddenStates));
  return concat.getResult();
}

// Rewrites a canonical GRU op into calls through extracted layer functions.
static void outlineGRUOpToLayerFunctions(NNGRUOp gruOp,
                                         mlir::RewriterBase &rewriter) {
  if (!gruOp || gruOp->getNumResults() != 2)
    return;

  std::optional<CanonicalGRUInfo> info = getCanonicalGRUInfo(gruOp);
  if (!info)
    return;

  llvm::SmallVector<mlir::arith::ConstantOp> recurrentConstants;
  for (mlir::Value recurrentOperand : gruOp.getRecurrentOperands()) {
    mlir::arith::ConstantOp recurrentConstant =
        getConstantDef(recurrentOperand);
    if (!recurrentConstant)
      return;
    recurrentConstants.push_back(recurrentConstant);
  }

  mlir::Operation *root = gruOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return;

  mlir::StringRef layerType =
      info->hasBias ? mlir::StringRef("gru_w_bias") : mlir::StringRef("gru");
  std::string baseName =
      makeUniqueLayerFunctionBaseName(module, layerType, info->layerCount);

  llvm::SmallVector<mlir::func::FuncOp> layerFuncs;
  layerFuncs.reserve(info->layerCount);
  for (int64_t layer = 0; layer < info->layerCount; ++layer) {
    mlir::func::FuncOp layerFunc = createGRULayerFunction(
        gruOp, recurrentConstants, *info, baseName, layer, rewriter);
    if (!layerFunc)
      return;
    layerFuncs.push_back(layerFunc);
  }

  mlir::Value currentSequence = gruOp.getInput();
  llvm::SmallVector<mlir::Value> finalHiddenStates;
  finalHiddenStates.reserve(info->layerCount);

  rewriter.setInsertionPoint(root);
  for (mlir::func::FuncOp layerFunc : layerFuncs) {
    auto call = rewriter.create<mlir::func::CallOp>(
        root->getLoc(), layerFunc.getSymName(), layerFunc.getResultTypes(),
        mlir::ValueRange{currentSequence, gruOp.getH0()});
    currentSequence = call.getResult(0);
    finalHiddenStates.push_back(call.getResult(1));
  }

  mlir::Value hiddenResult = buildStackedHiddenStateResult(
      finalHiddenStates, *info, root->getLoc(), rewriter);

  gruOp.getOutput().replaceAllUsesWith(currentSequence);
  gruOp.getHn().replaceAllUsesWith(hiddenResult);
  if (root->use_empty())
    rewriter.eraseOp(root);

  for (mlir::arith::ConstantOp recurrentConstant : recurrentConstants) {
    if (recurrentConstant->use_empty())
      rewriter.eraseOp(recurrentConstant);
  }
}

static bool extractCanonicalGRUs(mlir::func::FuncOp func) {
  llvm::SmallVector<NNGRUOp> matches;
  func.walk([&](mlir::Operation *op) {
    std::optional<NNGRUOp> match = matchCanonicalGRU(op);
    if (match)
      matches.push_back(*match);
  });

  if (matches.empty())
    return false;

  mlir::IRRewriter rewriter(func.getContext());
  for (NNGRUOp match : matches) {
    if (match && match->getBlock())
      outlineGRUOpToLayerFunctions(match, rewriter);
  }
  return true;
}

class GRUExtractor : public mlir::sculptor::LayerExtractor {
public:
  explicit GRUExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "gru"; }

  void extract(mlir::func::FuncOp func) const override {
    (void)extractCanonicalGRUs(func);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerGRUExtractor(LayerExtractors &extractors, MLIRContext *context) {
  extractors.push_back(std::make_unique<GRUExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
