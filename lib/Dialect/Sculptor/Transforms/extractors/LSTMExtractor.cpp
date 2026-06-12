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

using mlir::sculptor::NNLSTMLayerOp;
using mlir::sculptor::NNLSTMOp;

struct CanonicalLSTMInfo {
  mlir::RankedTensorType inputType;
  mlir::RankedTensorType hiddenStateType;
  mlir::RankedTensorType cellStateType;
  mlir::RankedTensorType sequenceResultType;
  mlir::RankedTensorType hiddenResultType;
  mlir::RankedTensorType cellResultType;
  mlir::RankedTensorType stateSliceType;
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

static std::optional<CanonicalLSTMInfo> getCanonicalLSTMInfo(NNLSTMOp lstmOp) {
  if (!lstmOp || !lstmOp.getBatchFirst())
    return std::nullopt;

  std::optional<mlir::RankedTensorType> inputType =
      getRankedTensorType(lstmOp.getInput().getType());
  std::optional<mlir::RankedTensorType> hiddenStateType =
      getRankedTensorType(lstmOp.getH0().getType());
  std::optional<mlir::RankedTensorType> cellStateType =
      getRankedTensorType(lstmOp.getC0().getType());
  std::optional<mlir::RankedTensorType> sequenceResultType =
      getRankedTensorType(lstmOp.getOutput().getType());
  std::optional<mlir::RankedTensorType> hiddenResultType =
      getRankedTensorType(lstmOp.getHn().getType());
  std::optional<mlir::RankedTensorType> cellResultType =
      getRankedTensorType(lstmOp.getCn().getType());
  if (!inputType || !hiddenStateType || !cellStateType || !sequenceResultType ||
      !hiddenResultType || !cellResultType || inputType->getRank() != 3 ||
      hiddenStateType->getRank() != 3 || cellStateType->getRank() != 3 ||
      sequenceResultType->getRank() != 3 || hiddenResultType->getRank() != 3 ||
      cellResultType->getRank() != 3)
    return std::nullopt;

  int64_t layerCount = lstmOp.getNumLayers();
  int64_t hiddenSize = lstmOp.getHiddenSize();
  int64_t batchSize = inputType->getDimSize(0);
  int64_t sequenceLength = inputType->getDimSize(1);
  if (layerCount < 1 || hiddenSize < 1 || batchSize < 1 || sequenceLength < 1)
    return std::nullopt;

  if (hiddenStateType->getShape() !=
          llvm::ArrayRef<int64_t>({layerCount, batchSize, hiddenSize}) ||
      cellStateType->getShape() != hiddenStateType->getShape() ||
      hiddenResultType->getShape() != hiddenStateType->getShape() ||
      cellResultType->getShape() != hiddenStateType->getShape() ||
      sequenceResultType->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}))
    return std::nullopt;

  int64_t operandsPerLayer = lstmOp.getHasBias() ? 4 : 2;
  if (static_cast<int64_t>(lstmOp.getRecurrentOperands().size()) !=
      layerCount * operandsPerLayer)
    return std::nullopt;

  auto stateSliceType = mlir::RankedTensorType::get(
      {1, batchSize, hiddenSize}, inputType->getElementType());

  return CanonicalLSTMInfo{
      *inputType,        *hiddenStateType, *cellStateType, *sequenceResultType,
      *hiddenResultType, *cellResultType,  stateSliceType, layerCount,
      batchSize,         sequenceLength,   hiddenSize,     lstmOp.getHasBias()};
}

static std::optional<NNLSTMOp> matchCanonicalLSTM(mlir::Operation *op) {
  auto lstmOp = llvm::dyn_cast_or_null<NNLSTMOp>(op);
  if (!lstmOp || !getCanonicalLSTMInfo(lstmOp))
    return std::nullopt;

  for (mlir::Value recurrentOperand : lstmOp.getRecurrentOperands()) {
    if (!getConstantDef(recurrentOperand))
      return std::nullopt;
  }

  return lstmOp;
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
                                                const CanonicalLSTMInfo &info) {
  return layer == 0 ? info.inputType : info.sequenceResultType;
}

// Creates one extracted LSTM layer function with cloned parameters.
static mlir::func::FuncOp createLSTMLayerFunction(
    NNLSTMOp lstmOp, llvm::ArrayRef<mlir::arith::ConstantOp> recurrentConstants,
    const CanonicalLSTMInfo &info, mlir::StringRef baseName, int64_t layer,
    mlir::RewriterBase &rewriter) {
  mlir::Operation *root = lstmOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};

  int64_t operandsPerLayer = info.hasBias ? 4 : 2;
  int64_t operandBase = layer * operandsPerLayer;
  mlir::RankedTensorType inputType = getLayerInputType(layer, info);
  llvm::SmallVector<mlir::Type, 3> inputTypes{inputType, info.hiddenStateType,
                                              info.cellStateType};
  llvm::SmallVector<mlir::Type, 3> outputTypes{
      info.sequenceResultType, info.stateSliceType, info.stateSliceType};

  std::string functionName = (baseName + "_" + std::to_string(layer)).str();
  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto layerFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  layerFunc->setAttr("layer_type", rewriter.getStringAttr(
                                       info.hasBias ? "lstm_w_bias" : "lstm"));

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

  auto layerOp = rewriter.create<NNLSTMLayerOp>(
      root->getLoc(), mlir::TypeRange(outputTypes), entryBlock->getArgument(0),
      entryBlock->getArgument(1), entryBlock->getArgument(2), wIh, wHh, bIh,
      bHh, lstmOp.getBatchFirstAttr(), lstmOp.getHasBiasAttr(),
      lstmOp.getHiddenSizeAttr(), rewriter.getI64IntegerAttr(layer),
      lstmOp.getNumLayersAttr());
  rewriter.create<mlir::func::ReturnOp>(
      root->getLoc(),
      mlir::ValueRange{layerOp.getOutput(), layerOp.getHn(), layerOp.getCn()});

  return layerFunc;
}

// Packs per-layer LSTM final states into the stacked result.
static mlir::Value
buildStackedLSTMStateResult(llvm::ArrayRef<mlir::Value> finalStates,
                            mlir::RankedTensorType resultType,
                            mlir::Location loc, mlir::RewriterBase &rewriter) {
  if (finalStates.size() == 1)
    return finalStates.front();

  auto concat = rewriter.create<mlir::tensor::ConcatOp>(
      loc, resultType, /*dim=*/0, mlir::ValueRange(finalStates));
  return concat.getResult();
}

// Rewrites a canonical LSTM op into calls through extracted layer functions.
static void outlineLSTMOpToLayerFunctions(NNLSTMOp lstmOp,
                                          mlir::RewriterBase &rewriter) {
  if (!lstmOp || lstmOp->getNumResults() != 3)
    return;

  std::optional<CanonicalLSTMInfo> info = getCanonicalLSTMInfo(lstmOp);
  if (!info)
    return;

  llvm::SmallVector<mlir::arith::ConstantOp> recurrentConstants;
  for (mlir::Value recurrentOperand : lstmOp.getRecurrentOperands()) {
    mlir::arith::ConstantOp recurrentConstant =
        getConstantDef(recurrentOperand);
    if (!recurrentConstant)
      return;
    recurrentConstants.push_back(recurrentConstant);
  }

  mlir::Operation *root = lstmOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return;

  mlir::StringRef layerType =
      info->hasBias ? mlir::StringRef("lstm_w_bias") : mlir::StringRef("lstm");
  std::string baseName =
      makeUniqueLayerFunctionBaseName(module, layerType, info->layerCount);

  llvm::SmallVector<mlir::func::FuncOp> layerFuncs;
  layerFuncs.reserve(info->layerCount);
  for (int64_t layer = 0; layer < info->layerCount; ++layer) {
    mlir::func::FuncOp layerFunc = createLSTMLayerFunction(
        lstmOp, recurrentConstants, *info, baseName, layer, rewriter);
    if (!layerFunc)
      return;
    layerFuncs.push_back(layerFunc);
  }

  mlir::Value currentSequence = lstmOp.getInput();
  llvm::SmallVector<mlir::Value> finalHiddenStates;
  llvm::SmallVector<mlir::Value> finalCellStates;
  finalHiddenStates.reserve(info->layerCount);
  finalCellStates.reserve(info->layerCount);

  rewriter.setInsertionPoint(root);
  for (mlir::func::FuncOp layerFunc : layerFuncs) {
    auto call = rewriter.create<mlir::func::CallOp>(
        root->getLoc(), layerFunc.getSymName(), layerFunc.getResultTypes(),
        mlir::ValueRange{currentSequence, lstmOp.getH0(), lstmOp.getC0()});
    currentSequence = call.getResult(0);
    finalHiddenStates.push_back(call.getResult(1));
    finalCellStates.push_back(call.getResult(2));
  }

  mlir::Value hiddenResult = buildStackedLSTMStateResult(
      finalHiddenStates, info->hiddenResultType, root->getLoc(), rewriter);
  mlir::Value cellResult = buildStackedLSTMStateResult(
      finalCellStates, info->cellResultType, root->getLoc(), rewriter);

  lstmOp.getOutput().replaceAllUsesWith(currentSequence);
  lstmOp.getHn().replaceAllUsesWith(hiddenResult);
  lstmOp.getCn().replaceAllUsesWith(cellResult);
  if (root->use_empty())
    rewriter.eraseOp(root);

  for (mlir::arith::ConstantOp recurrentConstant : recurrentConstants) {
    if (recurrentConstant->use_empty())
      rewriter.eraseOp(recurrentConstant);
  }
}

static bool extractCanonicalLSTMs(mlir::func::FuncOp func) {
  llvm::SmallVector<NNLSTMOp> matches;
  func.walk([&](mlir::Operation *op) {
    std::optional<NNLSTMOp> match = matchCanonicalLSTM(op);
    if (match)
      matches.push_back(*match);
  });

  if (matches.empty())
    return false;

  mlir::IRRewriter rewriter(func.getContext());
  for (NNLSTMOp match : matches) {
    if (match && match->getBlock())
      outlineLSTMOpToLayerFunctions(match, rewriter);
  }
  return true;
}

class LSTMExtractor : public mlir::sculptor::LayerExtractor {
public:
  explicit LSTMExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "lstm"; }

  void extract(mlir::func::FuncOp func) const override {
    (void)extractCanonicalLSTMs(func);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerLSTMExtractor(LayerExtractors &extractors, MLIRContext *context) {
  extractors.push_back(std::make_unique<LSTMExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
