#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Extraction/RewriteUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <optional>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace layer_utils = mlir::sculptor::layer_utils;
namespace rewrite_utils = mlir::sculptor::rewrite_utils;

namespace {

using mlir::sculptor::NNLSTMCellOp;

static std::optional<NNLSTMCellOp> matchCanonicalLSTMCell(mlir::Operation *op) {
  auto lstmCellOp = llvm::dyn_cast_or_null<NNLSTMCellOp>(op);
  if (!lstmCellOp)
    return std::nullopt;

  return lstmCellOp;
}

static mlir::arith::ConstantOp getConstantDef(mlir::Value value) {
  return value ? layer_utils::producerOfType<mlir::arith::ConstantOp>(value)
               : mlir::arith::ConstantOp();
}

static void outlineLSTMCellOpToLayerFunction(NNLSTMCellOp lstmCellOp,
                                             mlir::RewriterBase &rewriter) {
  if (!lstmCellOp || lstmCellOp->getNumResults() != 2)
    return;

  auto wIhConstant = getConstantDef(lstmCellOp.getWIh());
  auto wHhConstant = getConstantDef(lstmCellOp.getWHh());
  if (!wIhConstant || !wHhConstant)
    return;

  mlir::arith::ConstantOp bIhConstant;
  mlir::arith::ConstantOp bHhConstant;
  bool hasBias = lstmCellOp.getHasBias();
  if (hasBias) {
    bIhConstant = getConstantDef(lstmCellOp.getBIh());
    bHhConstant = getConstantDef(lstmCellOp.getBHh());
    if (!bIhConstant || !bHhConstant)
      return;
  }

  mlir::Operation *root = lstmCellOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return;

  mlir::StringRef layerType = hasBias ? mlir::StringRef("lstm_cell_w_bias")
                                      : mlir::StringRef("lstm_cell");
  llvm::SmallVector<mlir::Type, 3> inputTypes{lstmCellOp.getInput().getType(),
                                              lstmCellOp.getHPrev().getType(),
                                              lstmCellOp.getCPrev().getType()};
  llvm::SmallVector<mlir::Type, 2> outputTypes{lstmCellOp.getH().getType(),
                                               lstmCellOp.getC().getType()};
  std::string functionName =
      rewrite_utils::makeUniqueFunctionName(module, layerType);

  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto extractedFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  extractedFunc->setAttr("layer_type", rewriter.getStringAttr(layerType));

  mlir::Block *entryBlock = extractedFunc.addEntryBlock();
  mlir::IRMapping mapping;
  mapping.map(lstmCellOp.getInput(), entryBlock->getArgument(0));
  mapping.map(lstmCellOp.getHPrev(), entryBlock->getArgument(1));
  mapping.map(lstmCellOp.getCPrev(), entryBlock->getArgument(2));

  rewriter.setInsertionPointToStart(entryBlock);
  mlir::Value wIh = rewriter.clone(*wIhConstant, mapping)->getResult(0);
  mlir::Value wHh = rewriter.clone(*wHhConstant, mapping)->getResult(0);

  mlir::Value bIh;
  mlir::Value bHh;
  if (hasBias) {
    bIh = rewriter.clone(*bIhConstant, mapping)->getResult(0);
    bHh = rewriter.clone(*bHhConstant, mapping)->getResult(0);
  }

  auto outlinedLSTMCellOp = rewriter.create<NNLSTMCellOp>(
      root->getLoc(), mlir::TypeRange(outputTypes), entryBlock->getArgument(0),
      entryBlock->getArgument(1), entryBlock->getArgument(2), wIh, wHh, bIh,
      bHh, rewriter.getBoolAttr(hasBias));
  rewriter.create<mlir::func::ReturnOp>(
      root->getLoc(),
      mlir::ValueRange{outlinedLSTMCellOp.getH(), outlinedLSTMCellOp.getC()});

  rewriter.setInsertionPoint(root);
  auto call = rewriter.create<mlir::func::CallOp>(
      root->getLoc(), extractedFunc.getSymName(), outputTypes,
      mlir::ValueRange{lstmCellOp.getInput(), lstmCellOp.getHPrev(),
                       lstmCellOp.getCPrev()});

  lstmCellOp.getH().replaceAllUsesWith(call.getResult(0));
  lstmCellOp.getC().replaceAllUsesWith(call.getResult(1));
  if (root->use_empty())
    rewriter.eraseOp(root);

  if (bHhConstant && bHhConstant->use_empty())
    rewriter.eraseOp(bHhConstant);
  if (bIhConstant && bIhConstant->use_empty())
    rewriter.eraseOp(bIhConstant);
  if (wHhConstant->use_empty())
    rewriter.eraseOp(wHhConstant);
  if (wIhConstant->use_empty())
    rewriter.eraseOp(wIhConstant);
}

class LSTMCellExtractor : public mlir::sculptor::LayerExtractor {
public:
  explicit LSTMCellExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "lstm_cell"; }

  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter, matchCanonicalLSTMCell,
                                      outlineLSTMCellOpToLayerFunction);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerLSTMCellExtractor(LayerExtractors &extractors,
                               MLIRContext *context) {
  extractors.push_back(std::make_unique<LSTMCellExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
