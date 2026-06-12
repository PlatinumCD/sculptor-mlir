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

using mlir::sculptor::NNRNNCellOp;

static std::optional<NNRNNCellOp> matchCanonicalRNNCell(mlir::Operation *op) {
  auto rnnCellOp = llvm::dyn_cast_or_null<NNRNNCellOp>(op);
  if (!rnnCellOp)
    return std::nullopt;

  return rnnCellOp;
}

static mlir::arith::ConstantOp getConstantDef(mlir::Value value) {
  return value ? layer_utils::producerOfType<mlir::arith::ConstantOp>(value)
               : mlir::arith::ConstantOp();
}

static void outlineRNNCellOpToLayerFunction(NNRNNCellOp rnnCellOp,
                                            mlir::RewriterBase &rewriter) {
  if (!rnnCellOp || rnnCellOp->getNumResults() != 1)
    return;

  auto wIhConstant = getConstantDef(rnnCellOp.getWIh());
  auto wHhConstant = getConstantDef(rnnCellOp.getWHh());
  if (!wIhConstant || !wHhConstant)
    return;

  mlir::arith::ConstantOp bIhConstant;
  mlir::arith::ConstantOp bHhConstant;
  bool hasBias = rnnCellOp.getHasBias();
  if (hasBias) {
    bIhConstant = getConstantDef(rnnCellOp.getBIh());
    bHhConstant = getConstantDef(rnnCellOp.getBHh());
    if (!bIhConstant || !bHhConstant)
      return;
  }

  mlir::Operation *root = rnnCellOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return;

  mlir::StringRef layerType = hasBias ? mlir::StringRef("rnn_cell_w_bias")
                                      : mlir::StringRef("rnn_cell");
  llvm::SmallVector<mlir::Type, 2> inputTypes{rnnCellOp.getInput().getType(),
                                              rnnCellOp.getHPrev().getType()};
  llvm::SmallVector<mlir::Type, 1> outputTypes{rnnCellOp.getH().getType()};
  std::string functionName =
      rewrite_utils::makeUniqueFunctionName(module, layerType);

  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto extractedFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  extractedFunc->setAttr("layer_type", rewriter.getStringAttr(layerType));

  mlir::Block *entryBlock = extractedFunc.addEntryBlock();
  mlir::IRMapping mapping;
  mapping.map(rnnCellOp.getInput(), entryBlock->getArgument(0));
  mapping.map(rnnCellOp.getHPrev(), entryBlock->getArgument(1));

  rewriter.setInsertionPointToStart(entryBlock);
  mlir::Value wIh = rewriter.clone(*wIhConstant, mapping)->getResult(0);
  mlir::Value wHh = rewriter.clone(*wHhConstant, mapping)->getResult(0);

  mlir::Value bIh;
  mlir::Value bHh;
  if (hasBias) {
    bIh = rewriter.clone(*bIhConstant, mapping)->getResult(0);
    bHh = rewriter.clone(*bHhConstant, mapping)->getResult(0);
  }

  auto outlinedRNNCellOp = rewriter.create<NNRNNCellOp>(
      root->getLoc(), outputTypes.front(), entryBlock->getArgument(0),
      entryBlock->getArgument(1), wIh, wHh, bIh, bHh,
      rewriter.getBoolAttr(hasBias), rnnCellOp.getActivationAttr());
  rewriter.create<mlir::func::ReturnOp>(root->getLoc(),
                                        outlinedRNNCellOp.getH());

  rewriter.setInsertionPoint(root);
  auto call = rewriter.create<mlir::func::CallOp>(
      root->getLoc(), extractedFunc.getSymName(), outputTypes,
      mlir::ValueRange{rnnCellOp.getInput(), rnnCellOp.getHPrev()});

  rnnCellOp.getH().replaceAllUsesWith(call.getResult(0));
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

class RNNCellExtractor : public mlir::sculptor::LayerExtractor {
public:
  explicit RNNCellExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "rnn_cell"; }

  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter, matchCanonicalRNNCell,
                                      outlineRNNCellOpToLayerFunction);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerRNNCellExtractor(LayerExtractors &extractors,
                              MLIRContext *context) {
  extractors.push_back(std::make_unique<RNNCellExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
