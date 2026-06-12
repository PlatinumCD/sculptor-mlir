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

using mlir::sculptor::NNGRUCellOp;

static std::optional<NNGRUCellOp> matchCanonicalGRUCell(mlir::Operation *op) {
  auto gruCellOp = llvm::dyn_cast_or_null<NNGRUCellOp>(op);
  if (!gruCellOp)
    return std::nullopt;

  return gruCellOp;
}

static mlir::arith::ConstantOp getConstantDef(mlir::Value value) {
  return value ? layer_utils::producerOfType<mlir::arith::ConstantOp>(value)
               : mlir::arith::ConstantOp();
}

static void outlineGRUCellOpToLayerFunction(NNGRUCellOp gruCellOp,
                                            mlir::RewriterBase &rewriter) {
  if (!gruCellOp || gruCellOp->getNumResults() != 1)
    return;

  auto wIhConstant = getConstantDef(gruCellOp.getWIh());
  auto wHhConstant = getConstantDef(gruCellOp.getWHh());
  if (!wIhConstant || !wHhConstant)
    return;

  mlir::arith::ConstantOp bIhConstant;
  mlir::arith::ConstantOp bHhConstant;
  bool hasBias = gruCellOp.getHasBias();
  if (hasBias) {
    bIhConstant = getConstantDef(gruCellOp.getBIh());
    bHhConstant = getConstantDef(gruCellOp.getBHh());
    if (!bIhConstant || !bHhConstant)
      return;
  }

  mlir::Operation *root = gruCellOp.getOperation();
  auto module = root->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return;

  mlir::StringRef layerType = hasBias ? mlir::StringRef("gru_cell_w_bias")
                                      : mlir::StringRef("gru_cell");
  llvm::SmallVector<mlir::Type, 2> inputTypes{gruCellOp.getInput().getType(),
                                              gruCellOp.getHPrev().getType()};
  llvm::SmallVector<mlir::Type, 1> outputTypes{gruCellOp.getH().getType()};
  std::string functionName =
      rewrite_utils::makeUniqueFunctionName(module, layerType);

  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto extractedFunc = rewriter.create<mlir::func::FuncOp>(
      root->getLoc(), functionName, functionType);
  extractedFunc->setAttr("layer_type", rewriter.getStringAttr(layerType));

  mlir::Block *entryBlock = extractedFunc.addEntryBlock();
  mlir::IRMapping mapping;
  mapping.map(gruCellOp.getInput(), entryBlock->getArgument(0));
  mapping.map(gruCellOp.getHPrev(), entryBlock->getArgument(1));

  rewriter.setInsertionPointToStart(entryBlock);
  mlir::Value wIh = rewriter.clone(*wIhConstant, mapping)->getResult(0);
  mlir::Value wHh = rewriter.clone(*wHhConstant, mapping)->getResult(0);

  mlir::Value bIh;
  mlir::Value bHh;
  if (hasBias) {
    bIh = rewriter.clone(*bIhConstant, mapping)->getResult(0);
    bHh = rewriter.clone(*bHhConstant, mapping)->getResult(0);
  }

  auto outlinedGRUCellOp = rewriter.create<NNGRUCellOp>(
      root->getLoc(), outputTypes.front(), entryBlock->getArgument(0),
      entryBlock->getArgument(1), wIh, wHh, bIh, bHh,
      rewriter.getBoolAttr(hasBias));
  rewriter.create<mlir::func::ReturnOp>(root->getLoc(),
                                        outlinedGRUCellOp.getH());

  rewriter.setInsertionPoint(root);
  auto call = rewriter.create<mlir::func::CallOp>(
      root->getLoc(), extractedFunc.getSymName(), outputTypes,
      mlir::ValueRange{gruCellOp.getInput(), gruCellOp.getHPrev()});

  gruCellOp.getH().replaceAllUsesWith(call.getResult(0));
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

class GRUCellExtractor : public mlir::sculptor::LayerExtractor {
public:
  explicit GRUCellExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "gru_cell"; }

  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter, matchCanonicalGRUCell,
                                      outlineGRUCellOpToLayerFunction);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerGRUCellExtractor(LayerExtractors &extractors,
                              MLIRContext *context) {
  extractors.push_back(std::make_unique<GRUCellExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
