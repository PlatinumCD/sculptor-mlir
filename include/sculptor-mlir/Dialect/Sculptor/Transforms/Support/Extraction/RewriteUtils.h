#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_EXTRACTION_REWRITEUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_EXTRACTION_REWRITEUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cctype>
#include <string>

namespace mlir {
namespace sculptor {
namespace rewrite_utils {

// Builds a symbol-friendly lowercase stem from an extractor layer label.
inline std::string makeFunctionBaseName(StringRef layerType) {
  std::string baseName;
  baseName.reserve(layerType.size());

  for (char ch : layerType) {
    unsigned char c = static_cast<unsigned char>(ch);
    if (std::isalnum(c))
      baseName.push_back(static_cast<char>(std::tolower(c)));
    else if (std::isspace(c))
      baseName.push_back('_');
  }

  if (baseName.empty())
    baseName = "layer";

  return baseName;
}

// Chooses a module-local function name derived from the layer label.
inline std::string makeUniqueFunctionName(ModuleOp module,
                                          StringRef layerType) {
  unsigned functionIndex = 0;
  std::string baseName = makeFunctionBaseName(layerType);
  std::string functionName = baseName + "_" + std::to_string(functionIndex);
  while (module.lookupSymbol(functionName)) {
    ++functionIndex;
    functionName = baseName + "_" + std::to_string(functionIndex);
  }
  return functionName;
}

// Mirrors SSA value types into the signature vectors used by outlined funcs.
inline SmallVector<Type> collectValueTypes(llvm::ArrayRef<Value> values) {
  SmallVector<Type> types;
  types.reserve(values.size());
  for (Value value : values)
    types.push_back(value.getType());
  return types;
}

// Outlines a matched subgraph into a new function and replaces the root with
// a call using the caller-provided input and output boundaries.
inline void extractToFunction(Operation *root, llvm::ArrayRef<Operation *> ops,
                              llvm::ArrayRef<Value> inputs,
                              llvm::ArrayRef<Value> outputs,
                              RewriterBase &rewriter, StringRef layerType) {
  // Refuse partial matches that cannot produce a valid outlined function.
  if (!root || ops.empty() || outputs.empty())
    return;

  auto module = root->getParentOfType<ModuleOp>();
  if (!module)
    return;

  // Materialize the function signature and metadata in the parent module.
  SmallVector<Type> inputTypes = collectValueTypes(inputs);
  SmallVector<Type> outputTypes = collectValueTypes(outputs);
  std::string functionName = makeUniqueFunctionName(module, layerType);

  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto extractedFunc =
      rewriter.create<func::FuncOp>(root->getLoc(), functionName, functionType);
  extractedFunc->setAttr("layer_type", rewriter.getStringAttr(layerType));

  Block *entryBlock = extractedFunc.addEntryBlock();
  IRMapping mapping;
  for (unsigned i = 0; i < inputs.size(); ++i)
    mapping.map(inputs[i], entryBlock->getArgument(i));

  // Clone the matched body with external values remapped to block arguments.
  rewriter.setInsertionPointToStart(entryBlock);
  for (Operation *op : ops)
    rewriter.clone(*op, mapping);

  // Return the cloned values that correspond to the original output boundary.
  SmallVector<Value> mappedOutputs;
  for (Value output : outputs)
    mappedOutputs.push_back(mapping.lookupOrDefault(output));
  rewriter.create<func::ReturnOp>(root->getLoc(), mappedOutputs);

  // Replace the original subgraph boundary with a call to the outlined func.
  rewriter.setInsertionPoint(root);
  auto call = rewriter.create<func::CallOp>(
      root->getLoc(), extractedFunc.getSymName(), outputTypes, inputs);

  for (unsigned i = 0; i < outputs.size(); ++i) {
    Value output = outputs[i];
    output.replaceAllUsesWith(call.getResult(i));
  }

  // Drop now-dead matched ops from consumers back to producers.
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    if ((*it)->use_empty())
      rewriter.eraseOp(*it);
  }
}

// Outlines an existing canonical sculptor.nn op into a new function. The dynamic
// activation input remains a function argument, while static weight and bias
// constants are cloned into the outlined body.
template <typename CanonicalOpT, typename BuildOpFn>
inline void extractExistingSingleResultCanonicalOpToFunction(
    CanonicalOpT op, Value input, Value weightValue, Value biasValue,
    RewriterBase &rewriter, StringRef layerType, BuildOpFn buildOp) {
  if (!op || !input || !weightValue)
    return;

  Operation *root = op.getOperation();
  if (!root || root->getNumResults() != 1)
    return;

  auto weightConstant = weightValue.getDefiningOp<arith::ConstantOp>();
  if (!weightConstant)
    return;

  arith::ConstantOp biasConstant;
  if (biasValue) {
    biasConstant = biasValue.getDefiningOp<arith::ConstantOp>();
    if (!biasConstant)
      return;
  }

  auto module = root->template getParentOfType<ModuleOp>();
  if (!module)
    return;

  SmallVector<Type> inputTypes{input.getType()};
  SmallVector<Type> outputTypes{root->getResult(0).getType()};
  std::string functionName = makeUniqueFunctionName(module, layerType);

  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);
  rewriter.setInsertionPointToEnd(module.getBody());
  auto extractedFunc =
      rewriter.create<func::FuncOp>(root->getLoc(), functionName, functionType);
  extractedFunc->setAttr("layer_type", rewriter.getStringAttr(layerType));

  Block *entryBlock = extractedFunc.addEntryBlock();
  IRMapping mapping;
  mapping.map(input, entryBlock->getArgument(0));

  rewriter.setInsertionPointToStart(entryBlock);
  Operation *weightClone = rewriter.clone(*weightConstant, mapping);
  Value weight = weightClone->getResult(0);

  Value bias;
  if (biasConstant) {
    Operation *biasClone = rewriter.clone(*biasConstant, mapping);
    bias = biasClone->getResult(0);
  }

  Value result = buildOp(rewriter, root->getLoc(), outputTypes.front(),
                         entryBlock->getArgument(0), weight, bias);
  if (!result)
    return;

  rewriter.create<func::ReturnOp>(root->getLoc(), ValueRange{result});

  rewriter.setInsertionPoint(root);
  auto call = rewriter.create<func::CallOp>(
      root->getLoc(), extractedFunc.getSymName(), outputTypes,
      ValueRange{input});

  root->getResult(0).replaceAllUsesWith(call.getResult(0));

  if (root->use_empty())
    rewriter.eraseOp(root);
  if (biasConstant && biasConstant->use_empty())
    rewriter.eraseOp(biasConstant);
  if (weightConstant->use_empty())
    rewriter.eraseOp(weightConstant);
}

} // namespace rewrite_utils
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_EXTRACTION_REWRITEUTILS_H
