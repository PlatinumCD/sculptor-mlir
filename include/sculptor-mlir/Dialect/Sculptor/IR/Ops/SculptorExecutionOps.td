#ifndef SCULPTOR_EXECUTION_OPS_TD
#define SCULPTOR_EXECUTION_OPS_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.td"
include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.td"

class Sculptor_ExecutionOp<string mnem, list<Trait> traits = []>
    : Sculptor_Op<mnem, traits>;

//===----------------------------------------------------------------------===//
// sculptor.mvm
// Matrix-vector multiply execution abstraction.
//===----------------------------------------------------------------------===//

def Sculptor_MVMOp
    : Sculptor_ExecutionOp<"mvm"> {

  let summary = "Represent a matrix-vector multiply";

  let description = [{
    Captures a row-vector by matrix-vector multiply at the analog execution
    level. The vector operand has shape 1xNxf32, the matrix operand has shape
    MxNxf32, and the result has shape 1xMxf32.
  }];

  let arguments = (ins
    AnyRankedTensor:$vector,
    AnyRankedTensor:$matrix
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $vector `,` $matrix attr-dict `:` functional-type(operands, results)
  }];
}

#endif // SCULPTOR_EXECUTION_OPS_TD
