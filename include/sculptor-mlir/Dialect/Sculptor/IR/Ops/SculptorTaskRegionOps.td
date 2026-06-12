#ifndef SCULPTOR_TASK_REGION_OPS_TD
#define SCULPTOR_TASK_REGION_OPS_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.td"

//===----------------------------------------------------------------------===//
// sculptor.task_region
// Structural compiler region for one task boundary.
//===----------------------------------------------------------------------===//

def Sculptor_TaskRegionOp
  : Sculptor_Op<"task_region", [
      SingleBlockImplicitTerminator<"mlir::sculptor::YieldOp">
    ]> {

  let summary = "Group IR for one analog task boundary";

  let description = [{
    `sculptor.task_region` is a structural compiler operation used to draw an
    explicit boundary around one task-stage fragment. It is intentionally not a
    runtime scheduling operation. Later passes can outline these regions into
    task functions, while earlier passes can use the region as a precise box for
    named task-stage logic. The verifier only requires `kind` to be non-empty;
    individual passes decide which section kinds they understand.

    Region block arguments must match the operation operands. Values returned
    from the region are yielded with `sculptor.yield` and must match the operation
    result types.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    StrAttr:$kind,
    OptionalAttr<StrAttr>:$name
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `kind` `=` $kind (`name` `=` $name^)? `(` $inputs `)`
    $body attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.yield
// Terminator for sculptor.task_region.
//===----------------------------------------------------------------------===//

def Sculptor_YieldOp
  : Sculptor_Op<"yield", [Terminator]> {

  let summary = "Yield values from an analog task region";

  let description = [{
    `sculptor.yield` terminates an `sculptor.task_region`. It may yield zero, one,
    or many SSA values. The parent task region verifier checks that yielded
    values match the task region result types.
  }];

  let arguments = (ins
    Variadic<AnyType>:$values
  );

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict ($values^ `:` type($values))?
  }];
}

#endif // SCULPTOR_TASK_REGION_OPS_TD
