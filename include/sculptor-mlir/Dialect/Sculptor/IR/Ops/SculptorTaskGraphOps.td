#ifndef SCULPTOR_TASK_GRAPH_OPS_TD
#define SCULPTOR_TASK_GRAPH_OPS_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.td"
include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.td"

//===----------------------------------------------------------------------===//
// sculptor.task_graph.create
// Create a task graph handle
//===----------------------------------------------------------------------===//

def Sculptor_TaskGraphCreateOp
  : Sculptor_Op<"task_graph.create"> {

  let summary = "Create an analog task graph handle";

  let results = (outs
    Sculptor_TaskGraphType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.task_graph.input
// Create a graph input resource handle
//===----------------------------------------------------------------------===//

def Sculptor_TaskGraphInputOp
  : Sculptor_Op<"task_graph.input"> {

  let summary = "Create a graph input resource handle";

  let arguments = (ins
    Sculptor_TaskGraphType:$graph
  );

  let results = (outs
    Sculptor_TaskResourceType:$result
  );

  let assemblyFormat = [{
    $graph attr-dict `:` qualified(type($graph)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.task_graph.output
// Create a graph output resource handle
//===----------------------------------------------------------------------===//

def Sculptor_TaskGraphOutputOp
  : Sculptor_Op<"task_graph.output"> {

  let summary = "Create a graph output resource handle";

  let arguments = (ins
    Sculptor_TaskGraphType:$graph
  );

  let results = (outs
    Sculptor_TaskResourceType:$result
  );

  let assemblyFormat = [{
    $graph attr-dict `:` qualified(type($graph)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.task_graph.temporary
// Create an internal graph temporary resource handle
//===----------------------------------------------------------------------===//

def Sculptor_TaskGraphTemporaryOp
  : Sculptor_Op<"task_graph.temporary"> {

  let summary = "Create an internal graph temporary resource handle";

  let arguments = (ins
    Sculptor_TaskGraphType:$graph
  );

  let results = (outs
    Sculptor_TaskResourceType:$result
  );

  let assemblyFormat = [{
    $graph attr-dict `:` qualified(type($graph)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.task_graph.persistent
// Create a graph persistent resource handle
//===----------------------------------------------------------------------===//

def Sculptor_TaskGraphPersistentOp
  : Sculptor_Op<"task_graph.persistent"> {

  let summary = "Create a graph persistent resource handle";

  let arguments = (ins
    Sculptor_TaskGraphType:$graph
  );

  let results = (outs
    Sculptor_TaskResourceType:$result
  );

  let assemblyFormat = [{
    $graph attr-dict `:` qualified(type($graph)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.task.create
// Create a task node in a task graph
//===----------------------------------------------------------------------===//

def Sculptor_TaskCreateOp
  : Sculptor_Op<"task.create"> {

  let summary = "Create a task in an analog task graph";

  let description = [{
    Creates one task node associated with a task graph. The task records the
    callee it will invoke, its task metadata, its explicit dependencies, and
    whether it runs in the analog or digital domain.
  }];

  let arguments = (ins
    Sculptor_TaskGraphType:$graph,
    FlatSymbolRefAttr:$callee,
    StrAttr:$domain,
    StrAttr:$taskKind,
    StrAttr:$taskName,
    StrAttr:$sourceLayer,
    I64Attr:$sourceTaskOrdinal,
    Variadic<Sculptor_TaskResourceType>:$inputs,
    Variadic<Sculptor_TaskResourceType>:$outputs,
    Variadic<Sculptor_TaskType>:$dependencies
  );

  let results = (outs
    Sculptor_TaskType:$result
  );

  let hasVerifier = 1;
  let traits = [AttrSizedOperandSegments];

  let assemblyFormat = [{
    $graph `,` $callee `,` `domain` `=` $domain `,`
    `task_kind` `=` $taskKind `,`
    `task_name` `=` $taskName `,`
    `source_layer` `=` $sourceLayer `,`
    `source_task_ordinal` `=` $sourceTaskOrdinal `,`
    `inputs` `[` $inputs `]`
    `,`
    `outputs` `[` $outputs `]`
    `,`
    `deps` `[` $dependencies `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

#endif // SCULPTOR_TASK_GRAPH_OPS_TD
