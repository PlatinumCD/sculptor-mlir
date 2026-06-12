#ifndef SCULPTOR_GOLEM_OPS_TD
#define SCULPTOR_GOLEM_OPS_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.td"
include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.td"

//===----------------------------------------------------------------------===//
// sculptor.matrix.from_tensor
// Import a constant tensor into analog matrix storage
//===----------------------------------------------------------------------===//

def Sculptor_MatrixFromTensorOp
    : Sculptor_Op<"matrix.from_tensor"> {

  let summary = "Materialize analog matrix from a tensor";

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs
    Sculptor_MatrixType:$result
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.matrix.partition
// Create an analog array grid from a matrix
//===----------------------------------------------------------------------===//

def Sculptor_MatrixPartitionOp
    : Sculptor_Op<"matrix.partition"> {

  let summary = "Create an array partition view over an analog matrix";

  let description = [{
    Declares that an analog matrix tensor is realized as a grid of
    fixed-size analog arrays. The grid shape is derived from the
    matrix shape and array dimensions.
  }];

  let arguments = (ins
    Sculptor_MatrixType:$matrix
  );

  let results = (outs
    Sculptor_MatrixGridType:$result
  );

  let assemblyFormat = [{
    $matrix attr-dict
    `:` qualified(type($matrix)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.array.matrix.place
// Place an analog partition into the accelerator
//===----------------------------------------------------------------------===//

def Sculptor_ArrayMatrixPlaceOp
    : Sculptor_Op<"array.matrix.place"> {

  let summary = "Place an analog partition into the accelerator";

  let description = [{
    Given a grid of analog arrays, this op places a single array.
  }];

  let arguments = (ins
    Sculptor_MatrixGridType:$input,
    Index:$rowIndex,
    Index:$colIndex,
    Variadic<Index>:$indices
  );

  let assemblyFormat = [{
    $input `[` $rowIndex `,` $colIndex `]` `,` `array` `[` $indices `]`
    attr-dict
    `:` qualified(type($input))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.vector.from_tensor
// Import a constant tensor into analog vector storage
//===----------------------------------------------------------------------===//

def Sculptor_VectorFromTensorOp
    : Sculptor_Op<"vector.from_tensor"> {

  let summary = "Materialize analog vector from a tensor";

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs
    Sculptor_VectorType:$result
  );

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.vector.partition
// Create an analog vector-array slice from a vector
//===----------------------------------------------------------------------===//

def Sculptor_VectorPartitionOp
    : Sculptor_Op<"vector.partition"> {

  let summary = "Partition an analog vector into slices";

  let description = [{
    Creates an analog vector partition view for tiling.
  }];

  let arguments = (ins
    Sculptor_VectorType:$vector
  );

  let results = (outs
    Sculptor_VectorSliceType:$result
  );

  let assemblyFormat = [{
    $vector attr-dict
    `:` qualified(type($vector)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.array.vector.place
// Place a vector-array slice in the accelerator
//===----------------------------------------------------------------------===//

def Sculptor_ArrayVectorPlaceOp
  : Sculptor_Op<"array.vector.place"> {

  let summary = "Place an analog vector-array slice into an accelerator";

  let arguments = (ins
    Sculptor_VectorSliceType:$input,
    Index:$sliceIndex,
    Variadic<Index>:$indices
  );

  let assemblyFormat = [{
    $input `[` $sliceIndex `]` `,` `array` `[` $indices `]`
    attr-dict
    `:` qualified(type($input))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.array.set
// Program a logical accelerator array with a matrix tile
//===----------------------------------------------------------------------===//

def Sculptor_ArraySetOp
  : Sculptor_Op<"array.set"> {

  let summary = "Set a logical accelerator array from a matrix tile";

  let description = [{
    Programs a logical accelerator array with one concrete rank-2 f32 matrix
    tile and returns the logical array handle consumed by later array ops.
  }];

  let arguments = (ins
    AnyRankedTensor:$matrix
  );

  let results = (outs
    Sculptor_LogicalArrayType:$array
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $matrix attr-dict `:` type($matrix) `->` qualified(type($array))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.array.load
// Load a vector tile into a logical accelerator array
//===----------------------------------------------------------------------===//

def Sculptor_ArrayLoadOp
  : Sculptor_Op<"array.load"> {

  let summary = "Load a vector tile into a logical accelerator array";

  let description = [{
    Loads one rank-2 f32 vector tile into the logical accelerator array returned
    by sculptor.array.set.
  }];

  let arguments = (ins
    AnyRankedTensor:$vector,
    Sculptor_LogicalArrayType:$array
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $vector `,` $array attr-dict
    `:` type($vector) `,` qualified(type($array))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.array.execute
// Execute an array in the accelerator
//===----------------------------------------------------------------------===//

def Sculptor_ArrayExecuteOp
  : Sculptor_Op<"array.execute"> {

  let summary = "Execute a logical accelerator array";

  let arguments = (ins
    Sculptor_LogicalArrayType:$array
  );

  let results = (outs
    Sculptor_ArrayResultType:$result
  );

  let assemblyFormat = [{
    $array attr-dict `:` qualified(type($array)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.array.store
// Store an accelerator array result into tensor IR
//===----------------------------------------------------------------------===//

def Sculptor_ArrayStoreOp
  : Sculptor_Op<"array.store"> {

  let summary = "Store an accelerator array result into tensor IR";

  let arguments = (ins
    Sculptor_ArrayResultType:$input
  );

  let results = (outs
    AnyRankedTensor:$output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `->` type($output)
  }];
}

#endif // SCULPTOR_GOLEM_OPS_TD
