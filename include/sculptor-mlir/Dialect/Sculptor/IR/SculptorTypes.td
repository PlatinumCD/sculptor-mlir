#ifndef SCULPTOR_TYPES
#define SCULPTOR_TYPES

include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.td"

// =====-------------------------------------------=====
//       MatrixType
// =====-------------------------------------------=====

def Sculptor_MatrixType : Sculptor_Type<"Matrix", "matrix"> {
  let summary = "Analog matrix container type";

  let description = [{
    Represents a matrix intended for analog computation.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [8,8])">:$shape,
    "Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static ::mlir::ParseResult
    parseShapeAndElt(::mlir::AsmParser &parser,
                     ::llvm::SmallVector<int64_t> &shape,
                     ::mlir::Type &elementType);

    static void
    printShapeAndElt(::mlir::AsmPrinter &printer,
                     ::llvm::ArrayRef<int64_t> shape,
                     ::mlir::Type elementType);
  }];
}

// =====-------------------------------------------=====
//       VectorType
// =====-------------------------------------------=====

def Sculptor_VectorType : Sculptor_Type<"Vector", "vector"> {
  let summary = "Analog vector container type";

  let description = [{
    Represents a vector intended for analog computation.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$shape,
    "Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static ::mlir::ParseResult
    parseShapeAndElt(::mlir::AsmParser &parser,
                     ::llvm::SmallVector<int64_t> &shape,
                     ::mlir::Type &elementType);

    static void
    printShapeAndElt(::mlir::AsmPrinter &printer,
                     ::llvm::ArrayRef<int64_t> shape,
                     ::mlir::Type elementType);
  }];
}

// =====-------------------------------------------=====
//       MatrixGridType
// =====-------------------------------------------=====

def Sculptor_MatrixGridType : Sculptor_Type<"MatrixGrid", "matrix.grid"> {
  let summary = "Analog array grid view type";

  let description = [{
    Represents a 2D grid of analog arrays, each array having a fixed shape
    and element type.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$gridShape,
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$arrayShape,
    Sculptor_MatrixType:$matrix
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// =====-------------------------------------------=====
//       VectorSliceType
// =====-------------------------------------------=====

def Sculptor_VectorSliceType : Sculptor_Type<"VectorSlice", "vector.slice"> {
  let summary = "Analog array slice view type";

  let description = [{
    Represents a 1D view of analog vector arrays, each with a fixed shape and
    element type.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$gridShape,
    ArrayRefParameter<"int64_t", "Shape (e.g., [1,8])">:$arrayShape,
    Sculptor_VectorType:$vector
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// =====-------------------------------------------=====
//       LogicalArrayType
// =====-------------------------------------------=====

def Sculptor_LogicalArrayType
    : Sculptor_Type<"LogicalArray", "logical.array"> {
  let summary = "Analog logical array handle type";

  let description = [{
    Represents a logical accelerator array after a matrix tile has been set.
  }];
}

// =====-------------------------------------------=====
//       ArrayResultType
// =====-------------------------------------------=====

def Sculptor_ArrayResultType
    : Sculptor_Type<"ArrayResult", "array.result"> {
  let summary = "Analog array execution result type";

  let description = [{
    Represents the opaque result produced by executing a logical accelerator
    array before it is stored back into tensor IR.
  }];
}

// =====-------------------------------------------=====
//       TaskGraphType
// =====-------------------------------------------=====

def Sculptor_TaskGraphType : Sculptor_Type<"TaskGraph", "task_graph"> {
  let summary = "Analog task graph handle type";

  let description = [{
    Represents a lightweight handle to a task graph under construction in the
    analog IR.
  }];
}

// =====-------------------------------------------=====
//       TaskType
// =====-------------------------------------------=====

def Sculptor_TaskType : Sculptor_Type<"Task", "task"> {
  let summary = "Analog task handle type";

  let description = [{
    Represents a single scheduled task node inside an analog task graph.
  }];
}

// =====-------------------------------------------=====
//       RuntimeHandleType
// =====-------------------------------------------=====

def Sculptor_RuntimeHandleType : Sculptor_Type<"RuntimeHandle", "runtime_handle"> {
  let summary = "Analog runtime-owned opaque handle type";

  let description = [{
    Represents runtime-owned persistent state materialized by task execution and
    consumed by later graph tasks through explicit persistent resource slots.
  }];
}

// =====-------------------------------------------=====
//       TaskResourceType
// =====-------------------------------------------=====

def Sculptor_TaskResourceType : Sculptor_Type<"TaskResource", "task_resource"> {
  let summary = "Analog task graph resource handle type";

  let description = [{
    Represents one graph-owned resource slot carrying a concrete payload type
    used as a task input, task output, graph boundary value, or internal
    intermediate.
  }];

  let parameters = (ins
    "Type":$valueType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // SCULPTOR_TYPES
