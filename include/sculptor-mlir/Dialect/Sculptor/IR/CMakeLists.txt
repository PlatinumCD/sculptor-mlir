set(LLVM_TARGET_DEFINITIONS SculptorDialect.td)

mlir_tablegen(SculptorBase.h.inc -gen-dialect-decls)
mlir_tablegen(SculptorBase.cpp.inc -gen-dialect-defs)
mlir_tablegen(SculptorTypes.h.inc -gen-typedef-decls)
mlir_tablegen(SculptorTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(SculptorOps.h.inc -gen-op-decls)
mlir_tablegen(SculptorOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Ops/SculptorNNOps.td)
mlir_tablegen(SculptorNNOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SculptorNNOpsEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(SculptorIRIncGen)

add_dependencies(SculptorIncGen SculptorIRIncGen)
