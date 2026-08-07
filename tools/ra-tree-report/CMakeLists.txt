add_llvm_executable(sculptor-ra-tree-report
  sculptor-ra-tree-report.cpp
)

get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(sculptor-ra-tree-report PRIVATE
  MLIRSculptorTransforms

  ${MLIR_DIALECT_LIBS}
  ${MLIR_EXTENSION_LIBS}

  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

install(TARGETS sculptor-ra-tree-report
        EXPORT SculptorMLIRTargets
        RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
        COMPONENT sculptor-ra-tree-report)
