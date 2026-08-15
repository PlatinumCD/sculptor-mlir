add_llvm_executable(sculptor-split-tile-deployment
  sculptor-split-tile-deployment.cpp
)

get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(sculptor-split-tile-deployment PRIVATE
  MLIRSculptorTransforms

  ${MLIR_DIALECT_LIBS}
  ${MLIR_EXTENSION_LIBS}

  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

install(TARGETS sculptor-split-tile-deployment
        EXPORT SculptorMLIRTargets
        RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
        COMPONENT sculptor-split-tile-deployment)
