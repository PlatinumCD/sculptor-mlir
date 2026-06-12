add_library(MLIRSculptorRuntimeCommon OBJECT
  lib/TaskGraphRuntimeArgs.cpp
  lib/TaskGraphRuntimeBuffers.cpp
  lib/TaskGraphRuntimeGraph.cpp
  lib/TaskGraphRuntimeHandles.cpp
  lib/TaskGraphRuntimeState.cpp
)

target_include_directories(MLIRSculptorRuntimeCommon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(MLIRSculptorRuntimeCommon PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_compile_features(MLIRSculptorRuntimeCommon PUBLIC cxx_std_17)

set_target_properties(MLIRSculptorRuntimeCommon PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
