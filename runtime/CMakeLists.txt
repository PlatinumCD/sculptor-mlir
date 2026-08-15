set(SCULPTOR_RUNTIME_SOURCES
  src/basic_tile_runtime.cpp
  src/deployment_runtime.cpp
  src/heap_profile.cpp
  src/ready_queue.cpp
  src/routed_transport.cpp
  src/scratchpad_abi.cpp
  src/task_instance.cpp
  src/task_registry.cpp
  src/tile_abi.cpp
  src/transport.cpp
  src/mlir_runtime.cpp
)

add_library(golem-runtime STATIC ${SCULPTOR_RUNTIME_SOURCES})
add_library(sculptor::golem-runtime ALIAS golem-runtime)

target_compile_features(golem-runtime PUBLIC cxx_std_20)
target_include_directories(golem-runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(golem-runtime PROPERTIES OUTPUT_NAME golem-runtime)

if(SCULPTOR_MLIR_BUILD_RUNTIME_TESTS)
  target_compile_definitions(golem-runtime PRIVATE
    MITTENS_MEMREF_COPY_TEST_HOOK=1
  )
endif()

install(TARGETS golem-runtime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

if(SCULPTOR_MLIR_BUILD_RUNTIME_TESTS)
  add_executable(golem-runtime-test tests/runtime_test.cpp)
  target_compile_features(golem-runtime-test PRIVATE cxx_std_20)
  target_compile_definitions(golem-runtime-test PRIVATE
    MITTENS_MEMREF_COPY_TEST_HOOK=1
  )
  target_link_libraries(golem-runtime-test PRIVATE sculptor::golem-runtime)
  add_test(NAME golem-runtime-test COMMAND golem-runtime-test)
endif()
