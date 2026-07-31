cmake_minimum_required(VERSION 3.22)

project(
    sculptor-mlir
    VERSION 0.1.0 
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SCULPTOR_MLIR_BUILD_RUNTIME "Build Sculptor runtime libraries" ON)

find_package(Git QUIET)
set(SCULPTOR_COMPILER_REVISION "${PROJECT_VERSION}")
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --verify HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE SCULPTOR_GIT_REVISION_RESULT
    OUTPUT_VARIABLE SCULPTOR_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(SCULPTOR_GIT_REVISION_RESULT EQUAL 0)
    set(SCULPTOR_COMPILER_REVISION "${SCULPTOR_GIT_REVISION}")
  endif()
endif()
set(SCULPTOR_COMPILER_REVISION "${SCULPTOR_COMPILER_REVISION}" CACHE STRING
    "Revision recorded in Sculptor timing-model provenance")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR at: ${MLIR_DIR}")
message(STATUS "Found LLVM at: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH
  ${MLIR_CMAKE_DIR}
  ${LLVM_CMAKE_DIR}
)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

add_definitions(${LLVM_DEFINITIONS})

include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/runtime/common/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Put all executables in <build>/bin and libraries in <build>/lib.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(include)
if(SCULPTOR_MLIR_BUILD_RUNTIME)
  add_subdirectory(runtime)
endif()
add_subdirectory(lib)
add_subdirectory(tools)
