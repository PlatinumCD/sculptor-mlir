#!/usr/bin/env bash
set -euo pipefail

readonly TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${TEST_DIR}/../.." && pwd)"
readonly BUILD_DIR="${GOLEM_BUILD_ROOT:-${PROJECT_ROOT}/build}/runtime/host-tests"
readonly CXX="${CXX:-c++}"

mkdir -p -- "${BUILD_DIR}"

mapfile -t runtime_sources < <(
    find "${PROJECT_ROOT}/runtime/src" -maxdepth 1 -name '*.cpp' -print |
        sort
)

"${CXX}" \
    -std=c++20 \
    -O2 \
    -g \
    -Wall \
    -Wextra \
    -Wpedantic \
    -Werror \
    -DMITTENS_MEMREF_COPY_TEST_HOOK=1 \
    "-I${PROJECT_ROOT}/runtime/include" \
    "${runtime_sources[@]}" \
    "${PROJECT_ROOT}/platform/mlir-runtime.cpp" \
    "${TEST_DIR}/runtime_test.cpp" \
    -o "${BUILD_DIR}/runtime-test"

exec "${BUILD_DIR}/runtime-test"
