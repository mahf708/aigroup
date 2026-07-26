# ─────────────────────────────────────────────────────────────────────
# Standalone build of the emulator inference layer.
#
# Builds `src/inference` and its tests with no E3SM, no CIME, no MPI and
# no Fortran — the "desktop-friendly" path the 2026-07-21 discussion
# asked for.  Point it at a Catch2 v2 single header and it configures
# anywhere a C++17 compiler exists.
#
#   cmake -S components/emulators/standalone -B build \
#         -DCATCH2_INCLUDE_DIR=/path/to/catch2/single_include \
#         -DEMULATOR_ENABLE_PYTHON=ON
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#
# Inside an E3SM checkout, CATCH2_INCLUDE_DIR defaults to the copy in
# externals/ekat, so the flag can be dropped.
# ─────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.16)
project(emulator_inference_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

get_filename_component(EMULATORS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(INFERENCE_SRC ${EMULATORS_ROOT}/common/src/inference)
set(TESTS_SRC ${EMULATORS_ROOT}/common/tests)

# ── Catch2 ──────────────────────────────────────────────────────────
# Prefer an explicit path, then E3SM's ekat copy if we happen to be
# inside a checkout.
get_filename_component(E3SM_ROOT "${EMULATORS_ROOT}/../.." ABSOLUTE)
if(NOT CATCH2_INCLUDE_DIR)
  set(CATCH2_INCLUDE_DIR "${E3SM_ROOT}/externals/ekat/extern/Catch2/single_include"
      CACHE PATH "Directory containing catch2/catch.hpp")
endif()

set(EMULATOR_BUILD_TESTS ON CACHE BOOL "Build the inference unit tests")
if(EMULATOR_BUILD_TESTS AND NOT EXISTS "${CATCH2_INCLUDE_DIR}/catch2/catch.hpp")
  message(FATAL_ERROR
    "Catch2 v2 not found at ${CATCH2_INCLUDE_DIR}.\n"
    "Pass -DCATCH2_INCLUDE_DIR=<dir containing catch2/catch.hpp>, or\n"
    "-DEMULATOR_BUILD_TESTS=OFF to build the library only.")
endif()

# ── The library ─────────────────────────────────────────────────────
include(${INFERENCE_SRC}/inference.cmake)

# Build the DataView container alongside when it is present, so that the
# bridge between the two can be compiled and tested here too.  Absent,
# everything else still builds.
set(EMULATOR_DATA_SOURCES "")
set(EMULATOR_DATA_INCLUDES "")
if(EXISTS ${EMULATORS_ROOT}/common/src/data/data_view.hpp)
  file(GLOB EMULATOR_DATA_SOURCES ${EMULATORS_ROOT}/common/src/data/*.cpp)
  set(EMULATOR_DATA_INCLUDES ${EMULATORS_ROOT}/common/src/data)
  message(STATUS "Emulator inference: building against the DataView container")
endif()

add_library(emulator_common ${EMULATOR_INFERENCE_SOURCES} ${EMULATOR_DATA_SOURCES})
target_compile_features(emulator_common PUBLIC cxx_std_17)
target_include_directories(emulator_common PUBLIC
  ${EMULATOR_INFERENCE_INCLUDES} ${EMULATOR_DATA_INCLUDES})
target_link_libraries(emulator_common PUBLIC ${EMULATOR_INFERENCE_LIBS})
target_compile_definitions(emulator_common PUBLIC ${EMULATOR_INFERENCE_DEFS})

# ── Tests ───────────────────────────────────────────────────────────
if(EMULATOR_BUILD_TESTS)
  enable_testing()
  # inference_tests.cmake resolves its sources relative to its own
  # location, so including it from here works unchanged.
  include(${TESTS_SRC}/inference_tests.cmake)
endif()

# ── Demo driver ─────────────────────────────────────────────────────
add_executable(inference_demo ${CMAKE_CURRENT_SOURCE_DIR}/inference_demo.cpp)
target_link_libraries(inference_demo PRIVATE emulator_common)

message(STATUS "")
message(STATUS "Emulator inference standalone build")
message(STATUS "  python   : ${EMULATOR_ENABLE_PYTHON}")
message(STATUS "  onnx     : ${EMULATOR_ENABLE_ONNXRUNTIME}")
message(STATUS "  torch    : ${EMULATOR_ENABLE_TORCH}")
message(STATUS "  tests    : ${EMULATOR_BUILD_TESTS}")
message(STATUS "")
