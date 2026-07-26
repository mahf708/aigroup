# ─────────────────────────────────────────────────────────────────────
# inference_tests.cmake — test targets for the inference layer.
#
# Include from `common/tests/CMakeLists.txt`.  Expects CATCH2_INCLUDE_DIR
# and the `emulator_common` target, both of which the existing test
# CMakeLists already set up, plus the EMULATOR_ENABLE_* options that
# inference.cmake defined.
#
# Backend-specific tests are only added when their backend is compiled
# in, so a stub-only build still runs a full, green suite.
# ─────────────────────────────────────────────────────────────────────

set(EMULATOR_INFERENCE_TEST_DIR ${CMAKE_CURRENT_LIST_DIR})

# Where generate_test_models.py drops its fixtures.
set(EMULATOR_TEST_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_models)

function(emulator_add_inference_test name)
  add_executable(${name} ${EMULATOR_INFERENCE_TEST_DIR}/${name}.cpp)
  target_link_libraries(${name} PRIVATE emulator_common)
  target_include_directories(${name} PRIVATE ${CATCH2_INCLUDE_DIR})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# ── Always available ────────────────────────────────────────────────
emulator_add_inference_test(test_inference_tensor)
emulator_add_inference_test(test_inference_config)
emulator_add_inference_test(test_inference_registry)
emulator_add_inference_test(test_inference_stub_backend)

# ── DataView bridge ─────────────────────────────────────────────────
#
# Only meaningful where the DataView container exists (the
# `emulators/dataviews` work).  Detecting it rather than assuming it
# means this file works unchanged on a branch that has the data layer
# and on one that does not.
if(EXISTS "${EMULATOR_INFERENCE_DIR}/../data/data_view.hpp")
  emulator_add_inference_test(test_inference_data_view_bridge)
  target_include_directories(test_inference_data_view_bridge PRIVATE
    ${EMULATOR_INFERENCE_DIR}/../data)
  message(STATUS "Emulator inference tests: DataView bridge test enabled")
endif()

# ── Model fixtures for the ONNX and LibTorch tests ──────────────────
#
# Generated at configure time when a Python with torch is around.  A
# failure here is not fatal: the tests that need the fixtures check for
# them and report a skip, so a machine without torch still builds and
# runs everything else.
if(EMULATOR_ENABLE_ONNXRUNTIME OR EMULATOR_ENABLE_TORCH)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    file(MAKE_DIRECTORY ${EMULATOR_TEST_MODEL_DIR})
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
              ${EMULATOR_INFERENCE_TEST_DIR}/generate_test_models.py
              ${EMULATOR_TEST_MODEL_DIR}
      RESULT_VARIABLE _gen_result
      OUTPUT_VARIABLE _gen_output
      ERROR_VARIABLE _gen_error)
    if(_gen_result EQUAL 0)
      message(STATUS "Emulator inference tests: ${_gen_output}")
    else()
      message(WARNING
        "Emulator inference tests: could not generate model fixtures; "
        "ONNX/LibTorch tests will report skips.\n${_gen_error}")
    endif()
  else()
    message(WARNING
      "Emulator inference tests: no Python interpreter, so ONNX/LibTorch "
      "model fixtures were not generated; those tests will report skips.")
  endif()
endif()

# ── Python backend ──────────────────────────────────────────────────
if(EMULATOR_ENABLE_PYTHON)
  emulator_add_inference_test(test_inference_python)
  # Point the test at the in-tree helper package so that running the
  # suite needs no pip install.
  target_compile_definitions(test_inference_python PRIVATE
    EMULATOR_PYTHON_PACKAGE_DIR="${EMULATOR_INFERENCE_DIR}/python")
endif()

# ── ONNX Runtime backend ────────────────────────────────────────────
if(EMULATOR_ENABLE_ONNXRUNTIME)
  emulator_add_inference_test(test_inference_onnx)
  target_compile_definitions(test_inference_onnx PRIVATE
    EMULATOR_TEST_MODEL_DIR="${EMULATOR_TEST_MODEL_DIR}")
endif()

# ── LibTorch backend ────────────────────────────────────────────────
if(EMULATOR_ENABLE_TORCH)
  emulator_add_inference_test(test_inference_torch)
  target_compile_definitions(test_inference_torch PRIVATE
    EMULATOR_TEST_MODEL_DIR="${EMULATOR_TEST_MODEL_DIR}")
endif()
