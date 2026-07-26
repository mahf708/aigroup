# ─────────────────────────────────────────────────────────────────────
# inference.cmake — feature detection and source list for the inference
# layer.
#
# Include this from `common/src/CMakeLists.txt` and it sets:
#
#   EMULATOR_INFERENCE_SOURCES   sources to add to emulator_common
#   EMULATOR_INFERENCE_INCLUDES  include directories to expose
#   EMULATOR_INFERENCE_LIBS      libraries to link
#   EMULATOR_INFERENCE_DEFS      compile definitions (EMULATOR_HAVE_*)
#
# It is written as an include rather than an add_subdirectory so that it
# drops into either the `emulators/coupler-infrastructure` layout or the
# `emulators/dataviews` layout without either of their CMakeLists having
# to change shape.
#
# Every optional backend is OFF by default: a build with no ML libraries
# present must still configure, compile, and pass its tests.
# ─────────────────────────────────────────────────────────────────────

option(EMULATOR_ENABLE_PYTHON      "Build the embedded-Python inference backend" OFF)
option(EMULATOR_ENABLE_ONNXRUNTIME "Build the ONNX Runtime inference backend"    OFF)
option(EMULATOR_ENABLE_TORCH       "Build the LibTorch inference backend"        OFF)

set(EMULATOR_INFERENCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# ── Always built: no external dependencies ──────────────────────────
set(EMULATOR_INFERENCE_SOURCES
  ${EMULATOR_INFERENCE_DIR}/tensor.cpp
  ${EMULATOR_INFERENCE_DIR}/inference_config.cpp
  ${EMULATOR_INFERENCE_DIR}/inference_backend.cpp
  ${EMULATOR_INFERENCE_DIR}/backend_registry.cpp
  ${EMULATOR_INFERENCE_DIR}/create_inference_backend.cpp
  ${EMULATOR_INFERENCE_DIR}/stub_inference_backend.cpp
)
set(EMULATOR_INFERENCE_INCLUDES ${EMULATOR_INFERENCE_DIR})
set(EMULATOR_INFERENCE_LIBS "")
set(EMULATOR_INFERENCE_DEFS "")

# ── Python ──────────────────────────────────────────────────────────
if(EMULATOR_ENABLE_PYTHON)
  # Development.Embed (not Development.Module) is what an application
  # embedding the interpreter needs: it brings in libpython itself.
  find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)

  list(APPEND EMULATOR_INFERENCE_SOURCES
    ${EMULATOR_INFERENCE_DIR}/python_runtime.cpp
    ${EMULATOR_INFERENCE_DIR}/python_inference_backend.cpp
  )
  list(APPEND EMULATOR_INFERENCE_LIBS Python3::Python)
  list(APPEND EMULATOR_INFERENCE_DEFS EMULATOR_HAVE_PYTHON)

  # NumPy is a runtime requirement, not a build one — the backend wraps
  # memory through numpy.frombuffer rather than the NumPy C API, so no
  # headers are needed here.  Check for it anyway so the failure lands at
  # configure time with an actionable message instead of at the first
  # timestep of a queued job.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
    RESULT_VARIABLE _numpy_missing
    OUTPUT_VARIABLE _numpy_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_numpy_missing)
    message(FATAL_ERROR
      "EMULATOR_ENABLE_PYTHON=ON but numpy is not importable from "
      "${Python3_EXECUTABLE}. The Python inference backend exchanges data "
      "as NumPy arrays; install numpy into that interpreter.")
  endif()
  message(STATUS "Emulator inference: Python ${Python3_VERSION} "
                 "(numpy ${_numpy_version})")
endif()

# ── ONNX Runtime ────────────────────────────────────────────────────
if(EMULATOR_ENABLE_ONNXRUNTIME)
  # ONNX Runtime ships no CMake config package in its release tarballs,
  # so locate the pieces by hand.  ONNXRUNTIME_ROOT points at an
  # unpacked release (or anything with include/ and lib/).
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include include/onnxruntime
                  include/onnxruntime/core/session)
  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64)

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
      "EMULATOR_ENABLE_ONNXRUNTIME=ON but ONNX Runtime was not found. "
      "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime (the directory holding "
      "include/ and lib/).")
  endif()

  list(APPEND EMULATOR_INFERENCE_SOURCES
    ${EMULATOR_INFERENCE_DIR}/onnx_inference_backend.cpp)
  list(APPEND EMULATOR_INFERENCE_INCLUDES ${ONNXRUNTIME_INCLUDE_DIR})
  list(APPEND EMULATOR_INFERENCE_LIBS ${ONNXRUNTIME_LIBRARY})
  list(APPEND EMULATOR_INFERENCE_DEFS EMULATOR_HAVE_ONNXRUNTIME)
  message(STATUS "Emulator inference: ONNX Runtime ${ONNXRUNTIME_LIBRARY}")
endif()

# ── LibTorch ────────────────────────────────────────────────────────
if(EMULATOR_ENABLE_TORCH)
  # Works with either an unpacked libtorch distribution or the one
  # inside a pip-installed torch; for the latter, point CMAKE_PREFIX_PATH
  # at `python -c "import torch; print(torch.utils.cmake_prefix_path)"`.
  find_package(Torch REQUIRED)

  list(APPEND EMULATOR_INFERENCE_SOURCES
    ${EMULATOR_INFERENCE_DIR}/torch_inference_backend.cpp)
  list(APPEND EMULATOR_INFERENCE_LIBS ${TORCH_LIBRARIES})
  list(APPEND EMULATOR_INFERENCE_DEFS EMULATOR_HAVE_TORCH)
  message(STATUS "Emulator inference: LibTorch ${Torch_DIR}")
endif()

if(NOT EMULATOR_ENABLE_PYTHON AND NOT EMULATOR_ENABLE_ONNXRUNTIME
   AND NOT EMULATOR_ENABLE_TORCH)
  message(STATUS "Emulator inference: stub backend only "
                 "(enable others with -DEMULATOR_ENABLE_{PYTHON,ONNXRUNTIME,TORCH}=ON)")
endif()
