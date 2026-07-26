# `components/emulators` — inference layer prototype

This tree mirrors the layout of `components/emulators` on the E3SM
`emulators/coupler-infrastructure` branch, so that everything here drops
into an E3SM checkout at the same relative path with no edits.

It is not a fork of that branch. Only the files this work adds or
changes are present:

```
common/src/inference/          the inference layer (see its README)
common/tests/                  its unit tests
standalone/                    a no-CIME build harness and demo driver
```

## Why it lives in aigroup

`E3SM-Project/aigroup` is where the group shares prototypes, and this is
one — a point of departure to review and argue with before it goes into
the long-lived feature branch. The mirrored path is so that agreeing to
it is a copy, not a port.

## Dropping it into E3SM

```console
git checkout emulators/coupler-infrastructure   # or a branch off it
rsync -a /path/to/aigroup/components/emulators/ components/emulators/
```

Then two one-line hooks into the existing CMake:

**`components/emulators/common/src/CMakeLists.txt`** — replace the two
hardcoded inference sources with the generated list:

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/inference/inference.cmake)

add_library(emulator_common
  ${EMULATOR_INFERENCE_SOURCES}
  emulator.cpp
  emulator_c_api.cpp
  ${EMULATOR_COMMON_F90_SOURCES}
)
target_include_directories(emulator_common PUBLIC ${EMULATOR_INFERENCE_INCLUDES})
target_link_libraries(emulator_common PUBLIC ${EMULATOR_INFERENCE_LIBS})
target_compile_definitions(emulator_common PUBLIC ${EMULATOR_INFERENCE_DEFS})
```

**`components/emulators/common/tests/CMakeLists.txt`** — replace the two
hand-written inference test targets with:

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/inference_tests.cmake)
```

Nothing else changes. `emulator.cpp`, the coupler API, the Fortran
bridge, and `emulatoratm` are untouched.

## Compatibility with what is already there

The existing `test_inference_config.cpp` and
`test_inference_stub_backend.cpp` assertions are preserved verbatim and
still pass, including the ones that expect `create_backend(BackendType,
config)` to return a `"Stub"` and to fall back rather than throw on an
unknown type. New tests are added around them.

`InferenceConfig::input_channels` / `output_channels` and the flat
`infer(const double*, double*, int)` overload also still work; the flat
overload is now implemented once in the base class in terms of the
tensor interface.

## Relationship to `mahf708/emulators/dataviews`

The two are complementary and independent. `DataView` is the
coupler-facing container; the inference layer's `TensorView` is the
model-facing one. `common/src/inference/data_view_bridge.hpp` is the
whole of the mapping between them, and
`common/tests/test_inference_data_view_bridge.cpp` covers it.

Both are conditional: the bridge test is added only when
`common/src/data/data_view.hpp` exists, so this tree builds green on a
branch with the data layer and on one without. It has been checked
against both.

## Status

Built and tested with GCC 13 / C++17 in four configurations — stub only,
`+python` (CPython 3.11, NumPy 2.4), `+onnx` (ONNX Runtime 1.28),
`+torch` (LibTorch 2.13) — and against the `dataviews` branch's data
container. All suites green.
