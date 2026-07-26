# Emulator inference layer

Generic inference for E3SM emulators: one API for the coupler side, a
choice of backends underneath, and no ML library required to build or
test the code that surrounds it.

This directory answers the action item from the 2026-07-14 call ("a
rough draft of a *simple* inference setup as a point of departure") and
the approach the 2026-07-21 call settled on ("a C++ to Python bridge for
Python emulators", plus a memory-managed data container).

## Layout

| file | role |
|------|------|
| `tensor.hpp/.cpp` | `DType`, `TensorView`, `Tensor`, `TensorMap` — the data container for inference |
| `inference_config.hpp/.cpp` | `InferenceConfig`, namelist parsing, `TensorSpec` |
| `inference_backend.hpp/.cpp` | the abstract backend interface |
| `backend_registry.hpp/.cpp` | open, string-keyed factory registry |
| `create_inference_backend.hpp/.cpp` | the front door — include this one |
| `stub_inference_backend.hpp/.cpp` | dependency-free backend, always built |
| `python_runtime.hpp/.cpp` | embedded CPython: lifetime, GIL, zero-copy NumPy |
| `python_inference_backend.hpp/.cpp` | run a model written in Python |
| `onnx_inference_backend.hpp/.cpp` | run a serialized `.onnx` model |
| `torch_inference_backend.hpp/.cpp` | run a TorchScript archive |
| `data_view_bridge.hpp` | header-only glue to the `DataView` container |
| `inference.cmake` | feature detection and source list |
| `python/e3sm_emulator_inference/` | the Python side of the bridge |

## The shape of it

```cpp
#include "create_inference_backend.hpp"
using namespace emulator::inference;

// Once, at init.
auto config  = InferenceConfig::from_file("inference_in");
auto backend = create_backend(config);
backend->initialize();

// Buffers allocated once and reused for the whole run.
std::vector<double> state(ncol * nlev), tendency(ncol * nlev);

TensorMap inputs;
inputs.set(TensorView::from_doubles("state", state.data(), {ncol, nlev}));
TensorMap outputs;
outputs.set(TensorView::from_doubles("tendency", tendency.data(), {ncol, nlev}));

// Every timestep.
backend->infer(inputs, outputs);

// At the end.
backend->finalize();
```

Both maps belong to the caller. The backend writes into the caller's
output memory and allocates nothing per step — which is what a run of
half a million timesteps needs.

For a pointwise emulator that does not want to build a `TensorMap`, the
flat overload still works and is implemented in terms of the above:

```cpp
backend->infer(inputs.data(), outputs.data(), ncol);
```

## Choosing a backend

|  | `python` | `onnx` | `torch` | `stub` |
|--|----------|--------|---------|--------|
| runtime dependency | libpython + numpy | libonnxruntime | libtorch | none |
| model source | any Python | `.onnx` export | `torch.jit` archive | – |
| self-describing | if the model says so | yes | no | – |
| control flow | anything | awkward | with `script` | – |
| copies (CPU) | none | none | one each way | none |

**Start with `python`.** It is the only backend that can run the models
that actually exist — ACE2 among them — without an export step, and the
export step is exactly where a model's Python harness (normalization,
masking, diagnostics) tends to get lost. It is also the only one where
iterating on the model does not mean re-exporting an artifact.

**Move to `onnx` once a model is settled.** The `.onnx` file carries its
own input and output names, shapes, and dtypes, so `input_specs()`
reports what the model really wants and a field-packing mistake is
caught at startup rather than showing up as a bad forecast. No Python at
runtime is a real operational simplification on a compute node.

**`torch` is the middle ground** — a shorter step from PyTorch than an
ONNX export, no Python at runtime, but no introspection either, so the
config has to declare the interface and get it right.

**`stub` is for the plumbing.** Its `copy` and `affine` modes move real
data end to end, so a coupling test can assert that a field arrived,
not merely that nothing crashed.

## Config

Namelist-style, deliberately the same `key: value` format `atm_in`
already uses:

```
backend: onnx
model_path: /models/rad_emulator.onnx
device: cuda
batch_size: 512

input:  state:-1,72:f32
output: heating_rate:-1,72:f32

onnx.intra_op_threads: 4
```

A tensor spec is `name:shape:dtype`, with `-1` for a dynamic axis (in
practice the column count, which is a decomposition detail). Any key the
parser does not recognize becomes a backend option verbatim — that is
what lets a backend add a knob without touching the parser.

`device: cuda` without an index resolves to `cuda:<local_rank>`, so
ranks land on distinct GPUs without anything having to know the node
topology.

Per-backend options are documented in each backend's header.

## Adding a backend

Nothing in this directory needs editing. Subclass `InferenceBackend`,
then register:

```cpp
BackendRegistry::instance().register_backend(
    "my_backend",
    [](const InferenceConfig &cfg) {
      return std::make_shared<MyBackend>(cfg);
    },
    "What it does, for the diagnostics listing");
```

There is no enum to extend and no switch to edit. `BackendType` still
exists, and `create_backend(BackendType, config)` still works, but only
so that code written against the original stub-only interface keeps
compiling; new code should use names.

Built-ins are registered explicitly in `register_builtin_backends()`
rather than by static initializer, because a static initializer in a
static archive gets dropped by the linker when nothing references the
object file — a failure mode that looks like "my backend is not
registered" and takes an afternoon to find.

## The Python contract

```python
def create(config: dict):
    return MyEmulator(config)

class MyEmulator:
    def forward(self, inputs: dict, outputs: dict) -> None:
        outputs["tendency"][:] = self.net(inputs["state"])
```

The arrays in `inputs` and `outputs` are NumPy views over the
emulator's own buffers. `outputs["y"][:] = ...` writes straight into the
memory the coupler will read; nothing is copied in either direction. A
model that prefers to allocate can return a dict instead and the backend
copies it across, converting dtype.

Optional `input_specs()`, `output_specs()`, and `finalize()` are used
when present.

`config` carries `model_path`, `device`, `device_index`, `batch_size`,
`mpi_comm_fortran`, `local_rank`, `global_rank`, the declared specs, and
every option from the namelist. `e3sm_emulator_inference.ModelConfig`
does the string coercion; `reference_models.py` has worked examples of
every variant, including an MPI-aware one and a torch checkpoint loader.

Two implementation notes that matter:

- **The wrapping uses `numpy.frombuffer` on a `memoryview`, not the
  NumPy C API.** So the build needs no NumPy headers and cannot break on
  a NumPy ABI change. NumPy is still required at runtime.
- **The interpreter is started once and never finalized.** This is not
  an oversight. `Py_FinalizeEx()` followed by a second `Py_Initialize()`
  fails for any extension module holding C-level state — NumPy, and
  therefore PyTorch — with `cannot load module more than once per
  process`. A backend that finalized would work exactly once per run.
  Finalizing at exit instead would put arbitrary Python teardown after
  `MPI_Finalize()`. Backends still release their model references in
  `finalize()`, which is the memory that matters.

### Scaling

One interpreter per process, shared by every backend instance in it.
Under MPI each rank is its own process, so ranks never contend for the
GIL. A model that needs to communicate across ranks — a spatially
decomposed emulator doing halo exchange, which is what ACE2 at scale
would need — rebuilds the communicator on the Python side:

```python
from mpi4py import MPI
comm = MPI.Comm.f2py(config["mpi_comm_fortran"])
```

That is the *same* communicator the component was given, so the model is
collectives-compatible with the rest of the run.

## The data container

`DataView` (from the `emulators/dataviews` work) is the coupler-facing
container: doubles, fields × points, names, units, a decomposition for
parallel I/O. `TensorView` here is the model-facing one: any rank, any
dtype, no metadata.

Neither should absorb the other, and `data_view_bridge.hpp` is the whole
of the mapping between them — so `data/` and `inference/` stay
independent, and an emulator that only does I/O needs no inference layer.

```cpp
#include "data_view_bridge.hpp"

TensorMap inputs;
inputs.set(tensor_from_data_view(import_view, "state"));
TensorMap outputs;
outputs.set(tensor_from_data_view(export_view, "tendency"));
backend->infer(inputs, outputs);
```

Zero-copy: the backend writes directly into the export `DataView`'s
buffer.

## Building

Every backend is off by default; a build with no ML libraries present
still configures, compiles, and passes its whole test suite.

From `common/src/CMakeLists.txt`:

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/inference/inference.cmake)

add_library(emulator_common ${EMULATOR_INFERENCE_SOURCES} ...)
target_include_directories(emulator_common PUBLIC ${EMULATOR_INFERENCE_INCLUDES})
target_link_libraries(emulator_common PUBLIC ${EMULATOR_INFERENCE_LIBS})
target_compile_definitions(emulator_common PUBLIC ${EMULATOR_INFERENCE_DEFS})
```

and from `common/tests/CMakeLists.txt`:

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/inference_tests.cmake)
```

Flags:

```
-DEMULATOR_ENABLE_PYTHON=ON
-DEMULATOR_ENABLE_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime
-DEMULATOR_ENABLE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

## Testing without CIME

`../../standalone/` builds this directory and its tests with no E3SM, no
CIME, no MPI and no Fortran — the desktop-friendly path the 2026-07-21
discussion asked for:

```console
cmake -S components/emulators/standalone -B build \
      -DCATCH2_INCLUDE_DIR=/path/to/catch2/single_include \
      -DEMULATOR_ENABLE_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`inference_demo` in the same directory is a stand-in for the coupler:
it allocates buffers, builds a backend from a config file, and steps it,
printing what comes back. It is the quickest way to check that a new
model works before putting it anywhere near a coupled case.

```console
./build/inference_demo                    # stub, built-in config
./build/inference_demo inference_in 8 4   # your config, 8 columns, 4 steps
```
