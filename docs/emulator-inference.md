# Emulator Inference Layer

A generic inference layer for E3SM emulators, living at
`components/emulators/common/src/inference`. One API on the coupler
side, a choice of backends underneath, and no ML library needed to build
or test the code around it.

This follows through on two action items:

- **2026-07-14** — "a rough draft of a *simple* inference setup as a
  point of departure", and "structural similarities between components
  and processes with an eye toward data structures/architecture".
- **2026-07-21** — the group settled on "a C++ to Python bridge for
  Python emulators" over a fixed-operation framework, wanted a
  memory-managed data container, and wanted to test without CIME case
  builds.

!!! note "Where the code is"
    In this repo under
    [`components/emulators/`](https://github.com/E3SM-Project/aigroup/tree/main/components/emulators),
    mirroring the E3SM path exactly so that adopting it is a copy rather
    than a port. The E3SM home is the
    `emulators/coupler-infrastructure` branch.

## What was already there

The `emulators/coupler-infrastructure` branch had the skeleton: an
abstract `InferenceBackend`, an `InferenceConfig` with two channel
counts, a `BackendType` enum, a switch-statement factory, and a stub
backend that did nothing.

That is the right shape for a placeholder, but three things had to
change before a real model could go behind it:

1. **`infer(const double*, double*, int)` is too narrow.** Doubles only,
   two dimensions only, one unnamed input and one unnamed output. ACE2
   wants `[batch, channel, lat, lon]` float32 with named tensors.
2. **An enum plus a switch is closed.** Every new backend means editing
   core files, and a backend that only exists when a library is present
   means `#ifdef`s in the factory.
3. **Nothing carried a model's identity.** No model path, no device, no
   dtype, no per-backend options.

## What it looks like now

```cpp
#include "create_inference_backend.hpp"
using namespace emulator::inference;

auto config  = InferenceConfig::from_file("inference_in");
auto backend = create_backend(config);
backend->initialize();

TensorMap inputs;
inputs.set(TensorView::from_doubles("state", state.data(), {ncol, nlev}));
TensorMap outputs;
outputs.set(TensorView::from_doubles("tendency", tendency.data(), {ncol, nlev}));

backend->infer(inputs, outputs);   // every timestep

backend->finalize();
```

Both maps belong to the caller. The backend writes into the caller's
output memory and allocates nothing per step, which is what a run of
half a million timesteps needs.

The old flat call still works — it is now implemented once in the base
class in terms of the tensor interface — and the original tests pass
unchanged.

## The data container

The 07-21 call asked for a memory-managed data container. There are two,
and keeping them separate is deliberate:

| | `DataView` | `TensorView` |
|--|-----------|--------------|
| faces | the coupler | the model |
| dtype | `double` | f32 / f64 / i32 / i64 |
| rank | 2 (fields × points) | any |
| carries | names, units, decomposition | name and shape |
| owns memory | yes | no |

`DataView` comes from the `mahf708/emulators/dataviews` work and is
unchanged. `TensorView` is new here. Neither absorbs the other;
`data_view_bridge.hpp` is the whole of the mapping, so an emulator that
only does I/O needs no inference layer and a standalone inference test
needs no `DataView`.

```cpp
#include "data_view_bridge.hpp"

TensorMap inputs;
inputs.set(tensor_from_data_view(import_view, "state"));
TensorMap outputs;
outputs.set(tensor_from_data_view(export_view, "tendency"));
backend->infer(inputs, outputs);   // writes into export_view's buffer
```

This is the `is-a` caution from the 07-21 notes applied: a tensor is not
a kind of `DataView` and a `DataView` is not a kind of tensor. They
compose.

## The backends

Four, three of them real.

=== "python"

    Runs a model written in Python in an embedded interpreter.

    **This is the one that matters near term.** Every emulator we care
    about — ACE2 above all — exists as a PyTorch model with a Python
    harness around it. Rewriting those in C++ is not realistic, and
    exporting them is lossy for anything with data-dependent control
    flow, and is exactly where the harness (normalization, masking,
    diagnostics) tends to get lost.

    ```python
    def create(config: dict):
        return MyEmulator(config)

    class MyEmulator:
        def forward(self, inputs: dict, outputs: dict) -> None:
            outputs["tendency"][:] = self.net(inputs["state"])
    ```

    The arrays are NumPy views over the emulator's own buffers, so
    `outputs["y"][:] = ...` writes straight into the memory the coupler
    reads. Nothing is copied in either direction. `torch.from_numpy()`
    on an input is itself a view, so on CPU the whole path from MCT
    attribute vector to model input is copy-free.

=== "onnx"

    Runs a serialized `.onnx` model through ONNX Runtime.

    The backend to reach for once a model has stopped changing. A
    `.onnx` file describes itself, so `input_specs()` reports what the
    model really wants and a field-packing mistake is caught at startup
    instead of showing up as a bad forecast. No Python at runtime is a
    real operational simplification on a compute node.

=== "torch"

    Runs a TorchScript archive through LibTorch.

    The middle ground: a shorter step from PyTorch than an ONNX export,
    no Python at runtime, but no introspection either — the config has
    to declare the interface, and get it right.

=== "stub"

    Always built, no dependencies.

    Its `copy` and `affine` modes move real data end to end, so a
    coupling test can assert that a field *arrived*, not merely that
    nothing crashed. A no-op cannot tell "the pipeline works" from
    "nothing ran".

Adding a fifth touches nothing in the directory:

```cpp
BackendRegistry::instance().register_backend(
    "my_backend",
    [](const InferenceConfig &cfg) { return std::make_shared<MyBackend>(cfg); },
    "What it does");
```

## Configuration

Namelist-style, the same `key: value` format `atm_in` already uses:

``` { .yaml .annotate }
backend: onnx                                  # (1)!
model_path: /models/rad_emulator.onnx
device: cuda                                   # (2)!
batch_size: 512

input:  state:-1,72:f32                        # (3)!
output: heating_rate:-1,72:f32

onnx.intra_op_threads: 4                       # (4)!
```

1. Any registered backend name. Unknown names fail at startup with a
   list of what this build actually supports — the common case being a
   namelist asking for `torch` in a build configured without LibTorch.
2. Bare `cuda` resolves to `cuda:<local_rank>`, so ranks land on
   distinct GPUs without anything having to know the node topology.
3. `name:shape:dtype`, with `-1` for a dynamic axis. In practice that
   is the column count, which is a decomposition detail rather than a
   model one.
4. Any key the parser does not recognize becomes a backend option
   verbatim. That is what lets a backend add a knob without touching
   the parser.

## Running an emulator under MPI

Each rank is its own process and so its own interpreter; ranks never
contend for the GIL. A model that must communicate across ranks — a
spatially decomposed emulator doing halo exchange, which is what ACE2 at
scale would need — rebuilds the communicator on the Python side:

```python
from mpi4py import MPI
comm = MPI.Comm.f2py(config["mpi_comm_fortran"])
```

That is the *same* communicator the component was given, so the model is
collectives-compatible with the rest of the run.

## Testing without CIME

The 07-21 call wanted to work without CIME case builds. `standalone/`
builds the layer and its tests with no E3SM, no CIME, no MPI, and no
Fortran:

```console
cmake -S components/emulators/standalone -B build \
      -DCATCH2_INCLUDE_DIR=/path/to/catch2/single_include \
      -DEMULATOR_ENABLE_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`inference_demo` stands in for the coupler — it allocates buffers,
builds a backend from a config file, and steps it, printing what comes
back. The quickest way to check a new model before it goes anywhere near
a coupled case:

```console
./build/inference_demo my_inference_in 8 4    # 8 columns, 4 steps
```

## Two findings worth knowing

!!! warning "The Python interpreter must never be finalized"
    `Py_FinalizeEx()` followed by a second `Py_Initialize()` looks like
    it should work. It does not: any extension module holding C-level
    state — NumPy, and therefore PyTorch — refuses to load again with
    `cannot load module more than once per process`. A backend that
    finalized on the way out would work exactly once per process, which
    surfaces as a failure in the *second* instance of a multi-instance
    run and nowhere else.

    The interpreter is now a process-lifetime resource, like the MPI
    runtime. Backends still release their model references in
    `finalize()`, which is the memory that actually matters.

!!! warning "ONNX Runtime's type info is a non-owning view"
    `session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo()` returns a
    view into the `TypeInfo` temporary, which dies at the end of the
    full expression. The dangling read produces plausible-looking
    garbage element types rather than a crash. The `TypeInfo` has to be
    a named local.

## Status and what is next

Built and tested with GCC 13 / C++17 in four configurations — stub only,
`+python` (CPython 3.11, NumPy 2.4), `+onnx` (ONNX Runtime 1.28),
`+torch` (LibTorch 2.13) — and against the `dataviews` branch's data
container. All suites green.

- [ ] Wire `EmulatorAtm::run_impl` to a backend, replacing the TODOs in
      `atm.cpp` — this layer is the piece those TODOs were waiting on
- [ ] Map MCT coupling field lists onto `TensorSpec`s automatically, so
      the namelist does not restate what `seq_flds_mod` already knows
- [ ] Try ACE2 behind the Python backend: single rank first, then
      spatially decomposed with `mpi4py`
- [ ] Decide whether process-level emulation reuses this interface as-is
      or wants a narrower one
