/**
 * @file python_inference_backend.hpp
 * @brief Inference backend that runs a model written in Python.
 *
 * This header does not include `<Python.h>` — the interpreter state
 * lives behind a pimpl — so emulator code can include it freely without
 * inheriting Python's macro pollution or its include-order rules.
 */

#ifndef E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP

#include "inference_backend.hpp"

#include <memory>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Runs a Python model in an embedded interpreter, without copying.
 *
 * This is the backend that matters for the near-term work: essentially
 * every emulator we care about — ACE2 above all — exists as a PyTorch
 * model with a Python inference harness around it.  Rewriting those in
 * C++ is not realistic, and exporting them to TorchScript or ONNX is
 * lossy and fragile for anything with data-dependent control flow.
 * Calling the Python that already works is the honest path.
 *
 * ## The Python-side contract
 *
 * The backend imports a module and calls a factory in it to build a
 * model object:
 *
 * ```python
 * def create(config: dict):        # name overridable via python.factory
 *     return MyEmulator(config)
 *
 * class MyEmulator:
 *     def forward(self, inputs: dict, outputs: dict) -> None:
 *         outputs["y"][:] = self.net(inputs["x"])   # write in place
 *
 *     # optional
 *     def input_specs(self)  -> list[dict]: ...
 *     def output_specs(self) -> list[dict]: ...
 *     def finalize(self)     -> None: ...
 * ```
 *
 * `forward` receives two dicts of NumPy arrays.  The arrays in
 * `outputs` alias the emulator's own memory, so the idiomatic
 * implementation writes into them with `[:]` and returns None.  A model
 * that would rather allocate may instead **return** a dict of arrays
 * keyed the same way, and the backend copies them across (with dtype
 * conversion).  Both styles work; the in-place style avoids a copy per
 * field per step.
 *
 * Everything the emulator knows is passed to the factory in `config`:
 * `model_path`, `device`, `batch_size`, `mpi_comm_fortran`, `local_rank`,
 * `global_rank`, the declared tensor specs, and every entry from
 * `InferenceConfig::options` verbatim.
 *
 * ## Zero copy
 *
 * Input and output arrays are NumPy views over the emulator's buffers —
 * see PythonRuntime::wrap_array.  No data is copied on the way in, and
 * none on the way out when the model writes in place.  A model that
 * needs a torch tensor gets one for free with `torch.from_numpy()`,
 * which is itself a view; on CPU the whole path from MCT attribute
 * vector to model input is copy-free.
 *
 * ## Scaling out
 *
 * One interpreter per process, shared by every backend instance in that
 * process, started on first use and shut down with the last.  Under MPI
 * each rank is its own process, so ranks never contend for the GIL.  A
 * model that must communicate across ranks — a spatially decomposed
 * emulator doing halo exchange — rebuilds the communicator on the Python
 * side from the `mpi_comm_fortran` entry in its config dict:
 *
 * ```python
 * from mpi4py import MPI
 * comm = MPI.Comm.f2py(config["mpi_comm_fortran"])
 * ```
 *
 * GPU placement follows the same config: `device` is passed through, and
 * a bare `"cuda"` has already been resolved to `cuda:<local_rank>` by
 * InferenceConfig::device_index(), so ranks land on distinct GPUs
 * without the Python side having to know the node topology.
 *
 * ## Options
 *
 * | option            | default   | meaning                              |
 * |-------------------|-----------|--------------------------------------|
 * | `python.module`   | model_path| module name, or a path to a `.py`    |
 * | `python.factory`  | `create`  | callable in the module that builds it|
 * | `python.method`   | `forward` | method called each step              |
 * | `python.sys_path` | –         | `:`-separated dirs prepended to path |
 *
 * When `python.module` is unset, `model_path` is used, so a config need
 * only say `model_path: /path/to/my_emulator.py`.
 */
class PythonBackend : public InferenceBackend {
public:
  explicit PythonBackend(const InferenceConfig &config);
  ~PythonBackend() override;

  using InferenceBackend::infer;

  /**
   * @brief Start the interpreter, import the module, build the model.
   * @throws std::runtime_error carrying the Python traceback on failure
   */
  void initialize() override;

  /// @copydoc InferenceBackend::infer(const TensorMap&, TensorMap&)
  bool infer(const TensorMap &inputs, TensorMap &outputs) override;

  /** @brief Call the model's optional `finalize()` and drop references. */
  void finalize() override;

  std::string name() const override { return "Python"; }

  /**
   * @brief Specs from the model's `input_specs()` if it has one.
   *
   * Falls back to the config's declared inputs.  Only meaningful after
   * initialize().
   */
  std::vector<TensorSpec> input_specs() const override;

  /** @brief Specs from the model's `output_specs()`.  @see input_specs */
  std::vector<TensorSpec> output_specs() const override;

  /**
   * @brief Python's natural float width for scientific work.
   *
   * NumPy defaults to float64 and the wrapping is zero-copy for either
   * width, so there is no reason to make the emulator down-convert
   * before the call; a model that wants float32 can cast on its side,
   * or simply declare float32 specs.
   */
  DType preferred_dtype() const override { return DType::F64; }

  /** @brief Embedded interpreter version, or "" before initialize(). */
  std::string python_version() const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_PYTHON_INFERENCE_BACKEND_HPP
