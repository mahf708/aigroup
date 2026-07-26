/**
 * @file inference_config.hpp
 * @brief Backend-agnostic configuration for inference backends.
 *
 * One configuration struct serves every backend.  The fields that all
 * backends understand (which backend, which model, which device, what the
 * tensors look like) are typed; everything backend-specific lives in a
 * free-form string map so that adding a backend never requires touching
 * this header.
 */

#ifndef E3SM_EMULATOR_INFERENCE_CONFIG_HPP
#define E3SM_EMULATOR_INFERENCE_CONFIG_HPP

#include "tensor.hpp"

#include <map>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Legacy enumeration of backend types.
 *
 * Retained so that existing call sites built against the original
 * stub-only interface keep compiling.  New code should set
 * InferenceConfig::backend to a registered backend name instead — the
 * registry is open-ended, this enum is not.
 *
 * @see BackendRegistry
 */
enum class BackendType {
  STUB,   ///< No-op / analytic backend, no ML dependencies
  PYTHON, ///< Embedded CPython bridge
  ONNX,   ///< ONNX Runtime
  TORCH   ///< LibTorch (TorchScript)
};

/** @brief Registered backend name corresponding to a legacy BackendType. */
const char *backend_type_name(BackendType type);

/**
 * @brief Configuration for constructing an inference backend.
 *
 * ## Typical construction
 * ```cpp
 * InferenceConfig cfg;
 * cfg.backend    = "onnx";
 * cfg.model_path = "/path/to/emulator.onnx";
 * cfg.device     = "cpu";
 * cfg.inputs  = {TensorSpec("x", {-1, 12}, DType::F32)};
 * cfg.outputs = {TensorSpec("y", {-1,  4}, DType::F32)};
 * ```
 *
 * ## From a namelist-style file
 * ```cpp
 * auto cfg = InferenceConfig::from_file("inference_in");
 * ```
 */
struct InferenceConfig {
  // ── Core, understood by every backend ────────────────────────────

  /**
   * @brief Registered backend name ("stub", "python", "onnx", "torch", …).
   *
   * Empty means "stub".  Unknown names are an error at creation time
   * unless the caller opts into the fallback behaviour.
   */
  std::string backend = "stub";

  /**
   * @brief Path to the serialized model.
   *
   * Interpretation is backend-specific: an `.onnx` file for ONNX Runtime,
   * a TorchScript archive for LibTorch, an importable module name or
   * `.py` file for the Python bridge, ignored by the stub.
   */
  std::string model_path;

  /**
   * @brief Compute device: "cpu", "cuda", "cuda:N", "gpu".
   *
   * "cuda" without an index means "pick from the local MPI rank" — see
   * device_index() and local_rank.
   */
  std::string device = "cpu";

  /** @brief Number of samples per inference call (leading tensor axis). */
  int batch_size = 1;

  /** @brief Emit backend chatter on construction and each step. */
  bool verbose = false;

  // ── Parallel context ─────────────────────────────────────────────

  /**
   * @brief Fortran MPI communicator handle, or -1 if not running under MPI.
   *
   * Passed through to backends that can use it — most importantly the
   * Python bridge, where a model can rebuild the communicator with
   * `mpi4py.MPI.Comm.f2py(handle)` and do its own halo exchange.
   */
  int mpi_comm_fortran = -1;

  /** @brief Rank within this node, used for round-robin GPU assignment. */
  int local_rank = 0;

  /** @brief Global MPI rank, for logging and rank-dependent behaviour. */
  int global_rank = 0;

  // ── Tensor description ───────────────────────────────────────────

  /**
   * @brief Declared model inputs.
   *
   * May be left empty for backends that can introspect the model
   * (ONNX Runtime does; TorchScript and Python generally do not).
   * A negative extent marks a dynamic axis resolved from batch_size.
   */
  std::vector<TensorSpec> inputs;

  /** @brief Declared model outputs.  @see inputs */
  std::vector<TensorSpec> outputs;

  // ── Legacy flat interface ────────────────────────────────────────

  /**
   * @brief Number of input features per grid point.
   *
   * Used by the flat `infer(const double*, double*, int)` convenience
   * path, which presents the model as `[batch_size, input_channels]`.
   * Ignored when `inputs` is populated.
   */
  int input_channels = 0;

  /** @brief Number of output features per grid point.  @see input_channels */
  int output_channels = 0;

  // ── Backend-specific escape hatch ────────────────────────────────

  /**
   * @brief Free-form backend options.
   *
   * Keys are namespaced by convention, e.g. `python.module`,
   * `onnx.intra_op_threads`, `torch.jit_optimize`.  Documented per
   * backend in that backend's header.
   */
  std::map<std::string, std::string> options;

  // ── Option accessors ─────────────────────────────────────────────

  /** @brief Whether @p key is present in options. */
  bool has_option(const std::string &key) const;

  /** @brief String option, or @p fallback if absent. */
  std::string option(const std::string &key,
                     const std::string &fallback = "") const;

  /**
   * @brief Integer option, or @p fallback if absent or unparseable.
   * @throws std::runtime_error if present but not an integer
   */
  int option_int(const std::string &key, int fallback = 0) const;

  /** @brief Floating-point option.  @see option_int */
  double option_double(const std::string &key, double fallback = 0.0) const;

  /**
   * @brief Boolean option.
   *
   * Accepts true/false, yes/no, on/off, 1/0 (case-insensitive).
   * @throws std::runtime_error if present but not a recognized boolean
   */
  bool option_bool(const std::string &key, bool fallback = false) const;

  /** @brief Set an option (overwrites any existing value). */
  void set_option(const std::string &key, const std::string &value);

  // ── Derived queries ──────────────────────────────────────────────

  /** @brief Whether device names a CUDA/GPU device. */
  bool uses_gpu() const;

  /**
   * @brief Resolved GPU index.
   *
   * Returns the explicit index from "cuda:N", otherwise local_rank for a
   * bare "cuda"/"gpu", otherwise -1 for CPU.
   */
  int device_index() const;

  /**
   * @brief Effective input specs.
   *
   * Returns `inputs` when non-empty; otherwise synthesizes a single
   * tensor named "input" of shape `[batch_size, input_channels]` from the
   * legacy channel counts, so the flat interface and the tensor
   * interface describe the same thing.
   */
  std::vector<TensorSpec> effective_inputs() const;

  /** @brief Effective output specs.  @see effective_inputs */
  std::vector<TensorSpec> effective_outputs() const;

  /**
   * @brief Check internal consistency.
   * @throws std::runtime_error with a specific message on the first problem
   */
  void validate() const;

  /** @brief Multi-line human-readable dump, for logs. */
  std::string to_string() const;

  // ── Parsing ──────────────────────────────────────────────────────

  /**
   * @brief Parse a namelist-style configuration from text.
   *
   * The format deliberately matches what `atm.cpp` already parses for
   * `atm_in`: one `key: value` per line, `#` comments, blank lines
   * ignored.  Recognized keys:
   *
   * | key                | meaning                                  |
   * |--------------------|------------------------------------------|
   * | `backend`          | registered backend name                  |
   * | `model_path`       | path to the model                        |
   * | `device`           | cpu / cuda / cuda:N                      |
   * | `batch_size`       | integer                                  |
   * | `verbose`          | boolean                                  |
   * | `input_channels`   | integer (legacy flat interface)          |
   * | `output_channels`  | integer (legacy flat interface)          |
   * | `input`            | tensor spec, repeatable (see below)      |
   * | `output`           | tensor spec, repeatable                  |
   * | anything else      | stored verbatim in options               |
   *
   * A tensor spec is `name:shape:dtype`, where shape is comma-separated
   * and `-1` marks a dynamic axis; dtype defaults to f32.  For example:
   *
   * ```
   * backend: onnx
   * model_path: /models/rad_emulator.onnx
   * input:  state:-1,72:f32
   * output: heating_rate:-1,72:f32
   * onnx.intra_op_threads: 4
   * ```
   *
   * @throws std::runtime_error on a malformed line
   */
  static InferenceConfig from_string(const std::string &text);

  /**
   * @brief Parse a configuration file.
   * @throws std::runtime_error if the file cannot be opened
   * @see from_string for the format
   */
  static InferenceConfig from_file(const std::string &path);
};

/**
 * @brief Parse a `name:shape:dtype` tensor spec.
 *
 * Exposed because emulators sometimes build specs from coupler field
 * lists rather than from a config file.
 *
 * @throws std::runtime_error on a malformed spec
 */
TensorSpec parse_tensor_spec(const std::string &text);

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_CONFIG_HPP
