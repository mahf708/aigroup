/**
 * @file inference_backend.hpp
 * @brief Abstract interface implemented by every inference backend.
 */

#ifndef E3SM_EMULATOR_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_INFERENCE_BACKEND_HPP

#include "inference_config.hpp"
#include "tensor.hpp"

#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Abstract interface for running neural-network inference.
 *
 * ## The contract
 *
 * A backend is constructed from an InferenceConfig, loads its model in
 * initialize(), maps named input tensors to named output tensors in
 * infer(), and releases resources in finalize().
 *
 * Both the input and output TensorMaps are supplied by the *caller*.  The
 * backend writes into the caller's output memory and never allocates
 * per-step storage that the caller can see.  This matters inside a
 * coupled run: the emulator allocates its staging tensors once and reuses
 * them for the whole simulation.  Use output_specs() to learn the shapes
 * to allocate — for models that carry their own metadata (ONNX) this is
 * introspected; otherwise it reflects what the config declared.
 *
 * ## Two interfaces, one implementation
 *
 * Derived classes implement the tensor interface,
 * `infer(const TensorMap&, TensorMap&)`.  The flat convenience overload
 * `infer(const double*, double*, int)` is implemented once here in terms
 * of the tensor interface, presenting the model as a single unnamed
 * `[batch, channels]` F64 input and output.  It exists so that simple
 * pointwise emulators — and the existing coupler-side code — do not have
 * to build a TensorMap.
 *
 * Because the two are overloads of the same name, a derived class that
 * overrides one hides the other; every derived class in this directory
 * therefore carries a `using InferenceBackend::infer;` declaration.
 *
 * ## Lifecycle
 * ```cpp
 * auto backend = create_backend(config);   // constructs
 * backend->initialize();                   // loads the model
 * for (step) backend->infer(inputs, outputs);
 * backend->finalize();                     // releases the model
 * ```
 * Construction is cheap and must not touch the filesystem, the network,
 * or a GPU; everything expensive belongs in initialize().  That split
 * lets an emulator construct backends while parsing its namelist and only
 * pay for the ones it actually runs.
 */
class InferenceBackend {
public:
  explicit InferenceBackend(const InferenceConfig &config);
  virtual ~InferenceBackend();

  InferenceBackend(const InferenceBackend &) = delete;
  InferenceBackend &operator=(const InferenceBackend &) = delete;

  // ── Lifecycle ────────────────────────────────────────────────────

  /**
   * @brief Load the model and acquire resources.
   *
   * Idempotent: calling it on an already-initialized backend is a no-op.
   * Backends with nothing to load may leave the default in place.
   *
   * @throws std::runtime_error if the model cannot be loaded
   */
  virtual void initialize();

  /** @brief Whether initialize() has completed successfully. */
  bool is_initialized() const { return m_initialized; }

  /**
   * @brief Release the model and any acquired resources.
   *
   * Must be safe to call more than once, and safe to call without a
   * preceding initialize().
   */
  virtual void finalize();

  // ── Inference ────────────────────────────────────────────────────

  /**
   * @brief Run the model.
   *
   * @param inputs  Caller-owned input tensors, addressed by name
   * @param outputs Caller-owned, pre-allocated output tensors, written
   *                in place
   * @return true on success
   * @throws std::runtime_error if a required tensor is missing or has an
   *         incompatible shape
   */
  virtual bool infer(const TensorMap &inputs, TensorMap &outputs) = 0;

  /**
   * @brief Flat convenience overload for pointwise emulators.
   *
   * Wraps @p inputs as `[batch_size, config.input_channels]` and
   * @p outputs as `[batch_size, config.output_channels]`, both F64, both
   * unnamed (they take the names of the first declared input/output, or
   * "input"/"output"), then delegates to the tensor interface.
   *
   * @param inputs     `batch_size * input_channels` doubles
   * @param outputs    `batch_size * output_channels` doubles, written
   * @param batch_size Number of samples
   * @return true on success
   */
  virtual bool infer(const double *inputs, double *outputs,
                     int batch_size = 1);

  // ── Introspection ────────────────────────────────────────────────

  /** @brief Human-readable backend name, e.g. "ONNXRuntime". */
  virtual std::string name() const = 0;

  /**
   * @brief Tensors this backend expects as input.
   *
   * Defaults to the config's declared inputs.  Backends that can read
   * the model's own metadata override this and return the truth after
   * initialize().
   */
  virtual std::vector<TensorSpec> input_specs() const;

  /** @brief Tensors this backend produces.  @see input_specs */
  virtual std::vector<TensorSpec> output_specs() const;

  /**
   * @brief Element type this backend would rather receive.
   *
   * Emulators can use this to decide whether to stage a conversion from
   * the coupler's doubles.  Defaults to F32, which is what essentially
   * every trained model wants.
   */
  virtual DType preferred_dtype() const { return DType::F32; }

  /** @brief The configuration this backend was built from. */
  const InferenceConfig &config() const { return m_config; }

  // ── Helpers for implementers ─────────────────────────────────────

  /**
   * @brief Allocate output tensors matching output_specs().
   *
   * Convenience for callers that do not want to compute shapes by hand.
   * The returned tensors must outlive the TensorMap that views them.
   *
   * @param batch_size Value substituted for dynamic (negative) extents
   */
  std::vector<Tensor> allocate_outputs(int batch_size) const;

  /** @brief Allocate input tensors matching input_specs(). */
  std::vector<Tensor> allocate_inputs(int batch_size) const;

protected:
  /**
   * @brief Look up a required input, with a backend-tagged error message.
   * @throws std::runtime_error if @p name is absent from @p inputs
   */
  const TensorView &require(const TensorMap &inputs,
                            const std::string &tensor_name) const;

  /** @brief Look up a required output.  @see require */
  TensorView &require(TensorMap &outputs,
                      const std::string &tensor_name) const;

  /**
   * @brief Verify that a view's element count matches an expected shape.
   *
   * Dynamic (negative) extents in @p expected match anything, so this
   * checks the static extents and the total size implied by them.
   *
   * @throws std::runtime_error describing both shapes on mismatch
   */
  void check_shape(const TensorView &view, const Shape &expected,
                   const char *role) const;

  InferenceConfig m_config;  ///< Configuration, fixed at construction
  bool m_initialized = false; ///< Set by initialize(), cleared by finalize()
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_BACKEND_HPP
