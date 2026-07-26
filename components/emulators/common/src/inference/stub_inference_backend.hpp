/**
 * @file stub_inference_backend.hpp
 * @brief Dependency-free backend for exercising the data path.
 */

#ifndef E3SM_EMULATOR_STUB_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_STUB_INFERENCE_BACKEND_HPP

#include "inference_backend.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Analytic backend that needs no ML library.
 *
 * The stub exists so that the coupling path — pack fields, call the
 * emulator, unpack fields, hand them back to the coupler — can be built
 * and tested everywhere, including on machines with no Python, no
 * LibTorch, and no ONNX Runtime, and in CI.
 *
 * Its behaviour is selected by the `stub.mode` option:
 *
 * | mode       | effect                                              |
 * |------------|-----------------------------------------------------|
 * | `noop`     | leaves outputs untouched (default)                  |
 * | `zero`     | writes zeros to every output                        |
 * | `constant` | writes `stub.value` to every output                 |
 * | `copy`     | copies input i into output i, elementwise           |
 * | `affine`   | `out = stub.scale * in + stub.offset`, elementwise   |
 *
 * `copy` and `affine` pair the i-th supplied input with the i-th supplied
 * output positionally and require matching element counts; they are the
 * useful modes for checking that data really travels end to end, since a
 * no-op cannot distinguish "the pipeline works" from "nothing ran".
 *
 * `noop` is the default because it is what the original stub did, and
 * existing tests assert that outputs come back unchanged.
 */
class StubBackend : public InferenceBackend {
public:
  /** @brief What the stub writes into the output tensors. */
  enum class Mode {
    NOOP,     ///< Leave outputs untouched
    ZERO,     ///< Fill outputs with 0
    CONSTANT, ///< Fill outputs with `stub.value`
    COPY,     ///< out[i] = in[i]
    AFFINE    ///< out[i] = scale * in[i] + offset
  };

  explicit StubBackend(const InferenceConfig &config);
  ~StubBackend() override = default;

  // Keep the flat convenience overload visible alongside our override.
  using InferenceBackend::infer;

  /// @copydoc InferenceBackend::infer(const TensorMap&, TensorMap&)
  bool infer(const TensorMap &inputs, TensorMap &outputs) override;

  /// @copydoc InferenceBackend::finalize
  void finalize() override;

  /// @copydoc InferenceBackend::name
  std::string name() const override { return "Stub"; }

  /**
   * @brief The stub speaks the coupler's precision natively.
   *
   * Returning F64 tells emulators not to bother staging a conversion
   * when all they want is to test the plumbing.
   */
  DType preferred_dtype() const override { return DType::F64; }

  /** @brief Active mode, resolved from `stub.mode` at construction. */
  Mode mode() const { return m_mode; }

  /** @brief Parse a mode name; throws on an unrecognized name. */
  static Mode mode_from_string(const std::string &text);

  /** @brief Canonical name of a mode. */
  static const char *mode_name(Mode mode);

private:
  Mode m_mode = Mode::NOOP;
  double m_scale = 1.0;  ///< `stub.scale`, used by AFFINE
  double m_offset = 0.0; ///< `stub.offset`, used by AFFINE
  double m_value = 0.0;  ///< `stub.value`, used by CONSTANT
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_STUB_INFERENCE_BACKEND_HPP
