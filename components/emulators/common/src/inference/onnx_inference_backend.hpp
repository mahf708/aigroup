/**
 * @file onnx_inference_backend.hpp
 * @brief Inference backend backed by ONNX Runtime.
 *
 * ONNX Runtime's headers stay behind a pimpl so this header can be
 * included from any translation unit.
 */

#ifndef E3SM_EMULATOR_ONNX_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_ONNX_INFERENCE_BACKEND_HPP

#include "inference_backend.hpp"

#include <memory>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Runs a serialized `.onnx` model through ONNX Runtime.
 *
 * ONNX is the backend to reach for once a model has stopped changing.
 * A `.onnx` file is a self-describing artifact: it carries its own
 * input and output names, shapes, and dtypes, so the emulator can check
 * at startup that the coupling fields it packed actually match what the
 * model expects, instead of discovering a mismatch as garbage output.
 * It also has no Python at runtime, which makes it the easy backend to
 * deploy on a compute node and the one that behaves best under MPI.
 *
 * The cost is expressiveness: exporting a model with data-dependent
 * control flow is awkward, and the export step is one more thing to get
 * wrong.  For a model still under development, use the Python backend;
 * move to ONNX when the interface is settled.
 *
 * ## Introspection
 *
 * input_specs() and output_specs() report what the *model* declares
 * after initialize(), not what the config guessed.  An emulator can use
 * them to size its staging buffers, and validate() will complain if the
 * config contradicts the model.
 *
 * ## Precision
 *
 * Models are nearly always float32 while the coupler carries doubles.
 * The backend keeps a staging tensor for any input or output whose
 * caller-side dtype differs from the model's and converts through it;
 * when the dtypes agree, the caller's buffer is bound to the session
 * directly and no copy happens on either side.
 *
 * ## Options
 *
 * | option                    | default | meaning                       |
 * |---------------------------|---------|-------------------------------|
 * | `onnx.intra_op_threads`   | 1       | threads within one operator   |
 * | `onnx.inter_op_threads`   | 1       | operators run concurrently    |
 * | `onnx.graph_optimization` | `all`   | disable/basic/extended/all    |
 * | `onnx.log_level`          | `warning` | verbose/info/warning/error  |
 *
 * The thread defaults are 1 on purpose.  Inside a coupled run the ranks
 * already fill the node, and a backend that helpfully spawns a thread
 * pool per rank oversubscribes the CPUs badly.
 */
class OnnxBackend : public InferenceBackend {
public:
  explicit OnnxBackend(const InferenceConfig &config);
  ~OnnxBackend() override;

  using InferenceBackend::infer;

  /**
   * @brief Create the session and read the model's tensor metadata.
   * @throws std::runtime_error if the model cannot be loaded
   */
  void initialize() override;

  /// @copydoc InferenceBackend::infer(const TensorMap&, TensorMap&)
  bool infer(const TensorMap &inputs, TensorMap &outputs) override;

  /** @brief Release the session. */
  void finalize() override;

  std::string name() const override { return "ONNXRuntime"; }

  /** @brief Input tensors as declared by the model itself. */
  std::vector<TensorSpec> input_specs() const override;

  /** @brief Output tensors as declared by the model itself. */
  std::vector<TensorSpec> output_specs() const override;

  DType preferred_dtype() const override { return DType::F32; }

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_ONNX_INFERENCE_BACKEND_HPP
