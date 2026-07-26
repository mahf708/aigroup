/**
 * @file torch_inference_backend.hpp
 * @brief Inference backend backed by LibTorch (TorchScript).
 *
 * LibTorch's headers stay behind a pimpl; including this header does not
 * drag `torch/script.h` into the including translation unit, which
 * matters because that header is enormous and defines macros that
 * collide with Fortran-facing C code.
 */

#ifndef E3SM_EMULATOR_TORCH_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_TORCH_INFERENCE_BACKEND_HPP

#include "inference_backend.hpp"

#include <memory>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Runs a TorchScript module through LibTorch.
 *
 * This is the middle ground between the two other real backends.  Like
 * ONNX it needs no Python at runtime, so it deploys cleanly on a compute
 * node; unlike ONNX it comes from `torch.jit.script`/`torch.jit.trace`,
 * which is a shorter step from a working PyTorch model than an ONNX
 * export and keeps more of PyTorch's semantics — including, with
 * `script`, real control flow.
 *
 * Its weakness is introspection.  A TorchScript archive does not
 * reliably describe its own inputs and outputs, so unlike OnnxBackend
 * this backend cannot check the emulator's field packing against the
 * model.  Declare `input`/`output` specs in the config and they will be
 * used to size and order the call; get them wrong and the failure will
 * come from inside the model.
 *
 * ## Calling convention
 *
 * TorchScript takes positional arguments.  Inputs are passed in the
 * order the config declares them, falling back to the caller's
 * TensorMap insertion order when the config declares none — which is why
 * TensorMap preserves insertion order at all.
 *
 * The result may be a tensor, a tuple of tensors, or a dict of tensors;
 * all three are unpacked.  A dict is matched to outputs by name, a tuple
 * positionally, and a bare tensor fills the single declared output.
 *
 * ## Options
 *
 * | option           | default   | meaning                              |
 * |------------------|-----------|--------------------------------------|
 * | `torch.method`   | `forward` | module method to call                |
 * | `torch.threads`  | 1         | intra-op threads (see note below)    |
 * | `torch.grad`     | false     | keep autograd on during inference    |
 *
 * As with ONNX Runtime, one thread is the default because the ranks of
 * a coupled run already occupy the node's cores.
 */
class TorchBackend : public InferenceBackend {
public:
  explicit TorchBackend(const InferenceConfig &config);
  ~TorchBackend() override;

  using InferenceBackend::infer;

  /**
   * @brief Load the TorchScript archive onto the configured device.
   * @throws std::runtime_error if the archive cannot be loaded
   */
  void initialize() override;

  /// @copydoc InferenceBackend::infer(const TensorMap&, TensorMap&)
  bool infer(const TensorMap &inputs, TensorMap &outputs) override;

  /** @brief Drop the module. */
  void finalize() override;

  std::string name() const override { return "LibTorch"; }

  DType preferred_dtype() const override { return DType::F32; }

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_TORCH_INFERENCE_BACKEND_HPP
