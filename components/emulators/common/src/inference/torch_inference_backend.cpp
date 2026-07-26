/**
 * @file torch_inference_backend.cpp
 * @brief Implementation of the LibTorch (TorchScript) inference backend.
 */

#include "torch_inference_backend.hpp"

#include <ATen/Parallel.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace emulator {
namespace inference {

namespace {

/** @brief Map our DType onto a torch scalar type. */
torch::ScalarType to_torch_type(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return torch::kFloat32;
  case DType::F64:
    return torch::kFloat64;
  case DType::I32:
    return torch::kInt32;
  case DType::I64:
    return torch::kInt64;
  }
  return torch::kFloat32;
}

/** @brief Map a torch scalar type onto our DType. */
DType from_torch_type(torch::ScalarType type, const std::string &who) {
  switch (type) {
  case torch::kFloat32:
    return DType::F32;
  case torch::kFloat64:
    return DType::F64;
  case torch::kInt32:
    return DType::I32;
  case torch::kInt64:
    return DType::I64;
  default:
    throw std::runtime_error(
        "TorchBackend: output '" + who +
        "' has scalar type " + std::string(torch::toString(type)) +
        ", which this backend does not convert; supported types are "
        "float32, float64, int32, int64");
  }
}

/** @brief Copy a torch tensor into a caller-owned view, converting dtype. */
void copy_into(const at::Tensor &src, TensorView &dst) {
  // Bring it home and make it dense before reading raw memory: the
  // model may have produced a non-contiguous view, or left it on a GPU.
  const at::Tensor host = src.to(torch::kCPU).contiguous();

  if (host.numel() != dst.size()) {
    throw std::runtime_error(
        "TorchBackend: output '" + dst.name() + "' expects " +
        std::to_string(dst.size()) + " elements but the model produced " +
        std::to_string(host.numel()));
  }

  // copy_from compares element counts rather than shapes, so a [N,1]
  // result lands correctly in an [N] destination without reshaping
  // either side.
  const TensorView view(dst.name(), static_cast<const void *>(host.data_ptr()),
                        from_torch_type(host.scalar_type(), dst.name()),
                        Shape{host.numel()});
  dst.copy_from(view);
}

} // namespace

struct TorchBackend::Impl {
  torch::jit::script::Module module;
  torch::Device device = torch::Device(torch::kCPU);
  bool loaded = false;
};

// ─────────────────────────────────────────────────────────────────────

TorchBackend::TorchBackend(const InferenceConfig &config)
    : InferenceBackend(config), m_impl(new Impl()) {}

TorchBackend::~TorchBackend() = default;

void TorchBackend::initialize() {
  if (m_initialized)
    return;
  if (m_config.model_path.empty())
    throw std::runtime_error("TorchBackend: model_path is empty");

  if (m_config.uses_gpu()) {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error(
          "TorchBackend: device '" + m_config.device +
          "' was requested but this LibTorch build reports no CUDA support");
    }
    m_impl->device = torch::Device(torch::kCUDA, m_config.device_index());
  } else {
    m_impl->device = torch::Device(torch::kCPU);
  }

  const int threads = m_config.option_int("torch.threads", 1);
  if (threads > 0)
    at::set_num_threads(threads);

  // Autograd is off unconditionally: inference never needs it, and the
  // tape would grow without bound over a coupled run.  Reject the option
  // up front rather than silently ignoring it.
  if (m_config.option_bool("torch.grad", false)) {
    throw std::runtime_error(
        "TorchBackend: torch.grad=true is not supported; this backend runs "
        "under NoGradGuard so that a coupled run does not accumulate an "
        "autograd tape");
  }

  try {
    m_impl->module = torch::jit::load(m_config.model_path, m_impl->device);
  } catch (const c10::Error &e) {
    throw std::runtime_error("TorchBackend: failed to load '" +
                             m_config.model_path + "': " + e.what());
  }
  m_impl->module.eval();
  m_impl->loaded = true;

  if (m_config.verbose) {
    std::cout << "[TorchBackend] loaded " << m_config.model_path << " on "
              << m_impl->device.str() << std::endl;
  }
  m_initialized = true;
}

bool TorchBackend::infer(const TensorMap &inputs, TensorMap &outputs) {
  if (!m_initialized)
    throw std::runtime_error("TorchBackend::infer called before initialize()");

  const std::string method_name = m_config.option("torch.method", "forward");

  // Establish the argument order: config order when declared, otherwise
  // the order the caller inserted them.
  std::vector<std::string> order;
  for (const auto &spec : m_config.inputs)
    order.push_back(spec.name);
  if (order.empty())
    order = inputs.names();

  std::vector<torch::jit::IValue> args;
  args.reserve(order.size());
  // from_blob does not own its memory, so these tensors are only valid
  // while the caller's buffers are — which is exactly the duration of
  // this call.
  for (const auto &tensor_name : order) {
    const TensorView &view = require(inputs, tensor_name);
    const auto options =
        torch::TensorOptions().dtype(to_torch_type(view.dtype()));
    at::Tensor t = torch::from_blob(const_cast<void *>(view.raw()),
                                    view.shape(), options);
    if (m_impl->device.is_cuda())
      t = t.to(m_impl->device);
    args.emplace_back(std::move(t));
  }

  torch::NoGradGuard no_grad;

  c10::IValue result;
  try {
    if (method_name == "forward") {
      result = m_impl->module.forward(args);
    } else {
      auto method = m_impl->module.get_method(method_name);
      result = method(args);
    }
  } catch (const c10::Error &e) {
    throw std::runtime_error("TorchBackend: calling '" + method_name +
                             "' failed: " + e.what());
  }

  // Unpack whatever shape the module chose to return.
  if (result.isTensor()) {
    if (outputs.size() != 1) {
      throw std::runtime_error(
          "TorchBackend: the model returned a single tensor but the caller "
          "supplied " + std::to_string(outputs.size()) + " output tensors");
    }
    copy_into(result.toTensor(), outputs[0]);

  } else if (result.isTuple()) {
    const auto &elements = result.toTuple()->elements();
    if (elements.size() < outputs.size()) {
      throw std::runtime_error(
          "TorchBackend: the model returned a tuple of " +
          std::to_string(elements.size()) + " values but the caller supplied " +
          std::to_string(outputs.size()) + " output tensors");
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (!elements[i].isTensor()) {
        throw std::runtime_error(
            "TorchBackend: tuple element " + std::to_string(i) +
            " is not a tensor");
      }
      copy_into(elements[i].toTensor(), outputs[i]);
    }

  } else if (result.isGenericDict()) {
    const auto dict = result.toGenericDict();
    for (auto &view : outputs) {
      const auto it = dict.find(view.name());
      if (it == dict.end()) {
        throw std::runtime_error("TorchBackend: the model's returned dict has "
                                 "no entry '" + view.name() + "'");
      }
      if (!it->value().isTensor()) {
        throw std::runtime_error("TorchBackend: returned dict entry '" +
                                 view.name() + "' is not a tensor");
      }
      copy_into(it->value().toTensor(), view);
    }

  } else {
    throw std::runtime_error(
        "TorchBackend: the model returned a " +
        std::string(result.tagKind()) +
        "; this backend understands a Tensor, a Tuple of Tensors, or a "
        "Dict[str, Tensor]");
  }
  return true;
}

void TorchBackend::finalize() {
  if (m_impl->loaded) {
    m_impl->module = torch::jit::script::Module();
    m_impl->loaded = false;
  }
  m_initialized = false;
}

} // namespace inference
} // namespace emulator
