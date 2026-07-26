/**
 * @file onnx_inference_backend.cpp
 * @brief Implementation of the ONNX Runtime inference backend.
 */

#include "onnx_inference_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace emulator {
namespace inference {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return s;
}

/** @brief Map an ONNX element type onto our DType. */
DType from_onnx_type(ONNXTensorElementDataType type, const std::string &who) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return DType::F32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return DType::F64;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return DType::I32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return DType::I64;
  default:
    throw std::runtime_error(
        "OnnxBackend: tensor '" + who +
        "' has an element type this backend does not handle (ONNX type " +
        std::to_string(static_cast<int>(type)) +
        "); supported types are float, double, int32, int64");
  }
}

/** @brief Map our DType onto an ONNX element type. */
ONNXTensorElementDataType to_onnx_type(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case DType::F64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  case DType::I32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case DType::I64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

GraphOptimizationLevel parse_opt_level(const std::string &text) {
  const std::string t = lower(text);
  if (t == "disable" || t == "disabled" || t == "none")
    return ORT_DISABLE_ALL;
  if (t == "basic")
    return ORT_ENABLE_BASIC;
  if (t == "extended")
    return ORT_ENABLE_EXTENDED;
  if (t.empty() || t == "all")
    return ORT_ENABLE_ALL;
  throw std::runtime_error("OnnxBackend: unknown onnx.graph_optimization '" +
                           text + "'");
}

OrtLoggingLevel parse_log_level(const std::string &text) {
  const std::string t = lower(text);
  if (t == "verbose")
    return ORT_LOGGING_LEVEL_VERBOSE;
  if (t == "info")
    return ORT_LOGGING_LEVEL_INFO;
  if (t.empty() || t == "warning" || t == "warn")
    return ORT_LOGGING_LEVEL_WARNING;
  if (t == "error")
    return ORT_LOGGING_LEVEL_ERROR;
  if (t == "fatal")
    return ORT_LOGGING_LEVEL_FATAL;
  throw std::runtime_error("OnnxBackend: unknown onnx.log_level '" + text + "'");
}

/**
 * @brief The Ort::Env shared by every session in the process.
 *
 * ONNX Runtime wants exactly one environment per process; creating one
 * per session leaks thread pools and logging state.
 */
Ort::Env &shared_env(OrtLoggingLevel level) {
  static Ort::Env env(level, "e3sm_emulator");
  return env;
}

} // namespace

/** @brief One model tensor plus any staging needed to feed or drain it. */
struct OnnxTensorBinding {
  std::string name;
  Shape shape;      ///< From the model; negative extents are dynamic
  DType dtype;      ///< From the model
  Tensor staging;   ///< Used only when the caller's dtype differs
};

struct OnnxBackend::Impl {
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<OnnxTensorBinding> inputs;
  std::vector<OnnxTensorBinding> outputs;

  // Ort::Session::Run wants raw `const char*` arrays that outlive the
  // call, so the name strings are owned here and the pointers rebuilt
  // once at initialize().
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<const char *> input_name_ptrs;
  std::vector<const char *> output_name_ptrs;
};

// ─────────────────────────────────────────────────────────────────────

OnnxBackend::OnnxBackend(const InferenceConfig &config)
    : InferenceBackend(config), m_impl(new Impl()) {}

OnnxBackend::~OnnxBackend() = default;

void OnnxBackend::initialize() {
  if (m_initialized)
    return;
  if (m_config.model_path.empty())
    throw std::runtime_error("OnnxBackend: model_path is empty");

  Ort::Env &env = shared_env(parse_log_level(m_config.option("onnx.log_level")));

  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(m_config.option_int("onnx.intra_op_threads", 1));
  options.SetInterOpNumThreads(m_config.option_int("onnx.inter_op_threads", 1));
  options.SetGraphOptimizationLevel(
      parse_opt_level(m_config.option("onnx.graph_optimization")));

  if (m_config.uses_gpu()) {
    // Only present in a CUDA-enabled ONNX Runtime build; report the
    // reason clearly rather than silently running on the CPU, which
    // would look like a mysterious slowdown at scale.
    try {
      OrtCUDAProviderOptions cuda{};
      cuda.device_id = m_config.device_index();
      options.AppendExecutionProvider_CUDA(cuda);
    } catch (const Ort::Exception &e) {
      throw std::runtime_error(
          "OnnxBackend: device '" + m_config.device +
          "' was requested but the CUDA execution provider is not available "
          "in this ONNX Runtime build (" + e.what() + ")");
    }
  }

  try {
    m_impl->session = std::unique_ptr<Ort::Session>(
        new Ort::Session(env, m_config.model_path.c_str(), options));
  } catch (const Ort::Exception &e) {
    throw std::runtime_error("OnnxBackend: failed to load '" +
                             m_config.model_path + "': " + e.what());
  }

  // Read the model's own view of its interface.  This is the whole
  // reason to prefer ONNX for a settled model: the artifact tells us
  // what it wants instead of us having to trust the namelist.
  Ort::AllocatorWithDefaultOptions allocator;

  const std::size_t n_in = m_impl->session->GetInputCount();
  for (std::size_t i = 0; i < n_in; ++i) {
    auto name = m_impl->session->GetInputNameAllocated(i, allocator);
    // GetTensorTypeAndShapeInfo() returns a *non-owning* view into the
    // TypeInfo, so the TypeInfo has to be a named local: binding both
    // in one expression leaves the view dangling once the temporary
    // dies, and it then reads garbage element types.
    Ort::TypeInfo type_info = m_impl->session->GetInputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();

    OnnxTensorBinding binding;
    binding.name = name.get();
    binding.shape = info.GetShape();
    binding.dtype = from_onnx_type(info.GetElementType(), binding.name);
    m_impl->input_names.push_back(binding.name);
    m_impl->inputs.push_back(std::move(binding));
  }

  const std::size_t n_out = m_impl->session->GetOutputCount();
  for (std::size_t i = 0; i < n_out; ++i) {
    auto name = m_impl->session->GetOutputNameAllocated(i, allocator);
    Ort::TypeInfo type_info = m_impl->session->GetOutputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();

    OnnxTensorBinding binding;
    binding.name = name.get();
    binding.shape = info.GetShape();
    binding.dtype = from_onnx_type(info.GetElementType(), binding.name);
    m_impl->output_names.push_back(binding.name);
    m_impl->outputs.push_back(std::move(binding));
  }

  m_impl->input_name_ptrs.clear();
  for (const auto &n : m_impl->input_names)
    m_impl->input_name_ptrs.push_back(n.c_str());
  m_impl->output_name_ptrs.clear();
  for (const auto &n : m_impl->output_names)
    m_impl->output_name_ptrs.push_back(n.c_str());

  if (m_config.verbose) {
    std::cout << "[OnnxBackend] loaded " << m_config.model_path << "\n";
    for (const auto &b : m_impl->inputs) {
      std::cout << "  in  " << b.name << " " << shape_to_string(b.shape) << " "
                << dtype_name(b.dtype) << "\n";
    }
    for (const auto &b : m_impl->outputs) {
      std::cout << "  out " << b.name << " " << shape_to_string(b.shape) << " "
                << dtype_name(b.dtype) << "\n";
    }
    std::cout << std::flush;
  }
  m_initialized = true;
}

std::vector<TensorSpec> OnnxBackend::input_specs() const {
  if (!m_initialized)
    return InferenceBackend::input_specs();
  std::vector<TensorSpec> specs;
  for (const auto &b : m_impl->inputs)
    specs.emplace_back(b.name, b.shape, b.dtype);
  return specs;
}

std::vector<TensorSpec> OnnxBackend::output_specs() const {
  if (!m_initialized)
    return InferenceBackend::output_specs();
  std::vector<TensorSpec> specs;
  for (const auto &b : m_impl->outputs)
    specs.emplace_back(b.name, b.shape, b.dtype);
  return specs;
}

bool OnnxBackend::infer(const TensorMap &inputs, TensorMap &outputs) {
  if (!m_initialized)
    throw std::runtime_error("OnnxBackend::infer called before initialize()");

  std::vector<Ort::Value> in_values;
  std::vector<Ort::Value> out_values;
  in_values.reserve(m_impl->inputs.size());
  out_values.reserve(m_impl->outputs.size());

  // Bind inputs.  The caller's buffer is handed straight to the session
  // when the dtypes agree; otherwise it is converted into the staging
  // tensor allocated for exactly this purpose.
  for (auto &binding : m_impl->inputs) {
    const TensorView &view = require(inputs, binding.name);
    check_shape(view, binding.shape, "input");

    const void *data = view.raw();
    if (view.dtype() != binding.dtype) {
      if (binding.staging.size() != view.size() ||
          binding.staging.dtype() != binding.dtype) {
        binding.staging.resize(binding.dtype, view.shape());
        binding.staging.set_name(binding.name);
      }
      TensorView staged = binding.staging.view();
      staged.copy_from(view);
      data = binding.staging.raw();
    }

    const auto shape = view.shape();
    in_values.push_back(Ort::Value::CreateTensor(
        m_impl->memory_info, const_cast<void *>(data),
        static_cast<std::size_t>(view.size()) * dtype_size(binding.dtype),
        shape.data(), shape.size(), to_onnx_type(binding.dtype)));
  }

  // Bind outputs the same way, so ONNX Runtime writes into the
  // emulator's buffers rather than allocating fresh ones each step.
  std::vector<TensorView *> staged_outputs(m_impl->outputs.size(), nullptr);
  for (std::size_t i = 0; i < m_impl->outputs.size(); ++i) {
    auto &binding = m_impl->outputs[i];
    TensorView &view = require(outputs, binding.name);
    check_shape(view, binding.shape, "output");

    void *data = view.raw_mutable();
    if (view.dtype() != binding.dtype) {
      if (binding.staging.size() != view.size() ||
          binding.staging.dtype() != binding.dtype) {
        binding.staging.resize(binding.dtype, view.shape());
        binding.staging.set_name(binding.name);
      }
      data = binding.staging.raw();
      staged_outputs[i] = &view;
    }

    const auto shape = view.shape();
    out_values.push_back(Ort::Value::CreateTensor(
        m_impl->memory_info, data,
        static_cast<std::size_t>(view.size()) * dtype_size(binding.dtype),
        shape.data(), shape.size(), to_onnx_type(binding.dtype)));
  }

  try {
    m_impl->session->Run(Ort::RunOptions{nullptr},
                         m_impl->input_name_ptrs.data(), in_values.data(),
                         in_values.size(), m_impl->output_name_ptrs.data(),
                         out_values.data(), out_values.size());
  } catch (const Ort::Exception &e) {
    throw std::runtime_error(std::string("OnnxBackend: Run failed: ") +
                             e.what());
  }

  // Convert back any output that had to go through staging.
  for (std::size_t i = 0; i < staged_outputs.size(); ++i) {
    if (staged_outputs[i])
      staged_outputs[i]->copy_from(m_impl->outputs[i].staging.view());
  }
  return true;
}

void OnnxBackend::finalize() {
  m_impl->session.reset();
  m_impl->inputs.clear();
  m_impl->outputs.clear();
  m_impl->input_names.clear();
  m_impl->output_names.clear();
  m_impl->input_name_ptrs.clear();
  m_impl->output_name_ptrs.clear();
  m_initialized = false;
}

} // namespace inference
} // namespace emulator
