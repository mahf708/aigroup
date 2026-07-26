/**
 * @file inference_backend.cpp
 * @brief Shared implementation of the InferenceBackend base class.
 */

#include "inference_backend.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace emulator {
namespace inference {

namespace {

/**
 * @brief Per-sample element count implied by a spec.
 *
 * The flat interface treats the leading axis as the batch axis, so the
 * number of "channels" is the product of every remaining extent.  A
 * rank-0 or rank-1 spec has no trailing axes and contributes 1.
 */
std::int64_t channels_of(const TensorSpec &spec) {
  std::int64_t n = 1;
  for (std::size_t i = 1; i < spec.shape.size(); ++i)
    n *= spec.shape[i] < 0 ? 1 : spec.shape[i];
  return n;
}

} // namespace

InferenceBackend::InferenceBackend(const InferenceConfig &config)
    : m_config(config) {}

InferenceBackend::~InferenceBackend() = default;

void InferenceBackend::initialize() { m_initialized = true; }

void InferenceBackend::finalize() { m_initialized = false; }

std::vector<TensorSpec> InferenceBackend::input_specs() const {
  return m_config.effective_inputs();
}

std::vector<TensorSpec> InferenceBackend::output_specs() const {
  return m_config.effective_outputs();
}

bool InferenceBackend::infer(const double *inputs, double *outputs,
                             int batch_size) {
  const auto in_specs = input_specs();
  const auto out_specs = output_specs();

  const std::string in_name = in_specs.empty() ? "input" : in_specs[0].name;
  const std::string out_name = out_specs.empty() ? "output" : out_specs[0].name;

  // The flat interface is defined as [batch, channels] regardless of the
  // model's declared rank; a backend needing a richer layout is simply not
  // reachable through this path.
  const std::int64_t in_ch = m_config.input_channels > 0
                                 ? m_config.input_channels
                                 : (in_specs.empty() ? 0
                                                     : channels_of(in_specs[0]));
  const std::int64_t out_ch =
      m_config.output_channels > 0 ? m_config.output_channels
                                   : (out_specs.empty()
                                          ? 0
                                          : channels_of(out_specs[0]));

  TensorMap in_map;
  in_map.set(TensorView::from_doubles(in_name, inputs,
                                      {static_cast<std::int64_t>(batch_size),
                                       in_ch}));
  TensorMap out_map;
  out_map.set(TensorView::from_doubles(out_name, outputs,
                                       {static_cast<std::int64_t>(batch_size),
                                        out_ch}));

  return infer(in_map, out_map);
}

std::vector<Tensor> InferenceBackend::allocate_inputs(int batch_size) const {
  std::vector<Tensor> out;
  for (const auto &spec : input_specs())
    out.push_back(Tensor::from_spec(spec, batch_size));
  return out;
}

std::vector<Tensor> InferenceBackend::allocate_outputs(int batch_size) const {
  std::vector<Tensor> out;
  for (const auto &spec : output_specs())
    out.push_back(Tensor::from_spec(spec, batch_size));
  return out;
}

namespace {

/** @brief Append " (got: 'a' 'b')" to help diagnose a name mismatch. */
void append_available(std::ostringstream &os,
                      const std::vector<std::string> &have) {
  if (have.empty())
    return;
  os << " (got:";
  for (const auto &n : have)
    os << " '" << n << "'";
  os << ")";
}

} // namespace

const TensorView &InferenceBackend::require(const TensorMap &inputs,
                                            const std::string &tensor_name) const {
  if (const TensorView *v = inputs.find(tensor_name))
    return *v;
  std::ostringstream os;
  os << name() << ": required input tensor '" << tensor_name
     << "' was not supplied";
  append_available(os, inputs.names());
  throw std::runtime_error(os.str());
}

TensorView &InferenceBackend::require(TensorMap &outputs,
                                      const std::string &tensor_name) const {
  if (TensorView *v = outputs.find(tensor_name))
    return *v;
  std::ostringstream os;
  os << name() << ": required output tensor '" << tensor_name
     << "' was not supplied by the caller";
  append_available(os, outputs.names());
  throw std::runtime_error(os.str());
}

void InferenceBackend::check_shape(const TensorView &view,
                                   const Shape &expected,
                                   const char *role) const {
  bool ok = view.rank() == static_cast<int>(expected.size());
  if (ok) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] >= 0 && expected[i] != view.shape()[i]) {
        ok = false;
        break;
      }
    }
  }
  if (!ok) {
    throw std::runtime_error(name() + ": " + role + " tensor '" + view.name() +
                             "' has shape " + shape_to_string(view.shape()) +
                             ", expected " + shape_to_string(expected));
  }
}

} // namespace inference
} // namespace emulator
