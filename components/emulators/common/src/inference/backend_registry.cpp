/**
 * @file backend_registry.cpp
 * @brief Implementation of the inference backend registry.
 */

#include "backend_registry.hpp"

#include "stub_inference_backend.hpp"

#ifdef EMULATOR_HAVE_PYTHON
#include "python_inference_backend.hpp"
#endif
#ifdef EMULATOR_HAVE_ONNXRUNTIME
#include "onnx_inference_backend.hpp"
#endif
#ifdef EMULATOR_HAVE_TORCH
#include "torch_inference_backend.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <sstream>

namespace emulator {
namespace inference {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return s;
}

} // namespace

BackendRegistry &BackendRegistry::instance() {
  static BackendRegistry registry;
  return registry;
}

void BackendRegistry::register_backend(const std::string &name,
                                       Factory factory,
                                       const std::string &description) {
  m_entries[lower(name)] = Entry{std::move(factory), description};
}

bool BackendRegistry::is_registered(const std::string &name) const {
  return m_entries.find(lower(name)) != m_entries.end();
}

std::shared_ptr<InferenceBackend>
BackendRegistry::create(const std::string &name,
                        const InferenceConfig &config) const {
  const auto it = m_entries.find(lower(name));
  if (it == m_entries.end())
    return nullptr;
  return it->second.factory(config);
}

std::vector<std::string> BackendRegistry::available_backends() const {
  std::vector<std::string> names;
  names.reserve(m_entries.size());
  for (const auto &kv : m_entries)
    names.push_back(kv.first);
  return names; // std::map already keeps them sorted
}

std::string BackendRegistry::description(const std::string &name) const {
  const auto it = m_entries.find(lower(name));
  return it == m_entries.end() ? std::string() : it->second.description;
}

std::string BackendRegistry::summary() const {
  std::ostringstream os;
  for (const auto &kv : m_entries) {
    os << "  " << kv.first;
    if (!kv.second.description.empty())
      os << " — " << kv.second.description;
    os << "\n";
  }
  return os.str();
}

void BackendRegistry::clear() { m_entries.clear(); }

void register_builtin_backends() {
  static bool done = false;
  if (done)
    return;
  done = true;

  auto &reg = BackendRegistry::instance();

  reg.register_backend(
      "stub",
      [](const InferenceConfig &cfg) {
        return std::make_shared<StubBackend>(cfg);
      },
      "No-op / analytic backend for testing without ML dependencies");

#ifdef EMULATOR_HAVE_PYTHON
  reg.register_backend(
      "python",
      [](const InferenceConfig &cfg) {
        return std::make_shared<PythonBackend>(cfg);
      },
      "Embedded CPython bridge (zero-copy NumPy views of emulator memory)");
#endif

#ifdef EMULATOR_HAVE_ONNXRUNTIME
  reg.register_backend(
      "onnx",
      [](const InferenceConfig &cfg) {
        return std::make_shared<OnnxBackend>(cfg);
      },
      "ONNX Runtime session over a serialized .onnx model");
#endif

#ifdef EMULATOR_HAVE_TORCH
  reg.register_backend(
      "torch",
      [](const InferenceConfig &cfg) {
        return std::make_shared<TorchBackend>(cfg);
      },
      "LibTorch TorchScript module");
#endif
}

} // namespace inference
} // namespace emulator
