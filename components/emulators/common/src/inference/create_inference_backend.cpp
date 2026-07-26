/**
 * @file create_inference_backend.cpp
 * @brief Factory implementation, delegating to the backend registry.
 */

#include "create_inference_backend.hpp"
#include "stub_inference_backend.hpp"

#include <sstream>
#include <stdexcept>

namespace emulator {
namespace inference {

std::shared_ptr<InferenceBackend>
create_backend(const std::string &name, const InferenceConfig &config) {
  register_builtin_backends();

  const std::string wanted = name.empty() ? "stub" : name;
  auto &registry = BackendRegistry::instance();

  if (auto backend = registry.create(wanted, config))
    return backend;

  std::ostringstream os;
  os << "inference: no backend named '" << wanted
     << "' is available in this build.\n"
     << "Backends compiled in:\n"
     << registry.summary()
     << "Optional backends are enabled at configure time with "
        "-DEMULATOR_ENABLE_PYTHON=ON, -DEMULATOR_ENABLE_ONNXRUNTIME=ON, "
        "or -DEMULATOR_ENABLE_TORCH=ON.";
  throw std::runtime_error(os.str());
}

std::shared_ptr<InferenceBackend>
create_backend(const InferenceConfig &config) {
  return create_backend(config.backend, config);
}

std::shared_ptr<InferenceBackend>
create_backend(BackendType type, const InferenceConfig &config) {
  register_builtin_backends();

  // The enum overload predates the registry and callers rely on it always
  // returning something, so an unavailable backend degrades to the stub
  // rather than throwing.
  auto backend =
      BackendRegistry::instance().create(backend_type_name(type), config);
  if (backend)
    return backend;
  return std::make_shared<StubBackend>(config);
}

std::shared_ptr<InferenceBackend>
create_backend_from_file(const std::string &path) {
  const auto config = InferenceConfig::from_file(path);
  config.validate();
  return create_backend(config);
}

} // namespace inference
} // namespace emulator
