/**
 * @file backend_registry.hpp
 * @brief Open, string-keyed registry of inference backend factories.
 *
 * The registry is what makes this layer expandable.  Adding a backend
 * means writing a class and calling register_backend() — no enum to
 * extend, no switch statement to edit, no core file to recompile.  A
 * backend can therefore live in an optional library that is only built
 * when its dependency is present, or even in a component outside
 * `emulator_common` entirely.
 */

#ifndef E3SM_EMULATOR_INFERENCE_BACKEND_REGISTRY_HPP
#define E3SM_EMULATOR_INFERENCE_BACKEND_REGISTRY_HPP

#include "inference_backend.hpp"
#include "inference_config.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Process-wide registry mapping backend names to factories.
 *
 * ## Registering a backend
 * ```cpp
 * BackendRegistry::instance().register_backend(
 *     "my_backend",
 *     [](const InferenceConfig &cfg) {
 *       return std::make_shared<MyBackend>(cfg);
 *     },
 *     "One-line description for diagnostics");
 * ```
 *
 * Built-in backends are registered explicitly by
 * register_builtin_backends(), which the factory calls on first use.
 * Explicit registration is deliberate: static-initializer
 * self-registration is unreliable when backends are linked from a static
 * archive, because the linker discards object files nothing references.
 *
 * Not thread-safe for registration.  Register during initialization,
 * before spawning threads; lookup afterwards is read-only and safe.
 */
class BackendRegistry {
public:
  /** @brief Factory signature: config in, backend out. */
  using Factory =
      std::function<std::shared_ptr<InferenceBackend>(const InferenceConfig &)>;

  /** @brief The single process-wide registry. */
  static BackendRegistry &instance();

  /**
   * @brief Register (or replace) a backend factory.
   *
   * @param name        Lookup key; compared case-insensitively
   * @param factory     Callable that constructs the backend
   * @param description One-line summary shown by available_backends()
   */
  void register_backend(const std::string &name, Factory factory,
                        const std::string &description = "");

  /** @brief Whether @p name is registered (case-insensitive). */
  bool is_registered(const std::string &name) const;

  /**
   * @brief Construct a backend by name.
   *
   * @return The new backend, or nullptr if @p name is not registered
   */
  std::shared_ptr<InferenceBackend> create(const std::string &name,
                                           const InferenceConfig &config) const;

  /** @brief Registered names, sorted. */
  std::vector<std::string> available_backends() const;

  /** @brief Description supplied at registration, or "" if none. */
  std::string description(const std::string &name) const;

  /**
   * @brief Multi-line "name — description" listing.
   *
   * Used in error messages so a typo in a namelist reports what the
   * build actually supports rather than a bare failure.
   */
  std::string summary() const;

  /** @brief Remove all registrations.  Intended for tests. */
  void clear();

private:
  BackendRegistry() = default;

  struct Entry {
    Factory factory;
    std::string description;
  };

  std::map<std::string, Entry> m_entries; ///< Keyed by lowercased name
};

/**
 * @brief Register every backend compiled into this build.
 *
 * Idempotent.  Called automatically by create_backend(); call it
 * directly only when querying the registry before creating anything.
 *
 * Which backends this registers depends on the CMake feature flags:
 * "stub" always, plus "python" with `EMULATOR_ENABLE_PYTHON`, "onnx"
 * with `EMULATOR_ENABLE_ONNXRUNTIME`, and "torch" with
 * `EMULATOR_ENABLE_TORCH`.
 */
void register_builtin_backends();

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_BACKEND_REGISTRY_HPP
