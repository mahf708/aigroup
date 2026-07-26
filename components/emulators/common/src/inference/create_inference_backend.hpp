/**
 * @file create_inference_backend.hpp
 * @brief Front door for constructing inference backends.
 *
 * This is the header an emulator includes.  It pulls in the tensor
 * container, the config, the backend interface, and the registry, so a
 * component needs exactly one include to use the inference layer.
 */

#ifndef E3SM_EMULATOR_CREATE_INFERENCE_BACKEND_HPP
#define E3SM_EMULATOR_CREATE_INFERENCE_BACKEND_HPP

#include <memory>
#include <string>

#include "backend_registry.hpp"
#include "inference_backend.hpp"
#include "inference_config.hpp"
#include "tensor.hpp"

namespace emulator {
namespace inference {

/**
 * @brief Create the backend named by `config.backend`.
 *
 * Registers the built-in backends on first call, then looks up
 * `config.backend` in the registry.  The returned backend is
 * constructed but *not* initialized — call initialize() when you are
 * ready to pay for loading the model.
 *
 * @param config Backend configuration; `config.backend` selects the type
 * @return A new backend instance
 * @throws std::runtime_error if `config.backend` is not registered.  The
 *         message lists the backends this build actually supports, which
 *         is the common case when a namelist asks for "torch" in a build
 *         configured without LibTorch.
 */
std::shared_ptr<InferenceBackend> create_backend(const InferenceConfig &config);

/**
 * @brief Create a backend by name, overriding `config.backend`.
 * @see create_backend(const InferenceConfig&)
 */
std::shared_ptr<InferenceBackend> create_backend(const std::string &name,
                                                 const InferenceConfig &config);

/**
 * @brief Legacy enum-based factory.
 *
 * Retained for source compatibility with code written against the
 * original stub-only interface.  Unlike the string-based overloads this
 * one never throws: an unregistered type falls back to the stub backend,
 * matching the original behaviour.
 *
 * @param type   Backend type to create
 * @param config Configuration for the backend
 * @return A new backend instance, never nullptr
 */
std::shared_ptr<InferenceBackend> create_backend(BackendType type,
                                                 const InferenceConfig &config);

/**
 * @brief Create a backend from a namelist-style config file.
 *
 * Convenience for the common path of "read a file, build the backend it
 * describes".
 *
 * @param path Path to the config file
 * @see InferenceConfig::from_file for the format
 */
std::shared_ptr<InferenceBackend>
create_backend_from_file(const std::string &path);

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_CREATE_INFERENCE_BACKEND_HPP
