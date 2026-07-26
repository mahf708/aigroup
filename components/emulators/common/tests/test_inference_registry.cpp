// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"

#include <algorithm>
#include <vector>

namespace emulator {
namespace inference {
namespace test {

namespace {

/**
 * @brief A backend defined entirely outside the inference directory.
 *
 * Its whole purpose is to demonstrate the extensibility claim: a
 * component can add a backend without editing an enum, a switch, or any
 * file under `src/inference`.
 */
class CountingBackend : public InferenceBackend {
public:
  explicit CountingBackend(const InferenceConfig &config)
      : InferenceBackend(config) {}

  using InferenceBackend::infer;

  bool infer(const TensorMap &inputs, TensorMap &outputs) override {
    ++calls;
    for (auto &out : outputs)
      out.fill(static_cast<double>(calls));
    (void)inputs;
    return true;
  }

  void finalize() override { m_initialized = false; }
  std::string name() const override { return "Counting"; }

  int calls = 0;
};

} // namespace

TEST_CASE("built-in backends are registered", "[inference_registry]") {
  register_builtin_backends();
  auto &registry = BackendRegistry::instance();

  REQUIRE(registry.is_registered("stub"));
  REQUIRE(registry.is_registered("STUB")); // case-insensitive

  const auto names = registry.available_backends();
  REQUIRE(std::find(names.begin(), names.end(), "stub") != names.end());

  REQUIRE_FALSE(registry.description("stub").empty());
  REQUIRE(registry.summary().find("stub") != std::string::npos);
}

TEST_CASE("an out-of-tree backend can register itself",
          "[inference_registry]") {
  register_builtin_backends();
  auto &registry = BackendRegistry::instance();

  registry.register_backend(
      "counting",
      [](const InferenceConfig &cfg) {
        return std::make_shared<CountingBackend>(cfg);
      },
      "Test backend defined outside src/inference");

  REQUIRE(registry.is_registered("counting"));

  InferenceConfig config;
  config.backend = "counting";
  config.output_channels = 2;

  auto backend = create_backend(config);
  REQUIRE(backend->name() == "Counting");

  double inputs[2] = {0.0, 0.0};
  double outputs[2] = {0.0, 0.0};
  REQUIRE(backend->infer(inputs, outputs, 1));
  REQUIRE(outputs[0] == Approx(1.0));
  REQUIRE(backend->infer(inputs, outputs, 1));
  REQUIRE(outputs[0] == Approx(2.0));
}

TEST_CASE("an unknown backend name reports what is available",
          "[inference_registry]") {
  InferenceConfig config;
  config.backend = "definitely_not_a_backend";

  // The string-based factory throws — a namelist typo should stop the
  // run loudly rather than silently degrade to a no-op model.
  REQUIRE_THROWS_AS(create_backend(config), std::runtime_error);

  bool mentioned_available = false;
  try {
    create_backend(config);
  } catch (const std::runtime_error &e) {
    const std::string message = e.what();
    mentioned_available = message.find("stub") != std::string::npos;
  }
  REQUIRE(mentioned_available);
}

TEST_CASE("the legacy enum factory never returns null",
          "[inference_registry]") {
  InferenceConfig config;

  // Even for backends this build may not have compiled in, the enum
  // overload degrades to the stub instead of throwing, matching the
  // behaviour that existing callers were written against.
  for (auto type : {BackendType::STUB, BackendType::PYTHON, BackendType::ONNX,
                    BackendType::TORCH}) {
    auto backend = create_backend(type, config);
    REQUIRE(backend != nullptr);
  }
}

TEST_CASE("backends are constructed but not initialized",
          "[inference_registry]") {
  InferenceConfig config;
  auto backend = create_backend(config);

  // Construction must stay cheap: no model loading, no device grab.
  REQUIRE_FALSE(backend->is_initialized());
  backend->initialize();
  REQUIRE(backend->is_initialized());
  backend->finalize();
  REQUIRE_FALSE(backend->is_initialized());
}

TEST_CASE("allocate_outputs sizes from the specs", "[inference_registry]") {
  InferenceConfig config;
  config.outputs = {TensorSpec("a", {-1, 4}, DType::F32),
                    TensorSpec("b", {2, 3}, DType::F64)};
  auto backend = create_backend(config);

  auto tensors = backend->allocate_outputs(8);
  REQUIRE(tensors.size() == 2);
  REQUIRE(tensors[0].shape() == Shape{8, 4});
  REQUIRE(tensors[0].dtype() == DType::F32);
  REQUIRE(tensors[1].shape() == Shape{2, 3});
}

} // namespace test
} // namespace inference
} // namespace emulator
