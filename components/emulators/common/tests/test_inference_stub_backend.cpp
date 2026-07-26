// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "stub_inference_backend.hpp"

#include <vector>

namespace emulator {
namespace inference {
namespace test {

// ── The original stub expectations, unchanged ───────────────────────

TEST_CASE("StubBackend factory creation", "[stub_backend]") {
  InferenceConfig config;
  config.input_channels = 4;
  config.output_channels = 2;

  auto backend = create_backend(BackendType::STUB, config);

  REQUIRE(backend != nullptr);
  REQUIRE(backend->name() == "Stub");
}

TEST_CASE("StubBackend lifecycle", "[stub_backend]") {
  InferenceConfig config;
  config.input_channels = 4;
  config.output_channels = 2;

  auto backend = create_backend(BackendType::STUB, config);

  // Run inference (no-op: outputs unchanged)
  double inputs[4] = {1, 2, 3, 4};
  double outputs[2] = {99, 99};

  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(outputs[0] == 99.0);
  REQUIRE(outputs[1] == 99.0);

  // Finalize
  backend->finalize();
}

TEST_CASE("create_backend fallback for unknown type", "[stub_backend]") {
  InferenceConfig config;
  auto backend = create_backend(static_cast<BackendType>(999), config);

  REQUIRE(backend != nullptr);
  REQUIRE(backend->name() == "Stub"); // Falls back to stub
}

// ── Modes beyond the no-op ──────────────────────────────────────────

namespace {

/** @brief Build a stub in a given mode with an option or two. */
std::shared_ptr<InferenceBackend>
make_stub(const std::string &mode,
          const std::map<std::string, std::string> &extra = {}) {
  InferenceConfig config;
  config.input_channels = 3;
  config.output_channels = 3;
  config.set_option("stub.mode", mode);
  for (const auto &kv : extra)
    config.set_option(kv.first, kv.second);
  return create_backend(config);
}

} // namespace

TEST_CASE("stub mode parsing", "[stub_backend]") {
  REQUIRE(StubBackend::mode_from_string("") == StubBackend::Mode::NOOP);
  REQUIRE(StubBackend::mode_from_string("COPY") == StubBackend::Mode::COPY);
  REQUIRE(StubBackend::mode_from_string("passthrough") ==
          StubBackend::Mode::COPY);
  REQUIRE(StubBackend::mode_from_string("affine") == StubBackend::Mode::AFFINE);
  REQUIRE_THROWS_AS(StubBackend::mode_from_string("magic"),
                    std::runtime_error);

  REQUIRE(std::string(StubBackend::mode_name(StubBackend::Mode::ZERO)) ==
          "zero");
}

TEST_CASE("stub copy mode moves data end to end", "[stub_backend]") {
  auto backend = make_stub("copy");

  double inputs[3] = {1.0, 2.0, 3.0};
  double outputs[3] = {0.0, 0.0, 0.0};
  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(outputs[0] == Approx(1.0));
  REQUIRE(outputs[1] == Approx(2.0));
  REQUIRE(outputs[2] == Approx(3.0));
}

TEST_CASE("stub affine mode applies scale and offset", "[stub_backend]") {
  auto backend = make_stub("affine", {{"stub.scale", "2.0"},
                                      {"stub.offset", "0.5"}});

  double inputs[3] = {1.0, 2.0, 3.0};
  double outputs[3] = {0.0, 0.0, 0.0};
  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(outputs[0] == Approx(2.5));
  REQUIRE(outputs[1] == Approx(4.5));
  REQUIRE(outputs[2] == Approx(6.5));
}

TEST_CASE("stub constant and zero modes", "[stub_backend]") {
  double inputs[3] = {1.0, 2.0, 3.0};

  {
    auto backend = make_stub("constant", {{"stub.value", "7.25"}});
    double outputs[3] = {0.0, 0.0, 0.0};
    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(outputs[1] == Approx(7.25));
  }
  {
    auto backend = make_stub("zero");
    double outputs[3] = {5.0, 5.0, 5.0};
    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(outputs[0] == 0.0);
    REQUIRE(outputs[2] == 0.0);
  }
}

TEST_CASE("stub speaks the tensor interface too", "[stub_backend]") {
  InferenceConfig config;
  config.set_option("stub.mode", "copy");
  auto backend = create_backend(config);

  std::vector<double> in{1, 2, 3, 4, 5, 6};
  std::vector<float> out(6, 0.0f);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("x", in.data(), {2, 3}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("y", out.data(), {2, 3}));

  REQUIRE(backend->infer(inputs, outputs));

  // Copy mode pairs positionally and converts dtype on the way.
  REQUIRE(out[0] == Approx(1.0f));
  REQUIRE(out[5] == Approx(6.0f));
}

TEST_CASE("stub reports a mismatch instead of corrupting memory",
          "[stub_backend]") {
  InferenceConfig config;
  config.set_option("stub.mode", "copy");
  auto backend = create_backend(config);

  std::vector<double> in(6, 1.0);
  std::vector<double> out(4, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("x", in.data(), {6}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("y", out.data(), {4}));

  REQUIRE_THROWS_AS(backend->infer(inputs, outputs), std::runtime_error);
}

TEST_CASE("stub prefers doubles", "[stub_backend]") {
  InferenceConfig config;
  auto backend = create_backend(config);
  REQUIRE(backend->preferred_dtype() == DType::F64);
}

} // namespace test
} // namespace inference
} // namespace emulator
