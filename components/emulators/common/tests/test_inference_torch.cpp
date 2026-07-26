// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "torch_inference_backend.hpp"

#include <fstream>
#include <string>
#include <vector>

// Set by the test CMakeLists to where generate_test_models.py wrote.
#ifndef EMULATOR_TEST_MODEL_DIR
#define EMULATOR_TEST_MODEL_DIR ""
#endif

namespace emulator {
namespace inference {
namespace test {

namespace {

std::string model(const std::string &name) {
  return std::string(EMULATOR_TEST_MODEL_DIR) + "/" + name;
}

bool have(const std::string &name) {
  std::ifstream f(model(name));
  return f.good();
}

} // namespace

TEST_CASE("torch backend is registered", "[torch_backend]") {
  register_builtin_backends();
  REQUIRE(BackendRegistry::instance().is_registered("torch"));
}

TEST_CASE("torch backend runs a TorchScript module", "[torch_backend]") {
  if (!have("affine.pt")) {
    WARN("affine.pt not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "torch";
  config.model_path = model("affine.pt");
  config.inputs = {TensorSpec("x", {-1, 4}, DType::F32)};
  config.outputs = {TensorSpec("y", {-1, 4}, DType::F32)};

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(4, 0.0f);

  TensorMap inputs;
  inputs.set(TensorView::from_floats("x", in.data(), {1, 4}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("y", out.data(), {1, 4}));

  REQUIRE(backend->infer(inputs, outputs));

  // y = 2x + 1
  REQUIRE(out[0] == Approx(3.0f));
  REQUIRE(out[3] == Approx(9.0f));

  backend->finalize();
}

TEST_CASE("torch backend converts from doubles", "[torch_backend]") {
  if (!have("affine.pt")) {
    WARN("affine.pt not generated; run generate_test_models.py");
    return;
  }

  // The module is float32; feeding it float64 would silently produce a
  // float64 result, so declare the input as f32 and let the caller keep
  // its doubles.
  InferenceConfig config;
  config.backend = "torch";
  config.model_path = model("affine.pt");
  config.inputs = {TensorSpec("x", {-1, 4}, DType::F32)};
  config.outputs = {TensorSpec("y", {-1, 4}, DType::F64)};

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<double> out(4, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_floats("x", in.data(), {1, 4}));
  TensorMap outputs;
  // The destination is doubles; copy_into converts on the way back.
  outputs.set(TensorView::from_doubles("y", out.data(), {1, 4}));

  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(out[0] == Approx(3.0));
  REQUIRE(out[3] == Approx(9.0));

  backend->finalize();
}

TEST_CASE("torch backend unpacks a tuple result", "[torch_backend]") {
  if (!have("two_outputs.pt")) {
    WARN("two_outputs.pt not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "torch";
  config.model_path = model("two_outputs.pt");
  config.inputs = {TensorSpec("x", {-1, 3}, DType::F32)};

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<float> in{1.0f, 2.0f, 3.0f};
  std::vector<float> doubled(3, 0.0f), shifted(3, 0.0f);

  TensorMap inputs;
  inputs.set(TensorView::from_floats("x", in.data(), {1, 3}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("doubled", doubled.data(), {1, 3}));
  outputs.set(TensorView::from_floats("shifted", shifted.data(), {1, 3}));

  REQUIRE(backend->infer(inputs, outputs));

  // Tuple elements bind positionally, in the caller's insertion order.
  REQUIRE(doubled[2] == Approx(6.0f));
  REQUIRE(shifted[2] == Approx(103.0f));

  backend->finalize();
}

TEST_CASE("torch backend orders inputs from the config", "[torch_backend]") {
  if (!have("two_inputs.pt")) {
    WARN("two_inputs.pt not generated; run generate_test_models.py");
    return;
  }

  // TorchScript takes positional arguments, so the config's declared
  // input order is what decides which tensor is `a` and which is `b`.
  InferenceConfig config;
  config.backend = "torch";
  config.model_path = model("two_inputs.pt");
  config.inputs = {TensorSpec("a", {-1, 3}, DType::F32),
                   TensorSpec("b", {-1, 3}, DType::F32)};

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<float> a{10.0f, 20.0f, 30.0f};
  std::vector<float> b{1.0f, 2.0f, 3.0f};
  std::vector<float> out(3, 0.0f);

  // Inserted b-first to prove that config order, not insertion order,
  // is what wins when the config declares one.
  TensorMap inputs;
  inputs.set(TensorView::from_floats("b", b.data(), {1, 3}));
  inputs.set(TensorView::from_floats("a", a.data(), {1, 3}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("diff", out.data(), {1, 3}));

  REQUIRE(backend->infer(inputs, outputs));

  // a - b, not b - a
  REQUIRE(out[0] == Approx(9.0f));
  REQUIRE(out[2] == Approx(27.0f));

  backend->finalize();
}

TEST_CASE("torch backend handles a varying batch", "[torch_backend]") {
  if (!have("affine.pt")) {
    WARN("affine.pt not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "torch";
  config.model_path = model("affine.pt");
  config.inputs = {TensorSpec("x", {-1, 4}, DType::F32)};

  auto backend = create_backend(config);
  backend->initialize();

  for (int ncol : {1, 6, 23}) {
    std::vector<float> in(static_cast<std::size_t>(ncol) * 4, 2.0f);
    std::vector<float> out(in.size(), 0.0f);

    TensorMap inputs;
    inputs.set(TensorView::from_floats("x", in.data(), {ncol, 4}));
    TensorMap outputs;
    outputs.set(TensorView::from_floats("y", out.data(), {ncol, 4}));

    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(out.front() == Approx(5.0f));
    REQUIRE(out.back() == Approx(5.0f));
  }

  backend->finalize();
}

TEST_CASE("torch backend reports an output-count mismatch",
          "[torch_backend]") {
  if (!have("affine.pt")) {
    WARN("affine.pt not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "torch";
  config.model_path = model("affine.pt");
  config.inputs = {TensorSpec("x", {-1, 4}, DType::F32)};

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<float> in(4, 1.0f);
  std::vector<float> out_a(4, 0.0f), out_b(4, 0.0f);

  TensorMap inputs;
  inputs.set(TensorView::from_floats("x", in.data(), {1, 4}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("y", out_a.data(), {1, 4}));
  outputs.set(TensorView::from_floats("z", out_b.data(), {1, 4}));

  // The module returns a single tensor but two were asked for.
  REQUIRE_THROWS_AS(backend->infer(inputs, outputs), std::runtime_error);
  backend->finalize();
}

TEST_CASE("torch backend reports a missing file", "[torch_backend]") {
  InferenceConfig config;
  config.backend = "torch";
  config.model_path = "/nonexistent/model.pt";

  auto backend = create_backend(config);
  REQUIRE_THROWS_AS(backend->initialize(), std::runtime_error);
}

TEST_CASE("torch backend rejects torch.grad", "[torch_backend]") {
  if (!have("affine.pt")) {
    WARN("affine.pt not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "torch";
  config.model_path = model("affine.pt");
  config.set_option("torch.grad", "true");

  auto backend = create_backend(config);
  REQUIRE_THROWS_AS(backend->initialize(), std::runtime_error);
}

} // namespace test
} // namespace inference
} // namespace emulator
