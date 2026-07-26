// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "onnx_inference_backend.hpp"

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

TEST_CASE("onnx backend is registered", "[onnx_backend]") {
  register_builtin_backends();
  REQUIRE(BackendRegistry::instance().is_registered("onnx"));
}

TEST_CASE("onnx backend reads the model's own interface", "[onnx_backend]") {
  if (!have("affine.onnx")) {
    WARN("affine.onnx not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = model("affine.onnx");

  auto backend = create_backend(config);
  backend->initialize();

  // The point of ONNX: the artifact describes itself, so the emulator
  // never has to trust the namelist about names, shapes, or dtypes.
  const auto inputs = backend->input_specs();
  REQUIRE(inputs.size() == 1);
  REQUIRE(inputs[0].name == "state");
  REQUIRE(inputs[0].dtype == DType::F32);
  REQUIRE(inputs[0].shape.size() == 2);
  REQUIRE(inputs[0].shape[0] < 0); // dynamic batch axis
  REQUIRE(inputs[0].shape[1] == 4);

  const auto outputs = backend->output_specs();
  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].name == "tendency");

  backend->finalize();
}

TEST_CASE("onnx backend runs a float32 model", "[onnx_backend]") {
  if (!have("affine.onnx")) {
    WARN("affine.onnx not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = model("affine.onnx");

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(4, 0.0f);

  TensorMap inputs;
  inputs.set(TensorView::from_floats("state", in.data(), {1, 4}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("tendency", out.data(), {1, 4}));

  REQUIRE(backend->infer(inputs, outputs));

  // y = 2x + 1
  REQUIRE(out[0] == Approx(3.0f));
  REQUIRE(out[3] == Approx(9.0f));

  backend->finalize();
}

TEST_CASE("onnx backend stages double buffers", "[onnx_backend]") {
  if (!have("affine.onnx")) {
    WARN("affine.onnx not generated; run generate_test_models.py");
    return;
  }

  // This is the case that matters in a coupled run: the coupler hands
  // over doubles, the model is float32, and neither side has to know.
  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = model("affine.onnx");

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<double> in{1.0, 2.0, 3.0, 4.0};
  std::vector<double> out(4, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("state", in.data(), {1, 4}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("tendency", out.data(), {1, 4}));

  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(out[0] == Approx(3.0));
  REQUIRE(out[3] == Approx(9.0));

  backend->finalize();
}

TEST_CASE("onnx backend honours a dynamic batch axis", "[onnx_backend]") {
  if (!have("affine.onnx")) {
    WARN("affine.onnx not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = model("affine.onnx");

  auto backend = create_backend(config);
  backend->initialize();

  // Column count is a decomposition detail, so it changes between ranks
  // and between runs; the same session must handle both.
  for (int ncol : {1, 5, 17}) {
    std::vector<float> in(static_cast<std::size_t>(ncol) * 4, 2.0f);
    std::vector<float> out(in.size(), 0.0f);

    TensorMap inputs;
    inputs.set(TensorView::from_floats("state", in.data(), {ncol, 4}));
    TensorMap outputs;
    outputs.set(TensorView::from_floats("tendency", out.data(), {ncol, 4}));

    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(out.front() == Approx(5.0f));
    REQUIRE(out.back() == Approx(5.0f));
  }

  backend->finalize();
}

TEST_CASE("onnx backend handles multiple named tensors", "[onnx_backend]") {
  if (!have("two_in_two_out.onnx")) {
    WARN("two_in_two_out.onnx not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = model("two_in_two_out.onnx");

  auto backend = create_backend(config);
  backend->initialize();

  REQUIRE(backend->input_specs().size() == 2);
  REQUIRE(backend->output_specs().size() == 2);

  std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> b{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> sum(6, 0.0f), diff(6, 0.0f);

  // Deliberately supplied out of the model's declared order, to prove
  // that named lookup and not position is what binds them.
  TensorMap inputs;
  inputs.set(TensorView::from_floats("b", b.data(), {2, 3}));
  inputs.set(TensorView::from_floats("a", a.data(), {2, 3}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("diff", diff.data(), {2, 3}));
  outputs.set(TensorView::from_floats("sum", sum.data(), {2, 3}));

  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(sum[0] == Approx(2.0f));
  REQUIRE(sum[5] == Approx(7.0f));
  REQUIRE(diff[0] == Approx(0.0f));
  REQUIRE(diff[5] == Approx(5.0f));

  backend->finalize();
}

TEST_CASE("onnx backend reports a missing input", "[onnx_backend]") {
  if (!have("affine.onnx")) {
    WARN("affine.onnx not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = model("affine.onnx");

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<float> out(4, 0.0f);
  TensorMap inputs; // nothing supplied
  TensorMap outputs;
  outputs.set(TensorView::from_floats("tendency", out.data(), {1, 4}));

  REQUIRE_THROWS_AS(backend->infer(inputs, outputs), std::runtime_error);
  backend->finalize();
}

TEST_CASE("onnx backend rejects a wrong shape", "[onnx_backend]") {
  if (!have("affine.onnx")) {
    WARN("affine.onnx not generated; run generate_test_models.py");
    return;
  }

  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = model("affine.onnx");

  auto backend = create_backend(config);
  backend->initialize();

  // The model's second axis is fixed at 4; 3 must be caught before the
  // session reads past the end of the buffer.
  std::vector<float> in(3, 1.0f);
  std::vector<float> out(3, 0.0f);
  TensorMap inputs;
  inputs.set(TensorView::from_floats("state", in.data(), {1, 3}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("tendency", out.data(), {1, 3}));

  REQUIRE_THROWS_AS(backend->infer(inputs, outputs), std::runtime_error);
  backend->finalize();
}

TEST_CASE("onnx backend reports a missing file", "[onnx_backend]") {
  InferenceConfig config;
  config.backend = "onnx";
  config.model_path = "/nonexistent/model.onnx";

  auto backend = create_backend(config);
  REQUIRE_THROWS_AS(backend->initialize(), std::runtime_error);
}

TEST_CASE("onnx backend rejects an empty model path", "[onnx_backend]") {
  InferenceConfig config;
  config.backend = "onnx";

  auto backend = create_backend(config);
  REQUIRE_THROWS_AS(backend->initialize(), std::runtime_error);
}

} // namespace test
} // namespace inference
} // namespace emulator
