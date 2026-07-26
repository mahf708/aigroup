// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"
#include "python_inference_backend.hpp"

#include <string>
#include <vector>

// EMULATOR_PYTHON_PACKAGE_DIR is defined by the test CMakeLists and
// points at src/inference/python, so the reference models are importable
// without the developer having installed anything.
#ifndef EMULATOR_PYTHON_PACKAGE_DIR
#define EMULATOR_PYTHON_PACKAGE_DIR ""
#endif

namespace emulator {
namespace inference {
namespace test {

namespace {

constexpr const char *kModule = "e3sm_emulator_inference.reference_models";

/** @brief Config wired to a reference model, with the package importable. */
InferenceConfig python_config(const std::string &factory) {
  InferenceConfig config;
  config.backend = "python";
  config.set_option("python.module", kModule);
  config.set_option("python.factory", factory);
  config.set_option("python.sys_path", EMULATOR_PYTHON_PACKAGE_DIR);
  return config;
}

} // namespace

TEST_CASE("python backend is registered", "[python_backend]") {
  register_builtin_backends();
  REQUIRE(BackendRegistry::instance().is_registered("python"));
}

TEST_CASE("python identity model round-trips data", "[python_backend]") {
  auto config = python_config("create_identity");
  auto backend = create_backend(config);
  backend->initialize();

  std::vector<double> in{1.0, 2.0, 3.0, 4.0};
  std::vector<double> out(4, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("x", in.data(), {2, 2}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("x", out.data(), {2, 2}));

  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(out[0] == Approx(1.0));
  REQUIRE(out[3] == Approx(4.0));

  backend->finalize();
}

TEST_CASE("python model writes into emulator memory in place",
          "[python_backend]") {
  auto config = python_config("create_affine");
  config.set_option("scale", "3.0");
  config.set_option("offset", "0.5");

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<double> in{1.0, 2.0, 3.0};
  std::vector<double> out(3, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("x", in.data(), {3}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("y", out.data(), {3}));

  REQUIRE(backend->infer(inputs, outputs));

  // The Python side wrote through a NumPy view of `out`; no copy was
  // made in either direction.
  REQUIRE(out[0] == Approx(3.5));
  REQUIRE(out[1] == Approx(6.5));
  REQUIRE(out[2] == Approx(9.5));

  backend->finalize();
}

TEST_CASE("python model may return a dict instead", "[python_backend]") {
  auto config = python_config("create_returning");
  auto backend = create_backend(config);
  backend->initialize();

  std::vector<double> in{1.0, 2.0};
  std::vector<double> out(2, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("q", in.data(), {2}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("q", out.data(), {2}));

  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(out[0] == Approx(3.0));
  REQUIRE(out[1] == Approx(6.0));

  backend->finalize();
}

TEST_CASE("python backend converts precision", "[python_backend]") {
  // The coupler carries doubles; a model that wants float32 declares it
  // and the wrapping presents exactly that, still without a copy.
  auto config = python_config("create_affine");
  config.set_option("scale", "2.0");
  config.set_option("offset", "0.0");

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<float> in{1.0f, 2.0f, 3.0f};
  std::vector<float> out(3, 0.0f);

  TensorMap inputs;
  inputs.set(TensorView::from_floats("x", in.data(), {3}));
  TensorMap outputs;
  outputs.set(TensorView::from_floats("y", out.data(), {3}));

  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(out[2] == Approx(6.0f));

  backend->finalize();
}

TEST_CASE("python backend introspects declared specs", "[python_backend]") {
  auto config = python_config("create_spec_model");
  auto backend = create_backend(config);
  backend->initialize();

  const auto inputs = backend->input_specs();
  REQUIRE(inputs.size() == 1);
  REQUIRE(inputs[0].name == "state");
  REQUIRE(inputs[0].shape == Shape{-1, 6});
  REQUIRE(inputs[0].dtype == DType::F64);

  const auto outputs = backend->output_specs();
  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].name == "tendency");
  REQUIRE(outputs[0].shape == Shape{-1, 3});

  // An emulator can then size its buffers from what the model says.
  auto staging = backend->allocate_outputs(4);
  REQUIRE(staging.size() == 1);
  REQUIRE(staging[0].shape() == Shape{4, 3});

  backend->finalize();
}

TEST_CASE("python spec model computes over columns", "[python_backend]") {
  auto config = python_config("create_spec_model");
  auto backend = create_backend(config);
  backend->initialize();

  // Two columns, six input channels, three output channels.
  std::vector<double> in{1, 2, 3, 10, 20, 30,   //
                         4, 5, 6, 40, 50, 60};
  std::vector<double> out(6, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("state", in.data(), {2, 6}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("tendency", out.data(), {2, 3}));

  REQUIRE(backend->infer(inputs, outputs));

  REQUIRE(out[0] == Approx(11.0)); // 1 + 10
  REQUIRE(out[2] == Approx(33.0)); // 3 + 30
  REQUIRE(out[3] == Approx(44.0)); // 4 + 40
  REQUIRE(out[5] == Approx(66.0)); // 6 + 60

  backend->finalize();
}

TEST_CASE("python backend accepts a bare callable", "[python_backend]") {
  auto config = python_config("create_callable");
  config.set_option("scale", "5.0");

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<double> in{1.0, 2.0};
  std::vector<double> out(2, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("x", in.data(), {2}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("y", out.data(), {2}));

  REQUIRE(backend->infer(inputs, outputs));
  REQUIRE(out[1] == Approx(10.0));

  backend->finalize();
}

TEST_CASE("python column-physics model preserves layout",
          "[python_backend]") {
  // Two columns of four levels each; the smoother only touches interior
  // levels, so a transposed pack would be obvious in the result.
  auto config = python_config("create_column_physics");
  config.set_option("mixing_strength", "1.0");

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<double> in{0.0, 4.0, 0.0, 0.0,  //
                         0.0, 0.0, 8.0, 0.0};
  std::vector<double> out(8, -1.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("state", in.data(), {2, 4}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("tendency", out.data(), {2, 4}));

  REQUIRE(backend->infer(inputs, outputs));

  // Edge levels are untouched by the smoother, so their tendency is 0.
  REQUIRE(out[0] == Approx(0.0));
  REQUIRE(out[3] == Approx(0.0));
  // Interior level 1 of column 0: (0 + 2*4 + 0)/4 - 4 = -2
  REQUIRE(out[1] == Approx(-2.0));
  // Interior level 2 of column 1: (0 + 2*8 + 0)/4 - 8 = -4
  REQUIRE(out[6] == Approx(-4.0));

  backend->finalize();
}

TEST_CASE("python MPI-aware model runs serially", "[python_backend]") {
  // No communicator handle: the model must degrade to rank-local work
  // rather than failing, so the test suite needs no MPI.
  auto config = python_config("create_mpi_aware");
  auto backend = create_backend(config);
  backend->initialize();

  std::vector<double> in{1.0, 3.0, 5.0, 7.0};
  std::vector<double> out(4, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("state", in.data(), {4}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("state", out.data(), {4}));

  REQUIRE(backend->infer(inputs, outputs));
  for (double v : out)
    REQUIRE(v == Approx(4.0)); // mean of 1,3,5,7

  backend->finalize();
}

TEST_CASE("python backend survives repeated steps", "[python_backend]") {
  // A coupled run calls infer() many thousands of times; a reference
  // leak or a GIL mistake shows up as a crash or a slow death here.
  auto config = python_config("create_affine");
  config.set_option("scale", "1.0");
  config.set_option("offset", "1.0");

  auto backend = create_backend(config);
  backend->initialize();

  std::vector<double> state{0.0, 0.0};
  std::vector<double> next(2, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("x", state.data(), {2}));
  TensorMap outputs;
  outputs.set(TensorView::from_doubles("y", next.data(), {2}));

  for (int step = 0; step < 500; ++step) {
    REQUIRE(backend->infer(inputs, outputs));
    state = next;
  }
  REQUIRE(state[0] == Approx(500.0));

  backend->finalize();
}

TEST_CASE("python backend reports a bad module clearly", "[python_backend]") {
  auto config = python_config("create");
  config.set_option("python.module", "no_such_module_anywhere");

  auto backend = create_backend(config);
  REQUIRE_THROWS_AS(backend->initialize(), std::runtime_error);
}

TEST_CASE("python backend reports a missing factory", "[python_backend]") {
  auto config = python_config("no_such_factory");
  auto backend = create_backend(config);
  REQUIRE_THROWS_AS(backend->initialize(), std::runtime_error);
}

TEST_CASE("python backend rejects inference before initialize",
          "[python_backend]") {
  auto config = python_config("create_identity");
  auto backend = create_backend(config);

  TensorMap inputs, outputs;
  REQUIRE_THROWS_AS(backend->infer(inputs, outputs), std::runtime_error);
}

TEST_CASE("python interpreter survives backend churn", "[python_backend]") {
  // The interpreter is refcounted across backends; creating and
  // destroying several in sequence must neither finalize it early nor
  // leave it running.
  for (int i = 0; i < 3; ++i) {
    auto config = python_config("create_identity");
    auto backend = create_backend(config);
    backend->initialize();

    std::vector<double> in{static_cast<double>(i)};
    std::vector<double> out(1, 0.0);
    TensorMap inputs, outputs;
    inputs.set(TensorView::from_doubles("v", in.data(), {1}));
    outputs.set(TensorView::from_doubles("v", out.data(), {1}));

    REQUIRE(backend->infer(inputs, outputs));
    REQUIRE(out[0] == Approx(static_cast<double>(i)));
    backend->finalize();
  }
}

TEST_CASE("two python backends coexist", "[python_backend]") {
  auto first = create_backend(python_config("create_identity"));
  auto affine_config = python_config("create_affine");
  affine_config.set_option("scale", "2.0");
  affine_config.set_option("offset", "0.0");
  auto second = create_backend(affine_config);

  first->initialize();
  second->initialize();

  std::vector<double> in{5.0};
  std::vector<double> out_a(1, 0.0), out_b(1, 0.0);

  TensorMap inputs;
  inputs.set(TensorView::from_doubles("x", in.data(), {1}));
  TensorMap outputs_a, outputs_b;
  outputs_a.set(TensorView::from_doubles("x", out_a.data(), {1}));
  outputs_b.set(TensorView::from_doubles("y", out_b.data(), {1}));

  REQUIRE(first->infer(inputs, outputs_a));
  REQUIRE(second->infer(inputs, outputs_b));

  REQUIRE(out_a[0] == Approx(5.0));
  REQUIRE(out_b[0] == Approx(10.0));

  first->finalize();
  second->finalize();
}

} // namespace test
} // namespace inference
} // namespace emulator
