// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "create_inference_backend.hpp"

namespace emulator {
namespace inference {
namespace test {

// ── Original expectations, kept so the coupler-side code that was
//    written against the stub-only interface keeps working ───────────

TEST_CASE("InferenceConfig defaults", "[inference_config]") {
  InferenceConfig config;

  REQUIRE(config.input_channels == 0);
  REQUIRE(config.output_channels == 0);
  REQUIRE_FALSE(config.verbose);
}

TEST_CASE("InferenceConfig can be set", "[inference_config]") {
  InferenceConfig config;
  config.input_channels = 10;
  config.output_channels = 5;
  config.verbose = true;

  REQUIRE(config.input_channels == 10);
  REQUIRE(config.output_channels == 5);
  REQUIRE(config.verbose);
}

// ── Backend selection, device resolution, options ───────────────────

TEST_CASE("InferenceConfig defaults to the stub backend", "[inference_config]") {
  InferenceConfig config;
  REQUIRE(config.backend == "stub");
  REQUIRE(config.device == "cpu");
  REQUIRE(config.batch_size == 1);
  REQUIRE_FALSE(config.uses_gpu());
  REQUIRE(config.device_index() == -1);
}

TEST_CASE("device index follows local rank for a bare cuda",
          "[inference_config]") {
  InferenceConfig config;
  config.device = "cuda";
  config.local_rank = 3;

  REQUIRE(config.uses_gpu());
  REQUIRE(config.device_index() == 3);

  // An explicit index wins over the rank.
  config.device = "cuda:1";
  REQUIRE(config.device_index() == 1);

  config.device = "cuda:notanumber";
  REQUIRE_THROWS_AS(config.device_index(), std::runtime_error);
}

TEST_CASE("option accessors coerce and report", "[inference_config]") {
  InferenceConfig config;
  config.set_option("threads", "8");
  config.set_option("scale", "2.5");
  config.set_option("enabled", "yes");
  config.set_option("broken", "banana");

  REQUIRE(config.has_option("threads"));
  REQUIRE_FALSE(config.has_option("absent"));

  REQUIRE(config.option("threads") == "8");
  REQUIRE(config.option("absent", "fallback") == "fallback");
  REQUIRE(config.option_int("threads") == 8);
  REQUIRE(config.option_int("absent", 4) == 4);
  REQUIRE(config.option_double("scale") == Approx(2.5));
  REQUIRE(config.option_bool("enabled"));
  REQUIRE(config.option_bool("absent", true));

  REQUIRE_THROWS_AS(config.option_int("broken"), std::runtime_error);
  REQUIRE_THROWS_AS(config.option_bool("broken"), std::runtime_error);
}

TEST_CASE("effective specs synthesize from channel counts",
          "[inference_config]") {
  InferenceConfig config;
  config.input_channels = 6;
  config.output_channels = 2;

  const auto inputs = config.effective_inputs();
  REQUIRE(inputs.size() == 1);
  REQUIRE(inputs[0].name == "input");
  REQUIRE(inputs[0].shape == Shape{-1, 6});
  REQUIRE(inputs[0].dtype == DType::F64);

  const auto outputs = config.effective_outputs();
  REQUIRE(outputs.size() == 1);
  REQUIRE(outputs[0].shape == Shape{-1, 2});

  // Explicit specs take precedence over the legacy channel counts.
  config.inputs = {TensorSpec("state", {-1, 72}, DType::F32)};
  REQUIRE(config.effective_inputs().size() == 1);
  REQUIRE(config.effective_inputs()[0].name == "state");
}

TEST_CASE("validate rejects nonsense", "[inference_config]") {
  InferenceConfig config;
  REQUIRE_NOTHROW(config.validate());

  config.batch_size = 0;
  REQUIRE_THROWS_AS(config.validate(), std::runtime_error);

  config.batch_size = 1;
  config.inputs = {TensorSpec("", {4})};
  REQUIRE_THROWS_AS(config.validate(), std::runtime_error);

  config.inputs = {TensorSpec("x", {4, 0})};
  REQUIRE_THROWS_AS(config.validate(), std::runtime_error);
}

// ── Tensor spec parsing ─────────────────────────────────────────────

TEST_CASE("parse_tensor_spec reads name, shape and dtype",
          "[inference_config]") {
  const auto spec = parse_tensor_spec("state:-1,72:f32");
  REQUIRE(spec.name == "state");
  REQUIRE(spec.shape == Shape{-1, 72});
  REQUIRE(spec.dtype == DType::F32);

  // dtype defaults to f32
  const auto defaulted = parse_tensor_spec("y:4");
  REQUIRE(defaulted.dtype == DType::F32);
  REQUIRE(defaulted.shape == Shape{4});

  // A scalar has no extents at all.
  const auto scalar = parse_tensor_spec("t::f64");
  REQUIRE(scalar.shape.empty());
  REQUIRE(scalar.dtype == DType::F64);

  REQUIRE_THROWS_AS(parse_tensor_spec(":4:f32"), std::runtime_error);
  REQUIRE_THROWS_AS(parse_tensor_spec("x:a,b:f32"), std::runtime_error);
  REQUIRE_THROWS_AS(parse_tensor_spec("x:4:f32:extra"), std::runtime_error);
}

// ── Namelist parsing ────────────────────────────────────────────────

TEST_CASE("from_string parses a namelist-style config",
          "[inference_config]") {
  const auto config = InferenceConfig::from_string(R"(
# radiation emulator
backend: onnx
model_path: /models/rad.onnx
device: cuda:2
batch_size: 48
verbose: true
input:  state:-1,72:f32
input:  cosz:-1:f32
output: heating_rate:-1,72:f32
onnx.intra_op_threads: 4
)");

  REQUIRE(config.backend == "onnx");
  REQUIRE(config.model_path == "/models/rad.onnx");
  REQUIRE(config.device == "cuda:2");
  REQUIRE(config.batch_size == 48);
  REQUIRE(config.verbose);

  REQUIRE(config.inputs.size() == 2);
  REQUIRE(config.inputs[0].name == "state");
  REQUIRE(config.inputs[0].shape == Shape{-1, 72});
  REQUIRE(config.inputs[1].name == "cosz");

  REQUIRE(config.outputs.size() == 1);
  REQUIRE(config.outputs[0].name == "heating_rate");

  // Unrecognized keys become backend options rather than errors, which
  // is what lets a backend define new knobs without touching the parser.
  REQUIRE(config.option_int("onnx.intra_op_threads") == 4);

  REQUIRE_NOTHROW(config.validate());
}

TEST_CASE("from_string ignores blanks and comments", "[inference_config]") {
  const auto config = InferenceConfig::from_string(R"(

# nothing to see here

backend: stub
)");
  REQUIRE(config.backend == "stub");
  REQUIRE(config.options.empty());
}

TEST_CASE("from_string rejects a line with no colon", "[inference_config]") {
  REQUIRE_THROWS_AS(InferenceConfig::from_string("backend stub"),
                    std::runtime_error);
}

TEST_CASE("from_file reports a missing file", "[inference_config]") {
  REQUIRE_THROWS_AS(InferenceConfig::from_file("/nonexistent/inference_in"),
                    std::runtime_error);
}

TEST_CASE("to_string mentions the essentials", "[inference_config]") {
  InferenceConfig config;
  config.backend = "stub";
  config.input_channels = 3;
  config.output_channels = 1;

  const auto text = config.to_string();
  REQUIRE(text.find("stub") != std::string::npos);
  REQUIRE(text.find("input") != std::string::npos);
}

} // namespace test
} // namespace inference
} // namespace emulator
