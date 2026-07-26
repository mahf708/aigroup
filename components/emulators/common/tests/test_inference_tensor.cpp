// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "tensor.hpp"

#include <vector>

namespace emulator {
namespace inference {
namespace test {

TEST_CASE("dtype helpers", "[inference_tensor]") {
  REQUIRE(dtype_size(DType::F32) == 4);
  REQUIRE(dtype_size(DType::F64) == 8);
  REQUIRE(dtype_size(DType::I32) == 4);
  REQUIRE(dtype_size(DType::I64) == 8);

  REQUIRE(std::string(dtype_name(DType::F32)) == "f32");
  REQUIRE(std::string(dtype_name(DType::F64)) == "f64");

  REQUIRE(dtype_from_string("float32") == DType::F32);
  REQUIRE(dtype_from_string("DOUBLE") == DType::F64);
  REQUIRE(dtype_from_string("i64") == DType::I64);
  REQUIRE_THROWS_AS(dtype_from_string("bfloat16"), std::runtime_error);
}

TEST_CASE("shape helpers", "[inference_tensor]") {
  REQUIRE(num_elements({}) == 1); // scalar
  REQUIRE(num_elements({4}) == 4);
  REQUIRE(num_elements({2, 3, 4}) == 24);
  REQUIRE(shape_to_string({2, 3}) == "[2, 3]");
}

TEST_CASE("TensorSpec resolves dynamic dims", "[inference_tensor]") {
  TensorSpec spec("x", {-1, 8}, DType::F32);
  REQUIRE(spec.has_dynamic_dims());
  REQUIRE(spec.resolved_shape(16) == Shape{16, 8});

  TensorSpec fixed("y", {4, 8});
  REQUIRE_FALSE(fixed.has_dynamic_dims());
  REQUIRE(fixed.resolved_shape(99) == Shape{4, 8});
}

TEST_CASE("TensorView wraps existing memory", "[inference_tensor]") {
  std::vector<double> buffer{1, 2, 3, 4, 5, 6};
  auto view = TensorView::from_doubles("x", buffer.data(), {2, 3});

  REQUIRE(view.name() == "x");
  REQUIRE(view.dtype() == DType::F64);
  REQUIRE(view.rank() == 2);
  REQUIRE(view.size() == 6);
  REQUIRE(view.nbytes() == 48);
  REQUIRE(view.writable());
  REQUIRE(view.valid());

  // The view aliases the vector — no copy was made.
  view.set_element(0, 42.0);
  REQUIRE(buffer[0] == 42.0);
  REQUIRE(view.element(0) == 42.0);
}

TEST_CASE("TensorView respects read-only memory", "[inference_tensor]") {
  const std::vector<double> buffer{1, 2, 3};
  auto view = TensorView::from_doubles("x", buffer.data(), {3});

  REQUIRE_FALSE(view.writable());
  REQUIRE(view.element(1) == 2.0);
  REQUIRE_THROWS_AS(view.fill(0.0), std::runtime_error);
  REQUIRE_THROWS_AS(view.zero(), std::runtime_error);
  REQUIRE_THROWS_AS(view.set_element(0, 1.0), std::runtime_error);
}

TEST_CASE("TensorView typed access is checked", "[inference_tensor]") {
  std::vector<float> buffer{1.0f, 2.0f};
  auto view = TensorView::from_floats("x", buffer.data(), {2});

  REQUIRE(view.data_as<float>()[1] == 2.0f);
  REQUIRE_THROWS_AS(view.data_as<double>(), std::runtime_error);
}

TEST_CASE("TensorView reshape preserves element count", "[inference_tensor]") {
  std::vector<double> buffer(12, 0.0);
  auto view = TensorView::from_doubles("x", buffer.data(), {12});

  view.reshape({3, 4});
  REQUIRE(view.shape() == Shape{3, 4});
  REQUIRE_THROWS_AS(view.reshape({5, 4}), std::runtime_error);
}

TEST_CASE("TensorView converts on copy", "[inference_tensor]") {
  // This is the conversion the whole layer exists to make painless: the
  // coupler carries doubles, the model wants float32.
  std::vector<double> src{1.5, 2.5, 3.5, 4.5};
  std::vector<float> dst(4, 0.0f);

  auto src_view = TensorView::from_doubles("x", src.data(), {4});
  auto dst_view = TensorView::from_floats("x", dst.data(), {4});
  dst_view.copy_from(src_view);

  REQUIRE(dst[0] == Approx(1.5f));
  REQUIRE(dst[3] == Approx(4.5f));

  // ... and back the other way.
  std::vector<double> round_trip(4, 0.0);
  auto rt_view = TensorView::from_doubles("x", round_trip.data(), {2, 2});
  rt_view.copy_from(dst_view); // shapes differ, element counts match
  REQUIRE(round_trip[3] == Approx(4.5));
}

TEST_CASE("TensorView copy rejects size mismatch", "[inference_tensor]") {
  std::vector<double> src(4, 1.0);
  std::vector<double> dst(3, 0.0);
  auto src_view = TensorView::from_doubles("a", src.data(), {4});
  auto dst_view = TensorView::from_doubles("b", dst.data(), {3});

  REQUIRE_THROWS_AS(dst_view.copy_from(src_view), std::runtime_error);
}

TEST_CASE("TensorView fill and zero", "[inference_tensor]") {
  std::vector<double> buffer(5, 9.0);
  auto view = TensorView::from_doubles("x", buffer.data(), {5});

  view.fill(2.25);
  for (double v : buffer)
    REQUIRE(v == Approx(2.25));

  view.zero();
  for (double v : buffer)
    REQUIRE(v == 0.0);
}

TEST_CASE("Tensor owns and zeroes its storage", "[inference_tensor]") {
  Tensor tensor("t", DType::F32, {2, 3});

  REQUIRE(tensor.name() == "t");
  REQUIRE(tensor.dtype() == DType::F32);
  REQUIRE(tensor.size() == 6);
  REQUIRE(tensor.nbytes() == 24);

  auto view = tensor.view();
  for (std::int64_t i = 0; i < view.size(); ++i)
    REQUIRE(view.element(i) == 0.0);

  view.fill(1.5);
  REQUIRE(tensor.view().element(4) == Approx(1.5));
}

TEST_CASE("Tensor::from_spec resolves the batch axis", "[inference_tensor]") {
  TensorSpec spec("x", {-1, 4}, DType::F64);
  auto tensor = Tensor::from_spec(spec, 8);

  REQUIRE(tensor.shape() == Shape{8, 4});
  REQUIRE(tensor.size() == 32);
}

TEST_CASE("Tensor::resize reallocates", "[inference_tensor]") {
  Tensor tensor("t", DType::F32, {4});
  tensor.view().fill(7.0);

  tensor.resize(DType::F64, {2, 5});
  REQUIRE(tensor.dtype() == DType::F64);
  REQUIRE(tensor.size() == 10);
  REQUIRE(tensor.view().element(0) == 0.0); // resize zeroes
}

TEST_CASE("TensorMap keys by name and preserves order", "[inference_tensor]") {
  std::vector<double> a(2, 1.0), b(2, 2.0), c(2, 3.0);
  TensorMap map;
  map.set(TensorView::from_doubles("alpha", a.data(), {2}));
  map.set(TensorView::from_doubles("beta", b.data(), {2}));
  map.set(TensorView::from_doubles("gamma", c.data(), {2}));

  REQUIRE(map.size() == 3);
  REQUIRE(map.contains("beta"));
  REQUIRE_FALSE(map.contains("delta"));

  // Insertion order, which is what positional backends depend on.
  REQUIRE(map.names() == std::vector<std::string>{"alpha", "beta", "gamma"});
  REQUIRE(map[0].name() == "alpha");
  REQUIRE(map[2].name() == "gamma");

  REQUIRE(map.at("beta").element(0) == 2.0);
  REQUIRE_THROWS_AS(map.at("delta"), std::runtime_error);
  REQUIRE(map.find("delta") == nullptr);
  REQUIRE_THROWS_AS(map[3], std::runtime_error);
}

TEST_CASE("TensorMap replacement keeps position", "[inference_tensor]") {
  std::vector<double> a(2, 1.0), b(2, 2.0), replacement(2, 99.0);
  TensorMap map;
  map.set(TensorView::from_doubles("alpha", a.data(), {2}));
  map.set(TensorView::from_doubles("beta", b.data(), {2}));

  map.set(TensorView::from_doubles("alpha", replacement.data(), {2}));

  REQUIRE(map.size() == 2);
  REQUIRE(map[0].name() == "alpha");
  REQUIRE(map[0].element(0) == 99.0);
}

TEST_CASE("TensorMap supports range-for", "[inference_tensor]") {
  std::vector<double> a(2, 1.0), b(2, 2.0);
  TensorMap map;
  map.set(TensorView::from_doubles("alpha", a.data(), {2}));
  map.set(TensorView::from_doubles("beta", b.data(), {2}));

  double sum = 0.0;
  for (const auto &view : map)
    sum += view.element(0);
  REQUIRE(sum == Approx(3.0));
}

} // namespace test
} // namespace inference
} // namespace emulator
