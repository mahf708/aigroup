// Catch2 v2 single header
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// This test is only built when the DataView sources are present — see
// inference_tests.cmake.  It is the contract between the two containers:
// DataView owns the coupler-facing doubles, TensorView describes them to
// a model, and neither has to know about the other's internals.
#include "create_inference_backend.hpp"
#include "data_view_bridge.hpp"

#include <vector>

namespace emulator {
namespace inference {
namespace test {

namespace {

/** @brief An allocated DataView with `nfields` fields over `npoints`. */
DataView make_view(const std::string &name, int npoints, int nfields,
                   DataLayout layout) {
  DataView view(name, npoints, layout);
  for (int i = 0; i < nfields; ++i)
    view.add_field(FieldSpec("f" + std::to_string(i), "1", "field"));
  view.allocate();
  return view;
}

} // namespace

TEST_CASE("wrapping a DataView is zero-copy", "[data_view_bridge]") {
  DataView view = make_view("atm_import", 4, 3, DataLayout::POINT_MAJOR);
  view.fill(1.5);

  auto tensor = tensor_from_data_view(view, "state");

  REQUIRE(tensor.name() == "state");
  REQUIRE(tensor.dtype() == DType::F64);
  REQUIRE(tensor.shape() == Shape{4, 3}); // POINT_MAJOR → [npoints, nfields]
  REQUIRE(tensor.size() == 12);
  REQUIRE(tensor.writable());

  // Writing through the tensor writes into the DataView's buffer.
  tensor.set_element(0, 42.0);
  REQUIRE(view.data()[0] == 42.0);

  // ... and the other way round.
  view.data()[11] = 7.0;
  REQUIRE(tensor.element(11) == 7.0);
}

TEST_CASE("FIELD_MAJOR shape is transposed", "[data_view_bridge]") {
  DataView view = make_view("atm_export", 4, 3, DataLayout::FIELD_MAJOR);
  auto tensor = tensor_from_data_view(view, "state");

  REQUIRE(tensor.shape() == Shape{3, 4}); // [nfields, npoints]
}

TEST_CASE("an unallocated DataView is rejected", "[data_view_bridge]") {
  DataView view("empty", 4, DataLayout::POINT_MAJOR);
  view.add_field(FieldSpec("f0"));
  // No allocate() call.

  REQUIRE_THROWS_AS(tensor_from_data_view(view, "state"), std::runtime_error);
}

TEST_CASE("a single FIELD_MAJOR field wraps contiguously",
          "[data_view_bridge]") {
  DataView view = make_view("atm", 4, 3, DataLayout::FIELD_MAJOR);
  for (int p = 0; p < 4; ++p)
    view(1, p) = static_cast<double>(p) * 10.0;

  auto tensor = tensor_from_field(view, "f1");

  REQUIRE(tensor.name() == "f1");
  REQUIRE(tensor.shape() == Shape{4});
  REQUIRE(tensor.element(2) == Approx(20.0));

  tensor.set_element(3, 99.0);
  REQUIRE(view(1, 3) == Approx(99.0));
}

TEST_CASE("a POINT_MAJOR field is refused rather than aliased wrongly",
          "[data_view_bridge]") {
  // In POINT_MAJOR a field is strided, so there is no contiguous tensor
  // that describes it; silently handing back a wrong view would corrupt
  // data in a way that is very hard to trace back here.
  DataView view = make_view("atm", 4, 3, DataLayout::POINT_MAJOR);
  REQUIRE_THROWS_AS(tensor_from_field(view, "f1"), std::runtime_error);
}

TEST_CASE("a missing field is refused", "[data_view_bridge]") {
  DataView view = make_view("atm", 4, 3, DataLayout::FIELD_MAJOR);
  REQUIRE_THROWS_AS(tensor_from_field(view, "not_a_field"),
                    std::runtime_error);
}

TEST_CASE("a TensorMap can be built per field", "[data_view_bridge]") {
  DataView view = make_view("atm", 5, 3, DataLayout::FIELD_MAJOR);
  view(0, 0) = 1.0;
  view(2, 4) = 2.0;

  auto map = tensor_map_from_fields(view);

  REQUIRE(map.size() == 3);
  REQUIRE(map.names() == std::vector<std::string>{"f0", "f1", "f2"});
  REQUIRE(map.at("f0").element(0) == Approx(1.0));
  REQUIRE(map.at("f2").element(4) == Approx(2.0));
}

TEST_CASE("specs can be derived from a DataView", "[data_view_bridge]") {
  DataView view = make_view("atm", 6, 2, DataLayout::FIELD_MAJOR);

  const auto specs = specs_from_data_view(view, DType::F32);
  REQUIRE(specs.size() == 2);
  REQUIRE(specs[0].name == "f0");
  REQUIRE(specs[0].shape == Shape{6});
  REQUIRE(specs[0].dtype == DType::F32);

  const auto batched = specs_from_data_view(view, DType::F64, true);
  REQUIRE(batched[0].shape == Shape{-1, 6});
  REQUIRE(batched[0].dtype == DType::F64);
}

TEST_CASE("a DataView drives a backend end to end", "[data_view_bridge]") {
  // The shape of a real emulator step: import view in, export view out,
  // no copies and no intermediate buffers.
  DataView import_view = make_view("x2a", 4, 2, DataLayout::POINT_MAJOR);
  DataView export_view = make_view("a2x", 4, 2, DataLayout::POINT_MAJOR);

  for (int p = 0; p < 4; ++p) {
    import_view(0, p) = static_cast<double>(p);
    import_view(1, p) = static_cast<double>(p) * 2.0;
  }

  InferenceConfig config;
  config.set_option("stub.mode", "affine");
  config.set_option("stub.scale", "10.0");
  config.set_option("stub.offset", "1.0");
  auto backend = create_backend(config);
  backend->initialize();

  TensorMap inputs;
  inputs.set(tensor_from_data_view(import_view, "state"));
  TensorMap outputs;
  outputs.set(tensor_from_data_view(export_view, "tendency"));

  REQUIRE(backend->infer(inputs, outputs));

  // The backend wrote straight into the export DataView's buffer.
  REQUIRE(export_view(0, 0) == Approx(1.0));  // 10*0 + 1
  REQUIRE(export_view(1, 0) == Approx(1.0));  // 10*0 + 1
  REQUIRE(export_view(0, 3) == Approx(31.0)); // 10*3 + 1
  REQUIRE(export_view(1, 3) == Approx(61.0)); // 10*6 + 1

  backend->finalize();
}

} // namespace test
} // namespace inference
} // namespace emulator
