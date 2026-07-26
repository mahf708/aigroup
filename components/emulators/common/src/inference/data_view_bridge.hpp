/**
 * @file data_view_bridge.hpp
 * @brief Glue between the DataView data container and inference tensors.
 *
 * The two containers exist for different reasons and neither should
 * absorb the other.  DataView is the coupler-facing container: doubles,
 * two dimensions (fields × points), field names, units, and a
 * decomposition for parallel I/O.  TensorView is the model-facing
 * descriptor: any rank, any dtype, no metadata.  This header is the
 * whole of the mapping between them, so that `data/` and `inference/`
 * stay independent of each other — an emulator that only does I/O needs
 * no inference layer, and a standalone inference test needs no DataView.
 *
 * Header-only and opt-in: include it only where both are wanted.  It
 * requires the `data/` sources from the DataView work; if your build
 * does not have them, nothing else in `inference/` will miss it.
 */

#ifndef E3SM_EMULATOR_INFERENCE_DATA_VIEW_BRIDGE_HPP
#define E3SM_EMULATOR_INFERENCE_DATA_VIEW_BRIDGE_HPP

#include "data_view.hpp"
#include "tensor.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Zero-copy TensorView over an entire DataView buffer.
 *
 * The resulting shape follows the DataView's layout, so the caller does
 * not have to remember which way round it is:
 * - POINT_MAJOR → `[npoints, nfields]`, the layout an ML model wants
 * - FIELD_MAJOR → `[nfields, npoints]`
 *
 * The dtype is always F64, because that is what DataView stores.  A
 * model wanting F32 gets the conversion from the staging path in
 * whichever backend it uses, or from an explicit TensorView::copy_from.
 *
 * @param view DataView to wrap; must be allocated
 * @param name Tensor name the model knows this input by
 * @throws std::runtime_error if @p view is not allocated
 */
inline TensorView tensor_from_data_view(DataView &view,
                                        const std::string &name) {
  if (!view.is_allocated()) {
    throw std::runtime_error("inference: DataView '" + view.name() +
                             "' must be allocated before it can be wrapped "
                             "as a tensor");
  }
  const auto dims = view.shape();
  return TensorView::from_doubles(
      name, view.data(),
      {static_cast<std::int64_t>(dims.first),
       static_cast<std::int64_t>(dims.second)});
}

/** @brief Read-only overload.  @see tensor_from_data_view */
inline TensorView tensor_from_data_view(const DataView &view,
                                        const std::string &name) {
  if (!view.is_allocated()) {
    throw std::runtime_error("inference: DataView '" + view.name() +
                             "' must be allocated before it can be wrapped "
                             "as a tensor");
  }
  const auto dims = view.shape();
  return TensorView::from_doubles(
      name, view.data(),
      {static_cast<std::int64_t>(dims.first),
       static_cast<std::int64_t>(dims.second)});
}

/**
 * @brief Zero-copy TensorView over a single DataView field.
 *
 * Only valid for a FIELD_MAJOR DataView, where one field really is a
 * contiguous run of `npoints` doubles.  In a POINT_MAJOR view a field is
 * strided and cannot be described by a TensorView, which is contiguous
 * by construction; use tensor_from_data_view on the whole buffer and
 * have the model index the channel, or build the DataView FIELD_MAJOR.
 *
 * @param view       Allocated, FIELD_MAJOR DataView
 * @param field_name Field to wrap
 * @param name       Tensor name, defaulting to @p field_name
 * @throws std::runtime_error if the view is POINT_MAJOR or unallocated,
 *         or the field does not exist
 */
inline TensorView tensor_from_field(DataView &view,
                                    const std::string &field_name,
                                    const std::string &name = "") {
  if (!view.is_allocated()) {
    throw std::runtime_error("inference: DataView '" + view.name() +
                             "' must be allocated before wrapping a field");
  }
  if (view.layout() != DataLayout::FIELD_MAJOR) {
    throw std::runtime_error(
        "inference: field '" + field_name + "' of DataView '" + view.name() +
        "' is strided in a POINT_MAJOR layout and cannot be wrapped as a "
        "contiguous tensor; wrap the whole view instead, or build the "
        "DataView FIELD_MAJOR");
  }
  const int idx = view.field_index(field_name); // throws if absent
  return TensorView::from_doubles(name.empty() ? field_name : name,
                                  view.field_data(idx),
                                  {static_cast<std::int64_t>(view.num_points())});
}

/**
 * @brief Build a TensorMap naming each DataView field as its own tensor.
 *
 * The natural shape for models with one named input per coupling field,
 * which is how the Python backend's contract reads most naturally.
 * FIELD_MAJOR only, for the striding reason above.
 *
 * @param view DataView whose fields become tensors, each `[npoints]`
 * @throws std::runtime_error if @p view is POINT_MAJOR or unallocated
 */
inline TensorMap tensor_map_from_fields(DataView &view) {
  TensorMap map;
  for (int i = 0; i < view.num_fields(); ++i)
    map.set(tensor_from_field(view, view.field_spec(i).name));
  return map;
}

/**
 * @brief Derive TensorSpecs from a DataView's registered fields.
 *
 * Useful for filling in an InferenceConfig from the coupling field list
 * an emulator already parsed, rather than restating it in a namelist.
 *
 * @param view    DataView to describe
 * @param dtype   Element type to declare (the model's, not DataView's)
 * @param batched Whether to prepend a dynamic batch axis
 */
inline std::vector<TensorSpec>
specs_from_data_view(const DataView &view, DType dtype = DType::F32,
                     bool batched = false) {
  std::vector<TensorSpec> specs;
  specs.reserve(static_cast<std::size_t>(view.num_fields()));
  for (int i = 0; i < view.num_fields(); ++i) {
    Shape shape;
    if (batched)
      shape.push_back(-1);
    shape.push_back(static_cast<std::int64_t>(view.num_points()));
    specs.emplace_back(view.field_spec(i).name, shape, dtype);
  }
  return specs;
}

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_DATA_VIEW_BRIDGE_HPP
