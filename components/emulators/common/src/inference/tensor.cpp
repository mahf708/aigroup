/**
 * @file tensor.cpp
 * @brief Implementation of the inference tensor container.
 */

#include "tensor.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace emulator {
namespace inference {

namespace {

/**
 * @brief Dispatch a functor over the concrete C++ type behind a DType.
 *
 * Keeps the conversion table in one place instead of repeating a 4-way
 * switch in every helper.
 */
template <typename Fn> auto dispatch(DType dtype, Fn &&fn) -> decltype(fn(float{})) {
  switch (dtype) {
  case DType::F32:
    return fn(float{});
  case DType::F64:
    return fn(double{});
  case DType::I32:
    return fn(std::int32_t{});
  case DType::I64:
    return fn(std::int64_t{});
  }
  throw std::runtime_error("inference::Tensor: unknown DType");
}

/** @brief Read element @p i of a typed buffer as a double. */
double read_as_double(const void *data, DType dtype, std::int64_t i) {
  return dispatch(dtype, [&](auto tag) -> double {
    using T = decltype(tag);
    return static_cast<double>(static_cast<const T *>(data)[i]);
  });
}

/** @brief Write @p value into element @p i of a typed buffer. */
void write_from_double(void *data, DType dtype, std::int64_t i, double value) {
  dispatch(dtype, [&](auto tag) -> double {
    using T = decltype(tag);
    static_cast<T *>(data)[i] = static_cast<T>(value);
    return 0.0;
  });
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return s;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// DType helpers
// ─────────────────────────────────────────────────────────────────────

std::size_t dtype_size(DType dtype) {
  return dispatch(dtype, [](auto tag) -> std::size_t {
    // dispatch() is declared to return decltype(fn(float{})); the lambda
    // returns size_t uniformly so all branches agree.
    return sizeof(decltype(tag));
  });
}

const char *dtype_name(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return "f32";
  case DType::F64:
    return "f64";
  case DType::I32:
    return "i32";
  case DType::I64:
    return "i64";
  }
  return "unknown";
}

DType dtype_from_string(const std::string &text) {
  const std::string t = lower(text);
  if (t == "f32" || t == "float" || t == "float32" || t == "single" ||
      t == "real4")
    return DType::F32;
  if (t == "f64" || t == "double" || t == "float64" || t == "real8")
    return DType::F64;
  if (t == "i32" || t == "int" || t == "int32" || t == "integer")
    return DType::I32;
  if (t == "i64" || t == "long" || t == "int64")
    return DType::I64;
  throw std::runtime_error("inference: unknown dtype '" + text + "'");
}

std::int64_t num_elements(const Shape &shape) {
  std::int64_t n = 1;
  for (std::int64_t d : shape)
    n *= d;
  return n;
}

std::string shape_to_string(const Shape &shape) {
  std::ostringstream os;
  os << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i)
      os << ", ";
    os << shape[i];
  }
  os << ']';
  return os.str();
}

// ─────────────────────────────────────────────────────────────────────
// TensorSpec
// ─────────────────────────────────────────────────────────────────────

bool TensorSpec::has_dynamic_dims() const {
  return std::any_of(shape.begin(), shape.end(),
                     [](std::int64_t d) { return d < 0; });
}

Shape TensorSpec::resolved_shape(std::int64_t batch_size) const {
  Shape out = shape;
  for (std::int64_t &d : out) {
    if (d < 0)
      d = batch_size;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────
// TensorView
// ─────────────────────────────────────────────────────────────────────

TensorView::TensorView(std::string name, void *data, DType dtype, Shape shape)
    : m_name(std::move(name)), m_data(data), m_dtype(dtype),
      m_shape(std::move(shape)), m_writable(true) {}

TensorView::TensorView(std::string name, const void *data, DType dtype,
                       Shape shape)
    : m_name(std::move(name)), m_data(const_cast<void *>(data)), m_dtype(dtype),
      m_shape(std::move(shape)), m_writable(false) {}

TensorView TensorView::from_doubles(std::string name, double *data,
                                    Shape shape) {
  return TensorView(std::move(name), static_cast<void *>(data), DType::F64,
                    std::move(shape));
}

TensorView TensorView::from_doubles(std::string name, const double *data,
                                    Shape shape) {
  return TensorView(std::move(name), static_cast<const void *>(data),
                    DType::F64, std::move(shape));
}

TensorView TensorView::from_floats(std::string name, float *data, Shape shape) {
  return TensorView(std::move(name), static_cast<void *>(data), DType::F32,
                    std::move(shape));
}

TensorView TensorView::from_floats(std::string name, const float *data,
                                   Shape shape) {
  return TensorView(std::move(name), static_cast<const void *>(data),
                    DType::F32, std::move(shape));
}

std::size_t TensorView::nbytes() const {
  return static_cast<std::size_t>(size()) * dtype_size(m_dtype);
}

void *TensorView::raw_mutable() {
  if (!m_writable) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': write to a read-only view");
  }
  return m_data;
}

template <typename T> void TensorView::check_type() const {
  const bool ok = (std::is_same<T, float>::value && m_dtype == DType::F32) ||
                  (std::is_same<T, double>::value && m_dtype == DType::F64) ||
                  (std::is_same<T, std::int32_t>::value &&
                   m_dtype == DType::I32) ||
                  (std::is_same<T, std::int64_t>::value && m_dtype == DType::I64);
  if (!ok) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': typed access does not match dtype " +
                             dtype_name(m_dtype));
  }
}

// Explicit instantiations so the definition can live in the .cpp.
template void TensorView::check_type<float>() const;
template void TensorView::check_type<double>() const;
template void TensorView::check_type<std::int32_t>() const;
template void TensorView::check_type<std::int64_t>() const;

void TensorView::reshape(Shape shape) {
  if (num_elements(shape) != size()) {
    throw std::runtime_error(
        "inference::TensorView '" + m_name + "': reshape " +
        shape_to_string(m_shape) + " -> " + shape_to_string(shape) +
        " changes the element count");
  }
  m_shape = std::move(shape);
}

void TensorView::copy_from(const TensorView &src) {
  if (!m_writable) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': copy_from into a read-only view");
  }
  if (!src.valid() || !valid()) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': copy_from with an unbound view");
  }
  const std::int64_t n = size();
  if (src.size() != n) {
    throw std::runtime_error(
        "inference::TensorView '" + m_name + "': copy_from size mismatch (" +
        shape_to_string(src.shape()) + " -> " + shape_to_string(m_shape) + ")");
  }

  if (src.dtype() == m_dtype) {
    std::memcpy(m_data, src.raw(), nbytes());
    return;
  }
  for (std::int64_t i = 0; i < n; ++i) {
    write_from_double(m_data, m_dtype, i,
                      read_as_double(src.raw(), src.dtype(), i));
  }
}

void TensorView::fill(double value) {
  if (!m_writable) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': fill on a read-only view");
  }
  const std::int64_t n = size();
  for (std::int64_t i = 0; i < n; ++i)
    write_from_double(m_data, m_dtype, i, value);
}

void TensorView::zero() {
  if (!m_writable) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': zero on a read-only view");
  }
  if (m_data)
    std::memset(m_data, 0, nbytes());
}

double TensorView::element(std::int64_t index) const {
  if (index < 0 || index >= size()) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': element index out of range");
  }
  return read_as_double(m_data, m_dtype, index);
}

void TensorView::set_element(std::int64_t index, double value) {
  if (!m_writable) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': set_element on a read-only view");
  }
  if (index < 0 || index >= size()) {
    throw std::runtime_error("inference::TensorView '" + m_name +
                             "': element index out of range");
  }
  write_from_double(m_data, m_dtype, index, value);
}

// ─────────────────────────────────────────────────────────────────────
// Tensor
// ─────────────────────────────────────────────────────────────────────

Tensor::Tensor(std::string name, DType dtype, Shape shape)
    : m_name(std::move(name)), m_dtype(dtype), m_shape(std::move(shape)) {
  m_storage.assign(static_cast<std::size_t>(num_elements(m_shape)) *
                       dtype_size(m_dtype),
                   0);
}

Tensor Tensor::from_spec(const TensorSpec &spec, std::int64_t batch_size) {
  return Tensor(spec.name, spec.dtype, spec.resolved_shape(batch_size));
}

void Tensor::resize(DType dtype, Shape shape) {
  m_dtype = dtype;
  m_shape = std::move(shape);
  m_storage.assign(static_cast<std::size_t>(num_elements(m_shape)) *
                       dtype_size(m_dtype),
                   0);
}

void Tensor::set_name(std::string name) { m_name = std::move(name); }

TensorView Tensor::view() {
  return TensorView(m_name, static_cast<void *>(raw()), m_dtype, m_shape);
}

TensorView Tensor::view() const {
  return TensorView(m_name, static_cast<const void *>(raw()), m_dtype, m_shape);
}

// ─────────────────────────────────────────────────────────────────────
// TensorMap
// ─────────────────────────────────────────────────────────────────────

void TensorMap::set(TensorView view) {
  const std::string key = view.name();
  for (auto &entry : m_entries) {
    if (entry.name() == key) {
      entry = std::move(view);
      return;
    }
  }
  m_entries.push_back(std::move(view));
}

void TensorMap::set(const std::string &name, TensorView view) {
  view.set_name(name);
  set(std::move(view));
}

bool TensorMap::contains(const std::string &name) const {
  return find(name) != nullptr;
}

TensorView *TensorMap::find(const std::string &name) {
  for (auto &entry : m_entries) {
    if (entry.name() == name)
      return &entry;
  }
  return nullptr;
}

const TensorView *TensorMap::find(const std::string &name) const {
  for (const auto &entry : m_entries) {
    if (entry.name() == name)
      return &entry;
  }
  return nullptr;
}

TensorView &TensorMap::at(const std::string &name) {
  if (TensorView *v = find(name))
    return *v;
  throw std::runtime_error("inference::TensorMap: no tensor named '" + name +
                           "'");
}

const TensorView &TensorMap::at(const std::string &name) const {
  if (const TensorView *v = find(name))
    return *v;
  throw std::runtime_error("inference::TensorMap: no tensor named '" + name +
                           "'");
}

TensorView &TensorMap::operator[](std::size_t index) {
  if (index >= m_entries.size())
    throw std::runtime_error("inference::TensorMap: index out of range");
  return m_entries[index];
}

const TensorView &TensorMap::operator[](std::size_t index) const {
  if (index >= m_entries.size())
    throw std::runtime_error("inference::TensorMap: index out of range");
  return m_entries[index];
}

std::vector<std::string> TensorMap::names() const {
  std::vector<std::string> out;
  out.reserve(m_entries.size());
  for (const auto &entry : m_entries)
    out.push_back(entry.name());
  return out;
}

} // namespace inference
} // namespace emulator
