/**
 * @file tensor.hpp
 * @brief Minimal, dependency-free tensor container for inference I/O.
 *
 * The coupler side of an emulator speaks `double*` over flat MCT attribute
 * vectors; ML backends speak n-dimensional, usually single-precision tensors
 * with named inputs and outputs.  This header provides the small amount of
 * machinery needed to bridge the two without pulling LibTorch, ONNX Runtime,
 * or Python into `emulator_common`:
 *
 * - ::DType     — the element types we exchange with backends
 * - ::TensorView — a non-owning (name, pointer, dtype, shape) descriptor
 * - ::Tensor     — an owning buffer that exposes itself as a TensorView
 * - ::TensorMap  — an insertion-ordered collection of named tensors
 *
 * All tensors are row-major (C order), which is what NumPy, ONNX Runtime,
 * and contiguous `at::Tensor`s all use by default.
 */

#ifndef E3SM_EMULATOR_INFERENCE_TENSOR_HPP
#define E3SM_EMULATOR_INFERENCE_TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief Element types that can be exchanged with an inference backend.
 *
 * F32 is the default for ML models; F64 is what the coupler carries.
 * The integer types exist for models that take index/mask inputs.
 */
enum class DType {
  F32, ///< 32-bit IEEE float  (`float`)
  F64, ///< 64-bit IEEE float  (`double`)
  I32, ///< 32-bit signed int  (`std::int32_t`)
  I64  ///< 64-bit signed int  (`std::int64_t`)
};

/** @brief Size in bytes of one element of @p dtype. */
std::size_t dtype_size(DType dtype);

/** @brief Canonical lowercase name ("f32", "f64", "i32", "i64"). */
const char *dtype_name(DType dtype);

/**
 * @brief Parse a dtype from a string.
 *
 * Accepts the canonical names plus common aliases: "float"/"float32"/"single"
 * for F32, "double"/"float64" for F64, "int"/"int32" for I32, "long"/"int64"
 * for I64.  Case-insensitive.
 *
 * @throws std::runtime_error if @p text names no known dtype
 */
DType dtype_from_string(const std::string &text);

/** @brief Row-major tensor shape.  An empty shape denotes a scalar. */
using Shape = std::vector<std::int64_t>;

/** @brief Product of all extents in @p shape (1 for a scalar). */
std::int64_t num_elements(const Shape &shape);

/** @brief Render a shape as "[d0, d1, ...]" for logs and error messages. */
std::string shape_to_string(const Shape &shape);

/**
 * @brief Declarative description of one named tensor.
 *
 * Used in InferenceConfig to state what a model expects/produces, and
 * returned by backends that can introspect a loaded model.  A negative
 * extent marks a dynamic dimension (typically the leading batch axis).
 */
struct TensorSpec {
  std::string name;          ///< Tensor name as the model knows it
  Shape shape;               ///< Row-major shape; negative extent = dynamic
  DType dtype = DType::F32;  ///< Element type

  TensorSpec() = default;
  TensorSpec(std::string name_, Shape shape_, DType dtype_ = DType::F32)
      : name(std::move(name_)), shape(std::move(shape_)), dtype(dtype_) {}

  /** @brief Whether any extent is negative (i.e. resolved at runtime). */
  bool has_dynamic_dims() const;

  /**
   * @brief Resolve dynamic extents by substituting @p batch_size.
   *
   * Every negative extent is replaced by @p batch_size, which is almost
   * always only the leading axis.
   */
  Shape resolved_shape(std::int64_t batch_size) const;
};

/**
 * @brief Non-owning view of a contiguous, row-major tensor.
 *
 * A TensorView is a descriptor: it never allocates and never frees.  It is
 * how the emulator hands existing memory (a DataView buffer, an MCT
 * attribute vector, a stack array) to a backend, and how a backend hands
 * memory back.
 *
 * Views may be read-only; writing through a read-only view is rejected by
 * the copy helpers rather than silently permitted.
 */
class TensorView {
public:
  TensorView() = default;

  /**
   * @brief Construct a writable view.
   *
   * @param name   Tensor name (must match what the model expects)
   * @param data   Pointer to at least num_elements(shape) elements
   * @param dtype  Element type of @p data
   * @param shape  Row-major shape
   */
  TensorView(std::string name, void *data, DType dtype, Shape shape);

  /** @brief Construct a read-only view. */
  TensorView(std::string name, const void *data, DType dtype, Shape shape);

  /** @brief Writable view over a `double` buffer (the coupler's type). */
  static TensorView from_doubles(std::string name, double *data, Shape shape);

  /** @brief Read-only view over a `double` buffer. */
  static TensorView from_doubles(std::string name, const double *data,
                                 Shape shape);

  /** @brief Writable view over a `float` buffer. */
  static TensorView from_floats(std::string name, float *data, Shape shape);

  /** @brief Read-only view over a `float` buffer. */
  static TensorView from_floats(std::string name, const float *data,
                                Shape shape);

  // ── Accessors ────────────────────────────────────────────────────

  const std::string &name() const { return m_name; }
  void set_name(std::string name) { m_name = std::move(name); }

  DType dtype() const { return m_dtype; }
  const Shape &shape() const { return m_shape; }
  int rank() const { return static_cast<int>(m_shape.size()); }

  /** @brief Number of elements (product of extents). */
  std::int64_t size() const { return num_elements(m_shape); }

  /** @brief Number of bytes spanned by the view. */
  std::size_t nbytes() const;

  /** @brief Whether the underlying memory may be written. */
  bool writable() const { return m_writable; }

  /** @brief Whether the view points at anything at all. */
  bool valid() const { return m_data != nullptr; }

  /** @brief Raw read-only pointer. */
  const void *raw() const { return m_data; }

  /**
   * @brief Raw writable pointer.
   * @throws std::runtime_error if the view is read-only
   */
  void *raw_mutable();

  /**
   * @brief Typed read-only pointer.
   * @throws std::runtime_error if @p T does not match dtype()
   */
  template <typename T> const T *data_as() const {
    check_type<T>();
    return static_cast<const T *>(m_data);
  }

  /**
   * @brief Typed writable pointer.
   * @throws std::runtime_error if @p T mismatches dtype() or view is const
   */
  template <typename T> T *data_as() {
    check_type<T>();
    (void)raw_mutable(); // reuses the writability check / error message
    return static_cast<T *>(m_data);
  }

  /**
   * @brief Reinterpret this view with a new shape.
   *
   * The element count must be unchanged; no data is moved.
   * @throws std::runtime_error on element-count mismatch
   */
  void reshape(Shape shape);

  // ── Data movement ────────────────────────────────────────────────

  /**
   * @brief Element-wise copy from @p src, converting dtype if needed.
   *
   * This is the workhorse that lets a `double` coupler buffer feed an F32
   * model and vice versa.  Shapes need not match, only element counts —
   * a [2,3] source copies happily into a [6] destination.
   *
   * @throws std::runtime_error if this view is read-only or the element
   *         counts differ
   */
  void copy_from(const TensorView &src);

  /** @brief Set every element to @p value (converted to dtype()). */
  void fill(double value);

  /** @brief Set every byte to zero. */
  void zero();

  /**
   * @brief Read one element as a double, regardless of dtype.
   *
   * Convenience for tests and diagnostics; not a hot path.
   */
  double element(std::int64_t index) const;

  /** @brief Write one element from a double, converting to dtype(). */
  void set_element(std::int64_t index, double value);

private:
  template <typename T> void check_type() const;

  std::string m_name;
  void *m_data = nullptr;
  DType m_dtype = DType::F32;
  Shape m_shape;
  bool m_writable = false;
};

/**
 * @brief Owning, contiguous tensor.
 *
 * Used for staging buffers — for example converting a `double` coupler
 * buffer into the F32 layout a model wants, or holding model output before
 * it is scattered back into DataView fields.  Emulators are expected to
 * allocate these once during initialization and reuse them every step.
 */
class Tensor {
public:
  Tensor() = default;

  /** @brief Allocate a zero-filled tensor. */
  Tensor(std::string name, DType dtype, Shape shape);

  /** @brief Allocate from a spec, resolving dynamic dims with @p batch_size. */
  static Tensor from_spec(const TensorSpec &spec, std::int64_t batch_size = 1);

  // Movable, non-copyable: these can be large, so copies should be explicit.
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(Tensor &&) noexcept = default;

  /** @brief Reallocate (and zero) with a new dtype/shape. */
  void resize(DType dtype, Shape shape);

  const std::string &name() const { return m_name; }
  void set_name(std::string name);

  DType dtype() const { return m_dtype; }
  const Shape &shape() const { return m_shape; }
  std::int64_t size() const { return num_elements(m_shape); }
  std::size_t nbytes() const { return m_storage.size(); }

  /** @brief Writable view over this tensor's storage. */
  TensorView view();

  /** @brief Read-only view over this tensor's storage. */
  TensorView view() const;

  /** @brief Implicit-ish conversion for call sites that want a view. */
  operator TensorView() { return view(); }

  void *raw() { return m_storage.empty() ? nullptr : m_storage.data(); }
  const void *raw() const {
    return m_storage.empty() ? nullptr : m_storage.data();
  }

  template <typename T> T *data_as() { return view().data_as<T>(); }
  template <typename T> const T *data_as() const {
    return view().data_as<T>();
  }

private:
  std::string m_name;
  DType m_dtype = DType::F32;
  Shape m_shape;
  std::vector<unsigned char> m_storage;
};

/**
 * @brief Insertion-ordered map of named tensor views.
 *
 * Insertion order is preserved because several backends (LibTorch
 * TorchScript in particular) address their inputs positionally, while
 * others (ONNX Runtime, Python) address them by name.  Keeping both
 * addressing modes available lets one container serve every backend.
 */
class TensorMap {
public:
  TensorMap() = default;

  /**
   * @brief Insert or replace a tensor.
   *
   * The key is @p view.name(); inserting a name that already exists
   * overwrites the descriptor in place, preserving its position.
   */
  void set(TensorView view);

  /** @brief Insert or replace under an explicit key. */
  void set(const std::string &name, TensorView view);

  /** @brief Number of entries. */
  std::size_t size() const { return m_entries.size(); }
  bool empty() const { return m_entries.empty(); }

  /** @brief Whether @p name is present. */
  bool contains(const std::string &name) const;

  /**
   * @brief Look up by name.
   * @throws std::runtime_error if @p name is absent
   */
  TensorView &at(const std::string &name);
  const TensorView &at(const std::string &name) const;

  /** @brief Look up by name, or nullptr if absent. */
  TensorView *find(const std::string &name);
  const TensorView *find(const std::string &name) const;

  /**
   * @brief Positional access, in insertion order.
   * @throws std::runtime_error if @p index is out of range
   */
  TensorView &operator[](std::size_t index);
  const TensorView &operator[](std::size_t index) const;

  /** @brief Names in insertion order. */
  std::vector<std::string> names() const;

  void clear() { m_entries.clear(); }

  // Range-based for support.
  std::vector<TensorView>::iterator begin() { return m_entries.begin(); }
  std::vector<TensorView>::iterator end() { return m_entries.end(); }
  std::vector<TensorView>::const_iterator begin() const {
    return m_entries.begin();
  }
  std::vector<TensorView>::const_iterator end() const {
    return m_entries.end();
  }

private:
  std::vector<TensorView> m_entries;
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_TENSOR_HPP
