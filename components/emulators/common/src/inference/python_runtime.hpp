/**
 * @file python_runtime.hpp
 * @brief Embedded CPython interpreter management for the Python backend.
 *
 * Separated from python_inference_backend so that the fiddly parts —
 * whose job it is to call Py_Initialize, who holds the GIL, how a C++
 * buffer becomes a NumPy array without copying — live in one reviewable
 * place, and so that other emulator code (diagnostics, custom
 * parameterizations) can embed Python through the same machinery.
 *
 * This header includes `<Python.h>` and is therefore only available in
 * builds configured with `-DEMULATOR_ENABLE_PYTHON=ON`.
 */

#ifndef E3SM_EMULATOR_INFERENCE_PYTHON_RUNTIME_HPP
#define E3SM_EMULATOR_INFERENCE_PYTHON_RUNTIME_HPP

#include "tensor.hpp"

// Python.h insists on being included before any system header.
#include <Python.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace emulator {
namespace inference {

/**
 * @brief RAII owner of a borrowed-or-new Python reference.
 *
 * CPython reference counting is the main source of bugs in embedded
 * code.  PyRef makes ownership explicit: it always owns a *strong*
 * reference and always releases it.  Use PyRef::steal() for API calls
 * that return a new reference (most of them) and PyRef::borrow() for the
 * few that return a borrowed one.
 */
class PyRef {
public:
  PyRef() = default;

  /** @brief Take ownership of a new reference (does not incref). */
  static PyRef steal(PyObject *obj) { return PyRef(obj); }

  /** @brief Share a borrowed reference (increfs). */
  static PyRef borrow(PyObject *obj) {
    Py_XINCREF(obj);
    return PyRef(obj);
  }

  ~PyRef() { Py_XDECREF(m_obj); }

  PyRef(const PyRef &other) : m_obj(other.m_obj) { Py_XINCREF(m_obj); }
  PyRef &operator=(const PyRef &other) {
    if (this != &other) {
      Py_XINCREF(other.m_obj);
      Py_XDECREF(m_obj);
      m_obj = other.m_obj;
    }
    return *this;
  }
  PyRef(PyRef &&other) noexcept : m_obj(other.m_obj) { other.m_obj = nullptr; }
  PyRef &operator=(PyRef &&other) noexcept {
    if (this != &other) {
      Py_XDECREF(m_obj);
      m_obj = other.m_obj;
      other.m_obj = nullptr;
    }
    return *this;
  }

  PyObject *get() const { return m_obj; }
  explicit operator bool() const { return m_obj != nullptr; }

  /** @brief Relinquish ownership to a caller that will decref. */
  PyObject *release() {
    PyObject *out = m_obj;
    m_obj = nullptr;
    return out;
  }

  void reset() {
    Py_XDECREF(m_obj);
    m_obj = nullptr;
  }

private:
  explicit PyRef(PyObject *obj) : m_obj(obj) {}
  PyObject *m_obj = nullptr;
};

/**
 * @brief Scoped GIL acquisition.
 *
 * Every call into the CPython API must hold the GIL.  Construct one of
 * these around any block that touches Python and it will be released
 * again on the way out, including on an exception.
 *
 * Safe to nest, and safe on any thread — `PyGILState_Ensure` registers
 * the calling thread with the interpreter if it is not known yet.
 */
class PyGil {
public:
  PyGil() : m_state(PyGILState_Ensure()) {}
  ~PyGil() { PyGILState_Release(m_state); }

  PyGil(const PyGil &) = delete;
  PyGil &operator=(const PyGil &) = delete;

private:
  PyGILState_STATE m_state;
};

/**
 * @brief Process-wide embedded interpreter, shared by all backends.
 *
 * CPython supports exactly one interpreter per process for our purposes,
 * so PythonRuntime is a singleton rather than a per-backend object: the
 * first acquire() starts it and it then lives for the life of the
 * process.  Several emulator instances — several components, or several
 * instances of one component in a multi-instance run — share one
 * interpreter and one import cache.
 *
 * The interpreter is never finalized.  That is not an oversight; see
 * the comment on acquire() in the implementation for why restarting it
 * is impossible and tearing it down at exit is unwise.
 *
 * ## Parallelism
 *
 * Under MPI each rank is a separate process and so gets its own
 * interpreter; there is no cross-rank GIL contention.  Ranks that need
 * to talk to each other should do so inside Python, rebuilding the
 * communicator from the Fortran handle the backend passes in its config
 * dict (`mpi_comm_fortran`) via `mpi4py.MPI.Comm.f2py()`.  That is the
 * route by which a spatially decomposed model — an ACE2-scale emulator
 * with halo exchange — runs under this backend.
 *
 * Within a rank, the GIL serializes Python-level work, so threading is
 * not a way to speed up the Python side.  It does not serialize the ML
 * framework underneath: PyTorch releases the GIL inside its kernels, so
 * a rank's GPU work still overlaps with other ranks'.
 */
class PythonRuntime {
public:
  /**
   * @brief Get (and on first call, start) the interpreter.
   *
   * The returned handle does not own the interpreter — dropping the
   * last one does not shut it down.  It is a shared_ptr so that callers
   * can hold it without thinking about lifetime.
   *
   * @throws std::runtime_error if the interpreter cannot be started, or
   *         if NumPy is not importable from it
   */
  static std::shared_ptr<PythonRuntime> acquire();

  ~PythonRuntime();

  PythonRuntime(const PythonRuntime &) = delete;
  PythonRuntime &operator=(const PythonRuntime &) = delete;

  /**
   * @brief Whether this object started the interpreter.
   *
   * False when we attached to an interpreter someone else had already
   * initialized — in that case we must not finalize it on the way out.
   */
  bool owns_interpreter() const { return m_owns_interpreter; }

  /** @brief Python version string, for logging. */
  std::string version() const;

  // ── Module and object helpers (all require the GIL held) ─────────

  /**
   * @brief Prepend a directory to `sys.path`.
   *
   * Idempotent: a path already present is not added twice.
   */
  void add_sys_path(const std::string &path);

  /**
   * @brief Import a module by name, or load one from a `.py` file path.
   *
   * A @p spec ending in `.py` is loaded from that file via importlib,
   * with the containing directory added to `sys.path` first so the
   * module's own relative imports resolve.  Anything else is treated as
   * a dotted module name and imported normally.
   *
   * @throws std::runtime_error carrying the Python traceback on failure
   */
  PyRef import_module(const std::string &spec);

  /**
   * @brief Wrap raw memory as a NumPy array without copying.
   *
   * Builds `numpy.frombuffer(<memoryview of ptr>, dtype).reshape(shape)`.
   * Because `frombuffer` on a writable memoryview yields a writable
   * array, and a reshape of a contiguous array is a view, the returned
   * array aliases @p data directly — a Python model writing into it
   * writes into the emulator's buffer.
   *
   * Going through `frombuffer` rather than `PyArray_SimpleNewFromData`
   * is deliberate: it means this file compiles against nothing but
   * `Python.h`, so the build needs no NumPy headers and cannot break on
   * a NumPy ABI change.  NumPy is still required at *runtime*.
   *
   * @param data     Pointer to at least num_elements(shape) elements
   * @param dtype    Element type of @p data
   * @param shape    Row-major shape to present
   * @param writable Whether Python may modify the underlying memory
   * @throws std::runtime_error if NumPy is unavailable or the wrap fails
   */
  PyRef wrap_array(void *data, DType dtype, const Shape &shape, bool writable);

  /** @brief Wrap a TensorView, honouring its writability. */
  PyRef wrap_tensor(const TensorView &view, bool writable);

  /**
   * @brief Build a Python dict from string key/value pairs.
   *
   * Values are kept as strings; the Python side is responsible for any
   * coercion it wants.  Callers add typed entries afterwards with
   * dict_set_*.
   */
  PyRef make_dict(const std::map<std::string, std::string> &items);

  /** @brief Set a string entry on a dict. */
  void dict_set(PyObject *dict, const std::string &key,
                const std::string &value);

  /** @brief Set an integer entry on a dict. */
  void dict_set_int(PyObject *dict, const std::string &key, long value);

  /** @brief Set a boolean entry on a dict. */
  void dict_set_bool(PyObject *dict, const std::string &key, bool value);

  /** @brief Set an arbitrary object entry on a dict (steals nothing). */
  void dict_set_obj(PyObject *dict, const std::string &key, PyObject *value);

  /** @brief NumPy dtype name for a DType ("float32", "float64", …). */
  static const char *numpy_dtype_name(DType dtype);

  /**
   * @brief Format and clear the current Python exception.
   *
   * @param context Prefix for the message, e.g. "importing model module"
   * @return A multi-line message including the traceback when available,
   *         or "" if no exception was set
   */
  static std::string fetch_error(const std::string &context);

  /**
   * @brief Throw a std::runtime_error carrying the Python traceback.
   *
   * Call after any CPython API returns an error indicator.
   * @throws std::runtime_error always
   */
  [[noreturn]] static void throw_error(const std::string &context);

private:
  PythonRuntime();

  bool m_owns_interpreter = false;
  PyThreadState *m_saved_state = nullptr; ///< Main thread state, if we own it
  PyRef m_numpy;                          ///< Cached `numpy` module
};

} // namespace inference
} // namespace emulator

#endif // E3SM_EMULATOR_INFERENCE_PYTHON_RUNTIME_HPP
