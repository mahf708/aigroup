/**
 * @file python_runtime.cpp
 * @brief Embedded CPython interpreter management.
 */

#include "python_runtime.hpp"

#include <sstream>
#include <stdexcept>

namespace emulator {
namespace inference {

namespace {

/** @brief Convert a Python str/bytes object to std::string, best effort. */
std::string to_string(PyObject *obj) {
  if (!obj)
    return "";
  PyRef str = PyRef::steal(PyObject_Str(obj));
  if (!str)
    return "<unprintable>";
  const char *utf8 = PyUnicode_AsUTF8(str.get());
  return utf8 ? std::string(utf8) : std::string("<non-utf8>");
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────

PythonRuntime::PythonRuntime() {
  if (Py_IsInitialized()) {
    // Somebody else — an embedding host, or a previous runtime that has
    // not been torn down — already owns the interpreter.  Attach to it
    // and, importantly, do not finalize it in our destructor.
    m_owns_interpreter = false;
  } else {
    // Py_InitializeEx(0) skips signal-handler installation, which an
    // embedded interpreter inside a coupled model must not touch: MPI
    // and the driver own SIGINT/SIGTERM handling.
    Py_InitializeEx(0);
    if (!Py_IsInitialized())
      throw std::runtime_error("PythonRuntime: Py_InitializeEx failed");
    m_owns_interpreter = true;

    // Py_Initialize leaves the calling thread holding the GIL.  Release
    // it so that every subsequent access goes through PyGILState_Ensure
    // uniformly, including from threads the interpreter has never seen.
    m_saved_state = PyEval_SaveThread();
  }

  PyGil gil;
  m_numpy = PyRef::steal(PyImport_ImportModule("numpy"));
  if (!m_numpy) {
    const std::string err =
        fetch_error("importing numpy for the Python inference backend");
    throw std::runtime_error(
        err + "\nThe Python backend exchanges data with models as NumPy "
              "arrays; install numpy in the interpreter E3SM is linked "
              "against.");
  }
}

PythonRuntime::~PythonRuntime() {
  // Deliberately does not call Py_FinalizeEx(); see acquire().
}

std::shared_ptr<PythonRuntime> PythonRuntime::acquire() {
  // The interpreter is started once and then kept for the life of the
  // process.  Both halves of that are deliberate.
  //
  // It cannot be restarted.  Py_FinalizeEx() followed by a second
  // Py_Initialize() looks like it should work, but any extension module
  // holding C-level state — NumPy above all, and therefore PyTorch and
  // everything built on it — refuses to load a second time with
  // "cannot load module more than once per process".  A backend that
  // finalized on the way out would work exactly once per run, which is
  // the sort of bug that only shows up in the second nested case of a
  // test suite or the second instance of a multi-instance run.
  //
  // Nor should it be torn down at exit.  Running Py_FinalizeEx() from a
  // static destructor puts arbitrary Python teardown after
  // MPI_Finalize() and after the driver has released its resources,
  // which is a good way to turn a clean run into a crash in the last
  // second of a job.  The interpreter is a process-lifetime resource,
  // like the MPI runtime itself; the memory it holds is reclaimed when
  // the process exits.
  //
  // Backends still release *their* references in finalize().  That is
  // what frees model weights, which is the memory that actually matters
  // in a coupled run.
  static PythonRuntime *singleton = new PythonRuntime();
  static std::shared_ptr<PythonRuntime> handle(singleton,
                                               [](PythonRuntime *) {});
  return handle;
}

std::string PythonRuntime::version() const {
  PyGil gil;
  const char *v = Py_GetVersion();
  std::string s = v ? v : "unknown";
  // Py_GetVersion() returns "3.11.9 (main, ...) \n[GCC ...]"; keep the
  // first token, which is all a log line needs.
  const auto space = s.find(' ');
  return space == std::string::npos ? s : s.substr(0, space);
}

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

std::string PythonRuntime::fetch_error(const std::string &context) {
  if (!PyErr_Occurred())
    return "";

  PyObject *type = nullptr;
  PyObject *value = nullptr;
  PyObject *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  std::ostringstream os;
  os << "Python error while " << context << ":\n";
  if (type)
    os << "  " << to_string(type) << ": ";
  os << to_string(value) << "\n";

  // Render the traceback through the `traceback` module when we can; it
  // is the difference between a usable report and a bare exception name.
  if (traceback) {
    PyRef tb_mod = PyRef::steal(PyImport_ImportModule("traceback"));
    if (tb_mod) {
      PyRef lines = PyRef::steal(PyObject_CallMethod(
          tb_mod.get(), "format_exception", "OOO", type ? type : Py_None,
          value ? value : Py_None, traceback));
      if (lines) {
        const Py_ssize_t n = PyList_Size(lines.get());
        for (Py_ssize_t i = 0; i < n; ++i)
          os << to_string(PyList_GetItem(lines.get(), i));
      } else {
        PyErr_Clear();
      }
    } else {
      PyErr_Clear();
    }
  }

  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  return os.str();
}

void PythonRuntime::throw_error(const std::string &context) {
  std::string message = fetch_error(context);
  if (message.empty())
    message = "Python error while " + context + " (no exception set)";
  throw std::runtime_error(message);
}

// ─────────────────────────────────────────────────────────────────────
// Modules
// ─────────────────────────────────────────────────────────────────────

void PythonRuntime::add_sys_path(const std::string &path) {
  if (path.empty())
    return;

  PyObject *sys_path = PySys_GetObject("path"); // borrowed
  if (!sys_path)
    throw std::runtime_error("PythonRuntime: sys.path is unavailable");

  const Py_ssize_t n = PyList_Size(sys_path);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject *item = PyList_GetItem(sys_path, i); // borrowed
    if (item && PyUnicode_Check(item)) {
      const char *utf8 = PyUnicode_AsUTF8(item);
      if (utf8 && path == utf8)
        return; // already present
    }
  }

  PyRef entry = PyRef::steal(PyUnicode_FromString(path.c_str()));
  if (!entry || PyList_Insert(sys_path, 0, entry.get()) != 0)
    throw_error("adding '" + path + "' to sys.path");
}

PyRef PythonRuntime::import_module(const std::string &spec) {
  const bool is_file = spec.size() > 3 && spec.compare(spec.size() - 3, 3, ".py") == 0;

  if (!is_file) {
    PyRef mod = PyRef::steal(PyImport_ImportModule(spec.c_str()));
    if (!mod)
      throw_error("importing Python module '" + spec + "'");
    return mod;
  }

  // Load from an explicit file path.  Add its directory to sys.path
  // first so that a model file importing its siblings still works.
  const auto slash = spec.find_last_of('/');
  const std::string dir = slash == std::string::npos ? "." : spec.substr(0, slash);
  const std::string file = slash == std::string::npos ? spec : spec.substr(slash + 1);
  const std::string mod_name = file.substr(0, file.size() - 3);
  add_sys_path(dir);

  PyRef util = PyRef::steal(PyImport_ImportModule("importlib.util"));
  if (!util)
    throw_error("importing importlib.util");

  PyRef spec_obj = PyRef::steal(PyObject_CallMethod(
      util.get(), "spec_from_file_location", "ss", mod_name.c_str(),
      spec.c_str()));
  if (!spec_obj || spec_obj.get() == Py_None)
    throw_error("building an import spec for '" + spec + "'");

  PyRef mod = PyRef::steal(
      PyObject_CallMethod(util.get(), "module_from_spec", "O", spec_obj.get()));
  if (!mod)
    throw_error("creating a module from '" + spec + "'");

  // Register before executing so that a module referring to itself by
  // name (dataclasses, pickling, relative imports) resolves.
  if (PyDict_SetItemString(PyImport_GetModuleDict(), mod_name.c_str(),
                           mod.get()) != 0) {
    throw_error("registering module '" + mod_name + "' in sys.modules");
  }

  PyRef loader = PyRef::steal(PyObject_GetAttrString(spec_obj.get(), "loader"));
  if (!loader)
    throw_error("getting the loader for '" + spec + "'");
  PyRef result = PyRef::steal(
      PyObject_CallMethod(loader.get(), "exec_module", "O", mod.get()));
  if (!result)
    throw_error("executing Python model file '" + spec + "'");

  return mod;
}

// ─────────────────────────────────────────────────────────────────────
// Arrays
// ─────────────────────────────────────────────────────────────────────

const char *PythonRuntime::numpy_dtype_name(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return "float32";
  case DType::F64:
    return "float64";
  case DType::I32:
    return "int32";
  case DType::I64:
    return "int64";
  }
  return "float32";
}

PyRef PythonRuntime::wrap_array(void *data, DType dtype, const Shape &shape,
                                bool writable) {
  if (!data)
    throw std::runtime_error("PythonRuntime::wrap_array: null pointer");
  if (!m_numpy)
    throw std::runtime_error("PythonRuntime::wrap_array: numpy unavailable");

  const std::size_t nbytes =
      static_cast<std::size_t>(num_elements(shape)) * dtype_size(dtype);

  PyRef mv = PyRef::steal(PyMemoryView_FromMemory(
      static_cast<char *>(data), static_cast<Py_ssize_t>(nbytes),
      writable ? PyBUF_WRITE : PyBUF_READ));
  if (!mv)
    throw_error("creating a memoryview over emulator memory");

  PyRef flat = PyRef::steal(PyObject_CallMethod(
      m_numpy.get(), "frombuffer", "Os", mv.get(), numpy_dtype_name(dtype)));
  if (!flat)
    throw_error("calling numpy.frombuffer on emulator memory");

  // A 1-D array is already the right thing when the shape is 1-D; the
  // reshape below handles it uniformly and stays a view either way.
  PyRef dims = PyRef::steal(PyTuple_New(static_cast<Py_ssize_t>(shape.size())));
  if (!dims)
    throw_error("allocating a shape tuple");
  for (std::size_t i = 0; i < shape.size(); ++i) {
    PyObject *dim = PyLong_FromLongLong(static_cast<long long>(shape[i]));
    if (!dim)
      throw_error("building a shape tuple");
    PyTuple_SET_ITEM(dims.get(), static_cast<Py_ssize_t>(i), dim); // steals
  }

  PyRef array = PyRef::steal(
      PyObject_CallMethod(flat.get(), "reshape", "O", dims.get()));
  if (!array)
    throw_error("reshaping a NumPy view of emulator memory");

  return array;
}

PyRef PythonRuntime::wrap_tensor(const TensorView &view, bool writable) {
  if (writable && !view.writable()) {
    throw std::runtime_error("PythonRuntime: tensor '" + view.name() +
                             "' was requested as writable but the view is "
                             "read-only");
  }
  // const_cast is sound here: when writable is false the memoryview is
  // created read-only, so Python cannot reach through it to write.
  return wrap_array(const_cast<void *>(view.raw()), view.dtype(), view.shape(),
                    writable);
}

// ─────────────────────────────────────────────────────────────────────
// Dicts
// ─────────────────────────────────────────────────────────────────────

PyRef PythonRuntime::make_dict(const std::map<std::string, std::string> &items) {
  PyRef dict = PyRef::steal(PyDict_New());
  if (!dict)
    throw_error("allocating a Python dict");
  for (const auto &kv : items)
    dict_set(dict.get(), kv.first, kv.second);
  return dict;
}

void PythonRuntime::dict_set(PyObject *dict, const std::string &key,
                             const std::string &value) {
  PyRef val = PyRef::steal(PyUnicode_FromString(value.c_str()));
  if (!val || PyDict_SetItemString(dict, key.c_str(), val.get()) != 0)
    throw_error("setting dict entry '" + key + "'");
}

void PythonRuntime::dict_set_int(PyObject *dict, const std::string &key,
                                 long value) {
  PyRef val = PyRef::steal(PyLong_FromLong(value));
  if (!val || PyDict_SetItemString(dict, key.c_str(), val.get()) != 0)
    throw_error("setting dict entry '" + key + "'");
}

void PythonRuntime::dict_set_bool(PyObject *dict, const std::string &key,
                                  bool value) {
  PyObject *val = value ? Py_True : Py_False;
  if (PyDict_SetItemString(dict, key.c_str(), val) != 0)
    throw_error("setting dict entry '" + key + "'");
}

void PythonRuntime::dict_set_obj(PyObject *dict, const std::string &key,
                                 PyObject *value) {
  if (PyDict_SetItemString(dict, key.c_str(), value) != 0)
    throw_error("setting dict entry '" + key + "'");
}

} // namespace inference
} // namespace emulator
