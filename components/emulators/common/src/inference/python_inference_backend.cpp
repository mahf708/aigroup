/**
 * @file python_inference_backend.cpp
 * @brief Implementation of the embedded-Python inference backend.
 */

#include "python_inference_backend.hpp"
#include "python_runtime.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace emulator {
namespace inference {

/**
 * @brief Python-side state, hidden from the header.
 *
 * Holding the runtime handle here is what keeps the interpreter alive
 * for exactly as long as some backend needs it.
 */
struct PythonBackend::Impl {
  std::shared_ptr<PythonRuntime> runtime;
  PyRef module; ///< The imported model module
  PyRef model;  ///< The object returned by the factory
  std::vector<TensorSpec> introspected_inputs;
  std::vector<TensorSpec> introspected_outputs;
  std::string version;
};

namespace {

/** @brief Split "a:b:c" into its parts, skipping empties. */
std::vector<std::string> split_paths(const std::string &text) {
  std::vector<std::string> out;
  std::string item;
  std::istringstream is(text);
  while (std::getline(is, item, ':')) {
    if (!item.empty())
      out.push_back(item);
  }
  return out;
}

/**
 * @brief Read a list-of-dicts spec description from the Python model.
 *
 * Expects `[{"name": str, "shape": [int, ...], "dtype": str}, ...]`.
 * A model that does not implement the method gets an empty result, not
 * an error — spec introspection is optional.
 */
std::vector<TensorSpec> read_specs(PyObject *model, const char *method) {
  std::vector<TensorSpec> specs;
  if (!model || PyObject_HasAttrString(model, method) == 0)
    return specs;

  PyRef result = PyRef::steal(PyObject_CallMethod(model, method, nullptr));
  if (!result) {
    PyErr_Clear(); // Optional hook; a failure here is not fatal.
    return specs;
  }
  PyRef seq = PyRef::steal(PySequence_Fast(result.get(), "expected a sequence"));
  if (!seq) {
    PyErr_Clear();
    return specs;
  }

  const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject *entry = PySequence_Fast_GET_ITEM(seq.get(), i); // borrowed
    if (!PyDict_Check(entry))
      continue;

    TensorSpec spec;
    if (PyObject *name = PyDict_GetItemString(entry, "name")) {
      if (const char *utf8 = PyUnicode_AsUTF8(name))
        spec.name = utf8;
    }
    if (spec.name.empty())
      continue;

    if (PyObject *shape = PyDict_GetItemString(entry, "shape")) {
      PyRef dims =
          PyRef::steal(PySequence_Fast(shape, "expected a shape sequence"));
      if (dims) {
        const Py_ssize_t nd = PySequence_Fast_GET_SIZE(dims.get());
        for (Py_ssize_t d = 0; d < nd; ++d) {
          PyObject *dim = PySequence_Fast_GET_ITEM(dims.get(), d);
          spec.shape.push_back(PyLong_AsLongLong(dim));
        }
      } else {
        PyErr_Clear();
      }
    }
    if (PyObject *dtype = PyDict_GetItemString(entry, "dtype")) {
      if (const char *utf8 = PyUnicode_AsUTF8(dtype)) {
        try {
          spec.dtype = dtype_from_string(utf8);
        } catch (const std::exception &) {
          // Leave the default rather than rejecting the whole spec list.
        }
      }
    }
    specs.push_back(std::move(spec));
  }
  PyErr_Clear();
  return specs;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────

PythonBackend::PythonBackend(const InferenceConfig &config)
    : InferenceBackend(config), m_impl(new Impl()) {}

PythonBackend::~PythonBackend() {
  // finalize() is idempotent, and doing it here means a backend dropped
  // without an explicit finalize still releases its Python references
  // while the interpreter is guaranteed to be alive.
  try {
    finalize();
  } catch (...) {
    // Nothing useful to do during destruction.
  }
}

std::string PythonBackend::python_version() const { return m_impl->version; }

void PythonBackend::initialize() {
  if (m_initialized)
    return;

  const std::string module_spec =
      m_config.option("python.module", m_config.model_path);
  if (module_spec.empty()) {
    throw std::runtime_error(
        "PythonBackend: no model to load — set 'model_path' to a .py file "
        "or set the 'python.module' option to an importable module name");
  }
  const std::string factory_name = m_config.option("python.factory", "create");
  const std::string method_name = m_config.option("python.method", "forward");

  m_impl->runtime = PythonRuntime::acquire();
  m_impl->version = m_impl->runtime->version();

  PyGil gil;

  for (const auto &path : split_paths(m_config.option("python.sys_path")))
    m_impl->runtime->add_sys_path(path);

  m_impl->module = m_impl->runtime->import_module(module_spec);

  // Build the config dict handed to the factory.  Options go in first so
  // that a typed field always wins over a same-named option.
  PyRef cfg = m_impl->runtime->make_dict(m_config.options);
  PyObject *d = cfg.get();
  m_impl->runtime->dict_set(d, "backend", m_config.backend);
  m_impl->runtime->dict_set(d, "model_path", m_config.model_path);
  m_impl->runtime->dict_set(d, "device", m_config.device);
  m_impl->runtime->dict_set_int(d, "batch_size", m_config.batch_size);
  m_impl->runtime->dict_set_bool(d, "verbose", m_config.verbose);
  m_impl->runtime->dict_set_int(d, "mpi_comm_fortran", m_config.mpi_comm_fortran);
  m_impl->runtime->dict_set_int(d, "local_rank", m_config.local_rank);
  m_impl->runtime->dict_set_int(d, "global_rank", m_config.global_rank);
  m_impl->runtime->dict_set_int(d, "device_index", m_config.device_index());

  // Declared specs, as list-of-dicts mirroring what input_specs() returns.
  auto specs_to_py = [&](const std::vector<TensorSpec> &specs) {
    PyRef list = PyRef::steal(PyList_New(0));
    if (!list)
      PythonRuntime::throw_error("allocating a spec list");
    for (const auto &spec : specs) {
      PyRef entry = PyRef::steal(PyDict_New());
      if (!entry)
        PythonRuntime::throw_error("allocating a spec dict");
      m_impl->runtime->dict_set(entry.get(), "name", spec.name);
      m_impl->runtime->dict_set(entry.get(), "dtype", dtype_name(spec.dtype));
      PyRef shape = PyRef::steal(PyList_New(0));
      for (std::int64_t dim : spec.shape) {
        PyRef v = PyRef::steal(PyLong_FromLongLong(dim));
        PyList_Append(shape.get(), v.get());
      }
      m_impl->runtime->dict_set_obj(entry.get(), "shape", shape.get());
      PyList_Append(list.get(), entry.get());
    }
    return list;
  };
  PyRef in_specs = specs_to_py(m_config.effective_inputs());
  PyRef out_specs = specs_to_py(m_config.effective_outputs());
  m_impl->runtime->dict_set_obj(d, "inputs", in_specs.get());
  m_impl->runtime->dict_set_obj(d, "outputs", out_specs.get());

  PyRef factory =
      PyRef::steal(PyObject_GetAttrString(m_impl->module.get(), factory_name.c_str()));
  if (!factory) {
    PythonRuntime::throw_error("looking up factory '" + factory_name +
                               "' in module '" + module_spec + "'");
  }

  m_impl->model =
      PyRef::steal(PyObject_CallFunctionObjArgs(factory.get(), cfg.get(), nullptr));
  if (!m_impl->model) {
    PythonRuntime::throw_error("calling " + module_spec + "." + factory_name +
                               "(config)");
  }

  // The step method must exist now rather than failing mid-simulation.
  if (PyObject_HasAttrString(m_impl->model.get(), method_name.c_str()) == 0 &&
      PyCallable_Check(m_impl->model.get()) == 0) {
    throw std::runtime_error("PythonBackend: the object returned by " +
                             module_spec + "." + factory_name +
                             " has no '" + method_name +
                             "' method and is not itself callable");
  }

  m_impl->introspected_inputs = read_specs(m_impl->model.get(), "input_specs");
  m_impl->introspected_outputs = read_specs(m_impl->model.get(), "output_specs");

  if (m_config.verbose) {
    std::cout << "[PythonBackend] python " << m_impl->version << ", module '"
              << module_spec << "', factory '" << factory_name << "', method '"
              << method_name << "', device '" << m_config.device << "'"
              << std::endl;
  }
  m_initialized = true;
}

std::vector<TensorSpec> PythonBackend::input_specs() const {
  if (!m_impl->introspected_inputs.empty())
    return m_impl->introspected_inputs;
  return InferenceBackend::input_specs();
}

std::vector<TensorSpec> PythonBackend::output_specs() const {
  if (!m_impl->introspected_outputs.empty())
    return m_impl->introspected_outputs;
  return InferenceBackend::output_specs();
}

bool PythonBackend::infer(const TensorMap &inputs, TensorMap &outputs) {
  if (!m_initialized) {
    throw std::runtime_error(
        "PythonBackend::infer called before initialize()");
  }
  const std::string method_name = m_config.option("python.method", "forward");

  PyGil gil;

  // Wrap both sides as NumPy views over the emulator's own memory.
  // Nothing is copied here; the arrays alias the caller's buffers.
  PyRef in_dict = PyRef::steal(PyDict_New());
  PyRef out_dict = PyRef::steal(PyDict_New());
  if (!in_dict || !out_dict)
    PythonRuntime::throw_error("allocating the inference dicts");

  for (const auto &view : inputs) {
    PyRef array = m_impl->runtime->wrap_tensor(view, /*writable=*/false);
    m_impl->runtime->dict_set_obj(in_dict.get(), view.name(), array.get());
  }
  for (auto &view : outputs) {
    if (!view.writable()) {
      throw std::runtime_error("PythonBackend: output tensor '" + view.name() +
                               "' is read-only");
    }
    PyRef array = m_impl->runtime->wrap_tensor(view, /*writable=*/true);
    m_impl->runtime->dict_set_obj(out_dict.get(), view.name(), array.get());
  }

  PyRef result;
  if (PyObject_HasAttrString(m_impl->model.get(), method_name.c_str()) != 0) {
    PyRef method = PyRef::steal(
        PyObject_GetAttrString(m_impl->model.get(), method_name.c_str()));
    if (!method)
      PythonRuntime::throw_error("looking up '" + method_name + "'");
    result = PyRef::steal(PyObject_CallFunctionObjArgs(
        method.get(), in_dict.get(), out_dict.get(), nullptr));
  } else {
    // The factory returned a bare callable rather than an object with a
    // step method; call it directly.
    result = PyRef::steal(PyObject_CallFunctionObjArgs(
        m_impl->model.get(), in_dict.get(), out_dict.get(), nullptr));
  }
  if (!result)
    PythonRuntime::throw_error("calling the model's '" + method_name + "'");

  // A model that returned a dict of arrays wants us to copy them into
  // the caller's buffers; one that wrote in place returns None.
  if (result.get() != Py_None && PyDict_Check(result.get())) {
    for (auto &view : outputs) {
      PyObject *src = PyDict_GetItemString(result.get(), view.name().c_str());
      if (!src)
        continue; // Output the model chose not to produce; leave as-is.
      PyRef dst = m_impl->runtime->wrap_tensor(view, /*writable=*/true);
      PyRef numpy = PyRef::steal(PyImport_ImportModule("numpy"));
      if (!numpy)
        PythonRuntime::throw_error("importing numpy to copy model output");
      PyRef copied = PyRef::steal(PyObject_CallMethod(
          numpy.get(), "copyto", "OOs", dst.get(), src, "unsafe"));
      if (!copied) {
        PythonRuntime::throw_error("copying returned output '" + view.name() +
                                   "' into the emulator buffer");
      }
    }
  }
  return true;
}

void PythonBackend::finalize() {
  if (!m_impl->runtime) {
    m_initialized = false;
    return;
  }
  {
    PyGil gil;
    if (m_impl->model && PyObject_HasAttrString(m_impl->model.get(), "finalize")) {
      PyRef result =
          PyRef::steal(PyObject_CallMethod(m_impl->model.get(), "finalize", nullptr));
      if (!result) {
        // Report but do not throw: teardown should not mask the real
        // reason a run is ending.
        std::cerr << PythonRuntime::fetch_error("calling model.finalize()")
                  << std::endl;
      }
    }
    m_impl->model.reset();
    m_impl->module.reset();
  }
  // Dropping the last handle shuts the interpreter down.
  m_impl->runtime.reset();
  m_initialized = false;
}

} // namespace inference
} // namespace emulator
