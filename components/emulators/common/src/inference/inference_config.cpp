/**
 * @file inference_config.cpp
 * @brief Implementation of InferenceConfig parsing and accessors.
 */

#include "inference_config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace emulator {
namespace inference {

namespace {

std::string trim(const std::string &s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos)
    return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return s;
}

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  std::string item;
  std::istringstream is(s);
  while (std::getline(is, item, sep))
    out.push_back(trim(item));
  return out;
}

} // namespace

const char *backend_type_name(BackendType type) {
  switch (type) {
  case BackendType::STUB:
    return "stub";
  case BackendType::PYTHON:
    return "python";
  case BackendType::ONNX:
    return "onnx";
  case BackendType::TORCH:
    return "torch";
  }
  return "stub";
}

// ─────────────────────────────────────────────────────────────────────
// Tensor spec parsing
// ─────────────────────────────────────────────────────────────────────

TensorSpec parse_tensor_spec(const std::string &text) {
  // Format: name:d0,d1,...[:dtype]
  const auto parts = split(text, ':');
  if (parts.empty() || parts[0].empty()) {
    throw std::runtime_error("inference: tensor spec '" + text +
                             "' has no name");
  }
  TensorSpec spec;
  spec.name = parts[0];

  if (parts.size() > 1 && !parts[1].empty()) {
    for (const auto &dim : split(parts[1], ',')) {
      if (dim.empty())
        continue;
      try {
        spec.shape.push_back(std::stoll(dim));
      } catch (const std::exception &) {
        throw std::runtime_error("inference: tensor spec '" + text +
                                 "' has a non-integer extent '" + dim + "'");
      }
    }
  }
  if (parts.size() > 2 && !parts[2].empty())
    spec.dtype = dtype_from_string(parts[2]);
  if (parts.size() > 3) {
    throw std::runtime_error("inference: tensor spec '" + text +
                             "' has too many ':'-separated fields "
                             "(expected name:shape[:dtype])");
  }
  return spec;
}

// ─────────────────────────────────────────────────────────────────────
// Option accessors
// ─────────────────────────────────────────────────────────────────────

bool InferenceConfig::has_option(const std::string &key) const {
  return options.find(key) != options.end();
}

std::string InferenceConfig::option(const std::string &key,
                                    const std::string &fallback) const {
  const auto it = options.find(key);
  return it == options.end() ? fallback : it->second;
}

int InferenceConfig::option_int(const std::string &key, int fallback) const {
  const auto it = options.find(key);
  if (it == options.end())
    return fallback;
  try {
    return std::stoi(it->second);
  } catch (const std::exception &) {
    throw std::runtime_error("inference: option '" + key + "' = '" +
                             it->second + "' is not an integer");
  }
}

double InferenceConfig::option_double(const std::string &key,
                                      double fallback) const {
  const auto it = options.find(key);
  if (it == options.end())
    return fallback;
  try {
    return std::stod(it->second);
  } catch (const std::exception &) {
    throw std::runtime_error("inference: option '" + key + "' = '" +
                             it->second + "' is not a number");
  }
}

bool InferenceConfig::option_bool(const std::string &key, bool fallback) const {
  const auto it = options.find(key);
  if (it == options.end())
    return fallback;
  const std::string v = lower(trim(it->second));
  if (v == "true" || v == "yes" || v == "on" || v == "1" || v == ".true.")
    return true;
  if (v == "false" || v == "no" || v == "off" || v == "0" || v == ".false.")
    return false;
  throw std::runtime_error("inference: option '" + key + "' = '" + it->second +
                           "' is not a boolean");
}

void InferenceConfig::set_option(const std::string &key,
                                 const std::string &value) {
  options[key] = value;
}

// ─────────────────────────────────────────────────────────────────────
// Derived queries
// ─────────────────────────────────────────────────────────────────────

bool InferenceConfig::uses_gpu() const {
  const std::string d = lower(device);
  return d.rfind("cuda", 0) == 0 || d.rfind("gpu", 0) == 0 ||
         d.rfind("hip", 0) == 0;
}

int InferenceConfig::device_index() const {
  if (!uses_gpu())
    return -1;
  const auto colon = device.find(':');
  if (colon == std::string::npos)
    return local_rank; // bare "cuda" → round-robin by local rank
  try {
    return std::stoi(device.substr(colon + 1));
  } catch (const std::exception &) {
    throw std::runtime_error("inference: device '" + device +
                             "' has a non-integer index");
  }
}

std::vector<TensorSpec> InferenceConfig::effective_inputs() const {
  if (!inputs.empty())
    return inputs;
  if (input_channels <= 0)
    return {};
  return {TensorSpec("input", {-1, input_channels}, DType::F64)};
}

std::vector<TensorSpec> InferenceConfig::effective_outputs() const {
  if (!outputs.empty())
    return outputs;
  if (output_channels <= 0)
    return {};
  return {TensorSpec("output", {-1, output_channels}, DType::F64)};
}

void InferenceConfig::validate() const {
  if (batch_size <= 0) {
    throw std::runtime_error("inference: batch_size must be positive, got " +
                             std::to_string(batch_size));
  }
  if (input_channels < 0 || output_channels < 0) {
    throw std::runtime_error("inference: channel counts must be non-negative");
  }
  auto check = [](const std::vector<TensorSpec> &specs, const char *what) {
    for (const auto &spec : specs) {
      if (spec.name.empty())
        throw std::runtime_error(std::string("inference: unnamed ") + what +
                                 " tensor spec");
      for (std::int64_t d : spec.shape) {
        if (d == 0) {
          throw std::runtime_error(std::string("inference: ") + what +
                                   " tensor '" + spec.name +
                                   "' has a zero extent " +
                                   shape_to_string(spec.shape));
        }
      }
    }
  };
  check(inputs, "input");
  check(outputs, "output");
  (void)device_index(); // throws on a malformed "cuda:xyz"
}

std::string InferenceConfig::to_string() const {
  std::ostringstream os;
  os << "InferenceConfig\n";
  os << "  backend        : " << backend << "\n";
  os << "  model_path     : " << (model_path.empty() ? "<none>" : model_path)
     << "\n";
  os << "  device         : " << device;
  if (uses_gpu())
    os << " (index " << device_index() << ")";
  os << "\n";
  os << "  batch_size     : " << batch_size << "\n";
  os << "  mpi_comm (F)   : " << mpi_comm_fortran << "\n";
  os << "  rank           : " << global_rank << " (local " << local_rank
     << ")\n";

  auto dump = [&os](const std::string &label,
                    const std::vector<TensorSpec> &specs) {
    os << "  " << label << std::string(15 - std::min<std::size_t>(15, label.size()), ' ')
       << ": ";
    if (specs.empty()) {
      os << "<introspected from model>\n";
      return;
    }
    os << "\n";
    for (const auto &s : specs) {
      os << "      " << s.name << " " << shape_to_string(s.shape) << " "
         << dtype_name(s.dtype) << "\n";
    }
  };
  dump("inputs", effective_inputs());
  dump("outputs", effective_outputs());

  if (!options.empty()) {
    os << "  options        :\n";
    for (const auto &kv : options)
      os << "      " << kv.first << " = " << kv.second << "\n";
  }
  return os.str();
}

// ─────────────────────────────────────────────────────────────────────
// Parsing
// ─────────────────────────────────────────────────────────────────────

InferenceConfig InferenceConfig::from_string(const std::string &text) {
  InferenceConfig cfg;
  std::istringstream is(text);
  std::string line;
  int lineno = 0;

  while (std::getline(is, line)) {
    ++lineno;
    // Strip comments, but only when '#' starts the trimmed line or follows
    // whitespace, so paths and values containing '#' survive.
    const std::string raw = trim(line);
    if (raw.empty() || raw[0] == '#')
      continue;

    const auto colon = raw.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error("inference config line " +
                               std::to_string(lineno) + ": expected 'key: value', got '" +
                               raw + "'");
    }
    const std::string key = lower(trim(raw.substr(0, colon)));
    const std::string value = trim(raw.substr(colon + 1));

    if (key == "backend") {
      cfg.backend = lower(value);
    } else if (key == "model_path" || key == "model") {
      cfg.model_path = value;
    } else if (key == "device") {
      cfg.device = value;
    } else if (key == "batch_size") {
      cfg.batch_size = std::stoi(value);
    } else if (key == "verbose") {
      cfg.set_option("__verbose", value);
      cfg.verbose = cfg.option_bool("__verbose");
      cfg.options.erase("__verbose");
    } else if (key == "input_channels") {
      cfg.input_channels = std::stoi(value);
    } else if (key == "output_channels") {
      cfg.output_channels = std::stoi(value);
    } else if (key == "input") {
      cfg.inputs.push_back(parse_tensor_spec(value));
    } else if (key == "output") {
      cfg.outputs.push_back(parse_tensor_spec(value));
    } else {
      // Unknown keys are backend options, not errors — this is what keeps
      // the config format open to backends this file has never heard of.
      cfg.options[trim(raw.substr(0, colon))] = value;
    }
  }
  return cfg;
}

InferenceConfig InferenceConfig::from_file(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("inference: cannot open config file '" + path +
                             "'");
  }
  std::ostringstream buf;
  buf << ifs.rdbuf();
  return from_string(buf.str());
}

} // namespace inference
} // namespace emulator
