/**
 * @file stub_inference_backend.cpp
 * @brief Implementation of the dependency-free stub backend.
 */

#include "stub_inference_backend.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>

namespace emulator {
namespace inference {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return s;
}

} // namespace

StubBackend::Mode StubBackend::mode_from_string(const std::string &text) {
  const std::string t = lower(text);
  if (t.empty() || t == "noop" || t == "none")
    return Mode::NOOP;
  if (t == "zero" || t == "zeros")
    return Mode::ZERO;
  if (t == "constant" || t == "const")
    return Mode::CONSTANT;
  if (t == "copy" || t == "identity" || t == "passthrough")
    return Mode::COPY;
  if (t == "affine" || t == "linear")
    return Mode::AFFINE;
  throw std::runtime_error("StubBackend: unknown stub.mode '" + text +
                           "' (expected noop, zero, constant, copy, affine)");
}

const char *StubBackend::mode_name(Mode mode) {
  switch (mode) {
  case Mode::NOOP:
    return "noop";
  case Mode::ZERO:
    return "zero";
  case Mode::CONSTANT:
    return "constant";
  case Mode::COPY:
    return "copy";
  case Mode::AFFINE:
    return "affine";
  }
  return "noop";
}

StubBackend::StubBackend(const InferenceConfig &config)
    : InferenceBackend(config) {
  m_mode = mode_from_string(config.option("stub.mode"));
  m_scale = config.option_double("stub.scale", 1.0);
  m_offset = config.option_double("stub.offset", 0.0);
  m_value = config.option_double("stub.value", 0.0);
}

bool StubBackend::infer(const TensorMap &inputs, TensorMap &outputs) {
  switch (m_mode) {
  case Mode::NOOP:
    // Deliberately does nothing: the caller's outputs come back exactly
    // as they went in.
    break;

  case Mode::ZERO:
    for (auto &out : outputs)
      out.zero();
    break;

  case Mode::CONSTANT:
    for (auto &out : outputs)
      out.fill(m_value);
    break;

  case Mode::COPY:
  case Mode::AFFINE: {
    if (inputs.size() < outputs.size()) {
      throw std::runtime_error(
          "StubBackend: mode '" + std::string(mode_name(m_mode)) + "' pairs " +
          "inputs with outputs positionally, but got " +
          std::to_string(inputs.size()) + " input(s) for " +
          std::to_string(outputs.size()) + " output(s)");
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const TensorView &in = inputs[i];
      TensorView &out = outputs[i];
      if (in.size() != out.size()) {
        throw std::runtime_error(
            "StubBackend: input '" + in.name() + "' has " +
            std::to_string(in.size()) + " elements but output '" +
            out.name() + "' has " + std::to_string(out.size()));
      }
      if (m_mode == Mode::COPY) {
        out.copy_from(in);
      } else {
        for (std::int64_t k = 0; k < out.size(); ++k)
          out.set_element(k, m_scale * in.element(k) + m_offset);
      }
    }
    break;
  }
  }

  if (m_config.verbose) {
    std::cout << "[StubBackend] mode=" << mode_name(m_mode) << " inputs="
              << inputs.size() << " outputs=" << outputs.size() << std::endl;
  }
  return true;
}

void StubBackend::finalize() { m_initialized = false; }

} // namespace inference
} // namespace emulator
