/**
 * @file inference_demo.cpp
 * @brief Minimal driver showing what an emulator's run loop looks like.
 *
 * Stands in for the coupler: allocates buffers, builds a backend from a
 * config file, and steps it a few times, printing what comes back.  It
 * exists so that a new backend or a new model can be exercised without
 * standing up a coupled case — the "desktop-friendly" testing path.
 *
 * ```console
 * ./inference_demo                      # stub backend, built-in config
 * ./inference_demo inference_in         # whatever that file describes
 * ./inference_demo inference_in 8 4     # ... with 8 columns, 4 steps
 * ```
 */

#include "create_inference_backend.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace emulator::inference;

namespace {

/** @brief Print a buffer as `ncol` rows of `nchan` values. */
void dump(const std::string &label, const std::vector<double> &data, int ncol,
          int nchan) {
  std::cout << "  " << label << ":\n";
  for (int c = 0; c < ncol; ++c) {
    std::cout << "    col " << std::setw(3) << c << " |";
    for (int k = 0; k < nchan; ++k) {
      std::cout << " " << std::setw(9) << std::fixed << std::setprecision(3)
                << data[static_cast<std::size_t>(c) * nchan + k];
    }
    std::cout << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  const std::string config_path = argc > 1 ? argv[1] : "";
  const int ncol = argc > 2 ? std::atoi(argv[2]) : 4;
  const int nsteps = argc > 3 ? std::atoi(argv[3]) : 3;

  try {
    InferenceConfig config;
    if (config_path.empty()) {
      // A self-contained default so the demo does something useful with
      // no arguments at all.
      config.backend = "stub";
      config.input_channels = 3;
      config.output_channels = 3;
      config.set_option("stub.mode", "affine");
      config.set_option("stub.scale", "2.0");
      config.set_option("stub.offset", "1.0");
      config.verbose = true;
    } else {
      config = InferenceConfig::from_file(config_path);
    }
    config.batch_size = ncol;
    config.validate();

    std::cout << config.to_string() << "\n";

    register_builtin_backends();
    std::cout << "Backends available in this build:\n"
              << BackendRegistry::instance().summary() << "\n";

    auto backend = create_backend(config);
    backend->initialize();
    std::cout << "Backend: " << backend->name() << "\n\n";

    // An emulator allocates once and reuses the buffers every step;
    // nothing below allocates inside the loop.
    const auto in_specs = backend->input_specs();
    const auto out_specs = backend->output_specs();
    if (in_specs.empty() || out_specs.empty()) {
      std::cerr << "This config declares no tensors; set input_channels and "
                   "output_channels, or input:/output: specs.\n";
      return 1;
    }

    const int nin = static_cast<int>(
        in_specs[0].shape.size() > 1 ? in_specs[0].shape[1] : 1);
    const int nout = static_cast<int>(
        out_specs[0].shape.size() > 1 ? out_specs[0].shape[1] : 1);

    std::vector<double> inputs(static_cast<std::size_t>(ncol) * nin);
    std::vector<double> outputs(static_cast<std::size_t>(ncol) * nout, 0.0);
    for (std::size_t i = 0; i < inputs.size(); ++i)
      inputs[i] = static_cast<double>(i) * 0.5;

    TensorMap in_map;
    in_map.set(
        TensorView::from_doubles(in_specs[0].name, inputs.data(), {ncol, nin}));
    TensorMap out_map;
    out_map.set(TensorView::from_doubles(out_specs[0].name, outputs.data(),
                                         {ncol, nout}));

    for (int step = 0; step < nsteps; ++step) {
      std::cout << "step " << step << "\n";
      dump("in ", inputs, ncol, nin);
      backend->infer(in_map, out_map);
      dump("out", outputs, ncol, nout);
      std::cout << "\n";

      // Feed the output back in, the way an autoregressive emulator
      // would, when the widths line up.
      if (nin == nout)
        inputs = outputs;
    }

    backend->finalize();
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "inference_demo failed: " << e.what() << "\n";
    return 1;
  }
}
