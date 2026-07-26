#!/usr/bin/env python3
"""Generate the small ONNX and TorchScript models the C++ tests run.

The models are deliberately trivial — an affine map and a two-output
split — because the tests are checking the *plumbing*: that named
tensors reach the right place, that dtypes convert, that shapes are
validated, that outputs land in the caller's buffers.  A real model
would only make the failures harder to read.

CMake runs this at configure time when the toolchain allows.  It is
also runnable by hand::

    python3 generate_test_models.py /path/to/output/dir

Missing dependencies are not an error: the script writes what it can and
reports the rest, and the C++ tests skip whatever is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path


def write_torchscript(out_dir: Path) -> list[str]:
    """Write TorchScript archives; returns the names written."""
    try:
        import torch
    except ImportError:
        print("generate_test_models: torch not available, skipping TorchScript")
        return []

    written = []

    class Affine(torch.nn.Module):
        """y = 2x + 1, elementwise."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return 2.0 * x + 1.0

    path = out_dir / "affine.pt"
    torch.jit.script(Affine()).save(str(path))
    written.append(path.name)

    class TwoOutputs(torch.nn.Module):
        """Returns a tuple, to exercise multi-output unpacking."""

        def forward(self, x: torch.Tensor):
            return x * 2.0, x + 100.0

    path = out_dir / "two_outputs.pt"
    torch.jit.script(TwoOutputs()).save(str(path))
    written.append(path.name)

    class TwoInputs(torch.nn.Module):
        """Takes two positional tensors, to exercise argument ordering."""

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a - b

    path = out_dir / "two_inputs.pt"
    torch.jit.script(TwoInputs()).save(str(path))
    written.append(path.name)

    print(f"generate_test_models: wrote TorchScript {written}")
    return written


def write_onnx(out_dir: Path) -> list[str]:
    """Write ONNX models; returns the names written."""
    try:
        import torch
    except ImportError:
        print("generate_test_models: torch not available, skipping ONNX export")
        return []

    written = []

    class Affine(torch.nn.Module):
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return 2.0 * state + 1.0

    path = out_dir / "affine.onnx"
    dummy = torch.zeros(1, 4, dtype=torch.float32)
    torch.onnx.export(
        Affine(),
        (dummy,),
        str(path),
        input_names=["state"],
        output_names=["tendency"],
        # A dynamic batch axis is what a real emulator needs: the number
        # of columns per rank is a decomposition detail, not a model one.
        dynamic_axes={"state": {0: "ncol"}, "tendency": {0: "ncol"}},
        opset_version=17,
        dynamo=False,
    )
    written.append(path.name)

    class TwoInTwoOut(torch.nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor):
            return a + b, a - b

    path = out_dir / "two_in_two_out.onnx"
    dummy_a = torch.zeros(2, 3, dtype=torch.float32)
    dummy_b = torch.zeros(2, 3, dtype=torch.float32)
    torch.onnx.export(
        TwoInTwoOut(),
        (dummy_a, dummy_b),
        str(path),
        input_names=["a", "b"],
        output_names=["sum", "diff"],
        dynamic_axes={
            "a": {0: "ncol"},
            "b": {0: "ncol"},
            "sum": {0: "ncol"},
            "diff": {0: "ncol"},
        },
        opset_version=17,
        dynamo=False,
    )
    written.append(path.name)

    print(f"generate_test_models: wrote ONNX {written}")
    return written


def main(argv: list[str]) -> int:
    out_dir = Path(argv[1] if len(argv) > 1 else "test_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = write_torchscript(out_dir) + write_onnx(out_dir)
    if not written:
        print(
            "generate_test_models: nothing written; install torch to generate "
            "the ONNX and TorchScript fixtures"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
