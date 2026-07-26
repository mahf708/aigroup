"""Python side of the E3SM emulator inference bridge.

The C++ ``PythonBackend`` imports a module, calls a factory in it, and
then calls a method on whatever that factory returned, once per coupled
timestep.  Nothing in this package is *required* — a model can satisfy
the contract with a plain function and a class of its own — but using
these helpers means the boilerplate (spec declaration, dtype bookkeeping,
device selection, MPI communicator recovery) is written once.

The contract, in full::

    def create(config: dict) -> object: ...

    class Model:
        def forward(self, inputs: dict, outputs: dict) -> None | dict: ...
        def input_specs(self)  -> list[dict]: ...   # optional
        def output_specs(self) -> list[dict]: ...   # optional
        def finalize(self) -> None: ...             # optional

``inputs`` and ``outputs`` are dicts of NumPy arrays that *alias the
emulator's own memory*.  Writing ``outputs["y"][:] = ...`` writes
directly into the buffer the coupler will read; no copy happens in
either direction.  Rebinding the name (``outputs["y"] = ...``) does not
— it just replaces the dict entry and the emulator never sees it.  If
allocating is more natural, return a dict instead and the backend copies
it across for you.

Everything the C++ side knows arrives in ``config``: ``model_path``,
``device``, ``device_index``, ``batch_size``, ``mpi_comm_fortran``,
``local_rank``, ``global_rank``, the declared ``inputs``/``outputs``
specs, and every ``InferenceConfig`` option verbatim.  Option values are
strings, because they came from a namelist; :class:`ModelConfig` does the
coercion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "ModelConfig",
    "EmulatorModel",
    "spec",
    "get_mpi_comm",
    "resolve_device",
]


def spec(name: str, shape: Sequence[int], dtype: str = "f32") -> Dict[str, Any]:
    """Build one entry of an ``input_specs``/``output_specs`` list.

    Args:
        name: Tensor name the emulator will key its dict by.
        shape: Row-major extents; ``-1`` marks a dynamic (batch) axis.
        dtype: One of ``f32``/``f64``/``i32``/``i64``, or a NumPy-style
            alias such as ``float32``.

    Returns:
        A dict in the shape the C++ side parses.
    """
    return {"name": name, "shape": [int(d) for d in shape], "dtype": dtype}


class ModelConfig:
    """Typed accessors over the raw ``config`` dict from C++.

    Every value the backend puts in ``options`` arrives as a string, so
    ``config["n_layers"]`` is ``"4"`` and not ``4``.  This wrapper does
    the coercion in one place and leaves the underlying dict reachable
    as :attr:`raw` for anything it does not cover.
    """

    def __init__(self, config: Dict[str, Any]):
        self.raw = dict(config)

    # ── typed access ────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Raw value for ``key``, or ``default``."""
        return self.raw.get(key, default)

    def str(self, key: str, default: str = "") -> str:
        """Value for ``key`` as a string."""
        value = self.raw.get(key, default)
        return default if value is None else str(value)

    def int(self, key: str, default: int = 0) -> int:
        """Value for ``key`` as an int, tolerating a string source."""
        value = self.raw.get(key)
        return default if value is None or value == "" else int(value)

    def float(self, key: str, default: float = 0.0) -> float:
        """Value for ``key`` as a float, tolerating a string source."""
        value = self.raw.get(key)
        return default if value is None or value == "" else float(value)

    def bool(self, key: str, default: bool = False) -> bool:
        """Value for ``key`` as a bool.

        Accepts real bools and the string spellings a Fortran namelist
        might carry: ``true``/``yes``/``on``/``1``/``.true.``.
        """
        value = self.raw.get(key)
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "yes", "on", "1", ".true.")

    # ── convenience properties ──────────────────────────────────────

    @property
    def model_path(self) -> str:
        """Path the namelist gave for the model artifact."""
        return self.str("model_path")

    @property
    def device(self) -> str:
        """Requested device string, e.g. ``cpu`` or ``cuda:2``."""
        return self.str("device", "cpu")

    @property
    def device_index(self) -> int:
        """GPU index the C++ side resolved, or ``-1`` on CPU.

        Already accounts for ``local_rank``, so a bare ``cuda`` in the
        namelist has become a distinct GPU per rank by the time it gets
        here.
        """
        return self.int("device_index", -1)

    @property
    def batch_size(self) -> int:
        """Samples per ``forward`` call."""
        return self.int("batch_size", 1)

    @property
    def verbose(self) -> bool:
        """Whether the emulator asked for chatty output."""
        return self.bool("verbose", False)

    @property
    def local_rank(self) -> int:
        """This rank's index within its node."""
        return self.int("local_rank", 0)

    @property
    def global_rank(self) -> int:
        """This rank's index in the component communicator."""
        return self.int("global_rank", 0)

    @property
    def mpi_comm_fortran(self) -> int:
        """Fortran MPI communicator handle, or ``-1`` if serial.

        Pass to :func:`get_mpi_comm` to recover an ``mpi4py``
        communicator over the *same* communicator the component is using.
        """
        return self.int("mpi_comm_fortran", -1)

    @property
    def inputs(self) -> List[Dict[str, Any]]:
        """Input specs the namelist declared (possibly empty)."""
        return list(self.raw.get("inputs") or [])

    @property
    def outputs(self) -> List[Dict[str, Any]]:
        """Output specs the namelist declared (possibly empty)."""
        return list(self.raw.get("outputs") or [])

    def __repr__(self) -> str:
        return f"ModelConfig({self.raw!r})"


def get_mpi_comm(config: Dict[str, Any] | ModelConfig):
    """Recover the component's MPI communicator, or ``None``.

    The emulator passes the *Fortran* handle for the communicator its
    component was given, so a model that reconstructs it here is
    collectives-compatible with the rest of the component — which is what
    a spatially decomposed emulator needs in order to exchange halos.

    Args:
        config: The dict from the backend, or a :class:`ModelConfig`.

    Returns:
        An ``mpi4py.MPI.Comm``, or ``None`` when the run is serial or
        ``mpi4py`` is not installed.
    """
    cfg = config if isinstance(config, ModelConfig) else ModelConfig(config)
    handle = cfg.mpi_comm_fortran
    if handle < 0:
        return None
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.Comm.f2py(handle)


def resolve_device(config: Dict[str, Any] | ModelConfig) -> str:
    """Return the torch-style device string this rank should use.

    Falls back to ``"cpu"`` when CUDA was requested but is unavailable,
    rather than failing — a model asked to run on a GPU-less test machine
    should still run.  The C++ ONNX and LibTorch backends deliberately do
    the opposite and raise, because a silent CPU fallback inside a
    production run reads as an unexplained slowdown; here the Python
    model is in charge and can log whatever it likes.
    """
    cfg = config if isinstance(config, ModelConfig) else ModelConfig(config)
    index = cfg.device_index
    if index < 0:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{index % torch.cuda.device_count()}"


class EmulatorModel:
    """Base class handling the mechanical half of the contract.

    Subclasses override :meth:`predict` and, usually, declare their specs
    by setting :attr:`INPUTS` and :attr:`OUTPUTS`.  This class then takes
    care of matching names up, writing results into the emulator's
    buffers in place, and reporting the specs back to C++.

    A minimal model::

        class Doubler(EmulatorModel):
            INPUTS = [spec("x", [-1, 4], "f64")]
            OUTPUTS = [spec("y", [-1, 4], "f64")]

            def predict(self, inputs):
                return {"y": 2.0 * inputs["x"]}

        def create(config):
            return Doubler(config)
    """

    #: Input specs, as returned by :func:`spec`.  Empty means "whatever
    #: the namelist declared".
    INPUTS: List[Dict[str, Any]] = []

    #: Output specs.  @see INPUTS
    OUTPUTS: List[Dict[str, Any]] = []

    def __init__(self, config: Dict[str, Any] | ModelConfig | None = None):
        if config is None:
            config = {}
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(config)
        self.step_count = 0

    # ── to override ─────────────────────────────────────────────────

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Map input arrays to output arrays.

        Args:
            inputs: NumPy views over the emulator's input buffers.  Do
                not mutate these; they are the coupler's data.

        Returns:
            A dict keyed by output name.  Values may be any shape whose
            element count matches the destination, and any dtype — the
            copy into the emulator's buffer converts as needed.
        """
        raise NotImplementedError("EmulatorModel subclasses must implement predict()")

    # ── contract implementation ─────────────────────────────────────

    def forward(
        self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]
    ) -> None:
        """Called once per timestep by the C++ backend.

        Writes :meth:`predict`'s results into ``outputs`` in place, so
        nothing is copied on the way back to the coupler beyond the
        assignment itself.
        """
        produced = self.predict(inputs)
        if produced is None:
            # predict() wrote into the buffers itself.
            self.step_count += 1
            return

        for name, array in produced.items():
            target = outputs.get(name)
            if target is None:
                # A model may legitimately produce diagnostics the
                # emulator did not ask for this step; ignore them rather
                # than failing the run.
                continue
            source = np.asarray(array)
            if source.size != target.size:
                raise ValueError(
                    f"model produced {source.size} elements for output "
                    f"'{name}' but the emulator allocated {target.size}"
                )
            target[...] = source.reshape(target.shape)

        self.step_count += 1

    def input_specs(self) -> List[Dict[str, Any]]:
        """Declared input specs, for the C++ side to introspect."""
        return list(self.INPUTS)

    def output_specs(self) -> List[Dict[str, Any]]:
        """Declared output specs, for the C++ side to introspect."""
        return list(self.OUTPUTS)

    def finalize(self) -> None:
        """Release anything held.  Default does nothing."""
        return None
