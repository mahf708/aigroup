"""Reference models exercising every shape of the Python contract.

These are what the C++ tests run against, and they double as the worked
examples: between them they cover in-place writes, returned dicts, spec
introspection, MPI awareness, and a torch model loaded from a
checkpoint.  Each is small enough to read in one sitting.

Select one from a namelist with::

    backend: python
    python.module: e3sm_emulator_inference.reference_models
    python.factory: create_affine
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from . import EmulatorModel, ModelConfig, get_mpi_comm, resolve_device, spec


class IdentityModel(EmulatorModel):
    """Copies each input to the same-named output.

    The simplest thing that proves data actually made the round trip
    from the coupler through Python and back.
    """

    def forward(self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]):
        for name, target in outputs.items():
            source = inputs.get(name)
            if source is None:
                continue
            target[...] = source.reshape(target.shape)
        self.step_count += 1


class AffineModel(EmulatorModel):
    """``y = scale * x + offset``, written in place.

    Demonstrates reading tunables out of the namelist: ``scale`` and
    ``offset`` arrive as strings in ``options`` and
    :class:`ModelConfig` coerces them.
    """

    def __init__(self, config):
        super().__init__(config)
        self.scale = self.config.float("scale", 2.0)
        self.offset = self.config.float("offset", 1.0)

    def forward(self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]):
        for (_, source), (_, target) in zip(inputs.items(), outputs.items()):
            target[...] = (self.scale * source + self.offset).reshape(target.shape)
        self.step_count += 1


class ReturningModel(EmulatorModel):
    """Returns a dict instead of writing in place.

    The other half of the output contract: a model that finds it more
    natural to allocate can do so, and the backend copies the result
    into the emulator's buffers, converting dtype if it differs.
    """

    def forward(self, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]):
        self.step_count += 1
        return {name: np.asarray(source) * 3.0 for name, source in inputs.items()}


class SpecModel(EmulatorModel):
    """Declares its interface so the emulator can size buffers from it.

    A model with ``input_specs``/``output_specs`` lets an emulator call
    ``backend->output_specs()`` after ``initialize()`` and allocate
    exactly what the model produces, instead of restating the shapes in
    a namelist and hoping the two agree.
    """

    INPUTS = [spec("state", [-1, 6], "f64")]
    OUTPUTS = [spec("tendency", [-1, 3], "f64")]

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        state = inputs["state"]
        # Contract three input channels into one output channel each, so
        # a wrong-shaped buffer on either side shows up immediately.
        return {"tendency": state[:, :3] + state[:, 3:]}


class ColumnPhysicsModel(EmulatorModel):
    """A slightly less toy model: per-column vertical mixing.

    Stands in for a process-level emulator — the kind of thing that
    would replace a single parameterization rather than a whole
    component.  Input is ``[ncol, nlev]``, output is a tendency of the
    same shape, and the operator is a three-point smoother, which is
    enough to notice if columns or levels get transposed somewhere in
    the packing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.strength = self.config.float("mixing_strength", 0.25)

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        field = np.asarray(inputs["state"], dtype=np.float64)
        smoothed = field.copy()
        if field.shape[-1] >= 3:
            smoothed[..., 1:-1] = (
                field[..., :-2] + 2.0 * field[..., 1:-1] + field[..., 2:]
            ) / 4.0
        return {"tendency": self.strength * (smoothed - field)}


class MpiAwareModel(EmulatorModel):
    """Reports the rank layout it sees, and reduces across ranks.

    Exists to prove that a Python model really can talk to the rest of
    the component: it rebuilds the communicator from the Fortran handle
    the backend passed and runs an allreduce over it.  Falls back to
    rank-local behaviour when the run is serial or ``mpi4py`` is absent,
    so the test suite does not need MPI.
    """

    def __init__(self, config):
        super().__init__(config)
        self.comm = get_mpi_comm(self.config)
        self.rank = self.comm.Get_rank() if self.comm is not None else 0
        self.size = self.comm.Get_size() if self.comm is not None else 1

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        local = np.asarray(inputs["state"], dtype=np.float64)
        total = local.sum()
        if self.comm is not None:
            total = self.comm.allreduce(total)
        # Broadcast the global mean back over the local block, which is
        # only correct if the communicator really did span the ranks.
        return {"state": np.full_like(local, total / max(1, self.size * local.size))}


class TorchCheckpointModel(EmulatorModel):
    """Loads a PyTorch model from ``model_path`` and runs it.

    This is the pattern a real emulator follows, ACE2 included: the
    weights stay in PyTorch, the harness stays in Python, and the only
    thing C++ does is hand over buffers and ask for the next step.
    ``torch.from_numpy`` on the input is itself a view, so on CPU the
    whole path from MCT attribute vector to model input is copy-free.

    ``model_path`` may be a TorchScript archive or anything else
    ``torch.load`` accepts that yields a callable module.
    """

    def __init__(self, config):
        super().__init__(config)
        import torch

        self.torch = torch
        self.device = resolve_device(self.config)
        path = self.config.model_path
        if not path:
            raise ValueError(
                "TorchCheckpointModel needs model_path to point at a "
                "TorchScript archive or a torch.load-able module"
            )
        try:
            self.module = torch.jit.load(path, map_location=self.device)
        except Exception:
            self.module = torch.load(path, map_location=self.device, weights_only=False)
        self.module.eval()
        self.input_name = self.config.str("input_name", "state")
        self.output_name = self.config.str("output_name", "tendency")

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        torch = self.torch
        source = np.asarray(inputs[self.input_name])
        with torch.no_grad():
            # from_numpy is a view; .float() and .to() copy only when the
            # dtype or device actually differ.
            tensor = torch.from_numpy(source).float()
            if self.device != "cpu":
                tensor = tensor.to(self.device)
            result = self.module(tensor)
        return {self.output_name: result.detach().cpu().numpy()}


# ── Factories ────────────────────────────────────────────────────────
#
# The backend calls one of these by name (`python.factory`).  Keeping
# them as free functions rather than using the classes directly means a
# factory can do setup that does not belong in __init__ — reading a
# normalization table, sharding a checkpoint across ranks — without the
# C++ side needing to know.


def create(config: Dict[str, Any]) -> EmulatorModel:
    """Default factory: the identity model."""
    return IdentityModel(config)


def create_identity(config: Dict[str, Any]) -> EmulatorModel:
    """Build an :class:`IdentityModel`."""
    return IdentityModel(config)


def create_affine(config: Dict[str, Any]) -> EmulatorModel:
    """Build an :class:`AffineModel`."""
    return AffineModel(config)


def create_returning(config: Dict[str, Any]) -> EmulatorModel:
    """Build a :class:`ReturningModel`."""
    return ReturningModel(config)


def create_spec_model(config: Dict[str, Any]) -> EmulatorModel:
    """Build a :class:`SpecModel`."""
    return SpecModel(config)


def create_column_physics(config: Dict[str, Any]) -> EmulatorModel:
    """Build a :class:`ColumnPhysicsModel`."""
    return ColumnPhysicsModel(config)


def create_mpi_aware(config: Dict[str, Any]) -> EmulatorModel:
    """Build an :class:`MpiAwareModel`."""
    return MpiAwareModel(config)


def create_torch_checkpoint(config: Dict[str, Any]) -> EmulatorModel:
    """Build a :class:`TorchCheckpointModel`."""
    return TorchCheckpointModel(config)


def create_callable(config: Dict[str, Any]):
    """Return a bare callable rather than an object with ``forward``.

    The backend accepts this too: if the factory's result has no
    ``forward`` method but is itself callable, it is called directly.
    Useful for a one-line model that does not deserve a class.
    """
    scale = ModelConfig(config).float("scale", 10.0)

    def step(inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]) -> None:
        for (_, source), (_, target) in zip(inputs.items(), outputs.items()):
            target[...] = (scale * source).reshape(target.shape)

    return step
