import numpy as np
import pytest
import torch

from atlas.model import Atlas
from atlas.spec import (
    AtlasConfig,
    BackboneConfig,
    EstimatorConfig,
    ForcingConfig,
    GridSpec,
    LatentConfig,
    ProjectorConfig,
    VariableSet,
)

CHANNELS = ["u10m", "v10m", "t2m", "z500", "t850", "q850", "sst", "u850", "v850"]


def make_config(kind: str = "si", **estimator_options) -> AtlasConfig:
    """A deliberately tiny model: 9 channels on a 49x96 grid, 13x24 latent."""
    noise_dim = 32 if kind == "crps" else None
    if kind == "crps":
        estimator_options.setdefault("noise_dim", 32)
    return AtlasConfig(
        name=f"test-{kind}",
        variables=VariableSet.era5_atlas().subset(CHANNELS),
        grid=GridSpec.equiangular(49, 96),
        latent=LatentConfig(shape=(13, 24), history=1),
        backbone=BackboneConfig(
            embed_dim_state=32,
            embed_dim_history=16,
            depth=2,
            num_heads=4,
            patch=(2, 3),
            noise_dim=noise_dim,
        ),
        projector=ProjectorConfig(
            embed_dim_state=24, embed_dim_latent=8, depth=2, num_heads=4
        ),
        estimator=EstimatorConfig(kind=kind, options=estimator_options),
        forcings=ForcingConfig(cos_zenith=True, time_of_year=True),
    )


def make_model(kind: str = "si", excite: bool = True, **opts) -> Atlas:
    """Build a tiny model with identity normalisation.

    DiT final layers are zero-initialised by design, so an untrained model
    outputs exactly zero.  ``excite`` perturbs them so that tests of
    interventions and responses see a non-degenerate signal.
    """
    cfg = make_config(kind, **opts)
    model = Atlas(cfg).eval()
    n = len(cfg.variables)
    model.normalizers.state.set_stats(np.zeros(n), np.ones(n))
    model.normalizers.residual.set_stats(np.zeros(n), np.ones(n))
    if excite:
        torch.manual_seed(0)
        for net in (model.backbone, model.projector):
            torch.nn.init.normal_(net.final.linear.weight, std=0.02)
    return model


@pytest.fixture
def model() -> Atlas:
    return make_model("si", steps=3)


@pytest.fixture
def batch(model):
    """Two windows of spatially smooth synthetic fields.

    Real atmospheric increments are smooth, and several diagnostics only make
    sense on smooth data (a white-noise increment is unrepresentable in any
    coarse latent, by construction), so the fixture uses the synthetic store
    rather than raw Gaussian noise.
    """
    from atlas.data import SyntheticStore, WindowDataset, collate

    store = SyntheticStore(model.cfg.variables, model.cfg.grid, n_times=32, seed=1)
    ds = WindowDataset(store, history=1)
    return collate([ds[0], ds[7]])
