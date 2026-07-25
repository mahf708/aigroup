"""ATLAS -- Atmospheric Transformer in Latent Space.

A self-contained, configuration-driven reimplementation of the framework in
Kossaifi et al. (2026), *Demystifying Data-Driven Probabilistic Medium-Range
Weather Forecasting* (arXiv:2601.18111), built for experimentation rather than
for reproducing one checkpoint:

* the data description (channels, grid, vertical coordinate, forcings) is a
  config object, so E3SM output is a first-class target alongside ERA5;
* the three probabilistic estimators -- stochastic interpolants, EDM diffusion
  and spectrally-regularised CRPS -- share one backbone and one interface;
* the encoder is a swappable operator (bilinear, area-conservative, spectral,
  or learned) so the paper's Section 5 ablation is a config change;
* :mod:`atlas.probe` provides the tooling for latent-space and residual-stream
  analysis.

Nothing here requires ``physicsnemo``, ``natten``, ``timm`` or ``einops``;
``torch-harmonics`` and ``xarray`` are optional and degrade gracefully.
"""

from .backbones import LatentDiT, LocalProjector
from .estimators import (
    CRPSEstimator,
    EDMDiffusion,
    Estimator,
    StochasticInterpolant,
    build_estimator,
)
from .grids import Resampler, SphericalTransform
from .model import Atlas, RolloutState, StepOutput
from .normalize import ChannelNormalizer, NormalizerPair
from .spec import (
    AtlasConfig,
    BackboneConfig,
    Channel,
    EstimatorConfig,
    ForcingConfig,
    GridSpec,
    LatentConfig,
    ProjectorConfig,
    VariableSet,
)

__version__ = "0.1.0"

__all__ = [
    "Atlas",
    "AtlasConfig",
    "StepOutput",
    "RolloutState",
    "Channel",
    "VariableSet",
    "GridSpec",
    "LatentConfig",
    "BackboneConfig",
    "ProjectorConfig",
    "EstimatorConfig",
    "ForcingConfig",
    "LatentDiT",
    "LocalProjector",
    "Estimator",
    "StochasticInterpolant",
    "EDMDiffusion",
    "CRPSEstimator",
    "build_estimator",
    "Resampler",
    "SphericalTransform",
    "ChannelNormalizer",
    "NormalizerPair",
    "__version__",
]
