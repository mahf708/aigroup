"""Tools for asking what an ATLAS model has learned.

Start with :class:`~atlas.probe.record.Recorder` -- everything else consumes
its output.

============================  ==================================================
Module                        Question it answers
============================  ==================================================
:mod:`~atlas.probe.record`    What are the activations, and where on Earth?
:mod:`~atlas.probe.attention` What does a location attend to?
:mod:`~atlas.probe.probes`    Which physical quantities are linearly decodable,
                              and at what depth do they appear?
:mod:`~atlas.probe.sae`       What features exist that nobody thought to ask for?
:mod:`~atlas.probe.intervene` Which of them does the forecast actually depend on?
:mod:`~atlas.probe.diagnostics` Is the latent space itself well behaved?
============================  ==================================================
"""

from .attention import (
    attention_entropy,
    attention_rollout,
    great_circle_matrix,
    head_locality,
    nearest_token,
    teleconnection_map,
)
from .diagnostics import (
    latent_continuity,
    latent_eof,
    power_spectrum,
    reconstruction_report,
    spectral_ratio,
    trajectory_summary,
)
from .intervene import (
    ResponseResult,
    influence_matrix,
    latent_channel_response,
    noise_sensitivity,
    patch_site,
    steer,
)
from .probes import LinearProbe, builtin_targets, depth_sweep, fit_ridge, pool_tokens
from .record import RecordedSite, Recorder, to_xarray, token_field
from .sae import SAETrainConfig, TopKSAE, feature_maps, feature_summary, train_sae

__all__ = [
    "Recorder",
    "RecordedSite",
    "token_field",
    "to_xarray",
    "nearest_token",
    "teleconnection_map",
    "head_locality",
    "attention_entropy",
    "attention_rollout",
    "great_circle_matrix",
    "LinearProbe",
    "fit_ridge",
    "pool_tokens",
    "depth_sweep",
    "builtin_targets",
    "TopKSAE",
    "SAETrainConfig",
    "train_sae",
    "feature_maps",
    "feature_summary",
    "patch_site",
    "steer",
    "latent_channel_response",
    "influence_matrix",
    "noise_sensitivity",
    "ResponseResult",
    "power_spectrum",
    "spectral_ratio",
    "latent_continuity",
    "latent_eof",
    "trajectory_summary",
    "reconstruction_report",
]
