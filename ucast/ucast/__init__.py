"""U-Cast: a simple, efficient probabilistic weather forecaster.

A flexible PyTorch implementation of

    Cachay, Ruhling; Watson-Parris, Duncan; Yu, Rose.
    *U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster.*
    ICML 2026.  arXiv:2604.09041.  Reference code: https://github.com/Rose-STL-Lab/u-cast

Three ideas, no exotic machinery:

1. a **standard U-Net** with bottleneck self-attention, wrapped periodically in longitude
   (:mod:`ucast.nn`),
2. a **two-stage curriculum** -- long deterministic pre-training on weighted MAE, then short
   probabilistic fine-tuning on the fair CRPS (:mod:`ucast.train`, :mod:`ucast.losses`),
3. **MC dropout** as the only source of ensemble spread, with no noise-injection parameters
   (:mod:`ucast.forecaster`).

Quick start::

    from ucast import ExperimentConfig, Trainer
    from ucast.data import SyntheticForecastDataset

    config = ExperimentConfig.from_dict({"model": {"model_channels": 32, "channel_mult": [1, 2]}})
    trainer = Trainer(config, datasets={"train": SyntheticForecastDataset(), "val": SyntheticForecastDataset()})
    trainer.fit()

This implementation is written from the paper and the reference repository; it does **not** load the
published U-Cast checkpoints (the tensor layout of the attention and conditioning blocks differs).
"""

from __future__ import annotations

from .checkpoint import load_checkpoint, load_forecaster, load_weights_into, save_checkpoint
from .config import ExperimentConfig, load_config
from .ema import ExponentialMovingAverage
from .forecaster import UCast, build_forecaster, mc_dropout
from .grid import Grid, latitude_area_weights
from .inference import DeepEnsemble, evaluate
from .losses import WeightedCRPS, WeightedMAE, WeightedMSE, build_loss, crps_ensemble
from .metrics import ForecastMetrics, MetricCollection
from .nn import UCastUNet
from .normalization import Normalizer, compute_statistics
from .train import Trainer, train
from .variables import Variable, VariableSet, era5_variable_set

__version__ = "0.1.0"

__all__ = [
    "DeepEnsemble",
    "ExperimentConfig",
    "ExponentialMovingAverage",
    "ForecastMetrics",
    "Grid",
    "MetricCollection",
    "Normalizer",
    "Trainer",
    "UCast",
    "UCastUNet",
    "Variable",
    "VariableSet",
    "WeightedCRPS",
    "WeightedMAE",
    "WeightedMSE",
    "__version__",
    "build_forecaster",
    "build_loss",
    "compute_statistics",
    "crps_ensemble",
    "era5_variable_set",
    "evaluate",
    "latitude_area_weights",
    "load_checkpoint",
    "load_config",
    "load_forecaster",
    "load_weights_into",
    "mc_dropout",
    "save_checkpoint",
    "train",
]
