# U-Cast

A flexible, dependency-light PyTorch implementation of

> **U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster.**
> Salva Rühling Cachay, Duncan Watson-Parris, Rose Yu. ICML 2026. [arXiv:2604.09041](https://arxiv.org/abs/2604.09041).
> Reference code: <https://github.com/Rose-STL-Lab/u-cast>

U-Cast reaches GenCast/IFS-ENS-level probabilistic skill at 1.5° with a **standard U-Net**, no graph or
spherical machinery, in under 12 H200-GPU-days. It rests on three ideas:

1. **A plain U-Net backbone** (ADM/"DhariwalUNet"), scaled to 320 base channels (895M parameters), with
   self-attention only at the two coarsest levels and convolutions that wrap around longitude. The
   diffusion-era adaptive-LayerNorm conditioning is deleted — there is no timestep to condition on.
2. **A two-stage curriculum.** Long, cheap deterministic pre-training on area-weighted **MAE** (which
   *is* the CRPS for a single member, so the two stages share a loss landscape), then a short
   probabilistic fine-tuning on the fair ensemble **CRPS**. Stage 2 is ~15% of the total budget.
3. **MC dropout as the only stochasticity.** No noise-injection parameters; the dropout masks that
   regularise Stage 1 become the ensemble generator in Stage 2.

This implementation is written from the paper and the reference repository. It is **not**
checkpoint-compatible with the published U-Cast weights (the attention/conditioning tensor layout
differs) — it is a from-scratch training and inference stack you can point at your own data.

## What "flexible" means here

| | |
|---|---|
| **Any grid** | Arbitrary `(H, W)`, including non-powers-of-two like 121×240. Per-level sizes are recorded so the decoder resamples exactly. Regional (non-periodic) grids are a config flag. |
| **Any variable set** | `VariableSet` is the single source of truth for the channel axis. Loss weights (GraphCast pressure/surface conventions) are derived from it, and can be overridden per variable. |
| **Any data source** | Implement one `ForecastDataset` subclass and name it in the config (`builder: my_pkg.my_mod:MyDataset`). A WeatherBench-2 zarr/NetCDF adapter and a synthetic dataset ship with the package. |
| **Prescribed fields** | A variable can be an *input* without being *forecast* — AMIP-style prescribed SST — via `output_variables` plus `prescribed_policy`. |
| **No framework lock-in** | Pure PyTorch: no Lightning, no Hydra, no wandb required. `torch`, `numpy`, `pyyaml`. xarray/zarr only for the real-data adapter, wandb only if you ask for it. |
| **Composable configs** | Typed dataclasses, YAML with `base:` inheritance, and `dotted.key=value` CLI overrides. Stage 2's config is 30 lines because it inherits Stage 1's. |
| **Self-describing checkpoints** | Config, dataset spec and normalisation statistics travel with the weights, so `load_forecaster(path)` alone rebuilds a working model. |

## Install

```bash
pip install -e .              # torch, numpy, pyyaml
pip install -e '.[data]'      # + xarray/zarr/netCDF4 for real archives
pip install -e '.[dev]'       # + pytest
```

## Five-minute check (CPU, synthetic data)

```bash
ucast summary --config configs/synthetic_smoke.yaml   # shapes, channel accounting, parameter count
ucast train   --config configs/synthetic_smoke.yaml   # Stage 1
ucast train   --config configs/synthetic_smoke.yaml \
              train.stage=probabilistic train.max_epochs=1 \
              train.init_from=runs/smoke/last.ckpt train.output_dir=runs/smoke_prob
ucast score   --checkpoint 'runs/smoke_prob/best.ckpt' --ensemble-size 4 --steps 3
```

The synthetic fields are genuinely predictable, so the loss really does fall and the ensemble spread
really does grow when Stage 2 starts optimising CRPS. Use this to validate a cluster, a new dataset
adapter or a config change before spending GPU hours.

## The real thing: ERA5 at 1.5°

Both stages read the WeatherBench-2 1.5° ERA5 zarr
(`1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`).

```bash
# Stage 1 -- deterministic pre-training, weighted MAE, 100 epochs (~10 H200-days)
ucast train --config configs/era5_1p5deg_stage1.yaml \
    data.common.path=/path/to/era5.zarr \
    data.statistics=/path/to/wb2_stats_dir

# Stage 2 -- probabilistic CRPS fine-tuning, 2 members, 8 epochs (~1.2 H200-days)
ucast train --config configs/era5_1p5deg_stage2.yaml \
    data.common.path=/path/to/era5.zarr \
    data.statistics=/path/to/wb2_stats_dir \
    train.init_from=runs/era5_stage1/last.ckpt

# Stage 3 -- deep ensemble: repeat Stage 2 from the SAME Stage-1 checkpoint, K=4 seeds
for SEED in 11 22 33 44; do
  ucast train --config configs/era5_1p5deg_stage2.yaml \
      data.common.path=/path/to/era5.zarr data.statistics=/path/to/wb2_stats_dir \
      train.init_from=runs/era5_stage1/last.ckpt \
      train.seed=$SEED train.output_dir=runs/era5_stage2_seed$SEED
done

# Score the K x N member ensemble (15-day forecast = 30 steps at 12-hourly)
ucast score --checkpoint 'runs/era5_stage2_seed*/best.ckpt' \
    --ensemble-size 8 --steps 30 --batch-size 1 -o scores.json
```

`data.statistics` accepts a directory of WeatherBench-2 style `era5_mean.nc` / `era5_std.nc` /
`era5_residual_std.nc` files (the layout the reference repo ships), a `.npz` written by
`ucast stats`, `compute` to estimate them by streaming the training split, or `identity`.

Multi-GPU is `torchrun`; nothing in the config changes:

```bash
torchrun --nproc_per_node=4 -m ucast.cli train --config configs/era5_1p5deg_stage1.yaml ...
```

The effective batch size (48) is held fixed: gradient accumulation fills whatever
`batch_size_per_device` × ranks leaves over, so lowering `batch_size_per_device` for memory does not
change the optimisation.

## Python API

```python
from ucast import ExperimentConfig, Trainer, load_forecaster
from ucast.data import SyntheticForecastDataset

config = ExperimentConfig.from_dict({
    "model": {"model_channels": 64, "channel_mult": [1, 2, 3]},
    "train": {"stage": "probabilistic", "max_epochs": 8},
})
trainer = Trainer(config, datasets={"train": ..., "val": ...})
trainer.fit()

# Later, anywhere: one file is enough.
model, config, spec = load_forecaster("runs/era5_stage2/best.ckpt")
forecast = model.rollout(batch, steps=30, ensemble_size=8)   # (8, B, 30, C, H, W), physical units
```

### Bringing your own data

```python
from ucast.data.base import DatasetSpec, ForecastDataset
from ucast.grid import Grid
from ucast.variables import VariableSet

class E3SMForecastDataset(ForecastDataset):
    def __init__(self, variables, output_variables=None, window=2, rollout_steps=1, **kwargs):
        self.rollout_steps = rollout_steps
        self.spec = DatasetSpec(
            input_variables=VariableSet(variables),
            output_variables=VariableSet(output_variables or variables),
            grid=Grid.equiangular(180, 360),
            window=window,
            num_forcing_channels=4,
            num_static_channels=2,
            step_hours=6,
        )

    def __len__(self): ...

    def get_item(self, index):
        return {
            "dynamics": ...,   # (window + rollout_steps, C_in, H, W), physical units
            "forcings": ...,   # (window + rollout_steps, C_forcing, H, W)
            "statics": ...,    # (C_static, H, W)
            "time": ...,       # int64 seconds since epoch
        }
```

Then `data: {builder: my_package.e3sm:E3SMForecastDataset, common: {...}}`. Item shapes are validated
on access, so a mistake surfaces at the dataset, not three layers down in a training step.

## Layout

```
ucast/
├── variables.py     VariableSet: the channel axis, plus GraphCast loss weights
├── grid.py          lat/lon grids and exact spherical area weights
├── nn/
│   ├── layers.py    longitude-periodic convolutions, residual blocks, bottleneck attention
│   └── unet.py      the backbone (895M parameters at the paper's settings)
├── losses.py        weighted MAE/MSE and the fair (unbiased) ensemble CRPS
├── metrics.py       streaming, area-weighted RMSE / CRPS / spread / SSR / bias, DDP-reduced
├── normalization.py per-channel statistics plus the residual (increment) scaling
├── forecaster.py    UCast: windowing, residuals, MC-dropout ensembles, autoregressive rollout
├── ema.py           EMA weights (decay 0.9999) -- what the published skill refers to
├── optim/           Muon + AdamW split, hybrid optimizer, warmup/cosine schedule
├── data/            dataset contract, WB2 xarray/zarr adapter, synthetic data, clock forcings
├── train.py         the two-stage training loop (DDP, AMP, accumulation, checkpointing)
├── inference.py     deep ensembles, scoring, NetCDF/zarr forecast output
├── checkpoint.py    self-describing checkpoints and warm starts
├── config.py        typed config, YAML inheritance, CLI overrides
└── cli.py           ucast train | score | forecast | stats | summary
```

## Tests

```bash
pytest                    # 100+ tests, CPU-only, ~40 s
```

Beyond shape plumbing, the suite pins down the things that are easy to get quietly wrong:

- the fair CRPS matches a brute-force reference and is **unbiased in the ensemble size** (the biased
  estimator is visibly wrong at `M = 2`, which is exactly where U-Cast trains);
- circularly padded convolutions are **exactly equivariant** to longitude rotation;
- the paper's configuration has the **published 895M parameters**;
- area weights match analytic spherical cell areas, and the SSR of a calibrated ensemble is 1;
- an untrained model forecasts **persistence** (every residual branch is zero-initialised), and MC
  dropout is the *only* source of spread;
- the Stage 1 → Stage 2 → deep-ensemble handoff works, checkpoints round-trip bit-for-bit, and
  resuming continues from the saved step;
- training on the xarray adapter end to end, including WeatherBench-2's descending latitude,
  `(time, level, longitude, latitude)` dimension order, 6h→12h subsampling and NaN-over-land SST.

## Deviations from the reference implementation

Deliberate, and each one is a config flag if you want the original behaviour:

- **Layout.** `(B, C, lat, lon)` with periodic padding on the last axis, rather than the reference's
  `(lon, lat)` ordering. Padding is a property of the convolution module instead of a global
  monkey-patch of `torch.nn.functional.conv2d`.
- **Downsampling.** `avg_pool2d(..., ceil_mode=True)`, so an odd extent keeps its trailing row (the
  south pole) instead of discarding it. Set `model.pool_ceil_mode: false` for the original.
- **Attention.** `F.scaled_dot_product_attention` (fused/flash kernels) instead of a hand-written
  autograd function.
- **LR schedule.** A plain multiplicative warmup+cosine callable rather than a `_LRScheduler`
  subclass, so it composes with the hybrid optimizer.
- **Pole padding.** `latitude_padding: replicate` is offered as an alternative to the reference's
  zero padding, which injects artificial zeros next to the poles. Default remains `zeros`.

## Citation

```bibtex
@article{cachay2026ucast,
  title  = {U-Cast: A Surprisingly Simple and Efficient Frontier AI Probabilistic Weather Forecaster},
  author = {Cachay, Salva R{\"u}hling and Watson-Parris, Duncan and Yu, Rose},
  journal = {International Conference on Machine Learning},
  year   = {2026},
}
```

Muon is adapted from [Keller Jordan's reference implementation](https://kellerjordan.github.io/posts/muon/)
(MIT). The U-Net follows Dhariwal & Nichol (2021) / Karras et al. (2022); clock forcings follow
GraphCast (Apache-2.0).
