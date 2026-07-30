# U-Cast

A flexible PyTorch implementation of U-Cast lives in [`ucast/`](https://github.com/E3SM-Project/aigroup/tree/main/ucast)
in this repository.

!!! quote "The paper"
    **U-Cast: A Surprisingly Simple and Efficient Frontier Probabilistic AI Weather Forecaster.**
    Salva Rühling Cachay, Duncan Watson-Parris, Rose Yu. ICML 2026.
    [arXiv:2604.09041](https://arxiv.org/abs/2604.09041) ·
    [reference code](https://github.com/Rose-STL-Lab/u-cast)

## Why this is interesting for us

U-Cast matches GenCast and beats IFS ENS on probabilistic skill at 1.5° while training in **under 12
H200-GPU-days** and producing a 15-day ensemble forecast in **3 seconds**. That is an order of
magnitude less compute than the other frontier probabilistic models, and the architecture is a plain
U-Net — no graph network, no spherical harmonics, no diffusion sampler. It is the cheapest credible
route we have to a frontier probabilistic forecaster trained on our own data.

The recipe has exactly three ingredients:

1. **A standard U-Net** (the ADM/"DhariwalUNet" backbone from image diffusion) at 320 base channels,
   895M parameters, with self-attention only at the two coarsest levels and convolutions that wrap
   around longitude. The diffusion-era adaptive-LayerNorm conditioning is removed.
2. **A two-stage curriculum**: long, cheap deterministic pre-training on area-weighted **MAE**, then
   short probabilistic fine-tuning on the fair ensemble **CRPS**. MAE rather than MSE because for a
   single-member forecast the MAE *is* the CRPS, so the two stages optimise the same functional.
   Stage 2 is only ~15% of the training budget.
3. **Monte-Carlo dropout** as the only source of ensemble spread — no noise-injection layers, ~10%
   fewer parameters than the alternatives. The dropout that regularises Stage 1 becomes the ensemble
   generator in Stage 2.

Plus the [Muon](https://kellerjordan.github.io/posts/muon/) optimizer for the hidden weight matrices,
which is what makes an 8-epoch Stage 2 sufficient.

## What is in `ucast/`

A from-scratch training and inference stack, not a fork: pure PyTorch (no Lightning, no Hydra), with
the data layer behind a one-class interface so it can be pointed at E3SM output as easily as at ERA5.

- **Grid-agnostic**: any `(H, W)`, including non-powers-of-two such as 121×240; regional
  (non-periodic) grids are a config flag.
- **Variable-agnostic**: one `VariableSet` defines the channel axis, and the GraphCast
  pressure/surface loss weights follow from it.
- **Data-agnostic**: ship a `ForecastDataset` subclass and name it in the config. A WeatherBench-2
  zarr/NetCDF adapter and a CPU-sized synthetic dataset are included.
- **Prescribed fields**: a variable can be an input without being forecast — AMIP-style prescribed
  SST — which matters for our ACE2-style configurations.
- **Self-describing checkpoints**: config, dataset spec and normalisation statistics travel with the
  weights, so one file rebuilds a working model.

!!! note "Not checkpoint-compatible with the published weights"
    This is an independent implementation written from the paper and the reference repository; the
    attention and conditioning tensor layouts differ, so the released U-Cast checkpoints will not load
    into it. Use it to train, not to reproduce their exact weights.

## Install

```console
cd ucast
pip install -e '.[data,dev]'
```

See our [Python Environment Setup](python-envs.md) if you need an environment first.

## Five-minute check on a laptop

Everything below runs on CPU against synthetic advecting-wave fields, in well under a minute. Use it
to validate a machine, a new dataset adapter or a config change before spending GPU hours.

```console
ucast summary --config configs/synthetic_smoke.yaml
ucast train   --config configs/synthetic_smoke.yaml
ucast train   --config configs/synthetic_smoke.yaml \
              train.stage=probabilistic train.max_epochs=1 \
              train.init_from=runs/smoke/last.ckpt train.output_dir=runs/smoke_prob
ucast score   --checkpoint 'runs/smoke_prob/best.ckpt' --ensemble-size 4 --steps 3
```

The synthetic fields are genuinely predictable, so the training loss really falls — and the ensemble
spread really grows once Stage 2 starts optimising the CRPS, which is the paper's central claim about
dropout in miniature.

## The full recipe on ERA5 at 1.5°

Both stages read the WeatherBench-2 1.5° ERA5 zarr
(`1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`), 83 channels: five surface
variables plus six atmospheric variables on 13 pressure levels.

### Stage 1 — deterministic pre-training (~10 H200-days)

```console
ucast train --config configs/era5_1p5deg_stage1.yaml \
    data.common.path=/path/to/era5.zarr \
    data.statistics=/path/to/wb2_stats_dir
```

### Stage 2 — probabilistic CRPS fine-tuning (~1.2 H200-days)

```console
ucast train --config configs/era5_1p5deg_stage2.yaml \
    data.common.path=/path/to/era5.zarr \
    data.statistics=/path/to/wb2_stats_dir \
    train.init_from=runs/era5_stage1/last.ckpt
```

### Stage 3 — deep ensemble (K = 4 in the paper)

Repeat Stage 2 from the **same** Stage-1 checkpoint with different seeds, then score the checkpoints
together. Each extra member costs only one short Stage-2 run.

```console
for SEED in 11 22 33 44; do
  ucast train --config configs/era5_1p5deg_stage2.yaml \
      data.common.path=/path/to/era5.zarr data.statistics=/path/to/wb2_stats_dir \
      train.init_from=runs/era5_stage1/last.ckpt \
      train.seed=$SEED train.output_dir=runs/era5_stage2_seed$SEED
done

ucast score --checkpoint 'runs/era5_stage2_seed*/best.ckpt' \
    --ensemble-size 8 --steps 30 --batch-size 1 -o scores.json
```

Any config field can be overridden inline as `dotted.key=value`, parsed as YAML — so
`model.channel_mult=[1,2,2]` and `train.compile=true` both work.

### Multi-GPU

Nothing in the config changes:

```console
torchrun --nproc_per_node=4 -m ucast.cli train --config configs/era5_1p5deg_stage1.yaml ...
```

!!! tip "Out of memory?"
    Lower `data.batch_size_per_device`. The effective batch size stays at 48 — gradient accumulation
    fills the gap — so the optimisation is unchanged.

## Statistics

`data.statistics` accepts:

| value | meaning |
|---|---|
| a directory | WeatherBench-2 style `era5_mean.nc`, `era5_std.nc`, `era5_residual_std.nc` (the layout the reference repo ships) |
| a `.npz` path | written by `ucast stats --config ... -o stats.npz` |
| `compute` | estimate by streaming the training split — the option to use for E3SM output with no published statistics |
| `identity` | data is already standardised |

The residual standard deviation matters: the model predicts the *increment* from the last input frame,
and increments are far smaller than the states. Scaling them by `std / residual_std` puts every
channel — fast surface fields and slow stratospheric ones alike — on the same order.

## Metrics

Scores follow WeatherBench-2, so they are comparable to the public leaderboard: area-weighted RMSE of
the ensemble mean, fair CRPS, ensemble spread, spread/skill ratio
(`sqrt((M+1)/M) · spread / rmse`; 1 is calibrated, below 1 under-dispersive), MAE and bias. They are
accumulated as running sums, so a full year of initial conditions never holds more than one batch of
forecasts in memory, and they reduce across ranks with a single all-reduce.

## Applying it to E3SM data

Implement one class and name it in the config:

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

    def __len__(self):
        ...

    def get_item(self, index):
        return {
            "dynamics": ...,   # (window + rollout_steps, C_in, H, W), physical units
            "forcings": ...,   # (window + rollout_steps, C_forcing, H, W)
            "statics": ...,    # (C_static, H, W)
            "time": ...,       # int64 seconds since epoch
        }
```

```yaml
data:
  builder: my_package.e3sm:E3SMForecastDataset
  common: {path: /lcrc/group/e3sm/...}
```

Item shapes are validated on access, so a mistake surfaces at the dataset rather than three layers
down inside a training step. If your data is already regridded to a lat/lon zarr or NetCDF, the
built-in `era5` adapter may work as-is — it takes variable names, dimension names, level coordinate
and NaN handling from the config.

## Tests

```console
cd ucast && pytest
```

100+ CPU-only tests, about 40 seconds. They pin down the things that are easy to get quietly wrong:
the fair CRPS matches a brute-force reference and is unbiased in the ensemble size (the biased
estimator is visibly wrong at `M = 2`, exactly where U-Cast trains); circularly padded convolutions
are exactly equivariant to longitude rotation; the paper's configuration has the published 895M
parameters; area weights match analytic spherical cell areas; an untrained model forecasts
persistence and MC dropout is the only source of spread; and the whole Stage 1 → Stage 2 → deep
ensemble handoff round-trips through checkpoints.

## Further reading

The [`ucast/README.md`](https://github.com/E3SM-Project/aigroup/tree/main/ucast#readme) covers the
Python API, the module layout and the deliberate deviations from the reference implementation.
