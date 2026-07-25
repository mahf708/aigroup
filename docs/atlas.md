# ATLAS: Latent Probabilistic Forecasting

An implementation of **ATLAS** (Atmospheric Transformer in Latent Space) lives in
[`atlas/`](https://github.com/E3SM-Project/aigroup/tree/main/atlas). It follows
Kossaifi et al. (2026), *Demystifying Data-Driven Probabilistic Medium-Range Weather
Forecasting* ([arXiv:2601.18111](https://arxiv.org/abs/2601.18111)), but is written to
be retargeted at E3SM output rather than to reproduce one ERA5 checkpoint, and it ships
with tooling for asking what the model has learned.

!!! info "What you get"
    - The full framework — latent encoder, global-attention predictive DiT,
      local-attention projector — driven entirely by a config object.
    - All three probabilistic estimators from the paper on one shared backbone:
      stochastic interpolants, EDM diffusion, and spectrally-regularised CRPS.
    - E3SM-ready channel sets, grids, forcings (including CO2 and other scalars),
      and an `xarray` adapter.
    - `atlas.probe`: recording, attention analysis, linear probes, sparse
      autoencoders, and causal interventions.
    - PyTorch and NumPy only. `torch-harmonics`, `xarray` and `pyyaml` are optional.

## Quick start

```bash
cd atlas
pip install -e ".[dev]"

atlas info configs/dev_small.yaml      # summarise a config
atlas demo --steps 600                 # train on synthetic data and score it
pytest tests -q                        # 84 tests, ~6 s on CPU
```

`atlas demo` needs no data at all — it generates a synthetic advecting field, fits
normalisation, trains a small model, and reports ensemble RMSE, fair CRPS and the
spread-skill ratio. It is the fastest way to confirm an environment is working.

## How the model works

=== "The idea"

    Three components, each independently ablatable:

    1. **The latent space is not learned.** The encoder is bilinear downsampling from
       0.25° to 1° — 16× compression that discards exactly the scales that are not
       deterministically predictable at a 6-hour timestep. Latent channel *c* remains a
       coarse-grained version of physical variable *c*.
    2. **A global-attention DiT models the coarse conditional law** `ρ(r₁ | z₀, z₋₁)`,
       where `r` is a temporal *residual* and `z₋₁` supplies one step of history.
    3. **A local-attention DiT projects back up**, conditioned on the full-resolution
       current state: `D(r₁, x₀) ≈ x₁ − x₀`.

    The projector's reconstruction error sits an order of magnitude below the
    predictive model's, which is what makes the compression lossless in practice.

=== "Rollout"

    Two states are carried, and they are deliberately not the same object:

    - a coarse **latent state** `z`, advanced by the backbone in latent space;
    - the full-resolution **fine state** `x`, which conditions the projector.

    Re-encoding the decoded fine state at each step would feed the projector's own
    super-resolution artefacts back into the dynamics. Advancing them separately is
    what lets the model stay stable over 60 autoregressive steps without any
    autoregressive fine-tuning.

=== "Estimators"

    | Estimator | `kind` | Cost per step (paper, A100) |
    |---|---|---|
    | Stochastic interpolant | `si` | 94 s |
    | EDM diffusion | `edm` | 88 s |
    | Spectral CRPS | `crps` | 3.3 s |

    Switching is a one-line config change; the backbone is byte-identical. The CRPS
    variant samples in a single forward pass, which makes it the practical choice for
    large ensembles.

At the paper's hyperparameters the block stacks reproduce its parameter counts exactly:
2.39 B for the 12-block, 3328-wide backbone and 1.79 B for the 16-block, 2496-wide
projector.

## Retargeting to E3SM

Nothing about ERA5 is baked into the model — everything data-specific lives in
`AtlasConfig`. Start from `configs/e3sm_eam_1deg.yaml`.

```python
from atlas.spec import (
    AtlasConfig, VariableSet, GridSpec, LatentConfig, ForcingConfig,
)

cfg = AtlasConfig(
    variables=VariableSet.e3sm_eam(n_levels=24),                  # T_lev07-style names
    grid=GridSpec.equiangular(180, 360, descending=False,
                              include_poles=False),               # CAM cell centres
    latent=LatentConfig(shape=(45, 90), mode="area"),             # conserves means
    forcings=ForcingConfig(
        cos_zenith=True, time_of_year=True,
        static_fields=["PHIS", "LANDFRAC", "OCNFRAC"],
        scalar_fields=["co2vmr", "solar_constant"],
    ),
    dt_hours=6.0,
)
```

| Concern | Where it lives |
|---|---|
| Variable names, levels, units, loss weights | `VariableSet`, `Channel` |
| Latitude ordering, poles, regional domains | `GridSpec` (`descending`, `include_poles`, `periodic_lon`) |
| Encoder choice | `LatentConfig.mode`: `bilinear`, `area`, `spectral`, `learned` |
| Boundary conditions and external forcing | `ForcingConfig` |
| Model timestep vs. archive frequency | `dt_hours`, `WindowDataset(stride=...)` |

!!! warning "Regrid first"
    ATLAS tokenises with strided convolutions, so it needs a structured mesh. Native
    unstructured output (`ne30pg2` and friends) must be regridded — use
    `ncremap`/`TempestRemap` with a conservative map, which pairs naturally with
    `LatentConfig(mode="area")`.

!!! tip "Forced-response experiments"
    Scalar forcings are broadcast to full fields and supplied at inference:
    `model.rollout(..., scalars={"co2vmr": 4.2e-4})`. One trained model can then be
    driven along several pathways without retraining — which is the difference between
    a weather emulator and something usable for E3SM-style climate experiments.

Reading real data:

```python
import xarray as xr
from atlas.data import XarrayStore, WindowDataset

ds = xr.open_zarr("eamv3_1deg_6hourly.zarr")
store = XarrayStore(ds, cfg.variables, cfg.grid, level_dim="lev")
dataset = WindowDataset(store, history=cfg.latent.history)
```

`XarrayStore` reorders latitude and rolls longitude to match the `GridSpec`, so ERA5
(90→−90) and E3SM (−90→90) archives can back the same model.

## Investigating the latent space

!!! note "Read this before probing"
    ATLAS's latent is **not** a learned code — it is a bilinear downsampling. "What
    does the latent encode?" has a trivial answer: coarse-grained physical variables,
    in named channels, in physical units. The genuinely learned representation is the
    **token residual stream** inside the DiT.

    The flip side is that ATLAS is an unusually good subject for this work. The
    bottleneck between global dynamics and local physics is an explicit, named,
    physically-scaled field, so intervening on it asks a well-posed physical question in
    a way that editing a VAE code never does. And because the residual stream lives on a
    coarse token grid (91×120 at the paper's settings), every activation has a latitude
    and longitude.

```python
from atlas.probe import Recorder, depth_sweep, builtin_targets

rec = Recorder(model, sites=["backbone.blocks.*"])
rec.capture_attention("backbone.blocks.*.attn", query_indices=[q], stats=True)
with rec:
    out = model.step(x_cur, z0, history, times=times)

rec["backbone.blocks.11"]          # (B, 10920, 3328) tokens
rec.field("backbone.blocks.11")    # (B, 3328, 91, 120) maps
rec.token_latlon("backbone.blocks.11")
```

| Question | Tool |
|---|---|
| What are the activations, and where on Earth? | `Recorder`, `token_field`, `to_xarray` |
| What does a location attend to? | `teleconnection_map`, `head_locality`, `attention_entropy` |
| Which indices are linearly decodable, at what depth? | `depth_sweep`, `LinearProbe`, `builtin_targets` |
| What features exist that nobody asked for? | `train_sae`, `feature_maps`, `feature_summary` |
| Which of them does the forecast *depend* on? | `latent_channel_response`, `influence_matrix`, `patch_site`, `steer` |
| Where does ensemble spread come from? | `noise_sensitivity`, `latent_eof` |
| Is the latent space well behaved? | `reconstruction_report`, `spectral_ratio`, `latent_continuity` |

A few of these deserve comment.

**Depth sweeps separate copying from computing.** A target already decodable at block 0
was handed to the model by the input encoding; one that only appears mid-stack is being
constructed. The probes are deliberately linear — a non-linear probe decodes almost
anything from 3,328 dimensions and tells you nothing.

**Attention rows are teleconnection maps.** The backbone uses global attention, so each
row answers "when updating the atmosphere here, where does the model look?". The full
matrix cannot be stored (10,920² × 13 heads × 12 layers), so `capture_attention` takes
explicit `query_indices`. `head_locality` ranks heads by mean attended great-circle
distance — a fast way to find the few genuinely long-range heads worth studying.

**Interventions turn correlation into causation.** `influence_matrix` perturbs one
latent channel at a time and measures the response in every output channel, reusing the
*same* probabilistic sample for control and perturbation so the difference is not
sampling noise. The off-diagonal structure is an empirical read-out of the variable
couplings the projector has learned.

**Ensemble spread has directions.** `noise_sensitivity` returns the leading principal
directions of an ensemble's deviation from its mean — the model's own estimate of the
fastest-growing uncertainty, a learned analogue of singular vectors.

**The sampling path is itself an object of study.** `record_trajectory=True` returns the
full SDE path in latent space and `Recorder(mode="all")` captures the residual stream at
every solver step; `trajectory_summary` shows where in the integration the sample
actually acquires its structure.

Run the whole tour on a small model in about a minute:

```bash
python examples/latent_analysis.py --steps 400
```

## Ablating the encoder

Section 5 of the paper argues for the trivial encoder over a learned one. That
comparison is a `cfg.latent.mode` change:

| Mode | What it does |
|---|---|
| `bilinear` | The paper's choice: pure bilinear interpolation, no parameters |
| `area` | Area-weighted pooling; conserves each channel's global mean |
| `spectral` | Spherical-harmonic truncation (needs `torch-harmonics`) |
| `learned` | A trainable correction to the coarse field |

!!! note "`learned` is not a VAE, on purpose"
    A VAE-style encoder mixes all variables into a common channel space, and the paper
    attributes two failure modes to that mixing: a flat high-frequency tail in the
    latent spectrum, and a loss of temporal continuity. The learned mode here keeps
    channel identity and learns a correction, `z = B(x) + g(B(x), P(x))`, where `P` is a
    strided convolution that lets sub-grid structure inform the coarse value. Latent
    channel *c* still means physical variable *c*, so residual normalisation, the latent
    update `z + r` and every probe keep working. The correction is zero-initialised, so
    training starts exactly at the bilinear baseline.

    A full channel-mixing VAE latent is **not** implemented — it would break the
    physical channel correspondence the rest of the package depends on.

A learnable encoder is trained through the projector's reconstruction loss only;
`training_losses` detaches the latent target from the estimator, because a probabilistic
head chasing a target free to move does not converge.

```bash
atlas train cfg.yaml --parts encoder projector   # fit the autoencoder first
atlas train cfg.yaml --parts latent              # then the predictive model
```

Judge the result on the paper's own terms with `atlas.probe.diagnostics`:

- `reconstruction_report` — projector vs. naive-upsampling vs. persistence error
- `spectral_ratio` — power by wavenumber; a long flat high-frequency tail is the VAE
  pathology the paper reports
- `latent_continuity` — do temporally adjacent states stay adjacent in latent space?

## Training

The schedule follows the paper: linear warmup, cosine decay over a 100,000-step
*cycle*, then a restart at 0.8× the previous peak.

```bash
atlas stats configs/e3sm_eam_1deg.yaml --data eamv3.zarr --out stats.npz
atlas train configs/e3sm_eam_1deg.yaml --data eamv3.zarr --stats stats.npz \
    --batch-size 8 --parts latent projector
```

The two heads are independent — the projector is always trained on the *true* latent
residual, never a sampled one — so they can be optimised separately. That is what the
paper does: one projector, trained once, shared by all three estimators. Pass
`--parts projector` and then `--parts latent` to reproduce it.

## Relation to the reference implementation

The published inference code (`earth2studio/models/{px,nn}/atlas.py`) depends on
`physicsnemo`, `natten`, `timm` and `einops` and exists to run one released checkpoint.
This implementation is independent and trainable, so:

- Neighborhood attention uses a chunked gather rather than `natten` — exact,
  memory-bounded, and tested against unchunked evaluation, but slower at scale.
- Padding across the poles is topologically correct (`SphericalPad(mode="pole")`),
  not merely reflective.
- Diffusion preconditioning lives in the estimator, so one backbone serves all three
  objectives.
- **Weights are not interchangeable with the released checkpoint.**
- Without `torch-harmonics` the spherical transform falls back to a 2-D FFT. That is a
  different operator — fine as a regulariser and for qualitative spectra, and always
  reported by `SphericalTransform.backend`.
