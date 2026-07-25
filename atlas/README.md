# ATLAS

A flexible, self-contained implementation of **ATLAS** (Atmospheric Transformer in
Latent Space) from Kossaifi et al. (2026), *Demystifying Data-Driven Probabilistic
Medium-Range Weather Forecasting* ([arXiv:2601.18111](https://arxiv.org/abs/2601.18111)),
built for experimentation on E3SM-style data as well as ERA5 — and for finding out
what these models actually learn.

```
pip install -e ".[dev]"
atlas info configs/dev_small.yaml
atlas demo --steps 600            # trains a tiny model on synthetic data and scores it
python examples/latent_analysis.py --steps 400
```

Nothing beyond PyTorch and NumPy is required. `torch-harmonics` (true spherical
harmonics), `xarray` (real data), `pyyaml` (YAML configs) and `matplotlib` are optional
and degrade gracefully.

## What ATLAS is

Three ideas, and the implementation keeps them separable so each can be ablated:

1. **The latent space is not learned.** The encoder is bilinear downsampling from
   0.25° to 1° — 16× compression that removes exactly the scales that are not
   deterministically predictable at a 6-hour timestep. Latent channel *c* stays a
   coarse-grained version of physical variable *c*.
2. **A global-attention DiT models the coarse conditional law**
   `ρ(r₁ | z₀, z₋₁)`, where `r` is a *residual* and `z₋₁` is one step of history.
3. **A local-attention DiT projects back up**, conditioned on the full-resolution
   current state: `D(r₁, x₀) ≈ x₁ − x₀`. Its reconstruction error is an order of
   magnitude below the predictive model's, which is what makes the latent approach
   lossless in practice.

The paper's headline claim is that this framework is indifferent to the probabilistic
estimator. All three are implemented against one shared backbone:

| Estimator | Class | Cost per step (paper, A100) |
|---|---|---|
| Stochastic interpolant | `StochasticInterpolant` | 94 s |
| EDM diffusion | `EDMDiffusion` | 88 s |
| Spectrally-regularised CRPS | `CRPSEstimator` | 3.3 s |

Swapping between them is a one-line config change.

At the paper's hyperparameters the block stacks here reproduce its parameter counts
exactly — 2.39 B for the 12-block, 3328-wide backbone and 1.79 B for the 16-block,
2496-wide projector.

## Layout

```
src/atlas/
  spec.py          Channels, grids, and the config objects everything is built from
  grids.py         Resampling, sphere-consistent padding, positional encodings, SHT
  layers.py        DiT blocks; global / neighborhood / window attention
  backbones.py     LatentDiT (predictive) and LocalProjector (decoder)
  estimators.py    Stochastic interpolants, EDM diffusion, spectral CRPS
  model.py         Atlas: encode, predict, decode, rollout
  normalize.py     Separate state and residual statistics
  forcings.py      Cosine zenith angle, static maps, calendar and scalar forcings
  losses.py        Area-weighted losses; fair CRPS, ensemble RMSE, spread-skill ratio
  data.py          Store protocol, synthetic store, xarray/E3SM adapter, windowing
  train.py         Warmup + cosine-cycle trainer with EMA
  probe/           Latent-space and residual-stream analysis (see below)
configs/           ERA5 (SI / EDM / CRPS), E3SM EAM, and a laptop-sized dev config
examples/          A worked end-to-end analysis
```

## Using it with E3SM

Everything data-specific lives in `AtlasConfig`; nothing about ERA5 is baked into the
model. `configs/e3sm_eam_1deg.yaml` is a working starting point. The pieces that
matter:

- **Channels.** `VariableSet.e3sm_eam(n_levels=24)` builds `T_lev07`-style names on
  hybrid sigma-pressure levels. `Channel` carries units, level kind, sign constraints
  and per-channel loss weights, and any custom set works — the model only ever sees an
  ordered list of 2-D fields.
- **Grid.** `GridSpec.equiangular(180, 360, descending=False, include_poles=False)`
  matches CAM's cell-centre latitudes. Set `periodic_lon=False` for a regional domain
  and padding switches from circular to replication throughout.
  Native unstructured output (`ne30pg2` and friends) must be regridded to a structured
  mesh first — ATLAS tokenises with strided convolutions. Use
  `ncremap`/`TempestRemap` with a conservative map, which pairs naturally with
  `LatentConfig(mode="area")`.
- **Encoder.** `mode="area"` conserves the area-weighted global mean of every channel,
  which is usually what you want if the model will be judged on energy and water
  budgets. `mode="bilinear"` is the paper's choice, `mode="spectral"` truncates
  spherical harmonics, and `mode="learned"` trains a correction to the coarse field
  (see *Ablating the encoder* below).
- **Forcings.** `ForcingConfig` adds the cosine zenith angle, calendar encodings, static
  maps (`PHIS`, `LANDFRAC`, `OCNFRAC`), and **scalar forcings** such as `co2vmr` or the
  solar constant. The scalar path is what makes forced-response experiments possible:
  one trained model, several pathways, supplied at inference through
  `rollout(..., scalars={"co2vmr": 4.2e-4})`.
- **Timestep.** `dt_hours` and `WindowDataset(stride=...)` decouple the model timestep
  from the archive's output frequency.

Reading data:

```python
import xarray as xr
from atlas.data import XarrayStore, WindowDataset
from atlas.spec import AtlasConfig

cfg = AtlasConfig.load("configs/e3sm_eam_1deg.yaml")
ds = xr.open_zarr("eamv3_1deg_6hourly.zarr")
store = XarrayStore(ds, cfg.variables, cfg.grid, level_dim="lev")
dataset = WindowDataset(store, history=cfg.latent.history)
```

`XarrayStore` handles latitude ordering and longitude rolling, so ERA5 (90→−90) and
E3SM (−90→90) archives can back the same model. Pass `overrides={...}` for names the
default resolver does not recognise.

## Investigating the latent space

`atlas.probe` is built for this. One caveat first, because it changes what the
questions mean:

> **ATLAS's latent is not a learned code.** It is a bilinear downsampling. "What does
> the latent encode?" has a trivial answer — coarse-grained physical variables, in
> named channels, in physical units. The genuinely learned representation is the
> **token residual stream** inside the DiT, and that is where the interesting
> mechanistic questions live.

The flip side is that this makes ATLAS an unusually good subject. The bottleneck
between global dynamics and local physics is an explicit, named, physically-scaled
field, so an intervention on it is a well-posed physical question in a way that
editing a VAE code never is. And because the residual stream sits on a coarse token
grid (91×120 at the paper's settings), every activation has a latitude and longitude.

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
| Which physical indices are linearly decodable, at what depth? | `depth_sweep`, `LinearProbe`, `builtin_targets` |
| What features exist that nobody asked for? | `train_sae`, `feature_maps`, `feature_summary` |
| Which of them does the forecast *depend* on? | `latent_channel_response`, `influence_matrix`, `patch_site`, `steer` |
| Where does ensemble spread come from? | `noise_sensitivity`, `latent_eof` |
| Is the latent space well behaved? | `reconstruction_report`, `spectral_ratio`, `latent_continuity` |

A few notes on what these are good for.

**Depth sweeps separate copying from computing.** A target already decodable at block 0
was handed to the model by the input encoding. One that only appears mid-stack is being
constructed. Probes here are deliberately linear — a non-linear probe decodes almost
anything from 3,328 dimensions and tells you nothing.

**Attention rows are teleconnection maps.** The predictive backbone uses global
attention, so each row answers "when updating the atmosphere here, where does the model
look?". The full matrix is far too large to store (10,920² × 13 heads × 12 layers), so
`capture_attention` takes explicit `query_indices`; `head_locality` ranks heads by mean
attended great-circle distance, which is a fast way to find the few genuinely
long-range heads worth studying.

**Interventions are where correlational claims become causal.** `influence_matrix`
perturbs one latent channel at a time and measures the response in every output
channel, using the *same* probabilistic sample for control and perturbation so the
difference is not sampling noise. The off-diagonal structure is an empirical read-out
of the variable couplings the projector has learned.

**Ensemble spread has directions.** `noise_sensitivity` returns the leading principal
directions of an ensemble's deviation from its mean — the model's own estimate of the
fastest-growing uncertainty, a learned analogue of singular vectors.

**The sampling path is itself an object of study.** `record_trajectory=True` returns the
full SDE path in latent space, and `Recorder(mode="all")` captures the residual stream
at every solver step. `trajectory_summary` shows where in the integration the sample
actually acquires its structure; a schedule that spends its steps elsewhere is wasting
them.

`examples/latent_analysis.py` runs all of the above on a small model in about a minute.

## Ablating the encoder

Section 5 of the paper argues for the trivial encoder over a learned one. That
comparison is a config change:

```python
cfg.latent.mode = "bilinear"   # the paper's choice
cfg.latent.mode = "area"       # conserves area-weighted global means
cfg.latent.mode = "spectral"   # spherical-harmonic truncation (torch-harmonics)
cfg.latent.mode = "learned"    # trainable correction to the coarse field
```

`learned` deserves a word, because it is not a VAE. A VAE-style encoder mixes all
variables into a common channel space, and the paper attributes two failure modes to
that mixing: a flat high-frequency tail in the latent spectrum, and a loss of temporal
continuity. So the learned mode here keeps channel identity and learns a *correction*
to the coarse-grained field, `z = B(x) + g(B(x), P(x))`, where `P` is a strided
convolution that lets sub-grid structure inform the coarse value. Latent channel *c*
still means physical variable *c*, so residual normalisation, the latent update `z + r`
and every probe keep working. The correction is zero-initialised, so training starts
exactly at the bilinear baseline and any departure is measurable.

A learnable encoder is trained through the projector's reconstruction loss only —
`training_losses` detaches the latent target from the estimator, because a
probabilistic head chasing a target that is free to move does not converge. The paper's
recipe maps onto `--parts`:

```bash
atlas train cfg.yaml --parts encoder projector    # fit the autoencoder first
atlas train cfg.yaml --parts latent               # then the predictive model
```

Judge the result on the paper's own terms with `atlas.probe.diagnostics`:

- `reconstruction_report` — projector vs. naive-upsampling vs. persistence error
- `spectral_ratio` — power by wavenumber; a long flat high-frequency tail is the VAE
  pathology the paper reports
- `latent_continuity` — do temporally adjacent states stay adjacent in latent space?

A full channel-mixing VAE latent is a larger change and is **not** implemented: it
would break the physical channel correspondence that the whole `probe` package and the
`z + r` latent update rely on.

## Differences from the reference implementation

The published inference code (`earth2studio/models/{px,nn}/atlas.py`) depends on
`physicsnemo`, `natten`, `timm` and `einops` and is built to run one released
checkpoint. This implementation is independent and trainable, so it differs in ways
worth knowing:

- **No external attention kernel.** Neighborhood attention is implemented with a
  chunked gather (exact, memory-bounded, tested against unchunked evaluation).
  `natten` will be faster at scale; the semantics match.
- **Pole-consistent padding.** Crossing a pole, the physically adjacent cell is at the
  same latitude ring but 180° away in longitude. `SphericalPad(mode="pole")` does that;
  `"reflect"` reproduces the cheaper variant.
- **Preconditioning lives in the estimator**, not the network, so the same backbone
  weights serve all three objectives.
- **Weights are not interchangeable** with the released checkpoint.
- **Fallbacks are explicit.** Without `torch-harmonics` the spherical transform falls
  back to a 2-D FFT. That is a different operator; it is fine as a regulariser and for
  qualitative spectra, and it is reported by `SphericalTransform.backend`.

## Tests

```bash
pytest tests -q      # 84 tests, ~6 s on CPU
```

Beyond shapes, these check the properties that are easy to get quietly wrong: fair CRPS
is ensemble-size independent, a calibrated ensemble scores SSR ≈ 1, area weights follow
`sin(lat_u) − sin(lat_l)` and sum to one, neighborhood attention really is local,
chunked and unchunked attention agree, the cosine zenith angle has the right seasonal
and diurnal cycles, recording does not perturb the forward pass, and the rollout
advances the latent in latent space rather than re-encoding the decoded state.
