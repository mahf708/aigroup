"""A worked latent-space analysis, end to end, on synthetic data.

Runs in about a minute on a laptop CPU and exercises every question the
:mod:`atlas.probe` package is built to answer::

    python examples/latent_analysis.py --steps 400

The model here is far too small to say anything about the atmosphere.  The
point is the *shape* of the analysis: swap ``build()`` for a loaded checkpoint
and ``SyntheticStore`` for your archive, and the rest carries over unchanged.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from atlas.data import SyntheticStore, WindowDataset, collate
from atlas.model import Atlas
from atlas.probe import (
    LinearProbe,
    Recorder,
    builtin_targets,
    depth_sweep,
    feature_summary,
    head_locality,
    influence_matrix,
    latent_continuity,
    latent_eof,
    nearest_token,
    noise_sensitivity,
    patch_site,
    reconstruction_report,
    teleconnection_map,
    train_sae,
    trajectory_summary,
)
from atlas.probe.probes import pool_tokens
from atlas.probe.sae import SAETrainConfig
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
from atlas.train import TrainConfig, Trainer, fit_normalizers

CHANNELS = ["u10m", "v10m", "t2m", "msl", "z500", "t850", "u850", "v850", "q850", "sst"]


def build() -> Atlas:
    cfg = AtlasConfig(
        name="atlas-example",
        variables=VariableSet.era5_atlas().subset(CHANNELS),
        grid=GridSpec.equiangular(73, 144),
        latent=LatentConfig(shape=(19, 36), mode="bilinear", history=1),
        backbone=BackboneConfig(
            embed_dim_state=128, embed_dim_history=64, depth=6, num_heads=4, patch=(2, 3)
        ),
        projector=ProjectorConfig(
            embed_dim_state=96, embed_dim_latent=48, depth=4, num_heads=4
        ),
        estimator=EstimatorConfig(kind="si", options={"steps": 12}),
        forcings=ForcingConfig(cos_zenith=True, time_of_year=True),
    )
    return Atlas(cfg)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-times", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.manual_seed(0)

    model = build()
    cfg = model.cfg
    store = SyntheticStore(cfg.variables, cfg.grid, n_times=args.n_times, seed=11)
    dataset = WindowDataset(store, history=1)

    section("0. train a small model so the diagnostics have something to look at")
    fit_normalizers(model, dataset, max_samples=128)
    tcfg = TrainConfig(
        lr=6e-4, warmup_steps=20, cycle_steps=args.steps, cycles=1, batch_size=8,
        log_every=max(args.steps // 4, 1), checkpoint_every=0,
        out_dir="runs/example", device=args.device, amp_dtype=None,
    )
    Trainer(model, tcfg).fit(dataset, max_steps=args.steps)
    model.eval()

    n_eval = 64
    batch = collate([dataset[i] for i in range(n_eval)])
    x, times = batch["window"], batch["time"]
    x_cur, x_next = x[:, -1], batch["next"]
    z0, hist = model.encode(x_cur), model.encode(x[:, 0])
    gh = model.backbone.grid_hw

    # ------------------------------------------------------------------
    section("1. is the latent space itself well behaved?")
    # The paper's case for bilinear downsampling rests on two measurable
    # properties: low reconstruction error and temporal continuity.
    rep = reconstruction_report(model, x_cur, x_next, times)
    print(f"{'channel':<8} {'projector':>10} {'bilinear':>10} {'persistence':>12}")
    for i, name in enumerate(rep["channel_names"][:5]):
        print(
            f"{name:<8} {float(rep['projector_rmse'][i]):>10.4f} "
            f"{float(rep['bilinear_rmse'][i]):>10.4f} "
            f"{float(rep['persistence_rmse'][i]):>12.4f}"
        )
    print("  (projector should sit well below both baselines once trained)")

    series = torch.stack(
        [model.encode(torch.from_numpy(store.read(i))[None])[0] for i in range(64)]
    )
    cont = latent_continuity(series)
    print("\nlatent continuity:", {k: round(v, 4) for k, v in cont.items()})
    print("  step_ratio << 1 means temporally adjacent states stay adjacent in latent space")

    eof = latent_eof(series, k=4, area=model.latent_area)
    print("EOF variance fractions:", [round(float(v), 3) for v in eof.variance_fraction])

    # ------------------------------------------------------------------
    section("2. where does the ensemble's uncertainty live?")
    ns = noise_sensitivity(model, z0[:8], hist[:8], n_ensemble=16)
    spread = ns["channel_spread"]
    order = spread.argsort(descending=True)
    print("most uncertain latent channels:")
    for i in order[:5].tolist():
        print(f"  {cfg.variables[i].name:<8} spread={float(spread[i]):.4f}")
    print("leading spread-mode variance:", [round(float(v), 3) for v in ns["mode_variance"][:4]])

    traj = model.predict_latent(z0[:4], hist[:4], record_trajectory=True)
    ts = trajectory_summary(traj.trajectory)
    frac = ts["cumulative_fraction"]
    half = int((frac < 0.5).sum())
    print(
        f"\nSDE path: {len(frac)} steps, half the total displacement is reached by step {half}"
    )
    print("  (a schedule that front-loads displacement is spending its steps in the wrong place)")

    # ------------------------------------------------------------------
    section("3. what is in the residual stream?")
    rec = Recorder(model, sites=["backbone.blocks.*"], mode="last")
    rec.capture_attention(
        "backbone.blocks.*.attn",
        query_indices=[nearest_token(cfg.latent_grid, gh, 45.0, 300.0)],
        stats=True,
    )
    with rec:
        model.step(x_cur, z0, hist, times=times)

    acts = {n: rec[n] for n in rec.keys()}
    print("recorded:", list(acts), "->", tuple(acts[rec.keys()[0]].shape))

    targets = builtin_targets(cfg.variables, cfg.grid)
    print("\nlinear decodability by depth (validation R^2):")
    header = f"{'layer':<20}" + "".join(f"{k:>16}" for k in targets)
    print(header)
    ys = torch.stack([fn(x_cur) for fn in targets.values()], dim=1)
    sweep = depth_sweep(acts, ys, pooling="mean", n_components=32, val_fraction=0.3)
    for name in rec.keys():
        row = "".join(f"{float(v):>16.3f}" for v in sweep[name].r2)
        print(f"{name:<20}{row}")
    print("  a target that only becomes decodable deeper in the stack is being computed,")
    print("  not merely copied from the input encoding.")

    # ------------------------------------------------------------------
    section("4. what does a location attend to?")
    last = rec.keys()[-1]
    for name in rec.keys():
        cap = rec.attention(name)
        d = head_locality(
            cap.weights,
            cfg.latent_grid,
            gh,
            torch.tensor([nearest_token(cfg.latent_grid, gh, 45.0, 300.0)]),
        ).mean(0)
        ent = cap.entropy.mean(0)
        print(
            f"{name:<20} mean attended distance per head (km): "
            + " ".join(f"{float(v):7.0f}" for v in d)
            + f"   entropy: {float(ent.mean()):.2f}"
        )
    tmap = teleconnection_map(rec.attention(last).weights, gh, query=0)
    tlat, tlon = model.backbone.token_coords(cfg.latent_grid)
    peak = np.unravel_index(int(np.argmax(tmap)), tmap.shape)
    print(
        f"\nstrongest source for a query at (45N, 60W) in {last}: "
        f"({tlat[peak[0]]:.0f} lat, {tlon[peak[1]]:.0f} lon), weight {tmap[peak]:.3f}"
    )

    # ------------------------------------------------------------------
    section("5. unsupervised features (sparse autoencoder)")
    a = acts[last]
    sae, hist_sae = train_sae(
        a, SAETrainConfig(n_features=256, k=8, epochs=6, batch_size=1024)
    )
    print(
        f"explained variance {hist_sae['explained_variance'][-1]:.3f}, "
        f"dead features {int(hist_sae['dead'][-1])}/256"
    )
    print(f"{'feature':>8} {'fire rate':>10} {'concentration':>14} {'peak lat/lon':>16}")
    for s in feature_summary(sae, a, gh, top=5):
        loc = f"{tlat[s['peak_row']]:.0f}/{tlon[s['peak_col']]:.0f}"
        print(
            f"{int(s['feature']):>8} {s['fire_rate']:>10.3f} "
            f"{s['concentration']:>14.3f} {loc:>16}"
        )
    print("  concentration near 1 => a geographically sharp feature worth mapping")

    # ------------------------------------------------------------------
    section("6. what does the forecast causally depend on?")
    mat, names = influence_matrix(
        model, x_cur[:8], z0[:8], hist[:8], channels=CHANNELS[:5],
        times=times[:8], generator=torch.Generator().manual_seed(0),
    )
    print("relative response of each output channel to a 1-sigma latent perturbation")
    print(f"{'perturbed':<10}" + "".join(f"{n:>9}" for n in cfg.variables.names))
    for i, src in enumerate(names):
        print(f"{src:<10}" + "".join(f"{float(v):>9.3f}" for v in mat[i]))
    print("  off-diagonal structure = variable couplings the projector has learned")

    g = torch.Generator().manual_seed(3)
    base = model.step(x_cur[:8], z0[:8], hist[:8], times=times[:8], generator=g).state
    print("\nablating one block at a time (RMS change in the forecast):")
    for name in rec.keys():
        g2 = torch.Generator().manual_seed(3)
        with patch_site(model, name, lambda h: torch.zeros_like(h)):
            got = model.step(x_cur[:8], z0[:8], hist[:8], times=times[:8], generator=g2).state
        print(f"  {name:<20} {float((got - base).pow(2).mean().sqrt()):.4f}")

    # ------------------------------------------------------------------
    section("7. probing a single named diagnostic")
    probe = LinearProbe()
    feats = pool_tokens(acts[last], "mean", n_components=32)
    res = probe.fit(feats, targets["nino34"](x_cur).unsqueeze(1))
    print(f"nino34 from {last}: {res}")
    print("\ndone.")


if __name__ == "__main__":
    main()
