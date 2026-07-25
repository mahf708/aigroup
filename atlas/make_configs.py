"""Regenerate the YAML files in ``configs/``.

Kept as a script rather than hand-written YAML so the shipped configs are
guaranteed to round-trip through :class:`atlas.spec.AtlasConfig`.

    python make_configs.py
"""

from __future__ import annotations

from pathlib import Path

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

OUT = Path(__file__).parent / "configs"


def write(cfg: AtlasConfig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    cfg.save(path)
    reloaded = AtlasConfig.load(path)
    assert reloaded.to_dict() == cfg.to_dict(), f"{name} does not round-trip"
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")


def era5_base() -> AtlasConfig:
    """The paper's configuration: 2.4B backbone, 1.8B projector, 0.25 deg."""
    return AtlasConfig(
        name="atlas-era5-si",
        variables=VariableSet.era5_atlas(),
        grid=GridSpec.equiangular(721, 1440),
        latent=LatentConfig(shape=(181, 360), mode="bilinear", history=1),
        backbone=BackboneConfig(
            embed_dim_state=2496,
            embed_dim_history=832,
            depth=12,
            num_heads=13,
            patch=(2, 3),
            attention="global",
            pos_embed="sincos2d",
            grad_checkpoint_every=2,
        ),
        projector=ProjectorConfig(
            embed_dim_state=1728,
            embed_dim_latent=768,
            depth=16,
            num_heads=12,
            attention="neighborhood",
            kernel=(3, 3),
            grad_checkpoint_every=2,
        ),
        estimator=EstimatorConfig(
            kind="si",
            options={
                "epsilon": 1.0,
                "sample_method": "heun",
                "steps": 60,
                "noise_kind": "spherical",
            },
        ),
        forcings=ForcingConfig(
            cos_zenith=True,
            static_fields=["surface_geopotential", "land_sea_mask", "sst_mask"],
        ),
        dt_hours=6.0,
    )


def main() -> None:
    si = era5_base()
    write(si, "era5_atlas_si.yaml")

    edm = AtlasConfig.from_dict(si.to_dict())
    edm.name = "atlas-era5-edm"
    edm.estimator = EstimatorConfig(
        kind="edm", options={"sigma_data": 1.0, "steps": 40, "s_churn": 2.5}
    )
    write(edm, "era5_atlas_edm.yaml")

    crps = AtlasConfig.from_dict(si.to_dict())
    crps.name = "atlas-era5-crps"
    crps.backbone.noise_dim = 3328
    crps.estimator = EstimatorConfig(
        kind="crps", options={"noise_dim": 3328, "lambda_spectral": 0.1}
    )
    write(crps, "era5_atlas_crps.yaml")

    # E3SM: 1-degree regridded EAM output.  Cell-centre latitudes (no poles),
    # hybrid sigma levels, area-conservative encoding, and scalar forcings so
    # the same model can be driven with different CO2 pathways.
    e3sm = AtlasConfig(
        name="atlas-e3sm-eam-1deg",
        variables=VariableSet.e3sm_eam(n_levels=24),
        grid=GridSpec.equiangular(180, 360, descending=False, include_poles=False),
        latent=LatentConfig(shape=(45, 90), mode="area", history=1),
        backbone=BackboneConfig(
            embed_dim_state=1024,
            embed_dim_history=384,
            depth=12,
            num_heads=8,
            patch=(3, 3),
            attention="global",
            pos_embed="latlon",
            grad_checkpoint_every=2,
        ),
        projector=ProjectorConfig(
            embed_dim_state=768,
            embed_dim_latent=384,
            depth=10,
            num_heads=8,
            attention="neighborhood",
            kernel=(3, 3),
            pos_embed="latlon",
            grad_checkpoint_every=2,
        ),
        estimator=EstimatorConfig(kind="si", options={"steps": 40, "sample_method": "heun"}),
        forcings=ForcingConfig(
            cos_zenith=True,
            time_of_year=True,
            static_fields=["PHIS", "LANDFRAC", "OCNFRAC"],
            scalar_fields=["co2vmr", "solar_constant"],
        ),
        dt_hours=6.0,
    )
    write(e3sm, "e3sm_eam_1deg.yaml")

    # Small enough to train on a laptop; used by the tests and the demo.
    vs = VariableSet.era5_atlas()
    dev = AtlasConfig(
        name="atlas-dev",
        variables=vs.subset(
            ["u10m", "v10m", "t2m", "msl", "z500", "t850", "q850", "tcwv"]
        ),
        grid=GridSpec.equiangular(49, 96),
        latent=LatentConfig(shape=(13, 24), history=1),
        backbone=BackboneConfig(
            embed_dim_state=128, embed_dim_history=64, depth=6, num_heads=4, patch=(2, 3)
        ),
        projector=ProjectorConfig(
            embed_dim_state=96, embed_dim_latent=32, depth=4, num_heads=4
        ),
        estimator=EstimatorConfig(kind="si", options={"steps": 16}),
        forcings=ForcingConfig(cos_zenith=True, time_of_year=True),
    )
    write(dev, "dev_small.yaml")

    # Same model with a trainable encoder, for the Section 5 ablation.  Train it
    # as `--parts encoder projector`, then `--parts latent`.
    ablation = AtlasConfig.from_dict(dev.to_dict())
    ablation.name = "atlas-dev-learned-encoder"
    ablation.latent = LatentConfig(
        shape=(13, 24), mode="learned", history=1,
        learned_encoder_dim=64, learned_encoder_depth=3,
    )
    write(ablation, "dev_learned_encoder.yaml")


if __name__ == "__main__":
    main()
