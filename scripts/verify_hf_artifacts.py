#!/usr/bin/env python3
"""Assert that Hugging Face artifacts referenced by the tutorials still exist.

Uses ``HfApi().list_repo_files`` (no downloads) to enumerate files for each
repo and checks that an expected set of paths is present. Some checks are
exact paths; others require at least one file under a prefix.

Exits non-zero on any missing artifact, printing ``MISSING: <path>`` lines.
Picks up ``HF_TOKEN`` from the environment when the repos are gated.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


@dataclass
class RepoCheck:
    repo_id: str
    # Try these repo types in order; ACE2 repos sit at /<id> (model URL) but
    # may flip to dataset; be defensive.
    repo_types: tuple[str, ...] = ("model", "dataset")
    exact_paths: tuple[str, ...] = ()
    prefix_paths: tuple[str, ...] = ()
    files: list[str] = field(default_factory=list)


CHECKS: list[RepoCheck] = [
    RepoCheck(
        repo_id="allenai/ACE2-EAMv3",
        exact_paths=(
            "ace2_EAMv3_ckpt.tar",
            "initial_conditions/1971010100.nc",
        ),
        prefix_paths=("forcing_data/",),
    ),
    RepoCheck(
        repo_id="allenai/ACE2-ERA5",
        exact_paths=(
            "training_validation_data/normalization/centering.nc",
            "training_validation_data/normalization/scaling-full-field.nc",
            "training_validation_data/normalization/scaling-residual.nc",
        ),
        prefix_paths=("training_validation_data/training_validation/",),
    ),
]


def list_files(api: HfApi, check: RepoCheck) -> list[str]:
    last_err: Exception | None = None
    for repo_type in check.repo_types:
        try:
            return api.list_repo_files(check.repo_id, repo_type=repo_type)
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            last_err = e
            continue
    raise RuntimeError(
        f"could not list files for {check.repo_id} as any of {check.repo_types}: {last_err}"
    )


def main() -> int:
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    missing: list[str] = []
    for check in CHECKS:
        try:
            files = list_files(api, check)
        except Exception as e:
            print(f"MISSING: repo {check.repo_id} not accessible ({e})", file=sys.stderr)
            missing.append(check.repo_id)
            continue
        file_set = set(files)
        print(f"checked {check.repo_id}: {len(file_set)} files listed")
        for path in check.exact_paths:
            if path not in file_set:
                msg = f"MISSING: {check.repo_id}:{path}"
                print(msg, file=sys.stderr)
                missing.append(msg)
        for prefix in check.prefix_paths:
            if not any(f.startswith(prefix) for f in file_set):
                msg = f"MISSING: {check.repo_id}:{prefix}* (no files under prefix)"
                print(msg, file=sys.stderr)
                missing.append(msg)

    if missing:
        print(f"\n{len(missing)} artifact(s) missing", file=sys.stderr)
        return 1
    print("all referenced HuggingFace artifacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
