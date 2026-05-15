#!/usr/bin/env python3
"""Extract fenced YAML config blocks from docs/*.md.

Walks a docs directory, pulls every fenced block whose info-string starts
with ``yaml`` (also matches ``mkdocs-material`` annotated variants such as
``{ .yaml .annotate }``), and writes each that looks like an FME config to
a bucketed output directory.

Buckets are chosen by source filename:
  docs/ace2-inference.md     -> <out>/inference/
  docs/ace2-workflow.md      -> <out>/train/
  docs/ace2-spatial-decomp.md -> <out>/train/

A block must contain at least one of the top-level keys ``experiment_dir``,
``stepper``, ``checkpoint_path``, or ``forward_steps_in_memory`` to be
treated as a config (filters out narrative ``yaml`` snippets).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FENCE_OPEN_RE = re.compile(r"^(?P<indent>[ \t]*)```(?P<info>.*)$")

# Source filename -> bucket directory
BUCKETS = {
    "ace2-inference.md": "inference",
    "ace2-workflow.md": "train",
    "ace2-spatial-decomp.md": "train",
}

CONFIG_HINTS = (
    "experiment_dir",
    "stepper",
    "checkpoint_path",
    "forward_steps_in_memory",
)


def is_yaml_info(info: str) -> bool:
    info = info.strip()
    if info.startswith("yaml"):
        return True
    # mkdocs-material annotated form: "{ .yaml .annotate }" etc.
    if info.startswith("{") and ".yaml" in info:
        return True
    return False


def looks_like_config(block: str) -> bool:
    return any(hint in block for hint in CONFIG_HINTS)


def extract_yaml_blocks(md_text: str) -> list[str]:
    """Return raw YAML contents for every fenced yaml block in ``md_text``.

    Handles fences indented inside ``mkdocs`` admonitions (e.g. ``???
    example``) by capturing the opening indent and stripping it from each
    body line, and by requiring the same indent on the closing fence.
    """
    lines = md_text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        m = FENCE_OPEN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        indent = m.group("indent")
        info = m.group("info")
        # Match a closing fence at the same indent.
        close_re = re.compile(r"^" + re.escape(indent) + r"```\s*$")
        j = i + 1
        body: list[str] = []
        while j < len(lines) and not close_re.match(lines[j]):
            body.append(lines[j])
            j += 1
        if j >= len(lines):
            # Unterminated fence; bail on this one and keep scanning.
            i += 1
            continue
        if is_yaml_info(info):
            stripped = "\n".join(
                (line[len(indent):] if line.startswith(indent) else line)
                for line in body
            )
            blocks.append(stripped)
        i = j + 1
    return blocks


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <docs_dir> <out_dir>", file=sys.stderr)
        return 2
    docs_dir = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    if not docs_dir.is_dir():
        print(f"error: docs dir not found: {docs_dir}", file=sys.stderr)
        return 2

    extracted_any = False
    for md_path in sorted(docs_dir.glob("*.md")):
        bucket = BUCKETS.get(md_path.name)
        if bucket is None:
            continue
        blocks = extract_yaml_blocks(md_path.read_text())
        configs = [b for b in blocks if looks_like_config(b)]
        if not configs:
            print(f"note: {md_path.name}: 0 config-like yaml blocks", file=sys.stderr)
            continue
        bucket_dir = out_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        stem = md_path.stem
        for idx, block in enumerate(configs):
            out_path = bucket_dir / f"{stem}-{idx}.yaml"
            out_path.write_text(block)
            print(f"wrote {out_path}")
            extracted_any = True

    if not extracted_any:
        print("error: no config-like yaml blocks extracted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
