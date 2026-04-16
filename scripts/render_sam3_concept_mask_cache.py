#!/usr/bin/env python3
"""Render overlays from a SAM3 concept pseudo-mask manifest.

This is a small inspection utility for caches produced by
generate_sam3_concept_masks.py. It skips planned/pending records that do not yet
have a mask file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sam3_concept_mask_cache import iter_records, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SAM3 concept-mask cache overlays")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--output_dir", default=None, help="Override preview output directory")
    parser.add_argument("--max_records", type=int, default=32, help="Maximum rendered overlays")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    return parser.parse_args()


def _load_mask(path: Path) -> np.ndarray:
    payload = np.load(path)
    if "mask" not in payload:
        raise KeyError(f"Mask file is missing 'mask': {path}")
    return payload["mask"].astype(bool)


def _render_overlay(image_path: str, mask: np.ndarray, title: str, output_path: Path, alpha: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    if mask.shape[:2] != image_np.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {image_np.shape[:2]} for {image_path}"
        )
    overlay = image_np.copy()
    color = np.array([255, 80, 32], dtype=np.float32)
    overlay[mask] = ((1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", dpi=180, pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    cache_root = manifest_path.parent
    output_root = Path(args.output_dir) if args.output_dir else cache_root / "previews_rendered"

    rendered = 0
    skipped = 0
    for record in iter_records(manifest):
        if rendered >= int(args.max_records):
            break
        mask_rel = record.get("mask_path")
        image_path = record.get("image_path")
        if not mask_rel or not image_path:
            skipped += 1
            continue
        mask_path = cache_root / str(mask_rel)
        if not mask_path.exists():
            skipped += 1
            continue
        mask = _load_mask(mask_path)
        output_path = output_root / str(record.get("preview_path", f"record_{rendered:04d}.png"))
        title = f"{record.get('concept_index')}: {record.get('concept')}"
        _render_overlay(str(image_path), mask, title, output_path, float(args.alpha))
        rendered += 1

    print(json.dumps({"manifest": str(manifest_path), "rendered": rendered, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
