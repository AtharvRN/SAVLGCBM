#!/usr/bin/env python3
"""Precompute a single part-aligned annotation cache JSON for CUB.

This turns the per-image VLG/SAVLG annotation JSONs into one cache file keyed by
annotation base index. The cache keeps only concepts that map cleanly to CUB
parts via the GPT-generated concept->part mapping and only keeps exact parts
that are visible for that image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_savlg_cub_parts import (
    build_dataset_image_ids,
    canonicalize_concept_label,
    create_savlg_splits,
    load_images_index,
    load_mapping,
    load_part_locs,
    load_parts,
    preload_mapped_gt_concepts,
    resolve_base_index,
)
from scripts.visualize_savlg_examples import _load_args, _load_concepts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--cub_root", type=str, required=True)
    parser.add_argument("--mapping_json", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args_ns = parse_args()
    args = _load_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    _, _, _, _, test_dataset, _backbone = create_savlg_splits(args)
    if args_ns.max_images is not None:
        keep = min(args_ns.max_images, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(keep)))

    cub_root = Path(args_ns.cub_root)
    images_index = load_images_index(cub_root / "images.txt")
    part_names = load_parts(cub_root / "parts" / "parts.txt")
    part_locs = load_part_locs(cub_root / "parts" / "part_locs.txt", part_names)
    concept_to_parts = load_mapping(Path(args_ns.mapping_json))
    concepts = _load_concepts(args_ns.load_path, args)
    concept_to_idx = {canonicalize_concept_label(name): idx for idx, name in enumerate(concepts)}

    ann_split_dir = Path(args.annotation_dir) / f"{args.dataset}_test"
    if not ann_split_dir.is_dir():
        ann_split_dir = Path(args.annotation_dir) / f"{args.dataset}_val"

    dataset_base_indices = [resolve_base_index(test_dataset, i) for i in range(len(test_dataset))]
    image_ids_by_ds_idx = build_dataset_image_ids(test_dataset, images_index)
    image_part_names = {img_id: set(parts.keys()) for img_id, parts in part_locs.items()}
    items_by_base_idx = preload_mapped_gt_concepts(
        ann_split_dir=ann_split_dir,
        dataset_base_indices=dataset_base_indices,
        image_ids_by_ds_idx=image_ids_by_ds_idx,
        image_part_names_by_id=image_part_names,
        concept_to_parts=concept_to_parts,
        concept_to_idx=concept_to_idx,
    )

    ds_to_base_idx = {int(ds_idx): int(resolve_base_index(test_dataset, ds_idx)) for ds_idx in range(len(test_dataset))}
    base_to_image_id = {base_idx: image_ids_by_ds_idx[ds_idx] for ds_idx, base_idx in ds_to_base_idx.items() if ds_idx in image_ids_by_ds_idx}

    payload = {
        "meta": {
            "load_path": args_ns.load_path,
            "annotation_dir": args_ns.annotation_dir,
            "cub_root": args_ns.cub_root,
            "mapping_json": args_ns.mapping_json,
            "dataset": args.dataset,
            "split": "test",
            "num_dataset_images": len(test_dataset),
            "num_cached_images": len(items_by_base_idx),
            "num_cached_instances": int(sum(len(v) for v in items_by_base_idx.values())),
        },
        "items_by_base_idx": {
            str(base_idx): [{"label": label, "exact_parts": exact_parts} for label, exact_parts in items]
            for base_idx, items in sorted(items_by_base_idx.items())
        },
        "image_id_by_base_idx": {str(base_idx): int(image_id) for base_idx, image_id in sorted(base_to_image_id.items())},
    }

    out = Path(args_ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "output": str(out),
                "num_cached_images": payload["meta"]["num_cached_images"],
                "num_cached_instances": payload["meta"]["num_cached_instances"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
