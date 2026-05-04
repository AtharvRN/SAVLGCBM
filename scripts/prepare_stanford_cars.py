#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stanford_cars_common import (  # noqa: E402
    discover_stanford_cars_root,
    load_stanford_cars_records,
    stratified_split_train_val,
    summarize_manifest,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical Stanford Cars manifests.")
    parser.add_argument("--dataset_root", required=True, help="Root containing the Stanford Cars dataset.")
    parser.add_argument("--output_dir", default="data/stanford_cars", help="Directory for train/val/test manifests.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Validation fraction carved from the official train split.")
    parser.add_argument("--seed", type=int, default=6885, help="Seed for the deterministic train/val split.")
    parser.add_argument(
        "--keep_full_train_manifest",
        action="store_true",
        help="Also save the original official train split before val carving as full_train_manifest.jsonl.",
    )
    return parser.parse_args()


def rewrite_sample_indices(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    rewritten: List[Dict[str, Any]] = []
    for sample_index, row in enumerate(rows):
        item = dict(row)
        item["split"] = split
        item["sample_index"] = sample_index
        rewritten.append(item)
    return rewritten


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_root, layout_kind = discover_stanford_cars_root(dataset_root)
    records = load_stanford_cars_records(dataset_root)
    if "train" not in records or "test" not in records:
        raise RuntimeError("Expected at least train and test splits after parsing Stanford Cars records")

    official_train = rewrite_sample_indices(records["train"], "train")
    train_rows, val_rows = stratified_split_train_val(
        official_train,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    test_rows = rewrite_sample_indices(records["test"], "test")

    train_manifest = output_dir / "train_manifest.jsonl"
    val_manifest = output_dir / "val_manifest.jsonl"
    test_manifest = output_dir / "test_manifest.jsonl"
    write_jsonl(train_manifest, train_rows)
    write_jsonl(val_manifest, val_rows)
    write_jsonl(test_manifest, test_rows)
    if args.keep_full_train_manifest:
        write_jsonl(output_dir / "full_train_manifest.jsonl", official_train)

    class_names = sorted({str(row["class_name"]) for row in train_rows + val_rows + test_rows})
    (output_dir / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    summary: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "layout_root": str(layout_root),
        "layout_kind": layout_kind,
        "val_fraction": float(args.val_fraction),
        "seed": int(args.seed),
        "train": summarize_manifest(train_rows),
        "val": summarize_manifest(val_rows),
        "test": summarize_manifest(test_rows),
        "manifests": {
            "train": str(train_manifest),
            "val": str(val_manifest),
            "test": str(test_manifest),
        },
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
