#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from scipy.io import loadmat


def _load_leaf_id_to_wnid(meta_mat_path: Path) -> dict[int, str]:
    payload = loadmat(meta_mat_path, squeeze_me=True, struct_as_record=False)
    synsets = payload["synsets"]
    id_to_wnid: dict[int, str] = {}
    for syn in synsets:
        ilsvrc_id = int(syn.ILSVRC2012_ID)
        num_children = int(syn.num_children)
        if 1 <= ilsvrc_id <= 1000 and num_children == 0:
            id_to_wnid[ilsvrc_id] = str(syn.WNID)
    if len(id_to_wnid) != 1000:
        raise RuntimeError(f"expected 1000 leaf synsets, found {len(id_to_wnid)}")
    return id_to_wnid


def _load_val_labels(gt_path: Path) -> list[int]:
    labels: list[int] = []
    with gt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                labels.append(int(line))
    if len(labels) != 50000:
        raise RuntimeError(f"expected 50000 val labels, found {len(labels)}")
    return labels


def _reorganize(
    val_flat_dir: Path,
    out_dir: Path,
    labels: list[int],
    id_to_wnid: dict[int, str],
    copy_files: bool,
    overwrite: bool,
) -> tuple[int, int]:
    moved = 0
    skipped = 0
    op = shutil.copy2 if copy_files else shutil.move

    for idx, class_id in enumerate(labels, start=1):
        image_name = f"ILSVRC2012_val_{idx:08d}.JPEG"
        src = val_flat_dir / image_name
        if not src.is_file():
            raise FileNotFoundError(f"missing validation image: {src}")
        wnid = id_to_wnid[class_id]
        dst_dir = out_dir / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / image_name
        if dst.exists():
            if not overwrite:
                skipped += 1
                continue
            dst.unlink()
        op(src, dst)
        moved += 1
    return moved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize flat ImageNet val JPEGs into ImageFolder class directories using the ILSVRC2012 devkit."
    )
    parser.add_argument("--devkit_dir", type=Path, required=True)
    parser.add_argument("--val_flat_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    args = parser.parse_args()

    meta_mat_path = args.devkit_dir / "data" / "meta.mat"
    gt_path = args.devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not meta_mat_path.is_file():
        raise FileNotFoundError(f"missing devkit file: {meta_mat_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"missing devkit file: {gt_path}")
    if not args.val_flat_dir.is_dir():
        raise FileNotFoundError(f"missing val dir: {args.val_flat_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    id_to_wnid = _load_leaf_id_to_wnid(meta_mat_path)
    labels = _load_val_labels(gt_path)
    moved, skipped = _reorganize(
        val_flat_dir=args.val_flat_dir,
        out_dir=args.out_dir,
        labels=labels,
        id_to_wnid=id_to_wnid,
        copy_files=args.copy,
        overwrite=args.overwrite,
    )

    print(f"out_dir={args.out_dir}")
    print(f"moved={moved}")
    print(f"skipped={skipped}")
    print(f"class_dirs={len(list(args.out_dir.iterdir()))}")


if __name__ == "__main__":
    main()
