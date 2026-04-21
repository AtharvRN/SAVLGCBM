#!/usr/bin/env python3
"""Build a clean torchvision-style ImageNet root from official archives.

This script uses torchvision's official archive parsers, but generates
`meta.bin` from an extracted devkit directory so the build does not depend on
the devkit tarball MD5 check.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import scipy.io as sio
import torch
from torchvision.datasets.imagenet import META_FILE, parse_train_archive, parse_val_archive


def generate_meta_bin(root: Path, devkit_dir: Path) -> None:
    metafile = devkit_dir / "data" / "meta.mat"
    meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(", ")) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    with (devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt").open() as f:
        val_idcs = [int(x) for x in f.readlines()]
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    torch.save((wnid_to_classes, val_wnids), root / META_FILE)


def maybe_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--train-archive", required=True)
    parser.add_argument("--val-archive", required=True)
    parser.add_argument("--devkit-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    train_dir = root / "train"
    val_dir = root / "val"

    if args.force:
        if train_dir.exists():
            raise RuntimeError(f"Refusing to delete existing {train_dir}; remove it manually first.")
        if val_dir.exists():
            raise RuntimeError(f"Refusing to delete existing {val_dir}; remove it manually first.")

    maybe_symlink(Path(args.train_archive), root / "ILSVRC2012_img_train.tar")
    maybe_symlink(Path(args.val_archive), root / "ILSVRC2012_img_val.tar")

    meta_file = root / META_FILE
    if not meta_file.exists():
        generate_meta_bin(root, Path(args.devkit_dir))
        print(f"generated {meta_file}")
    else:
        print(f"meta already present: {meta_file}")

    if not train_dir.exists():
        print("parse_train_archive start")
        parse_train_archive(root)
        print("parse_train_archive done")
    else:
        print(f"train already present: {train_dir}")

    if not val_dir.exists():
        print("parse_val_archive start")
        parse_val_archive(root)
        print("parse_val_archive done")
    else:
        print(f"val already present: {val_dir}")


if __name__ == "__main__":
    main()
