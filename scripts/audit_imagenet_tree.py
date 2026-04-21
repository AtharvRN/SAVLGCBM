#!/usr/bin/env python3
"""Audit an extracted ImageNet train tree for count and zero-byte corruption."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import scipy.io as sio


def expected_counts_from_devkit(devkit_dir: Path) -> dict[str, int]:
    meta = sio.loadmat(devkit_dir / "data" / "meta.mat", squeeze_me=True)["synsets"]
    counts: dict[str, int] = {}
    for syn in meta:
        ilsvrc_id = int(getattr(syn, "ILSVRC2012_ID"))
        if ilsvrc_id > 1000:
            continue
        counts[str(getattr(syn, "WNID"))] = int(getattr(syn, "num_train_images"))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--devkit-dir", required=True)
    parser.add_argument("--show-top", type=int, default=50)
    args = parser.parse_args()

    train_root = Path(args.train_root)
    expected = expected_counts_from_devkit(Path(args.devkit_dir))

    corrupted = []
    mismatched = []
    total_files = 0
    zero_total = 0

    for cls in sorted(expected):
        d = train_root / cls
        actual = 0
        zero = 0
        if d.is_dir():
            with os.scandir(d) as it:
                for ent in it:
                    if ent.is_file(follow_symlinks=False):
                        actual += 1
                        total_files += 1
                        if ent.stat(follow_symlinks=False).st_size == 0:
                            zero += 1
                            zero_total += 1
        if actual != expected[cls]:
            mismatched.append((cls, actual, expected[cls], expected[cls] - actual))
        if zero:
            corrupted.append((cls, zero, actual))

    print(f"classes_total {len(expected)}")
    print(f"files_total {total_files}")
    print(f"zero_byte_total {zero_total}")
    print(f"corrupted_classes {len(corrupted)}")
    print(f"mismatched_classes {len(mismatched)}")

    if corrupted:
        print("== corrupted classes ==")
        for cls, zero, actual in sorted(corrupted, key=lambda x: (-x[1], x[0]))[: args.show_top]:
            print(f"{cls} zero_byte={zero} total={actual}")

    if mismatched:
        print("== mismatched classes ==")
        for cls, actual, exp, missing in sorted(mismatched, key=lambda x: (-x[3], x[0]))[: args.show_top]:
            print(f"{cls} actual={actual} expected={exp} missing={missing}")


if __name__ == "__main__":
    main()
