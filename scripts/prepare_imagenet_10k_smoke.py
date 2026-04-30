#!/usr/bin/env python
"""Prepare a tiny ImageNet ImageFolder subset with simple pseudo annotations.

This is intended for pipeline smoke tests when the full GDINO annotations are
not available on an interactive pod. It extracts a fixed number of images per
ImageNet class from the official train tar and writes full-image concept boxes
from imagenet_per_class.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

from scripts.train_savlg_imagenet_standalone import canonicalize_concept_label


def load_class_names(path: Path) -> list[str]:
    text = path.read_text(errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tar", required=True)
    parser.add_argument("--output_root", default="/tmp/imagenet_10k_train")
    parser.add_argument("--annotation_root", default="/tmp/imagenet_10k_annotations")
    parser.add_argument("--per_class_json", default="concept_files/imagenet_per_class.json")
    parser.add_argument("--class_file", default="concept_files/imagenet_classes.txt")
    parser.add_argument("--images_per_class", type=int, default=10)
    parser.add_argument("--concepts_per_class", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_tar = Path(args.train_tar)
    output_root = Path(args.output_root)
    annotation_root = Path(args.annotation_root)
    annotation_split = annotation_root / "imagenet_train"
    manifest_path = annotation_root / "manifest.json"

    if output_root.exists() and annotation_split.exists() and manifest_path.exists() and not args.force:
        print(
            json.dumps(
                {
                    "status": "exists",
                    "train_root": str(output_root),
                    "annotation_dir": str(annotation_root),
                    "manifest": str(manifest_path),
                },
                indent=2,
            )
        )
        return 0

    if args.force:
        shutil.rmtree(output_root, ignore_errors=True)
        shutil.rmtree(annotation_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    annotation_split.mkdir(parents=True, exist_ok=True)

    per_class = json.loads(Path(args.per_class_json).read_text())
    class_names = load_class_names(Path(args.class_file))
    total_images = 0
    classes: list[dict[str, object]] = []

    with tarfile.open(train_tar, "r|") as outer:
        class_idx = 0
        for member in outer:
            if not member.isfile() or not member.name.endswith(".tar"):
                continue
            wnid = Path(member.name).stem
            class_dir = output_root / wnid
            class_dir.mkdir(parents=True, exist_ok=True)
            class_name = class_names[class_idx] if class_idx < len(class_names) else wnid
            concepts = per_class.get(class_name, [])
            concepts = [canonicalize_concept_label(str(concept)) for concept in concepts]
            concepts = list(dict.fromkeys([concept for concept in concepts if concept]))[: args.concepts_per_class]
            if not concepts:
                concepts = [canonicalize_concept_label(class_name)]

            class_tar_file = outer.extractfile(member)
            if class_tar_file is None:
                continue
            image_count = 0
            with tarfile.open(fileobj=class_tar_file, mode="r|") as inner:
                for image_member in inner:
                    if image_count >= args.images_per_class:
                        break
                    if not image_member.isfile() or not image_member.name.lower().endswith((".jpeg", ".jpg", ".png")):
                        continue
                    extracted = inner.extractfile(image_member)
                    if extracted is None:
                        continue
                    output_path = class_dir / Path(image_member.name).name
                    output_path.write_bytes(extracted.read())
                    annotation = [{"image": output_path.name, "class_id": class_idx, "wnid": wnid}]
                    annotation.extend(
                        {"label": concept, "logit": 1.0, "box": [0.0, 0.0, 1.0, 1.0]}
                        for concept in concepts
                    )
                    (annotation_split / f"{total_images}.json").write_text(json.dumps(annotation))
                    total_images += 1
                    image_count += 1

            classes.append(
                {
                    "class_idx": class_idx,
                    "wnid": wnid,
                    "class_name": class_name,
                    "concepts": concepts,
                    "n_images": image_count,
                }
            )
            if (class_idx + 1) % 100 == 0:
                print(f"[subset] classes={class_idx + 1} images={total_images}", flush=True)
            class_idx += 1

    manifest = {
        "train_root": str(output_root),
        "annotation_dir": str(annotation_root),
        "n_images": total_images,
        "images_per_class": int(args.images_per_class),
        "classes": classes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": "created", **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
