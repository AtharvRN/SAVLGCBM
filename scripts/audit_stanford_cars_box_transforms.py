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
    annotation_concepts_from_payload,
    load_annotation_payload,
    load_jsonl,
    render_box_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw original-space and crop-space Stanford Cars boxes for audit.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="outputs/stanford_cars_box_audit")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--max_boxes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.manifest))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    written = 0
    for row in rows:
        payload = load_annotation_payload(Path(args.annotation_dir), args.split, str(row["image_id"]))
        concepts = annotation_concepts_from_payload(payload)
        if not concepts:
            continue
        result = render_box_audit(
            image_path=str(row["image_path"]),
            annotations=concepts,
            output_path=output_dir / f"{row['image_id']}.png",
            max_boxes=int(args.max_boxes),
        )
        manifest.append(
            {
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "audit_image": result["output_path"],
                "rendered_boxes": result["rendered_boxes"],
                "labels": result["labels"],
            }
        )
        written += 1
        if written >= int(args.max_images):
            break

    summary = {
        "split": args.split,
        "manifest": str(Path(args.manifest).resolve()),
        "annotation_dir": str(Path(args.annotation_dir).resolve()),
        "output_dir": str(output_dir),
        "written": written,
        "items": manifest,
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
