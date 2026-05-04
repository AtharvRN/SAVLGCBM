#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_savlg_imagenet_standalone_localization import normalize_maps  # noqa: E402
from scripts.stanford_cars_common import (  # noqa: E402
    annotation_concepts_from_payload,
    load_annotation_payload,
    load_jsonl,
    read_concepts,
)
from scripts.train_savlg_imagenet_standalone import (  # noqa: E402
    Config,
    build_model,
    configure_runtime,
    prepare_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stanford Cars spatial examples.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="outputs/stanford_cars_spatial_examples")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--map_normalization", default="concept_zscore_minmax", choices=["minmax", "sigmoid", "concept_zscore_minmax"])
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--max_concepts_per_image", type=int, default=3)
    return parser.parse_args()


def load_config(artifact_dir: Path, device: str) -> Config:
    payload = json.loads((artifact_dir / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload.setdefault("train_random_transforms", False)
    payload.setdefault("learn_spatial_residual_scale", False)
    payload["device"] = device
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    return Config(**payload)


def preprocess_image(image: Image.Image, input_size: int) -> torch.Tensor:
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(image)


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(artifact_dir, args.device)
    configure_runtime(cfg)
    concepts = read_concepts(artifact_dir / "concepts.txt")
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}

    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head_path = artifact_dir / "concept_head_best.pt"
    if not head_path.is_file():
        head_path = artifact_dir / "concept_head_last.pt"
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing concept head under {artifact_dir}")
    head.load_state_dict(torch.load(head_path, map_location=cfg.device))
    backbone.eval()
    head.eval()

    rows = load_jsonl(Path(args.manifest))
    manifest_items: List[Dict[str, Any]] = []
    written = 0
    for row in rows:
        payload = load_annotation_payload(Path(args.annotation_dir), args.split, str(row["image_id"]))
        annotations = annotation_concepts_from_payload(payload)
        if not annotations:
            continue
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            grouped[str(annotation.get("canonical_label") or annotation.get("label"))].append(annotation)
        selected = []
        for canonical_label, items in grouped.items():
            if canonical_label not in concept_to_idx:
                continue
            best_item = max(items, key=lambda item: float(item.get("score", 0.0)))
            selected.append((canonical_label, best_item))
        selected = sorted(selected, key=lambda item: float(item[1].get("score", 0.0)), reverse=True)
        if not selected:
            continue
        selected = selected[: max(1, int(args.max_concepts_per_image))]

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image_np = image.copy()
            tensor = preprocess_image(image, cfg.input_size).unsqueeze(0)
        inputs = prepare_images(tensor, cfg)
        with torch.no_grad():
            outputs = head(backbone(inputs))
        for concept_name, annotation in selected:
            concept_idx = concept_to_idx[concept_name]
            spatial_map = outputs["spatial_maps"][0, concept_idx]
            heatmap = F.interpolate(
                spatial_map.unsqueeze(0).unsqueeze(0),
                size=(image_np.size[1], image_np.size[0]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            heatmap = normalize_maps(heatmap.unsqueeze(0), args.map_normalization).squeeze(0).cpu().numpy()
            box = annotation.get("box_xyxy") or annotation.get("box")
            score = float(annotation.get("score", 0.0))

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].imshow(image_np)
            if isinstance(box, list) and len(box) == 4:
                x1, y1, x2, y2 = [float(value) for value in box]
                axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
            axes[0].set_title(f"Input + GDINO box\n{concept_name}")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="viridis")
            axes[1].set_title(f"Native map\nscore={score:.3f}")
            axes[1].axis("off")

            axes[2].imshow(image_np)
            axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
            axes[2].set_title(f"Overlay\n{row['class_name']}")
            axes[2].axis("off")

            fig.tight_layout()
            out_name = f"{row['image_id']}_{concept_name.replace(' ', '_')}.png"
            out_path = output_dir / out_name
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            manifest_items.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "concept": concept_name,
                    "score": score,
                    "box_xyxy": box,
                    "output_path": str(out_path),
                }
            )
        written += 1
        if written >= int(args.max_images):
            break

    summary = {
        "artifact_dir": str(artifact_dir),
        "manifest": str(Path(args.manifest).resolve()),
        "annotation_dir": str(Path(args.annotation_dir).resolve()),
        "split": args.split,
        "output_dir": str(output_dir),
        "items": manifest_items,
    }
    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
