#!/usr/bin/env python3
"""Render qualitative SAVLG part-localization examples on CUB.

This mirrors the existing SAVLG qualitative plots, but uses the GPT-mapped
concept->part supervision and official CUB part points instead of concept boxes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.savlg import build_savlg_concept_layer, create_savlg_splits, forward_savlg_backbone, forward_savlg_concept_layer
from scripts.evaluate_savlg_cub_parts import (
    IndexedDataset,
    build_dataset_image_ids,
    canonicalize_concept_label,
    disk_mask,
    load_images_index,
    load_mapping,
    load_part_locs,
    load_parts,
    merged_disk_mask,
    min_normalized_distance,
    point_in_any_disk,
    resolve_base_index,
    sample_path_from_dataset,
)
from scripts.evaluate_savlg_native_maps import _normalize_map_with_mode, _parse_csv_floats
from scripts.visualize_savlg_examples import _load_args, _load_concepts, _normalize_map, _overlay_heatmap


def _slugify(name: str) -> str:
    chars = []
    for ch in name.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "concept"


def _add_points(ax, points: Sequence[Tuple[float, float]], color: str = "#d62728", label: str | None = None) -> None:
    for i, (x, y) in enumerate(points):
        ax.scatter([x], [y], s=24, c=color, marker="o")
        if label and i == 0:
            ax.text(float(x) + 2.0, float(y) - 2.0, label, color=color, fontsize=10, weight="bold",
                    backgroundcolor=(1, 1, 1, 0.65))


def _add_disks(ax, points: Sequence[Tuple[float, float]], radius_px: float, color: str = "#d62728") -> None:
    for x, y in points:
        ax.add_patch(Circle((float(x), float(y)), radius=float(radius_px), fill=False, edgecolor=color, linewidth=2.0))


def _add_pred_point(ax, px: int, py: int, color: str = "#2ca02c", label: str | None = None) -> None:
    ax.scatter([px], [py], s=36, c=color, marker="x", linewidths=2.0)
    if label:
        ax.text(float(px) + 2.0, float(py) - 2.0, label, color=color, fontsize=10, weight="bold",
                backgroundcolor=(1, 1, 1, 0.65))


def _load_cache_or_scan(
    annotation_cache_json: str | None,
    test_dataset,
    args,
    cub_root: Path,
    mapping_json: Path,
    concepts: Sequence[str],
) -> Tuple[Dict[int, List[Tuple[str, List[str]]]], Dict[int, int], Dict[int, Dict[str, Tuple[float, float]]]]:
    images_index = load_images_index(cub_root / "images.txt")
    part_names = load_parts(cub_root / "parts" / "parts.txt")
    part_locs = load_part_locs(cub_root / "parts" / "part_locs.txt", part_names)
    concept_to_parts = load_mapping(mapping_json)
    concept_to_idx = {canonicalize_concept_label(name): idx for idx, name in enumerate(concepts)}
    del concept_to_idx  # kept for parity with evaluator behavior

    indexed = IndexedDataset(test_dataset)
    image_ids_by_ds_idx = build_dataset_image_ids(indexed.base, images_index)

    if annotation_cache_json:
        payload = json.loads(Path(annotation_cache_json).read_text())
        mapped = {
            int(base_idx): [(str(item["label"]), list(item["exact_parts"])) for item in items]
            for base_idx, items in (payload.get("items_by_base_idx") or {}).items()
        }
        return mapped, image_ids_by_ds_idx, part_locs

    raise RuntimeError("render_savlg_cub_parts.py currently requires --annotation_cache_json for speed and consistency")


def _render_for_index(
    load_path: str,
    annotation_dir: str,
    annotation_cache_json: str,
    cub_root: str,
    mapping_json: str,
    device: str,
    dataset_index: int,
    output_dir: str,
    concepts_per_figure: int,
    threshold: float,
    map_normalization: str,
    point_source: str,
    radius_fracs: Sequence[float],
    disk_radius_frac: float,
) -> None:
    args = _load_args(load_path, device, annotation_dir)
    _, _, _, _, test_dataset, backbone = create_savlg_splits(args)
    test_raw = test_dataset
    while hasattr(test_raw, "base_dataset"):
        test_raw = test_raw.base_dataset
    concepts = _load_concepts(load_path, args)
    class_names = __import__("data.utils", fromlist=["get_classes"]).get_classes(args.dataset)
    concept_to_idx = {canonicalize_concept_label(name): idx for idx, name in enumerate(concepts)}

    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts)).to(args.device)
    concept_layer.load_state_dict(torch.load(os.path.join(load_path, "concept_layer.pt"), map_location=args.device))
    concept_layer.eval()
    backbone.eval()

    mapped_gt_concepts_by_base_idx, image_ids_by_ds_idx, part_locs = _load_cache_or_scan(
        annotation_cache_json=annotation_cache_json,
        test_dataset=test_dataset,
        args=args,
        cub_root=Path(cub_root),
        mapping_json=Path(mapping_json),
        concepts=concepts,
    )

    image_tensor, target = test_dataset[dataset_index]
    raw_img, _ = test_raw[dataset_index]
    image_np = np.asarray(raw_img.convert("RGB"), dtype=np.float32) / 255.0
    img_w, img_h = raw_img.size
    diag = math.sqrt(float(img_h * img_h + img_w * img_w))
    disk_radius_px = float(disk_radius_frac) * diag

    with torch.no_grad():
        feats = forward_savlg_backbone(backbone, image_tensor.unsqueeze(0).to(args.device), args)
        global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)

    base_idx = resolve_base_index(test_dataset, dataset_index)
    image_id = image_ids_by_ds_idx[int(dataset_index)]
    image_parts = part_locs.get(image_id, {})
    gt_concepts = mapped_gt_concepts_by_base_idx.get(int(base_idx), [])
    gt_concepts = [(label, exact_parts) for label, exact_parts in gt_concepts if all(p in image_parts for p in exact_parts)]

    records = []
    for label, exact_parts in gt_concepts:
        cidx = concept_to_idx[label]
        native_map = spatial_maps.squeeze(0)[cidx].detach().cpu()
        native_up = F.interpolate(
            native_map.unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        if point_source == "pred_dist":
            score_map = torch.softmax(native_up.flatten(), dim=0).view_as(native_up)
        else:
            score_map = _normalize_map_with_mode(native_up, map_normalization)
        overlay = _overlay_heatmap(image_np, score_map if point_source == "pred_dist" else score_map)
        pred_mask = (score_map.cpu().numpy() >= float(threshold)).astype(np.uint8)
        pred_mask_t = torch.from_numpy(pred_mask.astype(np.bool_))

        points = [image_parts[p] for p in exact_parts]
        gt_mask = merged_disk_mask(img_h, img_w, points, disk_radius_px)
        p_flat = score_map.flatten()
        argmax_flat = int(p_flat.argmax().item())
        py = int(argmax_flat // score_map.shape[-1])
        px = int(argmax_flat % score_map.shape[-1])

        point_hits = {str(r): point_in_any_disk(px, py, points, float(r) * diag) for r in radius_fracs}
        point_in_mask = bool((pred_mask_t & gt_mask).any().item())
        inter = int((pred_mask_t & gt_mask).sum().item())
        fp = int((pred_mask_t & ~gt_mask).sum().item())
        fn = int((~pred_mask_t & gt_mask).sum().item())
        precision = float(inter / max(inter + fp, 1))
        recall = float(inter / max(inter + fn, 1))
        f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))

        records.append(
            {
                "concept_index": int(cidx),
                "concept_name": label,
                "exact_parts": exact_parts,
                "score_map": score_map,
                "target_mask": gt_mask.float(),
                "overlay": overlay,
                "pred_mask": pred_mask_t,
                "points": points,
                "pred_point": [px, py],
                "mean_normalized_distance": float(min_normalized_distance(px, py, points, diag)),
                "point_hits": point_hits,
                "point_in_mask": point_in_mask,
                "mask_iou": float((inter / max(int((pred_mask_t | gt_mask).sum().item()), 1))),
                "dice": float((2.0 * inter) / max(int(pred_mask_t.sum().item() + gt_mask.sum().item()), 1)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "threshold": float(threshold),
            }
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / "raw_values"
    raw_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_index": int(dataset_index),
        "predicted_class_index": int(target),
        "predicted_class": class_names[int(target)],
        "threshold": float(threshold),
        "point_source": point_source,
        "map_normalization": map_normalization,
        "disk_radius_frac": float(disk_radius_frac),
        "concepts": [],
    }

    for rec in records:
        stem = f"concept_{rec['concept_index']:03d}_{_slugify(rec['concept_name'])}"
        np.save(raw_path / f"{stem}_score_map.npy", rec["score_map"].cpu().numpy())
        np.save(raw_path / f"{stem}_target_mask.npy", rec["target_mask"].cpu().numpy())
        np.save(raw_path / f"{stem}_pred_mask.npy", rec["pred_mask"].cpu().numpy())
        np.savetxt(raw_path / f"{stem}_score_map.csv", rec["score_map"].cpu().numpy(), delimiter=",", fmt="%.9e")
        np.savetxt(raw_path / f"{stem}_target_mask.csv", rec["target_mask"].cpu().numpy(), delimiter=",", fmt="%.9e")
        np.savetxt(raw_path / f"{stem}_pred_mask.csv", rec["pred_mask"].cpu().numpy().astype(np.uint8), delimiter=",", fmt="%d")
        manifest["concepts"].append(
            {
                "concept_index": rec["concept_index"],
                "concept_name": rec["concept_name"],
                "exact_parts": rec["exact_parts"],
                "pred_point": rec["pred_point"],
                "mean_normalized_distance": rec["mean_normalized_distance"],
                "point_hits": rec["point_hits"],
                "point_in_mask": rec["point_in_mask"],
                "mask_iou": rec["mask_iou"],
                "dice": rec["dice"],
                "precision": rec["precision"],
                "recall": rec["recall"],
                "f1": rec["f1"],
            }
        )

    n_pages = math.ceil(len(records) / concepts_per_figure) if records else 1
    for page_idx in range(n_pages):
        page_records = records[page_idx * concepts_per_figure:(page_idx + 1) * concepts_per_figure]
        if not page_records:
            break
        rows = len(page_records)
        fig, axes = plt.subplots(rows, 4, figsize=(14, 4.2 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, 0)
        fig.suptitle(
            f"idx={dataset_index} class={class_names[int(target)]} | part-localized GT concepts {page_idx + 1}/{n_pages}",
            fontsize=16,
        )
        for row, rec in enumerate(page_records):
            axes[row, 0].imshow(image_np)
            axes[row, 0].set_title(f"Image\n{rec['concept_name']}")
            _add_disks(axes[row, 0], rec["points"], disk_radius_px, color="#d62728")
            _add_points(axes[row, 0], rec["points"], color="#d62728", label="GT")
            _add_pred_point(axes[row, 0], rec["pred_point"][0], rec["pred_point"][1], color="#2ca02c", label="Pred")

            axes[row, 1].imshow(rec["overlay"])
            axes[row, 1].contour(rec["pred_mask"].cpu().numpy().astype(np.float32), levels=[0.5], colors=["#2ca02c"], linewidths=1.8)
            _add_disks(axes[row, 1], rec["points"], disk_radius_px, color="#d62728")
            _add_pred_point(axes[row, 1], rec["pred_point"][0], rec["pred_point"][1], color="#2ca02c")
            subtitle = (
                "Part heatmap\n"
                f"dist={rec['mean_normalized_distance']:.4f} "
                f"maskIoU={rec['mask_iou']:.4f}\n"
                f"Dice={rec['dice']:.4f} "
                f"pointInMask={float(rec['point_in_mask']):.4f}\n"
                f"thr={rec['threshold']:.4f}"
            )
            axes[row, 1].set_title(subtitle)

            vmax = float(max(rec["score_map"].max().item(), rec["target_mask"].max().item(), 1e-6))
            axes[row, 2].imshow(rec["score_map"].cpu().numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
            axes[row, 2].set_title("Predicted score map")
            axes[row, 3].imshow(rec["target_mask"].cpu().numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
            axes[row, 3].set_title("Target part disk mask")
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        page_name = f"idx_{dataset_index}_parts_page_{page_idx + 1:02d}.png"
        fig.savefig(output_path / page_name, dpi=220, bbox_inches="tight")
        plt.close(fig)

    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--annotation_cache_json", type=str, required=True)
    parser.add_argument("--cub_root", type=str, required=True)
    parser.add_argument("--mapping_json", type=str, required=True)
    parser.add_argument("--dataset_index", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--concepts_per_figure", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--map_normalization", type=str, default="concept_zscore_minmax")
    parser.add_argument("--point_source", type=str, default="normalized_map", choices=["normalized_map", "pred_dist"])
    parser.add_argument("--radius_fracs", type=str, default="0.01,0.02,0.05,0.1")
    parser.add_argument("--disk_radius_frac", type=float, default=0.03)
    args = parser.parse_args()
    _render_for_index(
        load_path=args.load_path,
        annotation_dir=args.annotation_dir,
        annotation_cache_json=args.annotation_cache_json,
        cub_root=args.cub_root,
        mapping_json=args.mapping_json,
        device=args.device,
        dataset_index=args.dataset_index,
        output_dir=args.output_dir,
        concepts_per_figure=args.concepts_per_figure,
        threshold=args.threshold,
        map_normalization=args.map_normalization,
        point_source=args.point_source,
        radius_fracs=_parse_csv_floats(args.radius_fracs),
        disk_radius_frac=args.disk_radius_frac,
    )


if __name__ == "__main__":
    main()
