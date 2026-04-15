import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F

from data import utils as data_utils
from model.utils import get_bbox_iou
from methods.savlg import (
    _annotation_split_dir,
    _rasterize_box_target,
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    create_savlg_splits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)
from scripts.visualize_savlg_examples import (
    _add_bbox,
    _load_args,
    _load_concepts,
    _load_gt_boxes,
    _mask_to_bbox,
    _normalize_map,
    _overlay_heatmap,
    _union_boxes,
)


def _normalize_map_with_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.detach().float()
    mode = str(mode).lower()
    if mode == "minmax":
        return _normalize_map(x)
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "concept_zscore_minmax":
        x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
        return _normalize_map(x)
    raise ValueError(f"Unsupported map_normalization={mode}")


def _slugify(name: str) -> str:
    chars = []
    for ch in name.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "concept"


def _render_for_index(
    load_path: str,
    annotation_dir: str,
    device: str,
    dataset_index: int,
    output_dir: str,
    concepts_per_figure: int,
    heatmap_threshold: float,
    threshold_protocol: str,
    map_normalization: str,
    threshold_source: str,
) -> None:
    args = _load_args(load_path, device, annotation_dir)
    concepts = _load_concepts(load_path, args)
    class_names = data_utils.get_classes(args.dataset)
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}

    _, _, _, _, test_dataset, backbone = create_savlg_splits(args)
    test_raw = data_utils.get_data(f"{args.dataset}_val", None)

    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts)).to(args.device)
    concept_layer.load_state_dict(torch.load(os.path.join(load_path, "concept_layer.pt"), map_location=args.device))
    concept_layer.eval()
    backbone.eval()

    W_g = torch.load(os.path.join(load_path, "W_g.pt"), map_location=args.device).float()
    b_g = torch.load(os.path.join(load_path, "b_g.pt"), map_location=args.device).float()
    mean = torch.load(os.path.join(load_path, "proj_mean.pt"), map_location=args.device).float()
    std = torch.load(os.path.join(load_path, "proj_std.pt"), map_location=args.device).float()

    ann_dir = _annotation_split_dir(args.annotation_dir, args.dataset, "val")

    image_tensor, target = test_dataset[dataset_index]
    raw_img, _ = test_raw[dataset_index]
    image_np = np.asarray(raw_img.convert("RGB"), dtype=np.float32) / 255.0
    img_w, img_h = raw_img.size

    with torch.no_grad():
        feats = forward_savlg_backbone(backbone, image_tensor.unsqueeze(0).to(args.device), args)
        global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
        global_logits, spatial_logits, final_logits = compute_savlg_concept_logits(global_outputs, spatial_maps, args)
        proj = (final_logits - mean) / std
        class_logits = proj @ W_g.t() + b_g
        pred = int(class_logits.argmax(dim=-1).item())
        contrib = (W_g[pred] * proj.squeeze(0)).detach().cpu()

    ann_path = os.path.join(ann_dir, f"{int(dataset_index)}.json")
    ann_payload = json.load(open(ann_path))
    gt_boxes_by_concept = {}
    for ann in ann_payload[1:]:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label")
        if isinstance(label, str):
            label = data_utils.canonicalize_concept_label(label)
        box = ann.get("box")
        if not isinstance(label, str) or label not in concept_to_idx:
            continue
        if not isinstance(box, list) or len(box) != 4:
            continue
        gt_boxes_by_concept.setdefault(label, []).append([float(v) for v in box])

    gt_concept_indices = sorted(
        [concept_to_idx[name] for name in gt_boxes_by_concept.keys()],
        key=lambda cidx: float(contrib[cidx].item()),
        reverse=True,
    )
    records = []
    for cidx in gt_concept_indices:
        concept_name = concepts[cidx]
        native_map = spatial_maps.squeeze(0)[cidx].detach().cpu()
        native_up = F.interpolate(
            native_map.unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        gt_boxes = gt_boxes_by_concept.get(concept_name, [])
        target_mask_np = None
        for box in gt_boxes:
            box_mask = _rasterize_box_target(box=box, image_size=(img_w, img_h), args=args)
            if box_mask is None:
                continue
            if target_mask_np is None:
                target_mask_np = box_mask.copy()
            else:
                target_mask_np = np.maximum(target_mask_np, box_mask)
        if target_mask_np is None:
            continue
        target_mask = torch.from_numpy(target_mask_np).float()
        interp_logits = F.interpolate(
            native_map.unsqueeze(0).unsqueeze(0),
            size=target_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        interp_flat = interp_logits.flatten()
        p_k = torch.softmax(interp_flat, dim=0).view_as(target_mask)
        q_k = target_mask / target_mask.sum().clamp_min(1e-6)
        q_k_flat = q_k.flatten()
        pred_log_dist = F.log_softmax(interp_flat, dim=0)
        soft_align_kl = float(F.kl_div(pred_log_dist, q_k_flat, reduction="sum").item())

        gt_box = _union_boxes(gt_boxes)
        norm_up = _normalize_map_with_mode(native_up, map_normalization)
        pred_dist_up = torch.softmax(native_up.flatten(), dim=0).view_as(native_up)
        threshold_up = pred_dist_up if threshold_source == "pred_dist" else norm_up
        display_up = threshold_up
        overlay = _overlay_heatmap(image_np, display_up)
        if threshold_protocol == "meanthr":
            thr_value = float(threshold_up.mean().item())
        else:
            thr_value = float(heatmap_threshold)
        pred_mask = (threshold_up.cpu().numpy() >= thr_value).astype(np.uint8)
        pred_box = _mask_to_bbox(pred_mask)
        iou = None
        if gt_box is not None and pred_box is not None:
            iou = float(get_bbox_iou(pred_box, gt_box))

        records.append(
            {
                "concept_index": int(cidx),
                "concept_name": concept_name,
                "contribution": float(contrib[cidx].item()),
                "global_logit": float(global_logits.squeeze(0)[cidx].item()),
                "spatial_logit": float(spatial_logits.squeeze(0)[cidx].item()),
                "fused_logit": float(final_logits.squeeze(0)[cidx].item()),
                "target_mask": target_mask,
                "p_k": p_k,
                "q_k": q_k,
                "overlay": overlay,
                "display_map": display_up,
                "gt_box": gt_box,
                "pred_box": pred_box,
                "iou": iou,
                "soft_align_kl": soft_align_kl,
                "threshold_protocol": threshold_protocol,
                "threshold_value": thr_value,
            }
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / "raw_values"
    raw_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_index": int(dataset_index),
        "predicted_class": class_names[pred],
        "predicted_class_index": int(pred),
        "target_class_index": int(target),
        "heatmap_threshold": float(heatmap_threshold),
        "threshold_protocol": threshold_protocol,
        "map_normalization": map_normalization,
        "concepts_per_figure": int(concepts_per_figure),
        "concepts": [
            {
                "concept_index": rec["concept_index"],
                "concept_name": rec["concept_name"],
                "contribution": rec["contribution"],
                "global_logit": rec["global_logit"],
                "spatial_logit": rec["spatial_logit"],
                "fused_logit": rec["fused_logit"],
                "gt_box": rec["gt_box"],
                "pred_box": rec["pred_box"],
                "iou": rec["iou"],
                "soft_align_kl": rec["soft_align_kl"],
                "threshold_protocol": rec["threshold_protocol"],
                "threshold_value": rec["threshold_value"],
                "threshold_source": threshold_source,
            }
            for rec in records
        ],
    }

    for rec in records:
        stem = f"concept_{rec['concept_index']:03d}_{_slugify(rec['concept_name'])}"
        p_k_np = rec["p_k"].cpu().numpy()
        q_k_np = rec["q_k"].cpu().numpy()
        display_np = rec["display_map"].cpu().numpy()
        np.save(raw_path / f"{stem}_p_k.npy", p_k_np)
        np.save(raw_path / f"{stem}_q_k.npy", q_k_np)
        np.save(raw_path / f"{stem}_display_map.npy", display_np)
        np.savetxt(raw_path / f"{stem}_p_k.csv", p_k_np, delimiter=",", fmt="%.9e")
        np.savetxt(raw_path / f"{stem}_q_k.csv", q_k_np, delimiter=",", fmt="%.9e")

    n_pages = math.ceil(len(records) / concepts_per_figure)
    for page_idx in range(n_pages):
        page_records = records[page_idx * concepts_per_figure:(page_idx + 1) * concepts_per_figure]
        rows = len(page_records)
        fig, axes = plt.subplots(rows, 4, figsize=(14, 4.2 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, 0)
        fig.suptitle(
            f"idx={dataset_index} pred={class_names[pred]} target={int(target)} | GT concepts {page_idx + 1}/{n_pages}",
            fontsize=16,
        )
        for row, rec in enumerate(page_records):
            axes[row, 0].imshow(image_np)
            axes[row, 0].set_title(f"Image\n{rec['concept_name']}")
            _add_bbox(axes[row, 0], rec["gt_box"], color="#d62728", label="GT")
            _add_bbox(axes[row, 0], rec["pred_box"], color="#2ca02c", label="Pred")

            axes[row, 1].imshow(rec["overlay"])
            heatmap_label = "Pred-dist Heatmap" if threshold_source == "pred_dist" else "Normalized Heatmap"
            subtitle = (
                f"{heatmap_label}\n"
                f"g={rec['global_logit']:.7f}  s={rec['spatial_logit']:.7f}  f={rec['fused_logit']:.7f}\n"
                f"contrib={rec['contribution']:.7f}"
            )
            if rec["iou"] is not None:
                subtitle += f"  IoU={rec['iou']:.7f}"
            subtitle += f"\nKL={rec['soft_align_kl']:.7f}  thr={rec['threshold_value']:.7f}"
            axes[row, 1].set_title(subtitle)
            _add_bbox(axes[row, 1], rec["gt_box"], color="#d62728")
            _add_bbox(axes[row, 1], rec["pred_box"], color="#2ca02c")

            vmax = float(max(rec["p_k"].max().item(), rec["q_k"].max().item(), 1e-6))
            axes[row, 2].imshow(rec["p_k"].cpu().numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
            axes[row, 2].set_title(r"Predicted $p_k$")
            axes[row, 3].imshow(rec["q_k"].cpu().numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
            axes[row, 3].set_title(r"Target $q_k$")
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        page_name = f"idx_{dataset_index}_all_gt_page_{page_idx + 1:02d}.png"
        fig.savefig(output_path / page_name, dpi=220, bbox_inches="tight")
        plt.close(fig)

    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--dataset_index", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--concepts_per_figure", type=int, default=5)
    parser.add_argument("--heatmap_threshold", type=float, default=0.9)
    parser.add_argument("--threshold_protocol", type=str, default="explicit", choices=["explicit", "meanthr"])
    parser.add_argument("--map_normalization", type=str, default="concept_zscore_minmax")
    parser.add_argument(
        "--threshold_source",
        type=str,
        default="normalized_map",
        choices=["normalized_map", "pred_dist"],
    )
    args = parser.parse_args()
    _render_for_index(
        load_path=args.load_path,
        annotation_dir=args.annotation_dir,
        device=args.device,
        dataset_index=args.dataset_index,
        output_dir=args.output_dir,
        concepts_per_figure=args.concepts_per_figure,
        heatmap_threshold=args.heatmap_threshold,
        threshold_protocol=args.threshold_protocol,
        map_normalization=args.map_normalization,
        threshold_source=args.threshold_source,
    )


if __name__ == "__main__":
    main()
