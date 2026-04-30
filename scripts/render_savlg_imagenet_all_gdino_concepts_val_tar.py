#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
from scripts.eval_savlg_imagenet_standalone_localization import normalize_maps
from scripts.eval_savlg_imagenet_standalone_val_tar import (
    load_run_config,
    load_val_targets,
    resolve_source_run_dir,
)
from scripts.imagenet_annotation_index import (
    build_filename_to_annotation_path,
    load_annotation_payload,
    resolve_val_annotation_dir,
)
from scripts.train_savlg_imagenet_standalone import (
    Config,
    amp_dtype,
    build_gdino_target_sample,
    build_model,
    configure_runtime,
    prepare_images,
)


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render SAVLG ImageNet GT-concept decision panels from val tar and GDINO annotations."
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--devkit_dir", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--annotation_val_root", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_names", default="")
    parser.add_argument("--start_image", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=1)
    parser.add_argument("--concepts_per_figure", type=int, default=5)
    parser.add_argument("--heatmap_threshold", type=float, default=0.9)
    parser.add_argument("--threshold_protocol", default="explicit", choices=["explicit", "meanthr"])
    parser.add_argument("--map_normalization", default="concept_zscore_minmax", choices=["minmax", "sigmoid", "concept_zscore_minmax"])
    parser.add_argument("--threshold_source", default="normalized_map", choices=["normalized_map", "pred_dist"])
    return parser.parse_args()


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    x = x - x.min()
    return x / x.max().clamp_min(1e-6)


def normalize_map_with_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.detach().float()
    if x.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {tuple(x.shape)}")
    return normalize_maps(x.unsqueeze(0), mode).squeeze(0)


def overlay_heatmap(image_np: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    heat = normalize_map(heatmap).cpu().numpy()
    rgba = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip(0.55 * image_np + 0.45 * rgba, 0.0, 1.0)


def add_bbox(ax, bbox: Optional[Sequence[float]], color: str, label: str = "") -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [float(v) for v in bbox]
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0))
    if label:
        ax.text(
            x1,
            max(2.0, y1 - 4.0),
            label,
            color=color,
            fontsize=8,
            weight="bold",
            backgroundcolor=(1, 1, 1, 0.68),
        )


def union_boxes(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    valid = [box for box in boxes if isinstance(box, (list, tuple)) and len(box) == 4]
    if not valid:
        return None
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = max(area_a + area_b - inter, 1e-12)
    return float(inter / union)


def slugify(name: str) -> str:
    chars = []
    for ch in name.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "concept"


def iter_val_images(val_tar: Path, start_image: int, max_images: int) -> Iterable[Tuple[int, str, Image.Image]]:
    seen = 0
    with tarfile.open(val_tar, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            match = VAL_RE.search(image_name)
            if match is None:
                continue
            image_index = int(match.group(1))
            if image_index < start_image:
                continue
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                yield image_index, image_name, image.convert("RGB")
            seen += 1
            if max_images > 0 and seen >= max_images:
                break


def iter_val_images_by_name(val_tar: Path, image_names: Sequence[str]) -> Iterable[Tuple[int, str, Image.Image]]:
    requested = [name.strip() for name in image_names if name.strip()]
    remaining = set(requested)
    with tarfile.open(val_tar, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            if image_name not in remaining:
                continue
            match = VAL_RE.search(image_name)
            if match is None:
                continue
            image_index = int(match.group(1))
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                yield image_index, image_name, image.convert("RGB")
            remaining.remove(image_name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(f"Could not find requested images in tar: {', '.join(sorted(remaining))}")


def load_grouped_annotations(
    annotation_val_dir: Path,
    image_index: int,
    image_name: str,
    concept_to_idx: Dict[str, int],
    filename_to_annotation_path: Optional[Dict[str, Path]],
) -> Dict[str, Dict[str, Any]]:
    payload = load_annotation_payload(
        annotation_val_dir=annotation_val_dir,
        image_index_1based=image_index,
        image_name=image_name,
        filename_to_annotation_path=filename_to_annotation_path,
    )
    entries = payload[1:] if isinstance(payload, list) else payload.get("concepts", [])
    grouped: Dict[str, Dict[str, Any]] = {}
    for ann in entries:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label")
        if isinstance(label, str):
            label = data_utils.canonicalize_concept_label(label)
        if not isinstance(label, str) or label not in concept_to_idx:
            continue
        box = ann.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        rec = grouped.setdefault(label, {"boxes": [], "annotation_logit": float("-inf")})
        rec["boxes"].append([float(v) for v in box])
        rec["annotation_logit"] = max(float(rec["annotation_logit"]), float(ann.get("logit", 0.0)))
    return grouped


def render_image_panels(
    image_np: np.ndarray,
    image_index: int,
    image_name: str,
    pred_name: str,
    target_name: str,
    target_index: int,
    records: Sequence[Dict[str, Any]],
    concepts_per_figure: int,
    output_dir: Path,
) -> None:
    n_pages = max(1, math.ceil(len(records) / concepts_per_figure))
    for page_idx in range(n_pages):
        page_records = records[page_idx * concepts_per_figure : (page_idx + 1) * concepts_per_figure]
        rows = len(page_records)
        fig, axes = plt.subplots(rows, 4, figsize=(14, 4.2 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, 0)
        fig.suptitle(
            f"idx={image_index} pred={pred_name} target={target_index} | GT concepts {page_idx + 1}/{n_pages}",
            fontsize=16,
        )
        for row, rec in enumerate(page_records):
            axes[row, 0].imshow(image_np)
            axes[row, 0].set_title(f"Image\n{rec['concept_name']}")
            add_bbox(axes[row, 0], rec["gt_box"], "#d62728", "GT")
            add_bbox(axes[row, 0], rec["pred_box"], "#2ca02c", "Pred")

            axes[row, 1].imshow(rec["overlay"])
            heatmap_label = "Pred-dist Heatmap" if rec["threshold_source"] == "pred_dist" else "Normalized Heatmap"
            subtitle = (
                f"{heatmap_label}\n"
                f"g={rec['global_logit']:.3f}  s={rec['spatial_logit']:.3f}  f={rec['fused_logit']:.3f}\n"
                f"contrib={rec['contribution']:.3f}"
            )
            if rec["iou"] is not None:
                subtitle += f"  IoU={rec['iou']:.3f}"
            subtitle += f"\nKL={rec['soft_align_kl']:.3f}  thr={rec['threshold_value']:.3f}"
            axes[row, 1].set_title(subtitle)
            add_bbox(axes[row, 1], rec["gt_box"], "#d62728")
            add_bbox(axes[row, 1], rec["pred_box"], "#2ca02c")

            vmax = float(max(rec["p_k"].max().item(), rec["q_k"].max().item(), 1e-6))
            axes[row, 2].imshow(rec["p_k"].cpu().numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
            axes[row, 2].set_title(r"Predicted $p_k$")
            axes[row, 3].imshow(rec["q_k"].cpu().numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
            axes[row, 3].set_title(r"Target $q_k$")
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        out_name = f"{Path(image_name).stem}_all_gt_page_{page_idx + 1:02d}.png"
        fig.savefig(output_dir / out_name, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    source_run_dir = resolve_source_run_dir(artifact_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_values"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_run_config(source_run_dir, argparse.Namespace(device=args.device, batch_size=1))
    cfg.batch_size = 1
    cfg.workers = 0
    cfg.skip_final_layer = True
    configure_runtime(cfg)

    concepts = [line.strip() for line in (source_run_dir / "concepts.txt").read_text().splitlines() if line.strip()]
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    class_names = data_utils.get_classes("imagenet")
    targets = load_val_targets(Path(args.devkit_dir).resolve())

    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()

    norm = torch.load(artifact_dir / "final_layer_normalization.pt", map_location="cpu")
    linear_payload = torch.load(artifact_dir / "final_layer_dense.pt", map_location="cpu")
    feature_mean = norm["mean"].to(cfg.device).float()
    feature_std = norm["std"].to(cfg.device).float().clamp_min(1e-6)
    final_weight = linear_payload["weight"].to(cfg.device).float()
    final_bias = linear_payload["bias"].to(cfg.device).float()

    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)

    summary: Dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "source_run_dir": str(source_run_dir),
        "val_tar": str(Path(args.val_tar).resolve()),
        "annotation_dir": str(annotation_val_dir),
        "annotation_val_root": str(annotation_val_root) if annotation_val_root is not None else "",
        "output_dir": str(output_dir),
        "map_normalization": args.map_normalization,
        "threshold_source": args.threshold_source,
        "threshold_protocol": args.threshold_protocol,
        "heatmap_threshold": float(args.heatmap_threshold),
        "images": [],
    }

    if args.image_names:
        image_iter = iter_val_images_by_name(Path(args.val_tar).resolve(), args.image_names.split(","))
    else:
        image_iter = iter_val_images(Path(args.val_tar).resolve(), int(args.start_image), int(args.max_images))

    for image_index, image_name, image in image_iter:
        grouped = load_grouped_annotations(
            annotation_val_dir=annotation_val_dir,
            image_index=image_index,
            image_name=image_name,
            concept_to_idx=concept_to_idx,
            filename_to_annotation_path=filename_to_annotation_path,
        )
        if not grouped:
            continue

        image_np = np.asarray(image.resize((256, 256)).crop((16, 16, 240, 240)), dtype=np.float32) / 255.0
        tensor = data_utils.get_resnet_imagenet_preprocess()(image)
        batch = prepare_images(tensor.unsqueeze(0), cfg)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(cfg.amp),
                enabled=(str(cfg.device).startswith("cuda") and amp_dtype(cfg.amp) is not None),
            ):
                feats = backbone(batch)
                outputs = head(feats)
                global_logits = outputs["global_logits"].float().squeeze(0)
                spatial_logits = outputs["spatial_logits"].float().squeeze(0)
                fused_logits = outputs["final_logits"].float().squeeze(0)
                proj = ((fused_logits - feature_mean) / feature_std).float()
                class_logits = F.linear(proj.unsqueeze(0), final_weight, final_bias).squeeze(0)
        pred = int(class_logits.argmax().item())
        target = int(targets[image_index - 1])
        contrib = (final_weight[pred] * proj).detach().cpu()

        # Build low-res q_k masks using the same standalone target builder as training.
        annotation_payload = load_annotation_payload(
            annotation_val_dir=annotation_val_dir,
            image_index_1based=image_index,
            image_name=image_name,
            filename_to_annotation_path=filename_to_annotation_path,
        )
        _global_target, concept_ids, masks = build_gdino_target_sample(
            annotation_payload,
            image.size,
            concept_to_idx,
            len(concepts),
            cfg,
        )
        qk_by_cidx = {int(cidx): torch.from_numpy(masks[pos]).float() for pos, cidx in enumerate(concept_ids.tolist())}

        gt_concept_indices = sorted(
            [concept_to_idx[name] for name in grouped.keys() if concept_to_idx[name] in qk_by_cidx],
            key=lambda cidx: float(contrib[cidx].item()),
            reverse=True,
        )
        records: List[Dict[str, Any]] = []
        maps = outputs["spatial_maps"].float().squeeze(0).detach().cpu()
        for cidx in gt_concept_indices:
            concept_name = concepts[cidx]
            native_map = maps[cidx]
            q_k = qk_by_cidx[cidx]
            interp_logits = F.interpolate(
                native_map.unsqueeze(0).unsqueeze(0),
                size=q_k.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            p_k = torch.softmax(interp_logits.flatten(), dim=0).view_as(q_k)
            q_k_dist = q_k / q_k.sum().clamp_min(1e-6)
            pred_log_dist = F.log_softmax(interp_logits.flatten(), dim=0)
            soft_align_kl = float(F.kl_div(pred_log_dist, q_k_dist.flatten(), reduction="sum").item())

            native_up = F.interpolate(
                native_map.unsqueeze(0).unsqueeze(0),
                size=image_np.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            norm_up = normalize_map_with_mode(native_up, args.map_normalization)
            pred_dist_up = torch.softmax(native_up.flatten(), dim=0).view_as(native_up)
            threshold_up = pred_dist_up if args.threshold_source == "pred_dist" else norm_up
            overlay = overlay_heatmap(image_np, threshold_up)
            if args.threshold_protocol == "meanthr":
                thr_value = float(threshold_up.mean().item())
            else:
                thr_value = float(args.heatmap_threshold)
            pred_mask = (threshold_up.cpu().numpy() >= thr_value).astype(np.uint8)
            pred_box = mask_to_bbox(pred_mask)
            gt_box = union_boxes(grouped[concept_name]["boxes"])
            iou = None
            if gt_box is not None and pred_box is not None:
                iou = bbox_iou(pred_box, gt_box)

            rec = {
                "concept_index": int(cidx),
                "concept_name": concept_name,
                "contribution": float(contrib[cidx].item()),
                "global_logit": float(global_logits[cidx].item()),
                "spatial_logit": float(spatial_logits[cidx].item()),
                "fused_logit": float(fused_logits[cidx].item()),
                "annotation_logit": float(grouped[concept_name]["annotation_logit"]),
                "gt_box": gt_box,
                "pred_box": pred_box,
                "iou": iou,
                "soft_align_kl": soft_align_kl,
                "threshold_source": args.threshold_source,
                "threshold_value": thr_value,
                "overlay": overlay,
                "display_map": threshold_up.detach().cpu(),
                "p_k": p_k.detach().cpu(),
                "q_k": q_k_dist.detach().cpu(),
            }
            records.append(rec)

            stem = f"{Path(image_name).stem}_concept_{cidx:04d}_{slugify(concept_name)}"
            np.save(raw_dir / f"{stem}_p_k.npy", rec["p_k"].numpy())
            np.save(raw_dir / f"{stem}_q_k.npy", rec["q_k"].numpy())
            np.save(raw_dir / f"{stem}_display_map.npy", rec["display_map"].numpy())

        render_image_panels(
            image_np=image_np,
            image_index=image_index,
            image_name=image_name,
            pred_name=class_names[pred],
            target_name=class_names[target],
            target_index=target,
            records=records,
            concepts_per_figure=int(args.concepts_per_figure),
            output_dir=output_dir,
        )
        summary["images"].append(
            {
                "image_index": image_index,
                "image_name": image_name,
                "pred_class_index": pred,
                "pred_class": class_names[pred],
                "target_class_index": target,
                "target_class": class_names[target],
                "n_gt_concepts": len(records),
                "concepts": [
                    {
                        "concept_index": rec["concept_index"],
                        "concept_name": rec["concept_name"],
                        "contribution": rec["contribution"],
                        "global_logit": rec["global_logit"],
                        "spatial_logit": rec["spatial_logit"],
                        "fused_logit": rec["fused_logit"],
                        "annotation_logit": rec["annotation_logit"],
                        "gt_box": rec["gt_box"],
                        "pred_box": rec["pred_box"],
                        "iou": rec["iou"],
                        "soft_align_kl": rec["soft_align_kl"],
                        "threshold_value": rec["threshold_value"],
                    }
                    for rec in records
                ],
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
