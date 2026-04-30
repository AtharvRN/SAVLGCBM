import argparse
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from model.cbm import Backbone, ConceptLayer
from scripts.imagenet_annotation_index import (
    build_filename_to_annotation_path,
    load_annotation_payload,
    resolve_val_annotation_dir,
)


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render VLG-CBM ImageNet Grad-CAM concept maps from the official val tar."
    )
    parser.add_argument("--load_dir", required=True, help="VLG-CBM directory with args.txt, cbl.pt, concepts.txt.")
    parser.add_argument("--val_tar", required=True)
    parser.add_argument(
        "--annotation_dir",
        required=True,
        help="Directory containing imagenet_val/*.json, or the imagenet_val directory itself.",
    )
    parser.add_argument(
        "--annotation_val_root",
        default="",
        help="Optional reorganized ImageNet val ImageFolder root used when annotations are keyed by ImageFolder dataset index.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=24)
    parser.add_argument("--start_image", type=int, default=1, help="1-based ImageNet val index to start scanning.")
    parser.add_argument("--concepts_per_image", type=int, default=3)
    parser.add_argument("--page_size", type=int, default=6)
    parser.add_argument("--cam_threshold", type=float, default=0.5)
    parser.add_argument("--select_by", default="vlg_logit", choices=["vlg_logit", "annotation_logit"])
    parser.add_argument("--target_concepts", default="", help="Optional comma-separated concept names to restrict rendering.")
    return parser.parse_args()


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    x = x - x.min()
    return x / x.max().clamp_min(1e-6)


def overlay_heatmap(image_np: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    heat = normalize_map(heatmap).cpu().numpy()
    rgba = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip(0.55 * image_np + 0.45 * rgba, 0.0, 1.0)


def add_box(ax, box: Sequence[float], color: str, label: str = "") -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
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


def mask_to_box(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_iou(box_a: Optional[Sequence[float]], box_b: Optional[Sequence[float]]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def load_concepts(load_dir: Path) -> List[str]:
    return [data_utils.canonicalize_concept_label(line.strip()) for line in (load_dir / "concepts.txt").read_text().splitlines() if line.strip()]


def load_annotations(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
    concept_to_idx: Dict[str, int],
    filename_to_annotation_path: Dict[str, Path] | None = None,
) -> Dict[str, Dict[str, Any]]:
    payload = load_annotation_payload(
        annotation_val_dir=annotation_val_dir,
        image_index_1based=image_index_1based,
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


def gradcam_for_concepts(
    backbone: Backbone,
    cbl: ConceptLayer,
    image_tensor: torch.Tensor,
    concept_indices: Sequence[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = image_tensor.unsqueeze(0).to(device)
    backbone.zero_grad(set_to_none=True)
    cbl.zero_grad(set_to_none=True)
    _ = backbone.backbone(batch)
    feats = backbone.feature_vals[batch.device]
    pooled = feats.mean(dim=(2, 3))
    logits = cbl(pooled).float().squeeze(0)
    cams: List[torch.Tensor] = []
    for pos, concept_idx in enumerate(concept_indices):
        grad = torch.autograd.grad(
            logits[int(concept_idx)],
            feats,
            retain_graph=pos < len(concept_indices) - 1,
            create_graph=False,
        )[0]
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * feats).sum(dim=1, keepdim=False)).squeeze(0)
        cams.append(cam.detach())
    return torch.stack(cams, dim=0), logits.detach()


def iter_val_images(val_tar: Path, start_image: int):
    with tarfile.open(val_tar, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            match = VAL_RE.search(Path(member.name).name)
            if match is None:
                continue
            image_index = int(match.group(1))
            if image_index < int(start_image):
                continue
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                yield image_index, member.name, image.convert("RGB")


def save_page(records: Sequence[Dict[str, Any]], output_dir: Path, page_index: int) -> Path:
    rows = len(records)
    fig, axes = plt.subplots(rows, 3, figsize=(12, max(3.2 * rows, 3.2)), squeeze=False)
    for row, rec in enumerate(records):
        image_np = rec["image_np"]
        overlay = rec["overlay"]
        cam = rec["cam"]
        gt_box = rec["gt_box"]
        pred_box = rec["pred_box"]
        axes[row, 0].imshow(image_np)
        add_box(axes[row, 0], gt_box, "#d62728", "GT")
        axes[row, 0].set_title(f"{rec['image_name']}\n{rec['concept']}")
        axes[row, 1].imshow(overlay)
        add_box(axes[row, 1], gt_box, "#d62728", "GT")
        if pred_box is not None:
            add_box(axes[row, 1], pred_box, "#2ca02c", "CAM")
        axes[row, 1].set_title(f"Grad-CAM overlay\nIoU={rec['box_iou']:.3f}, logit={rec['vlg_logit']:.3f}")
        im = axes[row, 2].imshow(cam, cmap="jet", vmin=0.0, vmax=1.0)
        add_box(axes[row, 2], gt_box, "#d62728", "GT")
        if pred_box is not None:
            add_box(axes[row, 2], pred_box, "#2ca02c", "CAM")
        axes[row, 2].set_title("Normalized CAM")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
        for col in range(3):
            axes[row, col].axis("off")
    fig.tight_layout()
    path = output_dir / f"vlgcbm_gradcam_page_{page_index:03d}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    load_dir = Path(args.load_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
    with (load_dir / "args.txt").open("r", encoding="utf-8") as handle:
        model_args = json.load(handle)
    backbone = Backbone(model_args["backbone"], model_args["feature_layer"], args.device).eval()
    cbl = ConceptLayer.from_pretrained(str(load_dir), device=args.device).eval()
    concepts = load_concepts(load_dir)
    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    target_concepts = {
        data_utils.canonicalize_concept_label(item.strip())
        for item in args.target_concepts.split(",")
        if item.strip()
    }

    records: List[Dict[str, Any]] = []
    page_records: List[Dict[str, Any]] = []
    page_paths: List[str] = []
    page_index = 0
    images_used = 0
    for image_index, image_name, image in iter_val_images(Path(args.val_tar).resolve(), args.start_image):
        grouped = load_annotations(
            annotation_val_dir,
            image_index,
            Path(image_name).name,
            concept_to_idx,
            filename_to_annotation_path=filename_to_annotation_path,
        )
        if target_concepts:
            grouped = {concept: rec for concept, rec in grouped.items() if concept in target_concepts}
        if not grouped:
            continue
        image_tensor = backbone.preprocess(image)
        concept_names = list(grouped.keys())
        concept_indices = [concept_to_idx[name] for name in concept_names]
        cams, logits = gradcam_for_concepts(backbone, cbl, image_tensor, concept_indices, args.device)
        if args.select_by == "annotation_logit":
            order = sorted(range(len(concept_names)), key=lambda i: float(grouped[concept_names[i]]["annotation_logit"]), reverse=True)
        else:
            order = sorted(range(len(concept_names)), key=lambda i: float(logits[concept_indices[i]].item()), reverse=True)
        order = order[: max(int(args.concepts_per_image), 1)]
        image_np = np.asarray(image).astype(np.float32) / 255.0
        img_h, img_w = image_np.shape[:2]
        for i in order:
            concept = concept_names[i]
            cam_up = F.interpolate(
                cams[i].view(1, 1, *cams[i].shape),
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            cam_norm = normalize_map(cam_up)
            pred_mask = (cam_norm.cpu().numpy() >= float(args.cam_threshold)).astype(np.uint8)
            pred_box = mask_to_box(pred_mask)
            gt_box = union_boxes(grouped[concept]["boxes"])
            if gt_box is None:
                continue
            rec = {
                "image_index": int(image_index),
                "image_name": Path(image_name).name,
                "concept": concept,
                "concept_index": int(concept_to_idx[concept]),
                "vlg_logit": float(logits[concept_to_idx[concept]].item()),
                "annotation_logit": float(grouped[concept]["annotation_logit"]),
                "gt_box": gt_box,
                "pred_box": pred_box,
                "box_iou": box_iou(pred_box, gt_box),
                "image_np": image_np,
                "overlay": overlay_heatmap(image_np, cam_norm),
                "cam": cam_norm.cpu().numpy(),
            }
            page_records.append(rec)
            records.append({k: v for k, v in rec.items() if k not in {"image_np", "overlay", "cam"}})
            if len(page_records) >= int(args.page_size):
                page_index += 1
                page_paths.append(str(save_page(page_records, output_dir, page_index)))
                page_records.clear()
        images_used += 1
        if images_used >= int(args.max_images):
            break
    if page_records:
        page_index += 1
        page_paths.append(str(save_page(page_records, output_dir, page_index)))

    summary = {
        "load_dir": str(load_dir),
        "val_tar": str(Path(args.val_tar).resolve()),
        "annotation_val_dir": str(annotation_val_dir),
        "annotation_val_root": str(annotation_val_root) if annotation_val_root is not None else "",
        "output_dir": str(output_dir),
        "max_images": int(args.max_images),
        "images_used": images_used,
        "records": records,
        "pages": page_paths,
    }
    (output_dir / "vlgcbm_gradcam_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
