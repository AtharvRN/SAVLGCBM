import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from methods.savlg import (
    _annotation_split_dir,
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    create_savlg_splits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)
from model.utils import get_bbox_iou
from scripts.visualize_savlg_examples import (
    _load_args,
    _load_concepts,
    _normalize_map,
    _union_boxes,
)


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, target = self.base[idx]
        return idx, image, target


def _parse_csv_floats(x: str):
    return [float(v.strip()) for v in x.split(",") if v.strip()]


def _normalize_map_with_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.detach().float()
    mode = str(mode).lower()
    if x.ndim == 2:
        reduce_dims = None
    elif x.ndim == 3:
        reduce_dims = (1, 2)
    else:
        raise ValueError(f"Expected 2D or 3D maps, got shape {tuple(x.shape)}")
    if mode == "minmax":
        if reduce_dims is None:
            return _normalize_map(x)
        x = x - x.amin(dim=reduce_dims, keepdim=True)
        denom = x.amax(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        return x / denom
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "concept_zscore_minmax":
        if reduce_dims is None:
            x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
            return _normalize_map(x)
        mean = x.mean(dim=reduce_dims, keepdim=True)
        std = x.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = (x - mean) / std
        x = x - x.amin(dim=reduce_dims, keepdim=True)
        denom = x.amax(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        return x / denom
    raise ValueError(f"Unsupported map_normalization={mode}")


def _boxes_from_masks(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # masks: [K, H, W] bool on any device
    if masks.ndim != 3:
        raise ValueError(f"Expected [K,H,W] masks, got shape {tuple(masks.shape)}")
    valid = masks.flatten(1).any(dim=1)
    rows_any = masks.any(dim=2)
    cols_any = masks.any(dim=1)

    y1 = rows_any.float().argmax(dim=1)
    x1 = cols_any.float().argmax(dim=1)
    y2 = masks.shape[1] - 1 - rows_any.flip(1).float().argmax(dim=1)
    x2 = masks.shape[2] - 1 - cols_any.flip(1).float().argmax(dim=1)

    boxes = torch.stack([x1, y1, x2, y2], dim=1).to(torch.float32)
    return boxes, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--activation_thresholds", type=str, default="0.9")
    parser.add_argument("--box_iou_thresholds", type=str, default="0.1,0.3,0.5,0.7")
    parser.add_argument("--map_normalization", type=str, default="concept_zscore_minmax")
    parser.add_argument("--eval_subset_mode", type=str, default="gt_present", choices=["gt_present"])
    args_ns = parser.parse_args()

    args = _load_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    _, _, _, _, test_dataset, backbone = create_savlg_splits(args)
    if args_ns.max_images is not None:
        keep = min(args_ns.max_images, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(keep)))
    dataset = IndexedDataset(test_dataset)

    concepts = _load_concepts(args_ns.load_path, args)
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    ann_dir = _annotation_split_dir(args.annotation_dir, args.dataset, "val")

    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts)).to(args.device)
    concept_layer.load_state_dict(torch.load(os.path.join(args_ns.load_path, "concept_layer.pt"), map_location=args.device))
    concept_layer.eval()
    backbone.eval()

    loader = DataLoader(
        dataset,
        batch_size=args_ns.batch_size,
        shuffle=False,
        num_workers=args_ns.num_workers,
        pin_memory=False,
    )

    activation_thresholds = _parse_csv_floats(args_ns.activation_thresholds)
    iou_thresholds = _parse_csv_floats(args_ns.box_iou_thresholds)
    per_threshold_ious = {thr: [] for thr in activation_thresholds}

    with torch.no_grad():
        for batch in tqdm(loader, desc="native map eval"):
            indices, images, _targets = batch
            images = images.to(args.device, non_blocking=True)
            feats = forward_savlg_backbone(backbone, images, args)
            _global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            img_h, img_w = int(images.shape[-2]), int(images.shape[-1])

            for b, ds_idx in enumerate(indices.tolist()):
                ann_path = os.path.join(ann_dir, f"{int(ds_idx)}.json")
                if not os.path.exists(ann_path):
                    continue
                with open(ann_path, "r") as f:
                    payload = json.load(f)
                gt_boxes_by_concept = {}
                for ann in payload[1:]:
                    if not isinstance(ann, dict):
                        continue
                    label = ann.get("label")
                    box = ann.get("box")
                    if not isinstance(label, str) or label not in concept_to_idx:
                        continue
                    if not isinstance(box, list) or len(box) != 4:
                        continue
                    gt_boxes_by_concept.setdefault(label, []).append([float(v) for v in box])
                if not gt_boxes_by_concept:
                    continue
                concept_indices: List[int] = []
                gt_boxes: List[List[float]] = []
                for concept_name, concept_boxes in gt_boxes_by_concept.items():
                    gt_box = _union_boxes(concept_boxes)
                    if gt_box is None:
                        continue
                    concept_indices.append(concept_to_idx[concept_name])
                    gt_boxes.append(gt_box)
                if not concept_indices:
                    continue
                concept_idx_tensor = torch.as_tensor(concept_indices, device=spatial_maps.device, dtype=torch.long)
                maps_k_native = spatial_maps[b].index_select(0, concept_idx_tensor)
                maps_k = F.interpolate(
                    maps_k_native.unsqueeze(1),
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                norm_maps_k = _normalize_map_with_mode(maps_k, args_ns.map_normalization)
                gt_boxes_cpu = gt_boxes
                for thr in activation_thresholds:
                    pred_masks = norm_maps_k >= float(thr)
                    pred_boxes_t, valid_t = _boxes_from_masks(pred_masks)
                    pred_boxes = pred_boxes_t.cpu().tolist()
                    valid = valid_t.cpu().tolist()
                    for pred_box, is_valid, gt_box in zip(pred_boxes, valid, gt_boxes_cpu):
                        iou = 0.0 if not is_valid else float(get_bbox_iou(pred_box, gt_box))
                        per_threshold_ious[thr].append(iou)

    results = {
        "load_path": args_ns.load_path,
        "map_normalization": args_ns.map_normalization,
        "eval_subset_mode": args_ns.eval_subset_mode,
        "num_images": len(dataset),
        "num_gt_instances": int(len(next(iter(per_threshold_ious.values()))) if per_threshold_ious else 0),
        "threshold_metrics": {},
    }

    best_mean_iou_thr = None
    best_mean_iou = None
    best_box_acc = {iou_thr: {"threshold": None, "value": None} for iou_thr in iou_thresholds}

    for thr in activation_thresholds:
        ious = per_threshold_ious[thr]
        if not ious:
            mean_iou = 0.0
        else:
            mean_iou = float(sum(ious) / len(ious))
        box_acc = {}
        for iou_thr in iou_thresholds:
            value = 0.0 if not ious else float(sum(i >= iou_thr for i in ious) / len(ious))
            box_acc[str(iou_thr)] = value
            cur = best_box_acc[iou_thr]
            if cur["value"] is None or value > cur["value"]:
                cur["value"] = value
                cur["threshold"] = thr
        results["threshold_metrics"][str(thr)] = {
            "mean_iou": mean_iou,
            "box_acc": box_acc,
            "mAP": box_acc,
        }
        if best_mean_iou is None or mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_mean_iou_thr = thr

    results["best_mean_iou"] = {"threshold": best_mean_iou_thr, "value": best_mean_iou}
    results["best_box_acc"] = {str(k): v for k, v in best_box_acc.items()}

    output_path = Path(args_ns.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
