#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_savlg_imagenet_standalone_localization import (  # noqa: E402
    box_iou,
    boxes_from_masks,
    normalize_maps,
    parse_thresholds,
)
from scripts.stanford_cars_common import StanfordCarsManifestDataset, read_concepts  # noqa: E402
from scripts.train_savlg_imagenet_standalone import (  # noqa: E402
    Config,
    build_gdino_targets,
    build_loader,
    build_model,
    configure_runtime,
    prepare_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stanford Cars concept localization with native G-CBM maps.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_json", default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9,mean")
    parser.add_argument("--map_normalization", default="concept_zscore_minmax", choices=["minmax", "sigmoid", "concept_zscore_minmax"])
    parser.add_argument("--gt_threshold", type=float, default=0.0)
    parser.add_argument("--max_images", type=int, default=0)
    return parser.parse_args()


def load_config(artifact_dir: Path, args: argparse.Namespace) -> Config:
    payload = json.loads((artifact_dir / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload.setdefault("train_random_transforms", False)
    payload.setdefault("learn_spatial_residual_scale", False)
    payload["device"] = args.device
    payload["batch_size"] = int(args.batch_size)
    payload["workers"] = int(args.workers)
    payload["prefetch_factor"] = int(args.prefetch_factor)
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    return Config(**payload)


def update_metric(metrics: Dict[str, Any], threshold_key: str, iou: float) -> None:
    state = metrics.setdefault(
        threshold_key,
        {"instances": 0, "iou_sum": 0.0, "locacc_03": 0, "locacc_05": 0},
    )
    state["instances"] += 1
    state["iou_sum"] += float(iou)
    state["locacc_03"] += int(iou >= 0.3)
    state["locacc_05"] += int(iou >= 0.5)


def finalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    finalized: Dict[str, Any] = {}
    for key, state in metrics.items():
        count = max(int(state["instances"]), 1)
        finalized[key] = {
            "instances": int(state["instances"]),
            "MeanIoU": float(state["iou_sum"] / count),
            "LocAcc@0.3": float(state["locacc_03"] / count),
            "LocAcc@0.5": float(state["locacc_05"] / count),
        }
    return finalized


@torch.no_grad()
def evaluate(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg: Config,
    args: argparse.Namespace,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
) -> Dict[str, Any]:
    thresholds, include_mean = parse_thresholds(args.activation_thresholds)
    metrics: Dict[str, Any] = {}
    backbone.eval()
    head.eval()
    images_seen = 0
    instances_seen = 0
    for batch in loader:
        images = prepare_images(batch["images"], cfg)
        _global_targets, idx_pad, mask_pad, valid_pad = build_gdino_targets(
            batch["annotations"],
            batch["image_sizes"],
            concept_to_idx,
            n_concepts,
            cfg,
            cfg.device,
        )
        outputs = head(backbone(images))
        spatial_maps = F.interpolate(
            outputs["spatial_maps"],
            size=mask_pad.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        for batch_index in range(spatial_maps.shape[0]):
            valid = valid_pad[batch_index]
            if not bool(valid.any()):
                continue
            concept_ids = idx_pad[batch_index][valid]
            gt_masks = mask_pad[batch_index][valid] > float(args.gt_threshold)
            concept_maps = spatial_maps[batch_index].index_select(0, concept_ids)
            concept_maps = normalize_maps(concept_maps, args.map_normalization)
            gt_boxes, gt_box_valid = boxes_from_masks(gt_masks)
            for concept_offset in range(concept_maps.shape[0]):
                score_map = concept_maps[concept_offset]
                gt_box = gt_boxes[concept_offset : concept_offset + 1]
                gt_valid = gt_box_valid[concept_offset : concept_offset + 1]
                current_thresholds = list(thresholds)
                if include_mean:
                    current_thresholds.append(float(score_map.mean().item()))
                for threshold in current_thresholds:
                    pred_mask = score_map >= float(threshold)
                    pred_boxes, pred_valid = boxes_from_masks(pred_mask.unsqueeze(0))
                    iou = float(box_iou(pred_boxes, pred_valid, gt_box, gt_valid).item())
                    update_metric(metrics, f"{float(threshold):.6f}", iou)
                instances_seen += 1
            images_seen += 1
        if args.max_images > 0 and images_seen >= int(args.max_images):
            break
    finalized = finalize_metrics(metrics)
    best_locacc_05 = max(finalized.items(), key=lambda item: item[1]["LocAcc@0.5"]) if finalized else None
    return {
        "images_seen": images_seen,
        "concept_instances_seen": instances_seen,
        "metrics_by_threshold": finalized,
        "best_by_locacc_05": {
            "threshold": best_locacc_05[0],
            **best_locacc_05[1],
        }
        if best_locacc_05 is not None
        else None,
    }


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    cfg = load_config(artifact_dir, args)
    configure_runtime(cfg)

    concepts_path = artifact_dir / "concepts.txt"
    if not concepts_path.is_file():
        raise FileNotFoundError(f"Missing concepts.txt under {artifact_dir}")
    concepts = read_concepts(concepts_path)
    dataset = StanfordCarsManifestDataset(
        manifest_path=args.manifest,
        annotation_dir=args.annotation_dir,
        concepts=concepts,
        split=args.split,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        train_random_transforms=False,
    )
    if args.max_images > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(int(args.max_images), len(dataset)))))
    loader = build_loader(dataset, cfg, shuffle=False, drop_last=False)

    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head_path = artifact_dir / "concept_head_best.pt"
    if not head_path.is_file():
        head_path = artifact_dir / "concept_head_last.pt"
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing concept head under {artifact_dir}")
    head.load_state_dict(torch.load(head_path, map_location=cfg.device))

    result = evaluate(
        backbone,
        head,
        loader,
        cfg,
        args,
        dataset.concept_to_idx if hasattr(dataset, "concept_to_idx") else loader.dataset.dataset.concept_to_idx,
        len(concepts),
    )
    result.update(
        {
            "artifact_dir": str(artifact_dir),
            "manifest": str(Path(args.manifest).resolve()),
            "annotation_dir": str(Path(args.annotation_dir).resolve()),
            "split": args.split,
            "map_normalization": args.map_normalization,
        }
    )
    output_json = Path(args.output_json).resolve() if args.output_json else artifact_dir / f"{args.split}_localization_metrics.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
