import argparse
import json
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.concept_dataset import get_concept_dataloader
from data import utils as data_utils
from methods.common import load_run_info
from methods.salf import SpatialBackbone, build_spatial_concept_layer
from methods.savlg import (
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)
from model.cbm import Backbone, BackboneCLIP, ConceptLayer, NormalizationLayer, load_cbm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether GroundingDINO GT concepts are ranked among the "
            "highest-activated concepts of a trained CBM checkpoint."
        )
    )
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,20",
        help="Comma-separated top-k values for Hit@k / Recall@k / Precision@k.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Optional image cap for quick subset evaluation.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Alias for --num_images.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override checkpoint batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader workers for evaluation.",
    )
    parser.add_argument(
        "--savlg_score_source",
        type=str,
        default="final",
        choices=["final", "global", "spatial"],
        help="Which SAVLG concept logit source to rank.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path to save evaluation results.",
    )
    return parser.parse_args()


def _load_checkpoint_args(load_path: str, device_override: str | None, annotation_dir_override: str | None) -> Namespace:
    with open(os.path.join(load_path, "args.txt"), "r") as f:
        payload = json.load(f)
    if device_override is not None:
        payload["device"] = device_override
    if annotation_dir_override is not None:
        payload["annotation_dir"] = annotation_dir_override
    return Namespace(**payload)


def _load_concepts(load_path: str) -> List[str]:
    with open(os.path.join(load_path, "concepts.txt"), "r") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_num_images(args_ns: argparse.Namespace) -> int | None:
    if args_ns.num_images is not None:
        return int(args_ns.num_images)
    if args_ns.max_images is not None:
        return int(args_ns.max_images)
    return None


def _parse_ks(raw: str, n_concepts: int) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        k = int(item)
        if k <= 0:
            raise ValueError(f"Top-k must be positive, got {k}.")
        values.append(min(k, n_concepts))
    if not values:
        raise ValueError("No valid k values were provided.")
    return sorted(set(values))


def _subset_loader(loader: DataLoader, num_images: int | None) -> DataLoader:
    if num_images is None:
        return loader
    keep = min(int(num_images), len(loader.dataset))
    dataset = Subset(loader.dataset, list(range(keep)))
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        shuffle=False,
        pin_memory=getattr(loader, "pin_memory", False),
    )


def _build_test_loader(model_name: str, load_path: str, args: Namespace, concepts: List[str], batch_size_override: int | None, num_workers_override: int | None, num_images: int | None):
    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else getattr(args, "cbl_batch_size", getattr(args, "lf_batch_size", getattr(args, "batch_size", 64)))
    )
    num_workers = int(
        num_workers_override if num_workers_override is not None else getattr(args, "num_workers", 4)
    )

    if model_name == "lf_cbm":
        model = load_cbm(load_path, args.device)
        preprocess = model.preprocess
    elif model_name == "vlg_cbm":
        if args.backbone.startswith("clip_"):
            backbone = BackboneCLIP(
                args.backbone,
                use_penultimate=args.use_clip_penultimate,
                device=args.device,
            )
        else:
            backbone = Backbone(args.backbone, args.feature_layer, args.device)
            if os.path.exists(os.path.join(load_path, "backbone.pt")):
                backbone.backbone.load_state_dict(
                    torch.load(os.path.join(load_path, "backbone.pt"), map_location=args.device)
                )
        preprocess = backbone.preprocess
    elif model_name in {"salf_cbm", "savlg_cbm"}:
        spatial_stage = getattr(args, "savlg_spatial_stage", "conv5")
        backbone = SpatialBackbone(args.backbone, device=args.device, spatial_stage=spatial_stage)
        preprocess = backbone.preprocess
    else:
        raise NotImplementedError(f"Concept ranking for model_name={model_name} is not implemented.")

    if getattr(args, "annotation_dir", None) is None:
        raise ValueError("annotation_dir must be present in the checkpoint args or passed explicitly.")

    loader = get_concept_dataloader(
        args.dataset,
        "test",
        concepts,
        preprocess=preprocess,
        val_split=None,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        confidence_threshold=getattr(args, "cbl_confidence_threshold", 0.10),
        crop_to_concept_prob=0.0,
        label_dir=args.annotation_dir,
        use_allones=getattr(args, "allones_concept", False),
        seed=getattr(args, "seed", 42),
    )
    return _subset_loader(loader, num_images)


def _vlg_scores(load_path: str, args: Namespace, batch_images: torch.Tensor) -> torch.Tensor:
    if not hasattr(_vlg_scores, "_cache"):
        if args.backbone.startswith("clip_"):
            backbone = BackboneCLIP(
                args.backbone,
                use_penultimate=args.use_clip_penultimate,
                device=args.device,
            )
        else:
            backbone = Backbone(args.backbone, args.feature_layer, args.device)
            backbone_path = os.path.join(load_path, "backbone.pt")
            if os.path.exists(backbone_path):
                backbone.backbone.load_state_dict(torch.load(backbone_path, map_location=args.device))
        cbl = ConceptLayer.from_pretrained(load_path, args.device)
        normalization = NormalizationLayer.from_pretrained(load_path, args.device)
        backbone.eval()
        cbl.eval()
        normalization.eval()
        _vlg_scores._cache = (backbone, cbl, normalization)
    backbone, cbl, normalization = _vlg_scores._cache
    with torch.no_grad():
        embeddings = backbone(batch_images.to(args.device))
        return normalization(cbl(embeddings))


def _lf_scores(load_path: str, args: Namespace, batch_images: torch.Tensor) -> torch.Tensor:
    if not hasattr(_lf_scores, "_cache"):
        model = load_cbm(load_path, args.device)
        model.eval()
        _lf_scores._cache = model
    model = _lf_scores._cache
    with torch.no_grad():
        _, concept_scores = model(batch_images.to(args.device))
        return concept_scores


def _salf_scores(load_path: str, args: Namespace, batch_images: torch.Tensor) -> torch.Tensor:
    if not hasattr(_salf_scores, "_cache"):
        concepts = _load_concepts(load_path)
        backbone = SpatialBackbone(
            args.backbone,
            device=args.device,
            spatial_stage=getattr(args, "savlg_spatial_stage", "conv5"),
        )
        concept_layer = build_spatial_concept_layer(args, backbone.output_dim, len(concepts)).to(args.device)
        concept_layer.load_state_dict(
            torch.load(os.path.join(load_path, "concept_layer.pt"), map_location=args.device)
        )
        mean = torch.load(os.path.join(load_path, "proj_mean.pt"), map_location=args.device)
        std = torch.load(os.path.join(load_path, "proj_std.pt"), map_location=args.device).clamp_min(1e-6)
        backbone.eval()
        concept_layer.eval()
        _salf_scores._cache = (backbone, concept_layer, mean, std)
    backbone, concept_layer, mean, std = _salf_scores._cache
    with torch.no_grad():
        maps = concept_layer(backbone(batch_images.to(args.device)))
        pooled = F.adaptive_avg_pool2d(maps, 1).flatten(1)
        return (pooled - mean) / std


def _savlg_scores(load_path: str, args: Namespace, batch_images: torch.Tensor, score_source: str) -> torch.Tensor:
    if not hasattr(_savlg_scores, "_cache"):
        concepts = _load_concepts(load_path)
        backbone = SpatialBackbone(
            args.backbone,
            device=args.device,
            spatial_stage=getattr(args, "savlg_spatial_stage", "conv5"),
        )
        concept_layer = build_savlg_concept_layer(args, backbone, len(concepts)).to(args.device)
        concept_layer.load_state_dict(
            torch.load(os.path.join(load_path, "concept_layer.pt"), map_location=args.device)
        )
        mean = None
        std = None
        mean_path = os.path.join(load_path, "proj_mean.pt")
        std_path = os.path.join(load_path, "proj_std.pt")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path, map_location=args.device)
            std = torch.load(std_path, map_location=args.device).clamp_min(1e-6)
        backbone.eval()
        concept_layer.eval()
        _savlg_scores._cache = (backbone, concept_layer, mean, std)
    backbone, concept_layer, mean, std = _savlg_scores._cache
    with torch.no_grad():
        feats = forward_savlg_backbone(backbone, batch_images.to(args.device), args)
        global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
        global_logits, spatial_logits, final_logits = compute_savlg_concept_logits(
            global_outputs,
            spatial_maps,
            args,
        )
        if score_source == "global":
            return global_logits
        if score_source == "spatial":
            return spatial_logits
        scores = final_logits
        if mean is not None and std is not None:
            scores = (scores - mean) / std
        return scores


def _compute_batch_metrics(
    scores: torch.Tensor,
    gt_concepts: torch.Tensor,
    ks: List[int],
) -> dict:
    gt_mask = gt_concepts > 0
    valid_mask = gt_mask.sum(dim=1) > 0
    total_images = int(scores.shape[0])
    valid_images = int(valid_mask.sum().item())
    if valid_images == 0:
        return {
            "total_images": total_images,
            "valid_images": 0,
            "gt_instances": 0,
            "sum_gt_per_image": 0.0,
            "sum_best_rank": 0.0,
            "sum_best_rr": 0.0,
            "sum_gt_rank": 0.0,
            "sum_image_mean_gt_rank": 0.0,
            "sum_image_median_gt_rank": 0.0,
            "by_k": {str(k): {"hits": 0.0, "recall_macro": 0.0, "recall_micro_num": 0.0, "precision": 0.0} for k in ks},
        }

    scores = scores[valid_mask]
    gt_mask = gt_mask[valid_mask]
    batch_size, n_concepts = scores.shape
    ranks = torch.empty((batch_size, n_concepts), dtype=torch.long, device=scores.device)
    order = scores.argsort(dim=1, descending=True)
    positions = (
        torch.arange(1, n_concepts + 1, device=scores.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    ranks.scatter_(1, order, positions)

    gt_counts = gt_mask.sum(dim=1)
    max_rank = n_concepts + 1
    best_ranks = torch.where(
        gt_mask,
        ranks,
        torch.full_like(ranks, max_rank),
    ).min(dim=1).values

    masked_ranks = torch.where(gt_mask, ranks, torch.zeros_like(ranks))
    image_mean_gt_rank = masked_ranks.sum(dim=1).float() / gt_counts.float()
    image_median_gt_rank = []
    for row_ranks, row_mask in zip(ranks, gt_mask):
        selected = row_ranks[row_mask]
        image_median_gt_rank.append(selected.median().item())

    by_k = {}
    for k in ks:
        recovered = ((ranks <= k) & gt_mask).sum(dim=1)
        by_k[str(k)] = {
            "hits": float((recovered > 0).sum().item()),
            "recall_macro": float((recovered.float() / gt_counts.float()).sum().item()),
            "recall_micro_num": float(recovered.sum().item()),
            "precision": float((recovered.float() / float(k)).sum().item()),
        }

    return {
        "total_images": total_images,
        "valid_images": valid_images,
        "gt_instances": int(gt_counts.sum().item()),
        "sum_gt_per_image": float(gt_counts.float().sum().item()),
        "sum_best_rank": float(best_ranks.float().sum().item()),
        "sum_best_rr": float((1.0 / best_ranks.float()).sum().item()),
        "sum_gt_rank": float(ranks[gt_mask].float().sum().item()),
        "sum_image_mean_gt_rank": float(image_mean_gt_rank.sum().item()),
        "sum_image_median_gt_rank": float(sum(image_median_gt_rank)),
        "by_k": by_k,
    }


def _merge_metrics(accum: dict, batch_metrics: dict, ks: List[int]) -> None:
    for key in (
        "total_images",
        "valid_images",
        "gt_instances",
        "sum_gt_per_image",
        "sum_best_rank",
        "sum_best_rr",
        "sum_gt_rank",
        "sum_image_mean_gt_rank",
        "sum_image_median_gt_rank",
    ):
        accum[key] += batch_metrics[key]
    for k in ks:
        dst = accum["by_k"][str(k)]
        src = batch_metrics["by_k"][str(k)]
        for field in ("hits", "recall_macro", "recall_micro_num", "precision"):
            dst[field] += src[field]


def _finalize_metrics(accum: dict, ks: List[int], meta: dict) -> dict:
    valid_images = max(int(accum["valid_images"]), 1)
    gt_instances = max(int(accum["gt_instances"]), 1)
    output = {
        "metadata": meta,
        "num_images_total": int(accum["total_images"]),
        "num_images_with_gt_concepts": int(accum["valid_images"]),
        "num_gt_concept_instances": int(accum["gt_instances"]),
        "avg_gt_concepts_per_valid_image": float(accum["sum_gt_per_image"]) / float(valid_images),
        "mean_best_gt_rank": float(accum["sum_best_rank"]) / float(valid_images),
        "mean_reciprocal_best_gt_rank": float(accum["sum_best_rr"]) / float(valid_images),
        "mean_gt_rank": float(accum["sum_gt_rank"]) / float(gt_instances),
        "mean_image_mean_gt_rank": float(accum["sum_image_mean_gt_rank"]) / float(valid_images),
        "mean_image_median_gt_rank": float(accum["sum_image_median_gt_rank"]) / float(valid_images),
        "topk": {},
    }
    for k in ks:
        stats = accum["by_k"][str(k)]
        output["topk"][str(k)] = {
            "hit_rate": float(stats["hits"]) / float(valid_images),
            "recall_macro": float(stats["recall_macro"]) / float(valid_images),
            "recall_micro": float(stats["recall_micro_num"]) / float(gt_instances),
            "precision_at_k": float(stats["precision"]) / float(valid_images),
        }
    return output


def main() -> None:
    args_ns = _parse_args()
    run_info = load_run_info(args_ns.load_path)
    args = _load_checkpoint_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    model_name = run_info.get("model_name", "vlg_cbm")
    concepts = _load_concepts(args_ns.load_path)
    ks = _parse_ks(args_ns.ks, len(concepts))
    num_images = _resolve_num_images(args_ns)

    torch.manual_seed(getattr(args, "seed", 42))
    np.random.seed(getattr(args, "seed", 42))
    random.seed(getattr(args, "seed", 42))

    test_loader = _build_test_loader(
        model_name,
        args_ns.load_path,
        args,
        concepts,
        args_ns.batch_size,
        args_ns.num_workers,
        num_images,
    )
    logger.info(
        "Running concept ranking eval: model={} images={} ks={} source={}",
        model_name,
        len(test_loader.dataset),
        ks,
        args_ns.savlg_score_source if model_name == "savlg_cbm" else "default",
    )

    accum = {
        "total_images": 0.0,
        "valid_images": 0.0,
        "gt_instances": 0.0,
        "sum_gt_per_image": 0.0,
        "sum_best_rank": 0.0,
        "sum_best_rr": 0.0,
        "sum_gt_rank": 0.0,
        "sum_image_mean_gt_rank": 0.0,
        "sum_image_median_gt_rank": 0.0,
        "by_k": {str(k): {"hits": 0.0, "recall_macro": 0.0, "recall_micro_num": 0.0, "precision": 0.0} for k in ks},
    }

    for images, concept_one_hot, _targets in tqdm(test_loader):
        if model_name == "lf_cbm":
            scores = _lf_scores(args_ns.load_path, args, images)
        elif model_name == "vlg_cbm":
            scores = _vlg_scores(args_ns.load_path, args, images)
        elif model_name == "salf_cbm":
            scores = _salf_scores(args_ns.load_path, args, images)
        elif model_name == "savlg_cbm":
            scores = _savlg_scores(
                args_ns.load_path,
                args,
                images,
                args_ns.savlg_score_source,
            )
        else:
            raise NotImplementedError(f"Concept ranking for model_name={model_name} is not implemented.")

        batch_metrics = _compute_batch_metrics(scores.detach(), concept_one_hot.to(scores.device), ks)
        _merge_metrics(accum, batch_metrics, ks)

    meta = {
        "load_path": args_ns.load_path,
        "model_name": model_name,
        "dataset": args.dataset,
        "annotation_dir": args.annotation_dir,
        "score_source": args_ns.savlg_score_source if model_name == "savlg_cbm" else "default",
        "ks": ks,
        "num_images_requested": num_images,
    }
    result = _finalize_metrics(accum, ks, meta)

    logger.info(
        "Concept ranking complete: valid_images={} mean_best_gt_rank={:.2f} hit@{}={:.4f} recall@{}={:.4f}",
        result["num_images_with_gt_concepts"],
        result["mean_best_gt_rank"],
        ks[0],
        result["topk"][str(ks[0])]["hit_rate"],
        ks[0],
        result["topk"][str(ks[0])]["recall_macro"],
    )
    print(json.dumps(result, indent=2))

    if args_ns.output:
        output_path = Path(args_ns.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
