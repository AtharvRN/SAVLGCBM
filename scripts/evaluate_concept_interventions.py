import argparse
import json
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.concept_dataset import get_concept_dataloader
from methods.common import load_run_info
from methods.salf import SpatialBackbone, build_spatial_concept_layer
from methods.savlg import (
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)
from model.cbm import (
    Backbone,
    BackboneCLIP,
    ConceptLayer,
    FinalLayer,
    NormalizationLayer,
    load_cbm,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CBM interventions by replacing predicted concept values with "
            "GroundingDINO GT concepts and measuring class accuracy recovery."
        )
    )
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--ks",
        type=str,
        default="0,1,2,5,10,20",
        help="Comma-separated numbers of concept corrections to evaluate.",
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
        "--policy",
        type=str,
        default="oracle_error",
        choices=["oracle_error", "random"],
        help="Intervention policy for choosing which concepts to correct.",
    )
    parser.add_argument(
        "--random_trials",
        type=int,
        default=5,
        help="Number of random intervention trials when --policy=random.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path to save evaluation results.",
    )
    parser.add_argument(
        "--wrong_only",
        action="store_true",
        help=(
            "Add a wrong-only intervention block that conditions metrics on "
            "examples the baseline model gets wrong."
        ),
    )
    return parser.parse_args()


def _load_checkpoint_args(
    load_path: str,
    device_override: str | None,
    annotation_dir_override: str | None,
) -> Namespace:
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
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        k = int(item)
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}.")
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


def _build_test_loader(
    model_name: str,
    load_path: str,
    args: Namespace,
    concepts: List[str],
    batch_size_override: int | None,
    num_workers_override: int | None,
    num_images: int | None,
) -> DataLoader:
    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else getattr(
            args,
            "cbl_batch_size",
            getattr(args, "lf_batch_size", getattr(args, "batch_size", 64)),
        )
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
        raise NotImplementedError(f"Intervention evaluation for model_name={model_name} is not implemented.")

    if getattr(args, "annotation_dir", None) is None:
        raise ValueError("annotation_dir must be present in checkpoint args or passed explicitly.")

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


def _load_dense_linear_from_wg_bg(load_path: str, device: str) -> nn.Linear:
    w_g = torch.load(os.path.join(load_path, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_path, "b_g.pt"), map_location=device)
    linear = nn.Linear(w_g.shape[1], w_g.shape[0]).to(device)
    linear.load_state_dict({"weight": w_g, "bias": b_g})
    linear.eval()
    return linear


def _vlg_state(load_path: str, args: Namespace):
    if not hasattr(_vlg_state, "_cache"):
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
        final_layer = FinalLayer.from_pretrained(load_path, args.device)
        backbone.eval()
        cbl.eval()
        normalization.eval()
        final_layer.eval()
        _vlg_state._cache = (backbone, cbl, normalization, final_layer)
    return _vlg_state._cache


def _lf_state(load_path: str, args: Namespace):
    if not hasattr(_lf_state, "_cache"):
        model = load_cbm(load_path, args.device)
        model.eval()
        _lf_state._cache = model
    return _lf_state._cache


def _salf_state(load_path: str, args: Namespace):
    if not hasattr(_salf_state, "_cache"):
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
        final_layer = _load_dense_linear_from_wg_bg(load_path, args.device)
        backbone.eval()
        concept_layer.eval()
        _salf_state._cache = (backbone, concept_layer, mean, std, final_layer)
    return _salf_state._cache


def _savlg_state(load_path: str, args: Namespace):
    if not hasattr(_savlg_state, "_cache"):
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
        mean = torch.load(os.path.join(load_path, "proj_mean.pt"), map_location=args.device)
        std = torch.load(os.path.join(load_path, "proj_std.pt"), map_location=args.device).clamp_min(1e-6)
        final_layer = _load_dense_linear_from_wg_bg(load_path, args.device)
        backbone.eval()
        concept_layer.eval()
        _savlg_state._cache = (backbone, concept_layer, mean, std, final_layer)
    return _savlg_state._cache


def _gt_to_intervention_space(gt_concepts: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (gt_concepts - mean) / std


def _get_batch_model_state(
    model_name: str,
    load_path: str,
    args: Namespace,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_name == "lf_cbm":
        model = _lf_state(load_path, args)
        with torch.no_grad():
            logits, concept_space = model(images.to(args.device))
        mean = model.proj_mean.to(args.device)
        std = model.proj_std.to(args.device).clamp_min(1e-6)
        return concept_space, logits, lambda gt, _m=mean, _s=std: _gt_to_intervention_space(gt, _m, _s)
    if model_name == "vlg_cbm":
        backbone, cbl, normalization, final_layer = _vlg_state(load_path, args)
        with torch.no_grad():
            raw = cbl(backbone(images.to(args.device)))
            concept_space = normalization(raw)
            logits = final_layer(concept_space)
        return concept_space, logits, lambda gt, _m=normalization.mean.to(args.device), _s=normalization.std.to(args.device).clamp_min(1e-6): _gt_to_intervention_space(gt, _m, _s)
    if model_name == "salf_cbm":
        backbone, concept_layer, mean, std, final_layer = _salf_state(load_path, args)
        with torch.no_grad():
            maps = concept_layer(backbone(images.to(args.device)))
            pooled = F.adaptive_avg_pool2d(maps, 1).flatten(1)
            concept_space = (pooled - mean.to(args.device)) / std.to(args.device)
            logits = final_layer(concept_space)
        return concept_space, logits, lambda gt, _m=mean.to(args.device), _s=std.to(args.device): _gt_to_intervention_space(gt, _m, _s)
    if model_name == "savlg_cbm":
        backbone, concept_layer, mean, std, final_layer = _savlg_state(load_path, args)
        with torch.no_grad():
            feats = forward_savlg_backbone(backbone, images.to(args.device), args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            _, _, final_logits = compute_savlg_concept_logits(global_outputs, spatial_maps, args)
            concept_space = (final_logits - mean.to(args.device)) / std.to(args.device)
            logits = final_layer(concept_space)
        return concept_space, logits, lambda gt, _m=mean.to(args.device), _s=std.to(args.device): _gt_to_intervention_space(gt, _m, _s)
    raise NotImplementedError(f"Intervention evaluation for model_name={model_name} is not implemented.")


def _oracle_order(concept_space: torch.Tensor, gt_space: torch.Tensor) -> torch.Tensor:
    error = (concept_space - gt_space).abs()
    return error.argsort(dim=1, descending=True)


def _random_order(
    batch_size: int,
    n_concepts: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.rand((batch_size, n_concepts), generator=generator, device=device).argsort(dim=1, descending=True)


def _apply_intervention(
    concept_space: torch.Tensor,
    gt_space: torch.Tensor,
    order: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if k == 0:
        return concept_space
    pos = torch.empty_like(order)
    ranks = (
        torch.arange(order.shape[1], device=order.device, dtype=order.dtype)
        .unsqueeze(0)
        .expand(order.shape[0], -1)
    )
    pos.scatter_(1, order, ranks)
    mask = pos < k
    return torch.where(mask, gt_space, concept_space)


def _compute_batch_metrics(
    logits: torch.Tensor,
    intervened_logits_by_k: dict[int, torch.Tensor],
    targets: torch.Tensor,
    gt_present_mask: torch.Tensor,
    ks: List[int],
) -> dict:
    baseline_pred = logits.argmax(dim=1)
    baseline_correct = baseline_pred.eq(targets)
    wrong_mask = ~baseline_correct
    results = {
        "num_examples": int(targets.numel()),
        "num_examples_with_gt_concepts": int(gt_present_mask.sum().item()),
        "baseline_correct": float(baseline_correct.sum().item()),
        "baseline_wrong": float(wrong_mask.sum().item()),
        "by_k": {},
    }
    for k in ks:
        pred = intervened_logits_by_k[k].argmax(dim=1)
        correct = pred.eq(targets)
        changed = pred.ne(baseline_pred)
        recovered = wrong_mask & correct
        wrong_changed = wrong_mask & changed
        results["by_k"][str(k)] = {
            "correct": float(correct.sum().item()),
            "changed": float(changed.sum().item()),
            "recovered": float(recovered.sum().item()),
            "wrong_changed": float(wrong_changed.sum().item()),
        }
    return results


def _merge_metrics(accum: dict, batch_metrics: dict, ks: List[int]) -> None:
    for key in ("num_examples", "num_examples_with_gt_concepts", "baseline_correct", "baseline_wrong"):
        accum[key] += batch_metrics[key]
    for k in ks:
        dst = accum["by_k"][str(k)]
        src = batch_metrics["by_k"][str(k)]
        for field in ("correct", "changed", "recovered", "wrong_changed"):
            dst[field] += src[field]


def _finalize_metrics(accum: dict, ks: List[int], meta: dict, include_wrong_only: bool) -> dict:
    num_examples = max(int(accum["num_examples"]), 1)
    baseline_wrong_count = float(accum["baseline_wrong"])
    baseline_wrong_denom = max(baseline_wrong_count, 1.0)
    output = {
        "metadata": meta,
        "num_examples": int(accum["num_examples"]),
        "num_examples_with_gt_concepts": int(accum["num_examples_with_gt_concepts"]),
        "num_baseline_wrong": int(accum["baseline_wrong"]),
        "baseline_accuracy": float(accum["baseline_correct"]) / float(num_examples),
        "intervention_curve": {},
    }
    if include_wrong_only:
        output["wrong_only_intervention_curve"] = {}
    for k in ks:
        stats = accum["by_k"][str(k)]
        accuracy = float(stats["correct"]) / float(num_examples)
        output["intervention_curve"][str(k)] = {
            "accuracy": accuracy,
            "accuracy_gain": accuracy - output["baseline_accuracy"],
            "changed_prediction_rate": float(stats["changed"]) / float(num_examples),
            "recovery_rate_among_baseline_wrong": float(stats["recovered"]) / float(baseline_wrong_denom),
        }
        if include_wrong_only:
            wrong_only_accuracy = float(stats["recovered"]) / float(baseline_wrong_denom)
            output["wrong_only_intervention_curve"][str(k)] = {
                "wrong_only_accuracy_after_intervention": wrong_only_accuracy,
                "recovery_rate": wrong_only_accuracy,
                "flip_rate": float(stats["wrong_changed"]) / float(baseline_wrong_denom),
            }
    return output


def _evaluate_once(
    model_name: str,
    load_path: str,
    args: Namespace,
    test_loader: DataLoader,
    ks: List[int],
    policy: str,
    random_generator: torch.Generator | None = None,
) -> dict:
    accum = {
        "num_examples": 0.0,
        "num_examples_with_gt_concepts": 0.0,
        "baseline_correct": 0.0,
        "baseline_wrong": 0.0,
        "by_k": {
            str(k): {"correct": 0.0, "changed": 0.0, "recovered": 0.0, "wrong_changed": 0.0}
            for k in ks
        },
    }
    if model_name == "lf_cbm":
        final_layer = _lf_state(load_path, args).final
    elif model_name == "vlg_cbm":
        final_layer = _vlg_state(load_path, args)[3]
    elif model_name == "salf_cbm":
        final_layer = _salf_state(load_path, args)[4]
    elif model_name == "savlg_cbm":
        final_layer = _savlg_state(load_path, args)[4]
    else:
        raise NotImplementedError

    for images, concept_one_hot, targets in tqdm(test_loader):
        targets = targets.to(args.device)
        gt_binary = concept_one_hot.to(args.device)
        gt_present_mask = gt_binary.sum(dim=1) > 0
        concept_space, logits, gt_transform = _get_batch_model_state(model_name, load_path, args, images)
        gt_space = gt_transform(gt_binary)

        if policy == "oracle_error":
            order = _oracle_order(concept_space, gt_space)
        elif policy == "random":
            assert random_generator is not None
            order = _random_order(
                concept_space.shape[0],
                concept_space.shape[1],
                concept_space.device,
                random_generator,
            )
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        intervened_logits_by_k = {}
        for k in ks:
            intervened_concepts = _apply_intervention(concept_space, gt_space, order, k)
            with torch.no_grad():
                intervened_logits_by_k[k] = final_layer(intervened_concepts)

        batch_metrics = _compute_batch_metrics(logits, intervened_logits_by_k, targets, gt_present_mask, ks)
        _merge_metrics(accum, batch_metrics, ks)

    return accum


def _average_random_results(results: List[dict], ks: List[int], meta: dict, include_wrong_only: bool) -> dict:
    baseline_accuracy = float(np.mean([res["baseline_accuracy"] for res in results]))
    output = {
        "metadata": meta,
        "num_examples": int(results[0]["num_examples"]),
        "num_examples_with_gt_concepts": int(results[0]["num_examples_with_gt_concepts"]),
        "num_baseline_wrong": int(results[0]["num_baseline_wrong"]),
        "baseline_accuracy": baseline_accuracy,
        "intervention_curve": {},
    }
    if include_wrong_only:
        output["wrong_only_intervention_curve"] = {}
    for k in ks:
        key = str(k)
        accs = np.array([res["intervention_curve"][key]["accuracy"] for res in results], dtype=float)
        gains = np.array([res["intervention_curve"][key]["accuracy_gain"] for res in results], dtype=float)
        changed = np.array([res["intervention_curve"][key]["changed_prediction_rate"] for res in results], dtype=float)
        recovered = np.array([res["intervention_curve"][key]["recovery_rate_among_baseline_wrong"] for res in results], dtype=float)
        output["intervention_curve"][key] = {
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std()),
            "accuracy_gain_mean": float(gains.mean()),
            "accuracy_gain_std": float(gains.std()),
            "changed_prediction_rate_mean": float(changed.mean()),
            "changed_prediction_rate_std": float(changed.std()),
            "recovery_rate_among_baseline_wrong_mean": float(recovered.mean()),
            "recovery_rate_among_baseline_wrong_std": float(recovered.std()),
        }
        if include_wrong_only:
            wrong_acc = np.array(
                [res["wrong_only_intervention_curve"][key]["wrong_only_accuracy_after_intervention"] for res in results],
                dtype=float,
            )
            wrong_recovery = np.array(
                [res["wrong_only_intervention_curve"][key]["recovery_rate"] for res in results],
                dtype=float,
            )
            wrong_flip = np.array(
                [res["wrong_only_intervention_curve"][key]["flip_rate"] for res in results],
                dtype=float,
            )
            output["wrong_only_intervention_curve"][key] = {
                "wrong_only_accuracy_after_intervention_mean": float(wrong_acc.mean()),
                "wrong_only_accuracy_after_intervention_std": float(wrong_acc.std()),
                "recovery_rate_mean": float(wrong_recovery.mean()),
                "recovery_rate_std": float(wrong_recovery.std()),
                "flip_rate_mean": float(wrong_flip.mean()),
                "flip_rate_std": float(wrong_flip.std()),
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
        "Running concept intervention eval: model={} images={} ks={} policy={}",
        model_name,
        len(test_loader.dataset),
        ks,
        args_ns.policy,
    )

    meta = {
        "load_path": args_ns.load_path,
        "model_name": model_name,
        "dataset": args.dataset,
        "annotation_dir": args.annotation_dir,
        "ks": ks,
        "num_images_requested": num_images,
        "policy": args_ns.policy,
        "random_trials": int(args_ns.random_trials),
        "wrong_only": bool(args_ns.wrong_only),
    }

    if args_ns.policy == "random":
        random_results = []
        base_seed = int(getattr(args, "seed", 42))
        for trial in range(int(args_ns.random_trials)):
            logger.info("Random intervention trial {}/{}", trial + 1, args_ns.random_trials)
            generator = torch.Generator(device=args.device)
            generator.manual_seed(base_seed + trial)
            accum = _evaluate_once(
                model_name,
                args_ns.load_path,
                args,
                test_loader,
                ks,
                "random",
                random_generator=generator,
            )
            random_results.append(_finalize_metrics(accum, ks, meta | {"trial": trial}, args_ns.wrong_only))
        result = _average_random_results(random_results, ks, meta, args_ns.wrong_only)
        result["trials"] = random_results
    else:
        accum = _evaluate_once(
            model_name,
            args_ns.load_path,
            args,
            test_loader,
            ks,
            "oracle_error",
        )
        result = _finalize_metrics(accum, ks, meta, args_ns.wrong_only)

    logger.info(
        "Intervention eval complete: baseline_acc={:.4f}, acc@{}={:.4f}",
        result["baseline_accuracy"],
        ks[min(1, len(ks) - 1)],
        result["intervention_curve"][str(ks[min(1, len(ks) - 1)])].get("accuracy", result["intervention_curve"][str(ks[min(1, len(ks) - 1)])].get("accuracy_mean")),
    )
    print(json.dumps(result, indent=2))

    if args_ns.output:
        output_path = Path(args_ns.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
