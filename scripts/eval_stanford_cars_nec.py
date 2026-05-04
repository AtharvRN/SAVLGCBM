#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_savlg_imagenet_standalone_nec import (  # noqa: E402
    build_weight_sweep,
    load_final_layer_payload,
    load_saved_weight_sweep,
    parse_nec_values,
    resolve_final_layer_path,
    resolve_source_run_dir,
)
from scripts.stanford_cars_common import StanfordCarsManifestDataset, read_concepts  # noqa: E402
from scripts.train_savlg_imagenet_standalone import (  # noqa: E402
    Config,
    build_loader,
    build_model,
    configure_runtime,
    prepare_images,
    topk_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stanford Cars NEC accuracy from saved G-CBM artifacts.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--test_manifest", default="data/stanford_cars/test_manifest.jsonl")
    parser.add_argument("--annotation_dir", default="annotations/stanford_cars")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_json", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--save_truncated_weights", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def load_run_config(config_dir: Path, args: argparse.Namespace) -> Config:
    payload = json.loads((config_dir / "config.json").read_text())
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


@torch.no_grad()
def extract_features(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg: Config,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    seen = 0
    backbone.eval()
    head.eval()
    for batch in loader:
        images = prepare_images(batch["images"], cfg)
        outputs = head(backbone(images))
        features.append(outputs["final_logits"].detach().float().cpu())
        targets.append(batch["class_ids"].detach().cpu())
        seen += int(batch["class_ids"].shape[0])
        if max_samples > 0 and seen >= max_samples:
            break
    feature_tensor = torch.cat(features, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    if max_samples > 0:
        feature_tensor = feature_tensor[:max_samples]
        target_tensor = target_tensor[:max_samples]
    return feature_tensor, target_tensor


def evaluate_logits(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    return {
        "top1": topk_accuracy(logits, targets, k=1),
        "top5": topk_accuracy(logits, targets, k=5),
        "n": int(targets.shape[0]),
    }


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    source_run_dir = resolve_source_run_dir(artifact_dir)
    cfg = load_run_config(source_run_dir, args)
    configure_runtime(cfg)

    concepts_path = source_run_dir / "concepts.txt"
    if not concepts_path.is_file():
        raise FileNotFoundError(f"Missing concepts.txt under {source_run_dir}")
    concepts = read_concepts(concepts_path)

    dataset = StanfordCarsManifestDataset(
        manifest_path=args.test_manifest,
        annotation_dir=args.annotation_dir,
        concepts=concepts,
        split=args.split,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        train_random_transforms=False,
    )
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(int(args.max_samples), len(dataset)))))
    loader = build_loader(dataset, cfg, shuffle=False, drop_last=False)

    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head_path = source_run_dir / "concept_head_best.pt"
    if not head_path.is_file():
        head_path = source_run_dir / "concept_head_last.pt"
    if not head_path.is_file():
        raise FileNotFoundError(f"Could not find concept head under {source_run_dir}")
    head.load_state_dict(torch.load(head_path, map_location=cfg.device))

    features, targets = extract_features(
        backbone,
        head,
        loader,
        cfg,
        max_samples=int(args.max_samples),
    )

    final_layer_path = resolve_final_layer_path(artifact_dir)
    full_layer = load_final_layer_payload(final_layer_path)
    dense_logits = features @ full_layer["weight"].t() + full_layer["bias"]
    dense_metrics = evaluate_logits(dense_logits, targets)

    nec_values = parse_nec_values(args.nec_values)
    weight_sweep = load_saved_weight_sweep(artifact_dir, nec_values)
    if weight_sweep is None:
        save_dir = artifact_dir if args.save_truncated_weights else None
        weight_sweep = build_weight_sweep(full_layer["weight"], full_layer["bias"], nec_values, save_dir)

    sparse_results: List[Dict[str, Any]] = []
    for item in weight_sweep:
        logits = features @ item["weight"].t() + item["bias"]
        metrics = evaluate_logits(logits, targets)
        sparse_results.append(
            {
                "nec": int(item["nec"]),
                "top1": float(metrics["top1"]),
                "top5": float(metrics["top5"]),
                "nnz": int(item["nnz"]),
                "total": int(item["total"]),
                "weight_sparsity": float(item["weight_sparsity"]),
            }
        )

    acc_by_nec = {f"ACC@{item['nec']}": float(item["top1"]) for item in sparse_results}
    avgacc = sum(item["top1"] for item in sparse_results) / max(len(sparse_results), 1)
    result = {
        "artifact_dir": str(artifact_dir),
        "source_run_dir": str(source_run_dir),
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "split": args.split,
        "dense": dense_metrics,
        "sparse": sparse_results,
        "summary": {
            **acc_by_nec,
            "AVGACC": float(avgacc),
            "dense_top1": float(dense_metrics["top1"]),
            "dense_top5": float(dense_metrics["top5"]),
        },
    }
    output_json = Path(args.output_json).resolve() if args.output_json else artifact_dir / f"{args.split}_nec_metrics.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
