#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stanford_cars_common import (  # noqa: E402
    StanfordCarsManifestDataset,
    read_concepts,
)
from scripts.train_savlg_imagenet_standalone import (  # noqa: E402
    Config,
    DatasetView,
    apply_count_concept_filter,
    build_loader,
    build_model,
    compute_feature_stats_memmap,
    configure_runtime,
    extract_concept_features_to_memmap,
    init_global_head_from_vlg,
    make_optimizer,
    make_scheduler,
    precompute_target_store,
    save_checkpoint,
    select_subset_indices,
    train_dense_final_layer,
    train_one_epoch,
    train_sparse_final_layer,
    evaluate_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or precompute Stanford Cars G-CBM artifacts.")
    parser.add_argument("--mode", choices=["train", "precompute_targets"], default="train")
    parser.add_argument("--train_manifest", default="data/stanford_cars/train_manifest.jsonl")
    parser.add_argument("--val_manifest", default="data/stanford_cars/val_manifest.jsonl")
    parser.add_argument("--test_manifest", default="data/stanford_cars/test_manifest.jsonl")
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--concept_file", default="concept_files/stanford_cars_concepts_filtered.txt")
    parser.add_argument("--save_dir", default="saved_models/stanford_cars")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--feature_dir", default="")
    parser.add_argument("--precomputed_target_dir", default="data/stanford_cars/precomputed_targets")
    parser.add_argument("--precompute_splits", default="train,val,test")
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--disable_channels_last", action="store_true")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--disable_cudnn_benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=6885)

    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--mask_h", type=int, default=14)
    parser.add_argument("--mask_w", type=int, default=14)
    parser.add_argument("--min_image_bytes", type=int, default=2048)

    parser.add_argument("--filter_concepts_by_count", action="store_true")
    parser.add_argument("--concept_min_count", type=int, default=1)
    parser.add_argument("--concept_min_frequency", type=float, default=0.0)
    parser.add_argument("--concept_max_frequency", type=float, default=1.0)

    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")

    parser.add_argument("--global_pos_weight", type=float, default=1.0)
    parser.add_argument("--loss_global_w", type=float, default=1.0)
    parser.add_argument("--loss_mask_w", type=float, default=1.0)
    parser.add_argument("--loss_dice_w", type=float, default=0.0)

    parser.add_argument("--spatial_branch_mode", choices=["shared_stage", "multiscale_conv45"], default="multiscale_conv45")
    parser.add_argument("--spatial_stage", choices=["conv4", "conv5"], default="conv5")
    parser.add_argument("--residual_alpha", type=float, default=0.2)
    parser.add_argument("--residual_spatial_pooling", choices=["avg", "lse"], default="lse")
    parser.add_argument("--learn_spatial_residual_scale", action="store_true")

    parser.add_argument("--vlg_init_path", default="")
    parser.add_argument("--vlg_concepts_path", default="")
    parser.add_argument("--freeze_global_head", action="store_true")
    parser.add_argument("--skip_final_layer", action="store_true")
    parser.add_argument("--final_layer_type", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--feature_batch_size", type=int, default=256)
    parser.add_argument("--feature_workers", type=int, default=4)
    parser.add_argument("--feature_prefetch_factor", type=int, default=2)
    parser.add_argument("--feature_storage_dtype", choices=["fp16", "fp32"], default="fp16")

    parser.add_argument("--saga_lam", type=float, default=5e-4)
    parser.add_argument("--saga_n_iters", type=int, default=200)
    parser.add_argument("--saga_step_size", type=float, default=0.02)
    parser.add_argument("--saga_batch_size", type=int, default=4096)
    parser.add_argument("--saga_workers", type=int, default=0)
    parser.add_argument("--saga_prefetch_factor", type=int, default=2)
    parser.add_argument("--saga_verbose_every", type=int, default=20)
    parser.add_argument("--saga_table_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dense_lr", type=float, default=1e-3)
    parser.add_argument("--dense_n_iters", type=int, default=20)

    parser.add_argument("--concept_threshold", type=float, default=0.15)
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5)
    parser.add_argument("--spatial_target_mode", choices=["hard_iou", "soft_box"], default="soft_box")
    parser.add_argument("--spatial_loss_mode", choices=["bce", "soft_align"], default="soft_align")
    parser.add_argument("--patch_pos_weight", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--print_config", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        mode=args.mode,
        train_root="",
        train_manifest=args.train_manifest,
        annotation_dir=args.annotation_dir,
        concept_file=args.concept_file,
        val_root="",
        save_dir=args.save_dir,
        run_name=args.run_name,
        reuse_run_dir="",
        feature_dir=args.feature_dir,
        precomputed_target_dir=args.precomputed_target_dir,
        persist_feature_copy=False,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        val_split=0.0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=max(1, int(args.prefetch_factor)),
        persistent_workers=bool(args.persistent_workers and not args.disable_persistent_workers),
        pin_memory=bool(args.pin_memory and not args.disable_pin_memory),
        device=args.device,
        amp=args.amp,
        channels_last=bool(args.channels_last and not args.disable_channels_last),
        tf32=bool(args.tf32 and not args.disable_tf32),
        cudnn_benchmark=bool(args.cudnn_benchmark and not args.disable_cudnn_benchmark),
        seed=args.seed,
        min_image_bytes=args.min_image_bytes,
        input_size=args.input_size,
        train_random_transforms=False,
        mask_h=args.mask_h,
        mask_w=args.mask_w,
        patch_iou_thresh=args.patch_iou_thresh,
        concept_threshold=args.concept_threshold,
        spatial_target_mode=args.spatial_target_mode,
        spatial_loss_mode=args.spatial_loss_mode,
        filter_concepts_by_count=bool(args.filter_concepts_by_count),
        concept_min_count=max(0, int(args.concept_min_count)),
        concept_min_frequency=float(args.concept_min_frequency),
        concept_max_frequency=float(args.concept_max_frequency),
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        global_pos_weight=args.global_pos_weight,
        patch_pos_weight=args.patch_pos_weight,
        loss_global_w=args.loss_global_w,
        loss_mask_w=args.loss_mask_w,
        loss_dice_w=args.loss_dice_w,
        branch_arch="dual",
        spatial_branch_mode=args.spatial_branch_mode,
        spatial_stage=args.spatial_stage,
        residual_alpha=args.residual_alpha,
        profile_steps=0,
        warmup_steps=0,
        log_every=max(1, int(args.log_every)),
        save_every=max(1, int(args.save_every)),
        skip_final_layer=bool(args.skip_final_layer),
        final_layer_type=args.final_layer_type,
        saga_batch_size=args.saga_batch_size,
        saga_workers=max(0, int(args.saga_workers)),
        saga_prefetch_factor=max(1, int(args.saga_prefetch_factor)),
        saga_step_size=args.saga_step_size,
        saga_lam=args.saga_lam,
        saga_n_iters=args.saga_n_iters,
        saga_verbose_every=max(1, int(args.saga_verbose_every)),
        dense_lr=args.dense_lr,
        dense_n_iters=max(1, int(args.dense_n_iters)),
        feature_storage_dtype=args.feature_storage_dtype,
        saga_table_device=args.saga_table_device,
        vlg_init_path=args.vlg_init_path,
        vlg_concepts_path=args.vlg_concepts_path,
        freeze_global_head=bool(args.freeze_global_head),
        scheduler=args.scheduler,
        print_config=bool(args.print_config),
        residual_spatial_pooling=args.residual_spatial_pooling,
        learn_spatial_residual_scale=bool(args.learn_spatial_residual_scale),
        eval_every=1,
        feature_batch_size=max(1, int(args.feature_batch_size)),
        feature_workers=max(0, int(args.feature_workers)),
        feature_prefetch_factor=max(1, int(args.feature_prefetch_factor)),
    )


def load_run_concepts(cfg: Config) -> List[str]:
    precomputed_concepts = Path(cfg.precomputed_target_dir) / "concepts.txt"
    if precomputed_concepts.is_file():
        return read_concepts(precomputed_concepts)
    return read_concepts(Path(cfg.concept_file))


def build_dataset(
    manifest_path: str,
    *,
    split: str,
    cfg: Config,
    concepts: Sequence[str],
) -> StanfordCarsManifestDataset:
    return StanfordCarsManifestDataset(
        manifest_path=manifest_path,
        annotation_dir=cfg.annotation_dir,
        concepts=concepts,
        split=split,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        train_random_transforms=cfg.train_random_transforms,
    )


def build_run_dir(cfg: Config) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = cfg.run_name or f"gcbm_stanford_cars_{timestamp}"
    run_dir = Path(cfg.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_requested_splits(args: argparse.Namespace) -> List[str]:
    splits = [token.strip() for token in args.precompute_splits.split(",") if token.strip()]
    if not splits:
        raise ValueError("--precompute_splits did not contain any splits")
    return splits


def run_precompute_targets(cfg: Config, args: argparse.Namespace) -> Dict[str, Any]:
    concepts = read_concepts(Path(cfg.concept_file))
    datasets: Dict[str, StanfordCarsManifestDataset] = {}
    for split in load_requested_splits(args):
        manifest_value = getattr(args, f"{split}_manifest", "")
        if not manifest_value:
            continue
        manifest_path = Path(manifest_value)
        if not manifest_path.is_file():
            continue
        datasets[split] = build_dataset(str(manifest_path), split=split, cfg=cfg, concepts=concepts)
    if "train" not in datasets:
        raise ValueError("precompute_targets requires a train manifest")

    concept_filter_summary = apply_count_concept_filter(cfg, datasets["train"], list(datasets.values()))
    output_root = Path(cfg.precomputed_target_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "mode": "precompute_targets",
        "output_root": str(output_root),
        "concept_filter": concept_filter_summary,
        "splits": {},
    }
    for split, dataset in datasets.items():
        result["splits"][split] = precompute_target_store(dataset, output_root, cfg)
    (output_root / "concepts.txt").write_text("\n".join(datasets["train"].concepts) + "\n", encoding="utf-8")
    (output_root / "precompute_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if concept_filter_summary is not None:
        (output_root / "concept_filter_summary.json").write_text(json.dumps(concept_filter_summary, indent=2), encoding="utf-8")
    return result


def maybe_subset(dataset: StanfordCarsManifestDataset, *, max_images: int, seed: int) -> DatasetView:
    indices = select_subset_indices(
        dataset,
        list(range(len(dataset))),
        max_images=max_images,
        seed=seed,
        stratify=True,
    )
    return DatasetView(dataset, indices)


def run_training(cfg: Config) -> Dict[str, Any]:
    start_time = time.perf_counter()
    concepts = load_run_concepts(cfg)
    train_dataset_full = build_dataset(cfg.train_manifest, split="train", cfg=cfg, concepts=concepts)
    val_dataset_full = build_dataset(cfg.val_manifest, split="val", cfg=cfg, concepts=concepts)
    train_dataset_full.attach_precomputed_targets(cfg.precomputed_target_dir, cfg)
    val_dataset_full.attach_precomputed_targets(cfg.precomputed_target_dir, cfg)

    train_dataset = maybe_subset(train_dataset_full, max_images=cfg.max_train_images, seed=cfg.seed)
    val_dataset = maybe_subset(val_dataset_full, max_images=cfg.max_val_images, seed=cfg.seed + 1)
    concept_filter_summary = apply_count_concept_filter(cfg, train_dataset, [train_dataset, val_dataset])
    concepts = list(train_dataset.concepts)

    train_loader = build_loader(train_dataset, cfg, shuffle=True, drop_last=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False, drop_last=False)
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    init_global_head_from_vlg(head, cfg, concepts)
    optimizer = make_optimizer(head, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    scaler = None
    if cfg.amp == "fp16" and str(cfg.device).startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()

    run_dir = build_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (run_dir / "concepts.txt").write_text("\n".join(concepts) + "\n", encoding="utf-8")
    if concept_filter_summary is not None:
        (run_dir / "concept_filter_summary.json").write_text(json.dumps(concept_filter_summary, indent=2), encoding="utf-8")

    best_val = float("inf")
    train_history: List[Dict[str, Any]] = []
    val_history: List[Dict[str, Any]] = []
    best_path = run_dir / "concept_head_best.pt"
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            scaler,
            cfg,
            concept_to_idx=train_dataset.concept_to_idx,
            n_concepts=len(concepts),
            epoch=epoch,
        )
        val_metrics = evaluate_one_epoch(
            backbone,
            head,
            val_loader,
            cfg,
            concept_to_idx=val_dataset.concept_to_idx,
            n_concepts=len(concepts),
            split_name="val",
        )
        train_history.append({"epoch": epoch, **train_metrics})
        val_history.append({"epoch": epoch, **val_metrics})
        print(
            f"[epoch] {epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"train_ips={train_metrics['images_per_second']:.2f} "
            f"val_ips={val_metrics['images_per_second']:.2f}",
            flush=True,
        )
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            torch.save(head.state_dict(), best_path)
        if epoch % cfg.save_every == 0:
            save_checkpoint(run_dir, head, optimizer, epoch, cfg, train_metrics, val_metrics)
        if scheduler is not None:
            scheduler.step()

    torch.save(head.state_dict(), run_dir / "concept_head_last.pt")
    (run_dir / "train_metrics.json").write_text(json.dumps(train_history, indent=2), encoding="utf-8")
    (run_dir / "val_metrics.json").write_text(json.dumps(val_history, indent=2), encoding="utf-8")

    final_layer_summary: Optional[Dict[str, Any]] = None
    if not cfg.skip_final_layer:
        if best_path.is_file():
            head.load_state_dict(torch.load(best_path, map_location=cfg.device))
        feature_dir = Path(cfg.feature_dir).resolve() if cfg.feature_dir else (run_dir / "features")
        feature_train_loader = build_loader(
            train_dataset,
            cfg,
            shuffle=False,
            drop_last=False,
            batch_size=cfg.feature_batch_size,
            workers=cfg.feature_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=cfg.feature_prefetch_factor,
        )
        feature_val_loader = build_loader(
            val_dataset,
            cfg,
            shuffle=False,
            drop_last=False,
            batch_size=cfg.feature_batch_size,
            workers=cfg.feature_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=cfg.feature_prefetch_factor,
        )
        train_feature_path, train_target_path, train_extract_summary = extract_concept_features_to_memmap(
            backbone,
            head,
            feature_train_loader,
            cfg,
            split_name="train",
            output_dir=feature_dir,
        )
        val_feature_path, val_target_path, val_extract_summary = extract_concept_features_to_memmap(
            backbone,
            head,
            feature_val_loader,
            cfg,
            split_name="val",
            output_dir=feature_dir,
        )
        feature_mean, feature_std, norm_summary = compute_feature_stats_memmap(train_feature_path, cfg)
        normalization_payload = {
            "mean": feature_mean,
            "std": feature_std,
            "train_extraction": train_extract_summary,
            "val_extraction": val_extract_summary,
            "normalization": norm_summary,
        }
        torch.save(normalization_payload, run_dir / "final_layer_normalization.pt")
        final_layer_fn = train_dense_final_layer if cfg.final_layer_type == "dense" else train_sparse_final_layer
        final_layer_summary = final_layer_fn(
            train_feature_path=train_feature_path,
            train_target_path=train_target_path,
            val_feature_path=val_feature_path,
            val_target_path=val_target_path,
            feature_mean=feature_mean,
            feature_std=feature_std,
            cfg=cfg,
            n_classes=len(train_dataset_full.dataset.classes),
            run_dir=run_dir,
        )
        final_layer_summary["type"] = cfg.final_layer_type
        final_layer_summary["feature_extraction"] = {
            "train": train_extract_summary,
            "val": val_extract_summary,
            "normalization": norm_summary,
        }
        (run_dir / "final_layer_summary.json").write_text(json.dumps(final_layer_summary, indent=2), encoding="utf-8")

    result = {
        "mode": "train",
        "run_dir": str(run_dir),
        "best_val_loss": best_val,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "n_concepts": len(concepts),
        "concept_filter": concept_filter_summary,
        "final_layer": final_layer_summary,
        "elapsed_sec": time.perf_counter() - start_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    configure_runtime(cfg)
    if cfg.print_config:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    if cfg.mode == "precompute_targets":
        result = run_precompute_targets(cfg, args)
    else:
        result = run_training(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
