import argparse
import inspect
import json
import random
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_savlg_imagenet_standalone import (  # noqa: E402
    DatasetView,
    SafeImageFolderWithAnnotations,
    build_loader,
    build_model,
    compute_feature_stats_memmap,
    configure_runtime,
    feature_storage_dtype,
    prepare_images,
    split_train_val,
    autocast_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a SAVLG feature-key artifact for standalone GLM training."
    )
    parser.add_argument("--source_run_dir", required=True, help="Trained SAVLG run with config.json and concept_head_best.pt.")
    parser.add_argument("--output_dir", required=True, help="Output artifact dir with features/ and final_layer_normalization.pt.")
    parser.add_argument("--feature_key", choices=["global_logits", "spatial_logits", "final_logits"], default="global_logits")
    parser.add_argument("--checkpoint_name", default="concept_head_best.pt")
    parser.add_argument("--train_root", default="", help="Override train_root from source config.")
    parser.add_argument("--train_manifest", default="", help="Override train_manifest from source config.")
    parser.add_argument("--annotation_dir", default="", help="Override annotation_dir from source config.")
    parser.add_argument("--precomputed_target_dir", default="", help="Override precomputed_target_dir from source config.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="")
    parser.add_argument("--feature_storage_dtype", choices=["fp16", "fp32"], default="")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--max_train_images", type=int, default=-1, help="-1 keeps source config; >0 overrides.")
    parser.add_argument("--max_val_images", type=int, default=-1, help="-1 keeps source config; >0 overrides.")
    return parser.parse_args()


def load_config(source_run_dir: Path, args: argparse.Namespace) -> SimpleNamespace:
    payload = json.loads((source_run_dir / "config.json").read_text())
    cfg = SimpleNamespace(**payload)
    cfg.mode = "train"
    cfg.reuse_run_dir = getattr(cfg, "reuse_run_dir", "")
    cfg.feature_dir = ""
    cfg.skip_final_layer = True
    cfg.print_config = False
    cfg.batch_size = int(args.batch_size)
    cfg.workers = int(args.workers)
    cfg.prefetch_factor = int(args.prefetch_factor)
    cfg.persistent_workers = bool(args.workers > 0)
    cfg.pin_memory = False
    cfg.device = str(args.device)
    if args.amp:
        cfg.amp = str(args.amp)
    if args.feature_storage_dtype:
        cfg.feature_storage_dtype = str(args.feature_storage_dtype)
    cfg.log_every = int(args.log_every)
    if not hasattr(cfg, "train_manifest"):
        cfg.train_manifest = ""
    if not hasattr(cfg, "persistent_workers"):
        cfg.persistent_workers = False
    if not hasattr(cfg, "prefetch_factor"):
        cfg.prefetch_factor = 2
    if not hasattr(cfg, "feature_storage_dtype"):
        cfg.feature_storage_dtype = "fp16"
    if not hasattr(cfg, "channels_last"):
        cfg.channels_last = True
    if not hasattr(cfg, "tf32"):
        cfg.tf32 = True
    if not hasattr(cfg, "cudnn_benchmark"):
        cfg.cudnn_benchmark = True
    if not hasattr(cfg, "amp"):
        cfg.amp = "fp16"
    if args.train_root:
        cfg.train_root = str(args.train_root)
    if args.train_manifest:
        cfg.train_manifest = str(args.train_manifest)
    if args.annotation_dir:
        cfg.annotation_dir = str(args.annotation_dir)
    if args.precomputed_target_dir:
        cfg.precomputed_target_dir = str(args.precomputed_target_dir)
    if int(args.max_train_images) >= 0:
        cfg.max_train_images = int(args.max_train_images)
    if int(args.max_val_images) >= 0:
        cfg.max_val_images = int(args.max_val_images)
    return cfg


def set_local_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_source_concepts(source_run_dir: Path) -> list[str]:
    concepts_path = source_run_dir / "concepts.txt"
    if not concepts_path.exists():
        raise FileNotFoundError(f"Missing concepts.txt in source run: {concepts_path}")
    return [line.strip() for line in concepts_path.read_text().splitlines() if line.strip()]


def make_datasets(cfg: Any, source_run_dir: Path):
    concepts = load_source_concepts(source_run_dir)
    dataset_kwargs = {
        "root": cfg.train_root,
        "annotation_dir": cfg.annotation_dir,
        "concepts": concepts,
        "input_size": cfg.input_size,
        "min_image_bytes": cfg.min_image_bytes,
        "split": "train",
    }
    if "manifest" in inspect.signature(SafeImageFolderWithAnnotations).parameters:
        dataset_kwargs["manifest"] = cfg.train_manifest
    train_dataset_full = SafeImageFolderWithAnnotations(
        **dataset_kwargs,
    )
    # Targets are not needed for feature extraction, but attaching them catches
    # split/config mismatches when the precomputed store is available.
    if cfg.precomputed_target_dir:
        train_dataset_full.attach_precomputed_targets(cfg.precomputed_target_dir)
    if cfg.val_root:
        val_dataset_kwargs = {
            "root": cfg.val_root,
            "annotation_dir": cfg.annotation_dir,
            "concepts": concepts,
            "input_size": cfg.input_size,
            "min_image_bytes": cfg.min_image_bytes,
            "split": "val",
        }
        if "manifest" in inspect.signature(SafeImageFolderWithAnnotations).parameters:
            val_dataset_kwargs["manifest"] = ""
        val_dataset_full = SafeImageFolderWithAnnotations(
            **val_dataset_kwargs,
        )
        if cfg.precomputed_target_dir:
            val_dataset_full.attach_precomputed_targets(cfg.precomputed_target_dir)
        train_indices = list(range(len(train_dataset_full)))
        if cfg.max_train_images > 0:
            train_indices = train_indices[: cfg.max_train_images]
        val_indices = list(range(len(val_dataset_full)))
        if cfg.max_val_images > 0:
            val_indices = val_indices[: cfg.max_val_images]
        train_dataset = DatasetView(train_dataset_full, train_indices)
        val_dataset = DatasetView(val_dataset_full, val_indices)
    else:
        train_dataset, val_dataset = split_train_val(
            train_dataset_full,
            val_split=cfg.val_split,
            max_train_images=cfg.max_train_images,
            max_val_images=cfg.max_val_images,
            seed=cfg.seed,
        )
    return train_dataset_full, train_dataset, val_dataset


def extract_feature_key_to_memmap(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    loader: DataLoader,
    cfg: Any,
    split_name: str,
    output_dir: Path,
    feature_key: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    backbone.eval()
    head.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_examples = len(loader.dataset)
    target_path = output_dir / f"{split_name}_targets.npy"
    feature_path: Optional[Path] = None
    target_memmap = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.int64, shape=(total_examples,))
    feature_memmap: Optional[np.memmap] = None
    offset = 0
    start_time = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["images"], cfg)
        with torch.no_grad(), autocast_context(cfg):
            outputs = head(backbone(images))
        if feature_key not in outputs:
            raise KeyError(f"head output does not contain {feature_key}; keys={sorted(outputs)}")
        batch_features = outputs[feature_key].detach().float().cpu().numpy()
        batch_targets = batch["class_ids"].detach().cpu().numpy().astype(np.int64, copy=False)
        batch_size = int(batch_features.shape[0])
        if feature_memmap is None:
            feature_path = output_dir / f"{split_name}_features.npy"
            feature_memmap = np.lib.format.open_memmap(
                feature_path,
                mode="w+",
                dtype=feature_storage_dtype(cfg),
                shape=(total_examples, int(batch_features.shape[1])),
            )
        feature_memmap[offset : offset + batch_size] = batch_features.astype(feature_memmap.dtype, copy=False)
        target_memmap[offset : offset + batch_size] = batch_targets
        offset += batch_size
        if step % 10 == 0:
            feature_memmap.flush()
            target_memmap.flush()
        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{split_name}:{feature_key}] step={step}/{len(loader)} "
                f"n={offset} ips={offset / max(elapsed, 1e-6):.2f}",
                flush=True,
            )
        del images, outputs, batch_features, batch_targets
    if feature_memmap is None or feature_path is None:
        raise RuntimeError(f"No features extracted for split {split_name}")
    feature_memmap.flush()
    target_memmap.flush()
    elapsed = time.perf_counter() - start_time
    summary = {
        "stage": f"{split_name}_{feature_key}_extraction_summary",
        "feature_key": feature_key,
        "n_examples": offset,
        "n_features": int(feature_memmap.shape[1]),
        "images_per_second": offset / max(elapsed, 1e-6),
        "elapsed_sec": elapsed,
        "feature_path": str(feature_path),
        "target_path": str(target_path),
    }
    print(json.dumps(summary), flush=True)
    return feature_path, target_path, summary


def main() -> None:
    args = parse_args()
    source_run_dir = Path(args.source_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    feature_dir = output_dir / "features"
    cfg = load_config(source_run_dir, args)
    set_local_seed(cfg.seed)
    configure_runtime(cfg)

    train_dataset_full, train_dataset, val_dataset = make_datasets(cfg, source_run_dir)
    train_loader = build_loader(
        train_dataset,
        cfg,
        shuffle=False,
        drop_last=False,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )
    val_loader = build_loader(
        val_dataset,
        cfg,
        shuffle=False,
        drop_last=False,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )
    backbone, head = build_model(cfg, n_concepts=len(train_dataset.concepts))
    checkpoint_path = source_run_dir / args.checkpoint_name
    head.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({**vars(cfg), "feature_key": args.feature_key}, indent=2))
    (output_dir / "source_run_dir.txt").write_text(f"{source_run_dir}\n")
    (output_dir / "concepts.txt").write_text("\n".join(train_dataset.concepts))

    train_feature_path, train_target_path, train_summary = extract_feature_key_to_memmap(
        backbone, head, train_loader, cfg, "train", feature_dir, args.feature_key
    )
    val_feature_path, val_target_path, val_summary = extract_feature_key_to_memmap(
        backbone, head, val_loader, cfg, "val", feature_dir, args.feature_key
    )
    feature_mean, feature_std, norm_summary = compute_feature_stats_memmap(train_feature_path, cfg)
    torch.save(
        {
            "mean": feature_mean,
            "std": feature_std,
            "source_run_dir": str(source_run_dir),
            "feature_key": args.feature_key,
            "train_extraction": train_summary,
            "val_extraction": val_summary,
            "normalization": norm_summary,
        },
        output_dir / "final_layer_normalization.pt",
    )
    result = {
        "source_run_dir": str(source_run_dir),
        "output_dir": str(output_dir),
        "feature_key": args.feature_key,
        "n_classes": len(train_dataset_full.dataset.classes),
        "n_concepts": len(train_dataset.concepts),
        "features": {
            "train": str(train_feature_path),
            "train_targets": str(train_target_path),
            "val": str(val_feature_path),
            "val_targets": str(val_target_path),
        },
        "summaries": {
            "train": train_summary,
            "val": val_summary,
            "normalization": norm_summary,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
