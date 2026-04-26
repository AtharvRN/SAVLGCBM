import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_saga.elasticnet import glm_saga
from scripts.train_savlg_imagenet_standalone import Config, MemmapFeatureDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone GLM-SAGA lambda path sweep on saved SAVLG features.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saga_batch_size", type=int, default=512)
    parser.add_argument("--saga_workers", type=int, default=4)
    parser.add_argument("--saga_prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--lam_max", type=float, default=0.01)
    parser.add_argument("--max_glm_steps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--table_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    return parser.parse_args()


def parse_nec_values(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--nec_values must contain at least one integer")
    return values


def resolve_source_run_dir(artifact_dir: Path) -> Path:
    source_run_file = artifact_dir / "source_run_dir.txt"
    if source_run_file.exists():
        source_run_dir = Path(source_run_file.read_text().strip()).resolve()
        if source_run_dir.is_dir():
            return source_run_dir
    return artifact_dir


def load_config(config_dir: Path, device: str) -> Config:
    payload = json.loads((config_dir / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload["device"] = device
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    return Config(**payload)


def infer_n_classes(*target_paths: Path) -> int:
    max_class_id = -1
    for target_path in target_paths:
        targets = torch.from_numpy(np.load(target_path, mmap_mode="r"))
        if int(targets.shape[0]) == 0:
            continue
        max_class_id = max(max_class_id, int(targets.max().item()))
    if max_class_id < 0:
        raise RuntimeError("Could not infer class count from target files")
    return max_class_id + 1


def build_loader(
    feature_path: Path,
    target_path: Path,
    *,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    mean,
    std,
    include_index: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = MemmapFeatureDataset(
        feature_path=feature_path,
        target_path=target_path,
        mean=mean,
        std=std,
        include_index=include_index,
    )
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**kwargs)


def select_path_points_for_nec(path: Sequence[Dict[str, Any]], n_concepts: int, nec_values: Sequence[int]) -> List[Dict[str, Any]]:
    sparsities = [float((params["weight"].abs() > 1e-5).float().mean().item()) for params in path]
    selections: List[Dict[str, Any]] = []
    for nec in nec_values:
        target_sparsity = float(nec) / float(n_concepts)
        selected_idx = len(path) - 1
        for idx, sparsity in enumerate(sparsities):
            if sparsity >= target_sparsity:
                selected_idx = idx
                break
        params = path[selected_idx]
        weight = params["weight"]
        nnz = int((weight.abs() > 1e-5).sum().item())
        selections.append(
            {
                "nec": int(nec),
                "target_sparsity": target_sparsity,
                "path_index": int(selected_idx),
                "lambda": float(params["lam"]),
                "lr": float(params["lr"]),
                "nnz": nnz,
                "total": int(weight.numel()),
                "weight_sparsity": 1.0 - (nnz / max(int(weight.numel()), 1)),
                "metrics": params["metrics"],
            }
        )
    return selections


def serializable_path(path: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = []
    for idx, params in enumerate(path):
        weight = params["weight"]
        nnz = int((weight.abs() > 1e-5).sum().item())
        payload.append(
            {
                "index": idx,
                "lambda": float(params["lam"]),
                "lr": float(params["lr"]),
                "alpha": float(params["alpha"]),
                "time": float(params["time"]),
                "nnz": nnz,
                "total": int(weight.numel()),
                "weight_sparsity": 1.0 - (nnz / max(int(weight.numel()), 1)),
                "metrics": params["metrics"],
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    source_run_dir = resolve_source_run_dir(artifact_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else artifact_dir / f"glm_path_sweep_lam{str(args.lam_max).replace('.', 'p')}_k{args.max_glm_steps}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(source_run_dir, args.device)
    normalization_payload = torch.load(source_run_dir / "final_layer_normalization.pt", map_location="cpu")
    feature_mean = normalization_payload["mean"].float().cpu().numpy()
    feature_std = normalization_payload["std"].float().cpu().numpy()

    feature_root = artifact_dir / "features"
    if not feature_root.is_dir():
        feature_root = source_run_dir / "features"
    train_feature_path = feature_root / "train_features.npy"
    train_target_path = feature_root / "train_targets.npy"
    val_feature_path = feature_root / "val_features.npy"
    val_target_path = feature_root / "val_targets.npy"
    for path in (train_feature_path, train_target_path, val_feature_path, val_target_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing feature artifact: {path}")

    train_loader = build_loader(
        train_feature_path,
        train_target_path,
        batch_size=args.saga_batch_size,
        workers=args.saga_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.saga_prefetch_factor,
        mean=feature_mean,
        std=feature_std,
        include_index=True,
        shuffle=True,
    )
    val_loader = build_loader(
        val_feature_path,
        val_target_path,
        batch_size=args.saga_batch_size,
        workers=args.saga_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.saga_prefetch_factor,
        mean=feature_mean,
        std=feature_std,
        include_index=False,
        shuffle=False,
    )

    n_features = int(train_loader.dataset.features.shape[1])
    n_classes = infer_n_classes(train_target_path, val_target_path)
    linear = torch.nn.Linear(n_features, n_classes, bias=True).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    metadata = {"max_reg": {"nongrouped": args.lam_max}}
    start = time.perf_counter()
    output = glm_saga(
        linear,
        train_loader,
        args.step_size,
        args.n_iters,
        args.alpha,
        table_device=args.table_device,
        epsilon=args.epsilon,
        k=args.max_glm_steps,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_loader.dataset),
        n_classes=n_classes,
        verbose=args.verbose_every,
        eval_train=True,
        eval_val=True,
        eval_test=False,
    )
    elapsed = time.perf_counter() - start

    path = output["path"]
    best = output["best"]
    nec_values = parse_nec_values(args.nec_values)
    nec_selection = select_path_points_for_nec(path, n_features, nec_values)

    for item in nec_selection:
        params = path[item["path_index"]]
        torch.save(params["weight"].cpu(), output_dir / f"W_g@NEC={item['nec']}.pt")
        torch.save(params["bias"].cpu(), output_dir / f"b_g@NEC={item['nec']}.pt")

    torch.save({"path": path, "best": best}, output_dir / "glm_path.pt")

    payload = {
        "artifact_dir": str(artifact_dir),
        "source_run_dir": str(source_run_dir),
        "feature_root": str(feature_root),
        "output_dir": str(output_dir),
        "n_features": n_features,
        "n_classes": n_classes,
        "config": {
            "saga_batch_size": args.saga_batch_size,
            "saga_workers": args.saga_workers,
            "saga_prefetch_factor": args.saga_prefetch_factor,
            "step_size": args.step_size,
            "n_iters": args.n_iters,
            "lam_max": args.lam_max,
            "max_glm_steps": args.max_glm_steps,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "table_device": args.table_device,
            "verbose_every": args.verbose_every,
            "pin_memory": bool(args.pin_memory),
        },
        "elapsed_sec": elapsed,
        "best": {
            "lambda": float(best["lam"]),
            "lr": float(best["lr"]),
            "alpha": float(best["alpha"]),
            "time": float(best["time"]),
            "metrics": best["metrics"],
        },
        "path": serializable_path(path),
        "nec_selection": nec_selection,
    }
    (output_dir / "source_run_dir.txt").write_text(f"{source_run_dir}\n")
    (output_dir / "glm_path_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
