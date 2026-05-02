import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_savlg_imagenet_standalone import (
    Config,
    amp_dtype,
    build_model,
    configure_runtime,
    cuda_peak_stats_mb,
    prepare_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standalone SAVLG-CBM on full ImageNet val.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--val_root", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_source_run_dir(artifact_dir: Path) -> Path:
    source_run_file = artifact_dir / "source_run_dir.txt"
    if source_run_file.exists():
        source_run_dir = Path(source_run_file.read_text().strip()).resolve()
        if source_run_dir.is_dir():
            return source_run_dir
    return artifact_dir


def load_run_config(config_dir: Path, args: argparse.Namespace) -> Config:
    payload = json.loads((config_dir / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload.setdefault("train_random_transforms", True)
    payload["device"] = args.device
    payload["workers"] = args.workers
    payload["batch_size"] = args.batch_size
    payload["prefetch_factor"] = args.prefetch_factor
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    return Config(**payload)


def resolve_final_layer_path(artifact_dir: Path) -> Path:
    dense_path = artifact_dir / "final_layer_dense.pt"
    if dense_path.exists():
        return dense_path
    glm_path = artifact_dir / "final_layer_glm_saga.pt"
    if glm_path.exists():
        return glm_path
    raise FileNotFoundError(f"no final layer artifact found under {artifact_dir}")


def build_val_loader(val_root: Path, cfg: Config) -> DataLoader:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    dataset = ImageFolder(
        root=str(val_root),
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(cfg.input_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.workers,
        "pin_memory": cfg.pin_memory,
    }
    if cfg.workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(**kwargs)


def evaluate(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    final_layer: torch.nn.Linear,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    loader: DataLoader,
    cfg: Config,
) -> Dict[str, Any]:
    backbone.eval()
    head.eval()
    final_layer.eval()
    top1 = 0
    top5 = 0
    total = 0
    reset_start = time.perf_counter()
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader, start=1):
            images = prepare_images(images, cfg)
            targets = targets.to(cfg.device, non_blocking=cfg.pin_memory)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(cfg.amp),
                enabled=(str(cfg.device).startswith("cuda") and amp_dtype(cfg.amp) is not None),
            ):
                feats = backbone(images)
                outputs = head(feats)
                concept_logits = outputs["final_logits"].float()
                concept_logits = (concept_logits - feature_mean) / feature_std
                logits = final_layer(concept_logits)
            pred1 = logits.argmax(dim=-1)
            top1 += int((pred1 == targets).sum().item())
            pred5 = logits.topk(k=min(5, logits.shape[1]), dim=-1).indices
            top5 += int((pred5 == targets[:, None]).any(dim=1).sum().item())
            total += int(targets.numel())
            if step % 50 == 0:
                elapsed = time.perf_counter() - reset_start
                ips = total / max(elapsed, 1e-6)
                print(f"[eval] step={step}/{len(loader)} n={total} ips={ips:.2f}", flush=True)
    elapsed = time.perf_counter() - reset_start
    payload = {
        "n": total,
        "top1": top1 / max(total, 1),
        "top5": top5 / max(total, 1),
        "elapsed_sec": elapsed,
        "images_per_second": total / max(elapsed, 1e-6),
    }
    payload.update(cuda_peak_stats_mb(cfg))
    return payload


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    val_root = Path(args.val_root).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else artifact_dir / "full_val_eval.json"
    source_run_dir = resolve_source_run_dir(artifact_dir)

    cfg = load_run_config(source_run_dir, args)
    configure_runtime(cfg)

    concepts = [line.strip() for line in (source_run_dir / "concepts.txt").read_text().splitlines() if line.strip()]
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))

    final_layer_path = resolve_final_layer_path(artifact_dir)
    linear_payload = torch.load(final_layer_path, map_location="cpu")
    normalization_payload = torch.load(artifact_dir / "final_layer_normalization.pt", map_location="cpu")
    feature_mean = normalization_payload["mean"].to(cfg.device).float()
    feature_std = normalization_payload["std"].to(cfg.device).float().clamp_min(1e-6)

    val_loader = build_val_loader(val_root, cfg)
    n_classes = len(val_loader.dataset.classes)
    final_layer = torch.nn.Linear(feature_mean.numel(), n_classes, bias=True).to(cfg.device)
    final_layer.load_state_dict(
        {
            "weight": linear_payload["weight"].to(cfg.device).float(),
            "bias": linear_payload["bias"].to(cfg.device).float(),
        }
    )

    payload = {
        "artifact_dir": str(artifact_dir),
        "final_layer_path": str(final_layer_path),
        "val_root": str(val_root),
        "metrics": evaluate(backbone, head, final_layer, feature_mean, feature_std, val_loader, cfg),
    }
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
