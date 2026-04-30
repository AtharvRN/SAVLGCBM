import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils


VAL_RE = "ILSVRC2012_val_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate old-format VLG/standard sparse ImageNet model across NEC levels.")
    parser.add_argument("--load_dir", required=True, help="Directory containing W_g.pt, b_g.pt, proj_mean.pt, proj_std.pt, args.txt.")
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--devkit_dir", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=5000)
    parser.add_argument("--truncation_mode", choices=["global", "per_class"], default="global")
    parser.add_argument("--save_truncated_weights", action="store_true")
    return parser.parse_args()


def parse_nec_values(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--nec_values must contain at least one integer")
    return values


def load_val_targets(devkit_dir: Path) -> List[int]:
    payload = loadmat(devkit_dir / "data" / "meta.mat", squeeze_me=True, struct_as_record=False)
    synsets = payload["synsets"]
    id_to_wnid: Dict[int, str] = {}
    for syn in synsets:
        ilsvrc_id = int(syn.ILSVRC2012_ID)
        if 1 <= ilsvrc_id <= 1000 and int(syn.num_children) == 0:
            id_to_wnid[ilsvrc_id] = str(syn.WNID)
    if len(id_to_wnid) != 1000:
        raise RuntimeError(f"expected 1000 leaf synsets, got {len(id_to_wnid)}")
    wnids = sorted(id_to_wnid.values())
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    labels: List[int] = []
    with (devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                labels.append(class_to_idx[id_to_wnid[int(line)]])
    if len(labels) != 50000:
        raise RuntimeError(f"expected 50000 validation labels, got {len(labels)}")
    return labels


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def iter_tar_samples(val_tar: Path, targets: Sequence[int], transform: transforms.Compose):
    seen = 0
    with tarfile.open(val_tar, "r|") as tf:
        for member in tf:
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            if not (image_name.startswith(VAL_RE) and image_name.endswith(".JPEG")):
                continue
            image_idx = int(image_name[-13:-5])
            target = int(targets[image_idx - 1])
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                tensor = transform(image.convert("RGB"))
            seen += 1
            yield tensor, target, member.name
    if seen != 50000:
        raise RuntimeError(f"expected 50000 val images in tar, found {seen}")


def truncate_global(weight: torch.Tensor, nec: int) -> torch.Tensor:
    total_keep = int(nec) * int(weight.shape[0])
    total_keep = max(0, min(total_keep, int(weight.numel())))
    if total_keep == 0:
        return torch.zeros_like(weight)
    if total_keep == int(weight.numel()):
        return weight.clone().detach()
    threshold = weight.abs().flatten().kthvalue(int(weight.numel()) - total_keep + 1).values
    sparse_weight = weight.clone().detach()
    sparse_weight[weight.abs() < threshold] = 0
    return sparse_weight


def truncate_per_class(weight: torch.Tensor, nec: int) -> torch.Tensor:
    k = max(0, min(int(nec), int(weight.shape[1])))
    if k == 0:
        return torch.zeros_like(weight)
    if k == int(weight.shape[1]):
        return weight.clone().detach()
    topk_idx = weight.abs().topk(k=k, dim=1).indices
    sparse_weight = torch.zeros_like(weight)
    sparse_weight.scatter_(1, topk_idx, weight.gather(1, topk_idx))
    return sparse_weight


def build_weight_sweep(
    weight: torch.Tensor,
    bias: torch.Tensor,
    nec_values: Sequence[int],
    mode: str,
    save_dir: Path | None,
) -> List[Dict[str, Any]]:
    sweep = []
    for nec in nec_values:
        truncated = truncate_global(weight, nec) if mode == "global" else truncate_per_class(weight, nec)
        nnz = int((truncated.abs() > 1e-5).sum().item())
        if save_dir is not None:
            torch.save(truncated.cpu(), save_dir / f"W_g@NEC={int(nec)}.{mode}.pt")
            torch.save(bias.cpu(), save_dir / f"b_g@NEC={int(nec)}.{mode}.pt")
        sweep.append(
            {
                "nec": int(nec),
                "weight": truncated,
                "bias": bias,
                "nnz": nnz,
                "total": int(truncated.numel()),
                "weight_sparsity": 1.0 - (nnz / max(int(truncated.numel()), 1)),
            }
        )
    return sweep


def build_backbone(backbone_name: str, feature_layer: str, device: str):
    model, _ = data_utils.get_target_model(backbone_name, device)
    feature_vals: Dict[torch.device, torch.Tensor] = {}

    def hook(_module, _input, output):
        feature_vals[output.device] = output

    target_module = model
    for part in feature_layer.split("."):
        target_module = getattr(target_module, part)
    target_module.register_forward_hook(hook)
    model.eval()

    def forward(images: torch.Tensor) -> torch.Tensor:
        out = model(images)
        feats = feature_vals[out.device]
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        return torch.flatten(feats, 1).float()

    return forward


def evaluate(
    load_dir: Path,
    val_tar: Path,
    devkit_dir: Path,
    nec_values: Sequence[int],
    batch_size: int,
    device: str,
    log_every: int,
    truncation_mode: str,
    save_truncated_weights: bool,
) -> Dict[str, Any]:
    with (load_dir / "args.txt").open("r", encoding="utf-8") as handle:
        model_args = json.load(handle)

    weight = torch.load(load_dir / "W_g.pt", map_location="cpu").float()
    bias = torch.load(load_dir / "b_g.pt", map_location="cpu").float()
    proj_mean = torch.load(load_dir / "proj_mean.pt", map_location=device).float()
    proj_std = torch.load(load_dir / "proj_std.pt", map_location=device).float().clamp_min(1e-6)
    sweep = build_weight_sweep(
        weight,
        bias,
        nec_values,
        truncation_mode,
        load_dir if save_truncated_weights else None,
    )
    stacked_weights = torch.stack([item["weight"].to(device).float() for item in sweep], dim=0)
    stacked_biases = torch.stack([item["bias"].to(device).float() for item in sweep], dim=0)
    top1 = torch.zeros(len(sweep), dtype=torch.long)
    top5 = torch.zeros(len(sweep), dtype=torch.long)

    backbone = build_backbone(model_args["backbone"], model_args.get("feature_layer", "layer4"), device)
    targets = load_val_targets(devkit_dir)
    transform = build_transform()

    total = 0
    next_log = max(int(log_every), 1)
    start = time.perf_counter()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    images: List[torch.Tensor] = []
    batch_targets: List[int] = []

    def flush(last_name: str | None) -> None:
        nonlocal next_log, total
        if not images:
            return
        batch = torch.stack(images, dim=0).to(device, non_blocking=True)
        target_tensor = torch.tensor(batch_targets, dtype=torch.long, device=device)
        with torch.no_grad():
            feats = backbone(batch)
            normalized = (feats - proj_mean) / proj_std
            logits = torch.einsum("bc,koc->kbo", normalized, stacked_weights) + stacked_biases[:, None, :]
            pred1 = logits.argmax(dim=-1)
            pred5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            match1 = pred1.eq(target_tensor.unsqueeze(0))
            match5 = pred5.eq(target_tensor.view(1, -1, 1)).any(dim=-1)
            top1.add_(match1.sum(dim=1).cpu())
            top5.add_(match5.sum(dim=1).cpu())
        total += len(images)
        if total >= next_log:
            elapsed = time.perf_counter() - start
            print(f"[vlg-nec-eval] n={total} ips={total / max(elapsed, 1e-6):.2f} last={last_name}", flush=True)
            while next_log <= total:
                next_log += max(int(log_every), 1)
        images.clear()
        batch_targets.clear()

    for image, target, name in iter_tar_samples(val_tar, targets, transform):
        images.append(image)
        batch_targets.append(int(target))
        if len(images) >= batch_size:
            flush(name)
    flush(None)

    elapsed = time.perf_counter() - start
    results = []
    for idx, item in enumerate(sweep):
        results.append(
            {
                "nec": item["nec"],
                "nnz": item["nnz"],
                "total": item["total"],
                "weight_sparsity": item["weight_sparsity"],
                "top1": float(top1[idx].item() / max(total, 1)),
                "top5": float(top5[idx].item() / max(total, 1)),
            }
        )
    return {
        "load_dir": str(load_dir),
        "val_tar": str(val_tar),
        "devkit_dir": str(devkit_dir),
        "model_args": model_args,
        "truncation_mode": truncation_mode,
        "nec_values": list(nec_values),
        "n": total,
        "elapsed_sec": elapsed,
        "images_per_second": total / max(elapsed, 1e-6),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    load_dir = Path(args.load_dir).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else load_dir / f"nec_eval_{args.truncation_mode}.json"
    payload = evaluate(
        load_dir=load_dir,
        val_tar=Path(args.val_tar).resolve(),
        devkit_dir=Path(args.devkit_dir).resolve(),
        nec_values=parse_nec_values(args.nec_values),
        batch_size=args.batch_size,
        device=args.device,
        log_every=args.log_every,
        truncation_mode=args.truncation_mode,
        save_truncated_weights=args.save_truncated_weights,
    )
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
