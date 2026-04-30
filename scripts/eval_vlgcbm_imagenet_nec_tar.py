import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.cbm import Backbone, ConceptLayer, NormalizationLayer, amp_autocast_context


VAL_PREFIX = "ILSVRC2012_val_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate old VLG-CBM NEC weights on the ImageNet validation tar."
    )
    parser.add_argument("--load_dir", required=True, help="Directory with args.txt, cbl.pt, normalization stats, and W_g@NEC=*.pt.")
    parser.add_argument("--val_tar", required=True, help="Path to ILSVRC2012_img_val.tar.")
    parser.add_argument("--devkit_dir", required=True, help="Path to ILSVRC2012_devkit_t12.")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=5000)
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


def iter_tar_samples(val_tar: Path, targets: Sequence[int], transform):
    seen = 0
    with tarfile.open(val_tar, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            if not (image_name.startswith(VAL_PREFIX) and image_name.endswith(".JPEG")):
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


def evaluate(
    load_dir: Path,
    val_tar: Path,
    devkit_dir: Path,
    nec_values: Sequence[int],
    batch_size: int,
    device: str,
    log_every: int,
) -> Dict[str, object]:
    with (load_dir / "args.txt").open("r", encoding="utf-8") as handle:
        model_args = json.load(handle)

    backbone = Backbone.from_args(str(load_dir), device=device).eval()
    cbl = ConceptLayer.from_pretrained(str(load_dir), device=device).eval()
    normalization = NormalizationLayer.from_pretrained(str(load_dir), device=device).eval()

    weights = []
    biases = []
    weight_meta = []
    for nec in nec_values:
        weight = torch.load(load_dir / f"W_g@NEC={int(nec)}.pt", map_location="cpu").float()
        bias = torch.load(load_dir / f"b_g@NEC={int(nec)}.pt", map_location="cpu").float()
        nnz = int((weight.abs() > 1e-8).sum().item())
        weight_meta.append(
            {
                "nec": int(nec),
                "nnz": nnz,
                "total": int(weight.numel()),
                "weight_sparsity": 1.0 - nnz / max(int(weight.numel()), 1),
            }
        )
        weights.append(weight.to(device))
        biases.append(bias.to(device))
    stacked_weights = torch.stack(weights, dim=0)
    stacked_biases = torch.stack(biases, dim=0)

    targets = load_val_targets(devkit_dir)
    transform = backbone.preprocess
    top1 = torch.zeros(len(nec_values), dtype=torch.long)
    top5 = torch.zeros(len(nec_values), dtype=torch.long)

    total = 0
    next_log = max(int(log_every), 1)
    start = time.perf_counter()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    images: List[torch.Tensor] = []
    labels: List[int] = []

    def flush(last_name: str | None) -> None:
        nonlocal total, next_log
        if not images:
            return
        batch = torch.stack(images, dim=0).to(device, non_blocking=True)
        target = torch.tensor(labels, dtype=torch.long, device=device)
        with torch.no_grad(), amp_autocast_context(device):
            concept_logits = normalization(cbl(backbone(batch))).float()
            logits = torch.einsum("bc,koc->kbo", concept_logits, stacked_weights) + stacked_biases[:, None, :]
            pred1 = logits.argmax(dim=-1)
            pred5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            top1.add_(pred1.eq(target.unsqueeze(0)).sum(dim=1).cpu())
            top5.add_(pred5.eq(target.view(1, -1, 1)).any(dim=-1).sum(dim=1).cpu())
        total += len(images)
        if total >= next_log:
            elapsed = time.perf_counter() - start
            print(f"[vlgcbm-nec-tar] n={total} ips={total / max(elapsed, 1e-6):.2f} last={last_name}", flush=True)
            while next_log <= total:
                next_log += max(int(log_every), 1)
        images.clear()
        labels.clear()

    for image, target, name in iter_tar_samples(val_tar, targets, transform):
        images.append(image)
        labels.append(int(target))
        if len(images) >= batch_size:
            flush(name)
    flush(None)

    elapsed = time.perf_counter() - start
    results = []
    for idx, meta in enumerate(weight_meta):
        results.append(
            {
                **meta,
                "top1": float(top1[idx].item() / max(total, 1)),
                "top5": float(top5[idx].item() / max(total, 1)),
            }
        )
    payload: Dict[str, object] = {
        "load_dir": str(load_dir),
        "val_tar": str(val_tar),
        "devkit_dir": str(devkit_dir),
        "model_args": model_args,
        "nec_values": [int(v) for v in nec_values],
        "n": total,
        "elapsed_sec": elapsed,
        "images_per_second": total / max(elapsed, 1e-6),
        "results": results,
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        payload["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        payload["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return payload


def main() -> None:
    args = parse_args()
    load_dir = Path(args.load_dir).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else load_dir / "nec_eval_imagenet_val_tar.json"
    payload = evaluate(
        load_dir=load_dir,
        val_tar=Path(args.val_tar).resolve(),
        devkit_dir=Path(args.devkit_dir).resolve(),
        nec_values=parse_nec_values(args.nec_values),
        batch_size=args.batch_size,
        device=args.device,
        log_every=args.log_every,
    )
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
