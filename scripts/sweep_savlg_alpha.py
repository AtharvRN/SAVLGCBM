import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import utils as data_utils
from methods.savlg import (
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    create_savlg_splits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)


def _load_args(load_path: str) -> SimpleNamespace:
    with open(os.path.join(load_path, "args.txt"), "r") as f:
        payload = json.load(f)
    return SimpleNamespace(**payload)


def _load_concepts(load_path: str):
    with open(os.path.join(load_path, "concepts.txt"), "r") as f:
        return [line.strip() for line in f if line.strip()]


def _build_final_layer(load_path: str, num_concepts: int, num_classes: int, device: str) -> nn.Module:
    final_layer = nn.Linear(num_concepts, num_classes).to(device)
    W_g = torch.load(os.path.join(load_path, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_path, "b_g.pt"), map_location=device)
    final_layer.load_state_dict({"weight": W_g, "bias": b_g}, strict=True)
    final_layer.eval()
    return final_layer


def sweep_alpha(load_path: str, alphas, device: str, batch_size: int, num_workers: int):
    args = _load_args(load_path)
    args.device = device
    args.cbl_batch_size = batch_size
    args.num_workers = num_workers

    concepts = _load_concepts(load_path)
    classes = data_utils.get_classes(args.dataset)

    _, _, _, _, test_dataset, backbone = create_savlg_splits(args)
    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts)).to(device)
    concept_layer.load_state_dict(
        torch.load(os.path.join(load_path, "concept_layer.pt"), map_location=device),
        strict=True,
    )
    concept_layer.eval()
    backbone.eval()

    mean = torch.load(os.path.join(load_path, "proj_mean.pt"), map_location=device)
    std = torch.load(os.path.join(load_path, "proj_std.pt"), map_location=device)
    final_layer = _build_final_layer(load_path, len(concepts), len(classes), device)

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    alpha_values = [float(alpha) for alpha in alphas]
    correct = {alpha: 0 for alpha in alpha_values}
    total = 0

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            feats = forward_savlg_backbone(backbone, images, args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            global_logits, spatial_logits, _ = compute_savlg_concept_logits(
                global_outputs,
                spatial_maps,
                args,
            )
            for alpha in alpha_values:
                final_logits = global_logits + alpha * spatial_logits
                final_logits = (final_logits - mean.to(device)) / std.to(device)
                pred = final_layer(final_logits).argmax(dim=-1)
                correct[alpha] += int((pred == target).sum().item())
            total += int(target.numel())

    results = []
    original_alpha = float(getattr(args, "savlg_residual_spatial_alpha", 0.0))
    for alpha in alpha_values:
        acc = correct[alpha] / max(total, 1)
        results.append({"alpha": float(alpha), "test_accuracy": float(acc)})
    return {
        "load_path": load_path,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "original_alpha": original_alpha,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--alphas", type=str, default="0.0,0.05,0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    payload = sweep_alpha(
        load_path=args.load_path,
        alphas=alphas,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
