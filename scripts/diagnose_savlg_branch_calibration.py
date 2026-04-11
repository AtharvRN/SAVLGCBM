import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import utils as data_utils
from methods.savlg import (
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    create_savlg_splits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)


def _load_args(load_path: str, device: str, annotation_dir: str | None):
    with open(os.path.join(load_path, "args.txt"), "r") as f:
        payload = json.load(f)
    payload["device"] = device
    if annotation_dir is not None:
        payload["annotation_dir"] = annotation_dir
    return SimpleNamespace(**payload)


def _load_checkpoint_concepts(load_path: str):
    concepts_path = os.path.join(load_path, "concepts.txt")
    if os.path.exists(concepts_path):
        return data_utils.get_concepts(concepts_path, None)
    return None


def _infer_num_concepts_from_state_dict(state_dict: dict) -> int:
    for key in (
        "global_layer.model.0.weight",
        "global_layer.weight",
        "spatial_layer.weight",
    ):
        weight = state_dict.get(key)
        if weight is not None:
            return int(weight.shape[0])
    raise RuntimeError("Could not infer concept count from SAVLG concept_layer.pt")


def _summarize_tensor(x: torch.Tensor):
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "mean_abs": float(x.abs().mean().item()),
        "median_abs": float(x.abs().median().item()),
        "max_abs": float(x.abs().max().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    args_ns = parser.parse_args()

    args = _load_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    train_raw, val_raw, train_dataset, val_dataset, test_dataset, backbone = create_savlg_splits(args)
    split_to_dataset = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    dataset = split_to_dataset[args_ns.split]
    if args_ns.max_images is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args_ns.max_images, len(dataset)))))

    state_dict = torch.load(os.path.join(args_ns.load_path, "concept_layer.pt"), map_location=args.device)
    ckpt_concepts = _load_checkpoint_concepts(args_ns.load_path)
    num_concepts = _infer_num_concepts_from_state_dict(state_dict)
    if ckpt_concepts is not None and len(ckpt_concepts) != num_concepts:
        raise RuntimeError(
            f"Checkpoint concepts.txt length ({len(ckpt_concepts)}) does not match concept_layer.pt ({num_concepts})."
        )
    concepts = ckpt_concepts if ckpt_concepts is not None else list(range(num_concepts))
    concept_layer = build_savlg_concept_layer(args, backbone, num_concepts).to(args.device)
    concept_layer.load_state_dict(state_dict)
    concept_layer.eval()
    backbone.eval()

    loader = DataLoader(dataset, batch_size=args_ns.batch_size, shuffle=False, num_workers=args_ns.num_workers)

    all_global = []
    all_spatial = []
    all_fused = []
    all_ratio = []
    all_scaled_ratio = []
    all_sign_agree = []
    all_sign_agree_nonzero = []
    all_spatial_dominates = []

    alpha = float(getattr(args, "savlg_residual_spatial_alpha", 0.0))
    eps = 1e-6

    with torch.no_grad():
        for images, _targets in tqdm(loader, desc="SAVLG branch calibration"):
            images = images.to(args.device, non_blocking=True)
            feats = forward_savlg_backbone(backbone, images, args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            global_logits, spatial_logits, final_logits = compute_savlg_concept_logits(global_outputs, spatial_maps, args)

            g = global_logits.detach().float().cpu()
            s = spatial_logits.detach().float().cpu()
            f = final_logits.detach().float().cpu()
            scaled_s = alpha * s

            all_global.append(g)
            all_spatial.append(s)
            all_fused.append(f)
            all_ratio.append((s.abs() / (g.abs() + eps)).reshape(-1))
            all_scaled_ratio.append((scaled_s.abs() / (g.abs() + eps)).reshape(-1))

            sign_agree = torch.sign(g) == torch.sign(s)
            all_sign_agree.append(sign_agree.float().reshape(-1))

            nonzero_mask = (g.abs() > eps) | (s.abs() > eps)
            if bool(nonzero_mask.any()):
                all_sign_agree_nonzero.append(sign_agree[nonzero_mask].float().reshape(-1))

            all_spatial_dominates.append((scaled_s.abs() > g.abs()).float().reshape(-1))

    global_all = torch.cat(all_global, dim=0)
    spatial_all = torch.cat(all_spatial, dim=0)
    fused_all = torch.cat(all_fused, dim=0)
    ratio_all = torch.cat(all_ratio, dim=0)
    scaled_ratio_all = torch.cat(all_scaled_ratio, dim=0)
    sign_agree_all = torch.cat(all_sign_agree, dim=0)
    sign_agree_nonzero_all = torch.cat(all_sign_agree_nonzero, dim=0) if all_sign_agree_nonzero else torch.empty(0)
    spatial_dom_all = torch.cat(all_spatial_dominates, dim=0)

    result = {
        "load_path": args_ns.load_path,
        "split": args_ns.split,
        "num_images": int(global_all.shape[0]),
        "num_concepts": int(global_all.shape[1]),
        "alpha": alpha,
        "global_logits": _summarize_tensor(global_all),
        "spatial_logits": _summarize_tensor(spatial_all),
        "scaled_spatial_logits": _summarize_tensor(alpha * spatial_all),
        "fused_logits": _summarize_tensor(fused_all),
        "abs_ratio_spatial_over_global": _summarize_tensor(ratio_all),
        "abs_ratio_scaled_spatial_over_global": _summarize_tensor(scaled_ratio_all),
        "sign_agreement_fraction": float(sign_agree_all.mean().item()),
        "sign_agreement_fraction_nonzero": float(sign_agree_nonzero_all.mean().item()) if sign_agree_nonzero_all.numel() > 0 else None,
        "scaled_spatial_dominates_fraction": float(spatial_dom_all.mean().item()),
        "per_concept": {
            "global_mean_abs": global_all.abs().mean(dim=0).tolist(),
            "spatial_mean_abs": spatial_all.abs().mean(dim=0).tolist(),
            "scaled_spatial_mean_abs": (alpha * spatial_all).abs().mean(dim=0).tolist(),
            "fused_mean_abs": fused_all.abs().mean(dim=0).tolist(),
            "sign_agreement_fraction": (torch.sign(global_all) == torch.sign(spatial_all)).float().mean(dim=0).tolist(),
            "scaled_spatial_dominates_fraction": ((alpha * spatial_all).abs() > global_all.abs()).float().mean(dim=0).tolist(),
        },
    }

    output_path = Path(args_ns.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
