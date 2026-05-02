import argparse
import json
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
from model.cbm import Backbone, ConceptLayer
from scripts.imagenet_annotation_index import (
    build_filename_to_annotation_path,
    load_annotation_payload,
    resolve_val_annotation_dir,
)
from scripts.eval_savlg_imagenet_standalone_localization import (
    boxes_from_masks,
    finalize_threshold_metrics,
    normalize_maps,
    parse_thresholds,
    update_threshold_metrics,
)
from scripts.train_savlg_imagenet_standalone import (
    Config,
    build_gdino_targets,
    load_concepts as load_standalone_concepts,
)


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VLG-CBM Grad-CAM localization on ImageNet val tar using GDINO val annotations."
    )
    parser.add_argument("--load_dir", required=True, help="VLG-CBM dir with args.txt, cbl.pt, concepts.txt.")
    parser.add_argument("--val_tar", required=True)
    parser.add_argument(
        "--annotation_dir",
        required=True,
        help="Directory containing imagenet_val/*.json, or the imagenet_val directory itself.",
    )
    parser.add_argument(
        "--annotation_val_root",
        default="",
        help="Optional reorganized ImageFolder root used when annotations were generated. If set, annotations are matched by filename via ImageFolder ordering.",
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument(
        "--allowed_concept_file",
        default="",
        help="Optional concept file used to filter GDINO annotations to a paper/reference concept set while keeping checkpoint-native output indexing.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradcam_chunk_size", type=int, default=16)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9,mean")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument("--map_normalization", default="minmax", choices=["minmax", "sigmoid", "concept_zscore_minmax"])
    parser.add_argument("--gt_threshold", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=1024)
    return parser.parse_args()

def load_vlg_concepts(load_dir: Path) -> List[str]:
    return [
        data_utils.canonicalize_concept_label(line.strip())
        for line in (load_dir / "concepts.txt").read_text().splitlines()
        if line.strip()
    ]


def load_canonicalized_concepts(concept_file: Path) -> List[str]:
    return [
        data_utils.canonicalize_concept_label(line.strip())
        for line in concept_file.read_text().splitlines()
        if line.strip()
    ]


def load_vlg_concept_layer(load_dir: Path, device: str) -> nn.Module:
    cbl_path = load_dir / "cbl.pt"
    if cbl_path.is_file():
        return ConceptLayer.from_pretrained(str(load_dir), device=device).eval()

    weight_path = load_dir / "W_c.pt"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Expected either {cbl_path} or {weight_path}")
    weight = torch.load(weight_path, map_location=device).float()
    layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=False, device=device)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer.eval()


def load_annotation(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
    filename_to_annotation_path: Dict[str, Path] | None = None,
) -> List[Dict[str, Any]]:
    return load_annotation_payload(
        annotation_val_dir,
        image_index_1based,
        image_name,
        filename_to_annotation_path,
    )


def iter_tar_samples(
    val_tar: Path,
    annotation_val_dir: Path,
    transform,
    max_images: int,
    filename_to_annotation_path: Dict[str, Path] | None = None,
):
    seen = 0
    with tarfile.open(val_tar, "r|*") as tf:
        print(f"[vlg-gradcam-loc] streaming tar members from {val_tar}", flush=True)
        for member in tf:
            if not member.isfile():
                continue
            match = VAL_RE.search(Path(member.name).name)
            if match is None:
                continue
            image_index = int(match.group(1))
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                image = image.convert("RGB")
                image_size = (int(image.size[0]), int(image.size[1]))
                tensor = transform(image)
            annotation = load_annotation(
                annotation_val_dir,
                image_index,
                Path(member.name).name,
                filename_to_annotation_path,
            )
            seen += 1
            yield tensor, annotation, image_size, member.name
            if max_images > 0 and seen >= max_images:
                break


def predict_gradcam_maps(
    backbone: Backbone,
    cbl: ConceptLayer,
    images: torch.Tensor,
    concept_indices: torch.Tensor,
    chunk_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = images.to(device, non_blocking=True)
    backbone.zero_grad(set_to_none=True)
    cbl.zero_grad(set_to_none=True)
    _ = backbone.backbone(batch)
    feats = backbone.feature_vals[batch.device].float()
    feats.requires_grad_(True)
    logits = cbl(feats.mean(dim=(2, 3))).float()
    selected_logits = logits.index_select(1, concept_indices.to(logits.device))
    cams: List[torch.Tensor] = []
    n_concepts = int(selected_logits.shape[1])
    for start in range(0, n_concepts, int(chunk_size)):
        stop = min(start + int(chunk_size), n_concepts)
        chunk_logits = selected_logits[:, start:stop]
        cur = int(chunk_logits.shape[1])
        grad_outputs = torch.zeros((cur, chunk_logits.shape[0], cur), device=device, dtype=chunk_logits.dtype)
        diag = torch.arange(cur, device=device)
        grad_outputs[diag, :, diag] = 1.0
        grads = torch.autograd.grad(
            outputs=chunk_logits,
            inputs=feats,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=stop < n_concepts,
            create_graph=False,
        )[0]
        alpha = grads.mean(dim=(-1, -2))
        chunk_cams = torch.einsum("kbc,bchw->kbhw", alpha, feats)
        cams.append(F.relu(chunk_cams).permute(1, 0, 2, 3).contiguous())
    return torch.cat(cams, dim=1), logits.detach()


def make_target_config(device: str) -> Config:
    return Config(
        mode="train",
        train_root="",
        train_manifest="",
        annotation_dir="",
        concept_file="concept_files/imagenet_filtered.txt",
        val_root="",
        save_dir="",
        run_name="",
        reuse_run_dir="",
        feature_dir="",
        precomputed_target_dir="",
        persist_feature_copy=False,
        max_train_images=0,
        max_val_images=0,
        val_split=0.1,
        epochs=0,
        batch_size=1,
        workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        device=device,
        amp="off",
        channels_last=False,
        tf32=True,
        cudnn_benchmark=True,
        seed=6885,
        min_image_bytes=2048,
        input_size=224,
        train_random_transforms=True,
        mask_h=7,
        mask_w=7,
        patch_iou_thresh=0.5,
        concept_threshold=0.15,
        spatial_target_mode="soft_box",
        spatial_loss_mode="soft_align",
        filter_concepts_by_count=False,
        concept_min_count=1,
        concept_min_frequency=0.0,
        concept_max_frequency=1.0,
        optimizer="sgd",
        lr=0.0,
        weight_decay=0.0,
        momentum=0.0,
        global_pos_weight=1.0,
        patch_pos_weight=1.0,
        loss_global_w=1.0,
        loss_mask_w=0.25,
        loss_dice_w=0.0,
        branch_arch="dual",
        spatial_branch_mode="multiscale_conv45",
        spatial_stage="conv5",
        residual_alpha=0.0,
        profile_steps=0,
        warmup_steps=0,
        log_every=0,
        save_every=1,
        skip_final_layer=True,
        final_layer_type="dense",
        saga_batch_size=512,
        saga_workers=0,
        saga_prefetch_factor=2,
        saga_step_size=0.1,
        saga_lam=0.0005,
        saga_n_iters=80,
        saga_verbose_every=10,
        dense_lr=1e-3,
        dense_n_iters=20,
        feature_storage_dtype="fp16",
        saga_table_device="cpu",
        print_config=False,
    )


def evaluate_batch(
    images: List[torch.Tensor],
    annotations: List[List[Dict[str, Any]]],
    image_sizes: List[Tuple[int, int]],
    backbone: Backbone,
    cbl: ConceptLayer,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    target_cfg: Config,
    args: argparse.Namespace,
    allowed_concepts: set[str] | None,
    thresholds: Sequence[float],
    include_mean_threshold: bool,
    box_iou_thresholds: Sequence[float],
    threshold_raw: Dict[str, Any],
    distribution: Dict[str, float],
) -> Tuple[int, int]:
    if allowed_concepts is not None:
        annotations = [
            [
                item
                for item in annotation
                if data_utils.canonicalize_concept_label(str(item.get("label", item.get("name", "")))) in allowed_concepts
            ]
            for annotation in annotations
        ]
    global_targets, mask_indices, mask_targets, mask_valid = build_gdino_targets(
        annotations,
        image_sizes,
        concept_to_idx,
        n_concepts,
        target_cfg,
        args.device,
    )
    del global_targets
    present_concepts = sorted({int(x) for x in mask_indices[mask_valid].detach().cpu().tolist() if int(x) >= 0})
    if not present_concepts:
        return 0, 0
    union_indices = torch.tensor(present_concepts, dtype=torch.long, device=args.device)
    union_pos = {concept_idx: pos for pos, concept_idx in enumerate(present_concepts)}
    cams_union, _ = predict_gradcam_maps(
        backbone,
        cbl,
        torch.stack(images, dim=0),
        union_indices,
        int(args.gradcam_chunk_size),
        args.device,
    )
    cams_union = F.interpolate(
        cams_union,
        size=mask_targets.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).float()

    images_with_targets = 0
    instances_seen = 0
    mask_indices = mask_indices.to(args.device)
    mask_targets = mask_targets.to(args.device).float()
    mask_valid = mask_valid.to(args.device)
    for batch_index in range(cams_union.shape[0]):
        valid = mask_valid[batch_index]
        if not bool(valid.any()):
            continue
        concept_ids = mask_indices[batch_index][valid]
        gt = mask_targets[batch_index][valid]
        target_mass = gt.flatten(1).sum(dim=1)
        target_valid = target_mass > 0
        if not bool(target_valid.any()):
            continue
        concept_ids = concept_ids[target_valid]
        gt = gt[target_valid]
        cam_positions = torch.tensor([union_pos[int(idx)] for idx in concept_ids.detach().cpu().tolist()], dtype=torch.long, device=args.device)
        pred = cams_union[batch_index].index_select(0, cam_positions)
        gt_masks = gt > float(args.gt_threshold)
        gt_boxes, gt_box_valid = boxes_from_masks(gt_masks)
        score_maps = normalize_maps(pred, args.map_normalization)

        pred_dist = F.softmax(pred.flatten(1), dim=1).view_as(pred)
        gt_dist = gt.flatten(1) / gt.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-6)
        pred_dist_flat = pred_dist.flatten(1)
        soft_inter = torch.minimum(pred_dist_flat, gt_dist).sum(dim=1)
        soft_union = torch.maximum(pred_dist_flat, gt_dist).sum(dim=1).clamp_min(1e-6)
        argmax = pred_dist_flat.argmax(dim=1)
        dist_point_hit = gt_masks.flatten(1).gather(1, argmax[:, None]).squeeze(1).float()

        count = int(gt.shape[0])
        images_with_targets += 1
        instances_seen += count
        distribution["instances"] += count
        distribution["soft_iou_sum"] += float((soft_inter / soft_union).sum().item())
        distribution["mass_in_gt_sum"] += float((pred_dist * gt_masks.float()).flatten(1).sum(dim=1).sum().item())
        distribution["point_hit_sum"] += float(dist_point_hit.sum().item())
        for threshold in thresholds:
            update_threshold_metrics(
                threshold_raw,
                str(threshold),
                score_maps,
                score_maps >= float(threshold),
                gt_masks,
                gt_boxes,
                gt_box_valid,
                box_iou_thresholds,
            )
        if include_mean_threshold:
            update_threshold_metrics(
                threshold_raw,
                "mean",
                score_maps,
                score_maps >= score_maps.mean(dim=(1, 2), keepdim=True),
                gt_masks,
                gt_boxes,
                gt_box_valid,
                box_iou_thresholds,
            )
    return images_with_targets, instances_seen


def main() -> None:
    args = parse_args()
    load_dir = Path(args.load_dir).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else load_dir / "gradcam_localization_val_tar.json"
    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
    with (load_dir / "args.txt").open("r", encoding="utf-8") as handle:
        model_args = json.load(handle)
    backbone = Backbone(model_args["backbone"], model_args["feature_layer"], args.device).eval()
    cbl = load_vlg_concept_layer(load_dir, args.device)
    concepts = load_vlg_concepts(load_dir)
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    allowed_concepts = None
    if args.allowed_concept_file:
        allowed_concepts = set(load_canonicalized_concepts(Path(args.allowed_concept_file).resolve()))
    target_cfg = make_target_config(args.device)
    thresholds, include_mean_threshold = parse_thresholds(args.activation_thresholds)
    box_iou_thresholds = [float(x.strip()) for x in args.box_iou_thresholds.split(",") if x.strip()]
    threshold_raw: Dict[str, Any] = {}
    distribution: Dict[str, float] = {"instances": 0, "soft_iou_sum": 0.0, "mass_in_gt_sum": 0.0, "point_hit_sum": 0.0}

    images: List[torch.Tensor] = []
    annotations: List[List[Dict[str, Any]]] = []
    image_sizes: List[Tuple[int, int]] = []
    total_images = 0
    images_with_targets = 0
    start = time.perf_counter()
    for image, annotation, image_size, name in iter_tar_samples(
        Path(args.val_tar).resolve(),
        annotation_val_dir,
        backbone.preprocess,
        int(args.max_images),
        filename_to_annotation_path,
    ):
        images.append(image)
        annotations.append(annotation)
        image_sizes.append(image_size)
        if len(images) >= int(args.batch_size):
            cur_with_targets, _ = evaluate_batch(
                images,
                annotations,
                image_sizes,
                backbone,
                cbl,
                concept_to_idx,
                len(concepts),
                target_cfg,
                args,
                allowed_concepts,
                thresholds,
                include_mean_threshold,
                box_iou_thresholds,
                threshold_raw,
                distribution,
            )
            total_images += len(images)
            images_with_targets += cur_with_targets
            if args.log_every > 0 and total_images % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[vlg-gradcam-loc] n={total_images} images_with_targets={images_with_targets} "
                    f"instances={int(distribution['instances'])} ips={total_images / max(elapsed, 1e-6):.2f} last={name}",
                    flush=True,
                )
            images.clear()
            annotations.clear()
            image_sizes.clear()
    if images:
        cur_with_targets, _ = evaluate_batch(
            images,
            annotations,
            image_sizes,
            backbone,
            cbl,
            concept_to_idx,
            len(concepts),
            target_cfg,
            args,
            allowed_concepts,
            thresholds,
            include_mean_threshold,
            box_iou_thresholds,
            threshold_raw,
            distribution,
        )
        total_images += len(images)
        images_with_targets += cur_with_targets

    elapsed = time.perf_counter() - start
    instances = max(int(distribution["instances"]), 1)
    payload = {
        "load_dir": str(load_dir),
        "val_tar": str(Path(args.val_tar).resolve()),
        "annotation_val_dir": str(annotation_val_dir),
        "annotation_val_root": str(annotation_val_root) if annotation_val_root is not None else "",
        "n_concepts": len(concepts),
        "allowed_concept_file": str(Path(args.allowed_concept_file).resolve()) if args.allowed_concept_file else "",
        "allowed_concept_count": len(allowed_concepts) if allowed_concepts is not None else len(concepts),
        "config": {
            "batch_size": int(args.batch_size),
            "gradcam_chunk_size": int(args.gradcam_chunk_size),
            "device": args.device,
            "map_normalization": args.map_normalization,
            "activation_thresholds": args.activation_thresholds,
            "box_iou_thresholds": args.box_iou_thresholds,
            "gt_threshold": float(args.gt_threshold),
            "max_images": int(args.max_images),
        },
        "metrics": {
            "images_seen": total_images,
            "images_with_targets": images_with_targets,
            "instances": int(distribution["instances"]),
            "elapsed_sec": elapsed,
            "images_per_second": total_images / max(elapsed, 1e-6),
            "distribution_metrics": {
                "soft_iou": float(distribution["soft_iou_sum"] / instances),
                "mass_in_gt": float(distribution["mass_in_gt_sum"] / instances),
                "point_hit": float(distribution["point_hit_sum"] / instances),
            },
            "threshold_metrics": finalize_threshold_metrics(threshold_raw, box_iou_thresholds),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
