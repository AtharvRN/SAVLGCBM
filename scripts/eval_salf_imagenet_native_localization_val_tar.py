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
from PIL import Image
from torchvision import transforms
from transformers import ResNetForImageClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
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
from scripts.eval_salf_imagenet_nec_tar import SoftmaxPooling2D
from scripts.train_savlg_imagenet_standalone import Config, build_gdino_targets


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SALF-CBM native concept maps on ImageNet val tar.")
    parser.add_argument("--load_dir", required=True, help="SALF dir with W_c.pt, concepts.txt.")
    parser.add_argument(
        "--allowed_concept_file",
        default="",
        help="Optional concept file used to filter GDINO annotations to a paper/reference concept set while keeping checkpoint-native output indexing.",
    )
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument(
        "--annotation_val_root",
        default="",
        help="Optional reorganized ImageNet val ImageFolder root used when annotations are keyed by ImageFolder dataset index.",
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--map_size", default="12,12")
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9,mean")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument("--map_normalization", default="concept_zscore_minmax", choices=["minmax", "sigmoid", "concept_zscore_minmax"])
    parser.add_argument("--gt_threshold", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=2048)
    return parser.parse_args()


def parse_map_size(raw: str) -> Tuple[int, int]:
    parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--map_size must be H,W")
    return parts[0], parts[1]


def load_salf_concepts(load_dir: Path) -> List[str]:
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


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_annotation(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
    filename_to_annotation_path: Dict[str, Path] | None = None,
) -> List[Dict[str, Any]]:
    return load_annotation_payload(
        annotation_val_dir=annotation_val_dir,
        image_index_1based=image_index_1based,
        image_name=image_name,
        filename_to_annotation_path=filename_to_annotation_path,
    )


def iter_tar_samples(
    val_tar: Path,
    annotation_val_dir: Path,
    transform: transforms.Compose,
    max_images: int,
    filename_to_annotation_path: Dict[str, Path] | None = None,
):
    seen = 0
    with tarfile.open(val_tar, "r|*") as tf:
        print(f"[salf-native-loc] streaming tar members from {val_tar}", flush=True)
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
                filename_to_annotation_path=filename_to_annotation_path,
            )
            seen += 1
            yield tensor, annotation, image_size, member.name
            if max_images > 0 and seen >= max_images:
                break


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
        mask_h=12,
        mask_w=12,
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


class SALFNativeModel(torch.nn.Module):
    def __init__(self, load_dir: Path, device: str, map_size: Tuple[int, int]) -> None:
        super().__init__()
        self.device = device
        self.map_size = map_size
        target_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)
        target_model.eval()
        self.backbone = torch.nn.Sequential(*list(target_model.resnet.children())[:-1]).to(device).eval()
        self.w_c = torch.load(load_dir / "W_c.pt", map_location=device).float()
        self.proj_mean = torch.load(load_dir / "proj_mean.pt", map_location=device).float().flatten()
        self.proj_std = torch.load(load_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
        self.pool = SoftmaxPooling2D(map_size).to(device)

    def forward_maps(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images.to(self.device)).last_hidden_state.float()
        if tuple(feats.shape[-2:]) != self.map_size:
            feats = F.interpolate(feats, size=self.map_size, mode="bilinear", align_corners=False)
        maps = F.conv2d(feats, self.w_c)
        pooled = self.pool(maps).flatten(1)
        pooled_norm = (pooled - self.proj_mean) / self.proj_std
        maps_norm = (maps - self.proj_mean.view(1, -1, 1, 1)) / self.proj_std.view(1, -1, 1, 1)
        # Match classifier scale for distribution metrics, while retaining spatial variation.
        return maps_norm - maps_norm.mean(dim=(2, 3), keepdim=True) + pooled_norm[:, :, None, None]


def evaluate_batch(
    images: List[torch.Tensor],
    annotations: List[List[Dict[str, Any]]],
    image_sizes: List[Tuple[int, int]],
    model: SALFNativeModel,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    target_cfg: Config,
    args: argparse.Namespace,
    allowed_concepts: set[str] | None,
    threshold_raw: Dict[str, Any],
    distribution: Dict[str, float],
    thresholds: Sequence[float],
    include_mean_threshold: bool,
    box_iou_thresholds: Sequence[float],
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
    if not bool(mask_valid.any()):
        return 0, 0
    with torch.no_grad():
        maps = model.forward_maps(torch.stack(images, dim=0))
        maps = F.interpolate(maps, size=mask_targets.shape[-2:], mode="bilinear", align_corners=False).float()

    images_with_targets = 0
    instances_seen = 0
    for batch_index in range(maps.shape[0]):
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
        pred = maps[batch_index].index_select(0, concept_ids)
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
    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
    output_json = Path(args.output_json).resolve() if args.output_json else load_dir / "native_localization_val_tar.json"
    map_size = parse_map_size(args.map_size)
    concepts = load_salf_concepts(load_dir)
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    allowed_concepts = None
    if args.allowed_concept_file:
        allowed_concepts = set(load_canonicalized_concepts(Path(args.allowed_concept_file).resolve()))
    target_cfg = make_target_config(args.device)
    target_cfg.mask_h, target_cfg.mask_w = map_size
    model = SALFNativeModel(load_dir, args.device, map_size).eval()
    transform = build_transform()
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
        transform,
        int(args.max_images),
        filename_to_annotation_path=filename_to_annotation_path,
    ):
        images.append(image)
        annotations.append(annotation)
        image_sizes.append(image_size)
        if len(images) >= int(args.batch_size):
            cur_with_targets, _ = evaluate_batch(
                images,
                annotations,
                image_sizes,
                model,
                concept_to_idx,
                len(concepts),
                target_cfg,
                args,
                allowed_concepts,
                threshold_raw,
                distribution,
                thresholds,
                include_mean_threshold,
                box_iou_thresholds,
            )
            total_images += len(images)
            images_with_targets += cur_with_targets
            if args.log_every > 0 and total_images % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[salf-native-loc] n={total_images} images_with_targets={images_with_targets} "
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
            model,
            concept_to_idx,
            len(concepts),
            target_cfg,
            args,
            allowed_concepts,
            threshold_raw,
            distribution,
            thresholds,
            include_mean_threshold,
            box_iou_thresholds,
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
            "device": args.device,
            "map_size": list(map_size),
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
