import argparse
import json
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_savlg_imagenet_standalone_localization import (
    boxes_from_masks,
    box_iou,
    finalize_threshold_metrics,
    normalize_maps,
    parse_thresholds,
    update_threshold_metrics,
)
from scripts.imagenet_annotation_index import (
    build_filename_to_annotation_path,
    load_annotation_payload,
    resolve_val_annotation_dir,
)
from scripts.train_savlg_imagenet_standalone import (
    Config,
    amp_dtype,
    build_gdino_targets,
    build_model,
    configure_runtime,
    load_concepts,
    prepare_images,
)


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standalone ImageNet SAVLG spatial maps on official val tar using GDINO val annotations."
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--val_tar", required=True)
    parser.add_argument(
        "--annotation_dir",
        required=True,
        help="Directory containing imagenet_val/*.json, or the imagenet_val directory itself.",
    )
    parser.add_argument(
        "--annotation_val_root",
        default="",
        help="Optional reorganized ImageNet val ImageFolder root used when annotations are keyed by ImageFolder dataset index.",
    )
    parser.add_argument("--devkit_dir", default="", help="Optional ImageNet devkit, only used to sanity-check val label count.")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9,mean")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument(
        "--map_normalization",
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "concept_zscore_minmax"],
    )
    parser.add_argument("--gt_threshold", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=1000)
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
    payload.setdefault("learn_spatial_residual_scale", False)
    payload["device"] = args.device
    payload["batch_size"] = int(args.batch_size)
    payload["workers"] = 0
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    return Config(**payload)


def load_val_label_count(devkit_dir: Path) -> Optional[int]:
    if not devkit_dir:
        return None
    labels_path = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    meta_path = devkit_dir / "data" / "meta.mat"
    if not labels_path.is_file() or not meta_path.is_file():
        return None
    # Load meta too, matching the classification eval's validation that this is a real devkit.
    loadmat(meta_path, squeeze_me=True, struct_as_record=False)
    with labels_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_annotation(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
    filename_to_annotation_path: Optional[Dict[str, Path]] = None,
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
    filename_to_annotation_path: Optional[Dict[str, Path]] = None,
):
    seen = 0
    with tarfile.open(val_tar, "r|*") as tf:
        print(f"[loc-val] streaming tar members from {val_tar}", flush=True)
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


def evaluate_batch(
    images: List[torch.Tensor],
    annotations: List[List[Dict[str, Any]]],
    image_sizes: List[Tuple[int, int]],
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    cfg: Config,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    args: argparse.Namespace,
    threshold_raw: Dict[str, Any],
    distribution: Dict[str, float],
    thresholds: Sequence[float],
    include_mean_threshold: bool,
    box_iou_thresholds: Sequence[float],
) -> Tuple[int, int]:
    batch = prepare_images(torch.stack(images, dim=0), cfg)
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype(cfg.amp),
            enabled=(str(cfg.device).startswith("cuda") and amp_dtype(cfg.amp) is not None),
        ):
            feats = backbone(batch)
            outputs = head(feats)
            global_targets, mask_indices, mask_targets, mask_valid = build_gdino_targets(
                annotations,
                image_sizes,
                concept_to_idx,
                n_concepts,
                cfg,
                cfg.device,
            )
            del global_targets
            spatial_maps = F.interpolate(
                outputs["spatial_maps"],
                size=mask_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).float()

    images_with_targets = 0
    instance_count = 0
    for batch_index in range(spatial_maps.shape[0]):
        valid = mask_valid[batch_index]
        if not bool(valid.any()):
            continue
        concept_ids = mask_indices[batch_index][valid]
        gt = mask_targets[batch_index][valid]
        target_mass = gt.flatten(1).sum(dim=1)
        target_valid = target_mass > 0
        if not bool(target_valid.any()):
            continue
        images_with_targets += 1
        concept_ids = concept_ids[target_valid]
        gt = gt[target_valid]
        pred = spatial_maps[batch_index].index_select(0, concept_ids)
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
        instance_count += int(gt.shape[0])
        distribution["instances"] += int(gt.shape[0])
        distribution["soft_iou_sum"] += float((soft_inter / soft_union).sum().item())
        distribution["mass_in_gt_sum"] += float((pred_dist * gt_masks.float()).flatten(1).sum(dim=1).sum().item())
        distribution["point_hit_sum"] += float(dist_point_hit.sum().item())

        for threshold in thresholds:
            key = str(threshold)
            update_threshold_metrics(
                threshold_raw,
                key,
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
    return images_with_targets, instance_count


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    source_run_dir = resolve_source_run_dir(artifact_dir)
    val_tar = Path(args.val_tar).resolve()
    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else artifact_dir / "localization_imagenet_val_tar.json"
    )
    cfg = load_run_config(source_run_dir, args)
    configure_runtime(cfg)

    concepts = load_concepts(str(source_run_dir / "concepts.txt"))
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()

    thresholds, include_mean_threshold = parse_thresholds(args.activation_thresholds)
    box_iou_thresholds = [float(x.strip()) for x in args.box_iou_thresholds.split(",") if x.strip()]
    threshold_raw: Dict[str, Any] = {}
    distribution: Dict[str, float] = {
        "instances": 0,
        "soft_iou_sum": 0.0,
        "mass_in_gt_sum": 0.0,
        "point_hit_sum": 0.0,
    }
    transform = build_transform(cfg.input_size)
    label_count = load_val_label_count(Path(args.devkit_dir).resolve()) if args.devkit_dir else None

    start = time.perf_counter()
    images: List[torch.Tensor] = []
    annotations: List[List[Dict[str, Any]]] = []
    image_sizes: List[Tuple[int, int]] = []
    total_images = 0
    images_with_targets = 0
    for image_tensor, annotation, image_size, name in iter_tar_samples(
        val_tar,
        annotation_val_dir,
        transform,
        int(args.max_images),
        filename_to_annotation_path=filename_to_annotation_path,
    ):
        images.append(image_tensor)
        annotations.append(annotation)
        image_sizes.append(image_size)
        if len(images) >= int(cfg.batch_size):
            batch_with_targets, _ = evaluate_batch(
                images,
                annotations,
                image_sizes,
                backbone,
                head,
                cfg,
                concept_to_idx,
                len(concepts),
                args,
                threshold_raw,
                distribution,
                thresholds,
                include_mean_threshold,
                box_iou_thresholds,
            )
            total_images += len(images)
            images_with_targets += batch_with_targets
            if args.log_every > 0 and total_images % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[loc-val] n={total_images} images_with_targets={images_with_targets} "
                    f"instances={int(distribution['instances'])} ips={total_images / max(elapsed, 1e-6):.2f} last={name}",
                    flush=True,
                )
            images.clear()
            annotations.clear()
            image_sizes.clear()
    if images:
        batch_with_targets, _ = evaluate_batch(
            images,
            annotations,
            image_sizes,
            backbone,
            head,
            cfg,
            concept_to_idx,
            len(concepts),
            args,
            threshold_raw,
            distribution,
            thresholds,
            include_mean_threshold,
            box_iou_thresholds,
        )
        total_images += len(images)
        images_with_targets += batch_with_targets

    elapsed = time.perf_counter() - start
    instances = max(int(distribution["instances"]), 1)
    payload = {
        "artifact_dir": str(artifact_dir),
        "source_run_dir": str(source_run_dir),
        "val_tar": str(val_tar),
        "annotation_val_dir": str(annotation_val_dir),
        "annotation_val_root": str(annotation_val_root) if annotation_val_root is not None else "",
        "devkit_label_count": label_count,
        "n_concepts": len(concepts),
        "config": {
            "batch_size": int(cfg.batch_size),
            "device": cfg.device,
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
