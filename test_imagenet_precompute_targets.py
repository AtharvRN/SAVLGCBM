import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.train_savlg_imagenet_standalone import (
    Config,
    PREPROCESS_RESIZE_SIZE,
    PrecomputedTargetStore,
    SafeImageFolderWithAnnotations,
    build_gdino_target_sample,
    precompute_target_store,
    resize_short_edge_size,
    transform_box_for_model_input,
)


def make_cfg(tmp_path: Path, *, input_size: int = 224) -> Config:
    return Config(
        mode="precompute_targets",
        train_root=str(tmp_path / "train"),
        train_manifest="",
        annotation_dir=str(tmp_path / "annotations"),
        concept_file="",
        val_root="",
        save_dir=str(tmp_path / "runs"),
        run_name="",
        reuse_run_dir="",
        feature_dir="",
        precomputed_target_dir=str(tmp_path / "targets"),
        persist_feature_copy=False,
        max_train_images=0,
        max_val_images=0,
        val_split=0.1,
        epochs=1,
        batch_size=2,
        workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        device="cpu",
        amp="none",
        channels_last=False,
        tf32=False,
        cudnn_benchmark=False,
        seed=1,
        min_image_bytes=0,
        input_size=input_size,
        train_random_transforms=False,
        mask_h=14,
        mask_w=14,
        patch_iou_thresh=0.5,
        concept_threshold=0.15,
        spatial_target_mode="soft_box",
        spatial_loss_mode="soft_align",
        filter_concepts_by_count=False,
        concept_min_count=1,
        concept_min_frequency=0.0,
        concept_max_frequency=1.0,
        optimizer="sgd",
        lr=1e-3,
        weight_decay=0.0,
        momentum=0.9,
        global_pos_weight=1.0,
        patch_pos_weight=1.0,
        loss_global_w=1.0,
        loss_mask_w=1.0,
        loss_dice_w=0.0,
        branch_arch="dual",
        spatial_branch_mode="multiscale_conv45",
        spatial_stage="conv5",
        residual_alpha=0.2,
        profile_steps=1,
        warmup_steps=0,
        log_every=1000,
        save_every=1,
        skip_final_layer=True,
        final_layer_type="dense",
        saga_batch_size=2,
        saga_workers=0,
        saga_prefetch_factor=2,
        saga_step_size=0.1,
        saga_lam=1e-3,
        saga_n_iters=1,
        saga_verbose_every=1,
        dense_lr=1e-3,
        dense_n_iters=1,
        feature_storage_dtype="fp16",
        saga_table_device="cpu",
        vlg_init_path="",
        vlg_concepts_path="",
        freeze_global_head=False,
        scheduler="none",
        print_config=False,
    )


def write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def write_annotation(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"img_path": None}, *entries]))


def make_dataset(tmp_path: Path, cfg: Config) -> SafeImageFolderWithAnnotations:
    train_root = Path(cfg.train_root)
    ann_root = Path(cfg.annotation_dir) / "imagenet_train"
    write_image(train_root / "class0" / "img0.JPEG", (500, 250))
    write_image(train_root / "class0" / "img1.JPEG", (250, 250))
    write_annotation(
        ann_root / "0.json",
        [
            {"label": "left object", "logit": 0.9, "box": [0, 0, 10, 10]},
            {"label": "right object", "logit": 0.9, "box": [300, 50, 400, 200]},
            {"label": "normalized object", "logit": 0.9, "box": [0.4, 0.25, 0.6, 0.75]},
        ],
    )
    write_annotation(
        ann_root / "1.json",
        [
            {"label": "right object", "logit": 0.9, "box": [50, 50, 200, 200]},
            {"label": "below threshold", "logit": 0.01, "box": [50, 50, 200, 200]},
        ],
    )
    return SafeImageFolderWithAnnotations(
        root=cfg.train_root,
        annotation_dir=cfg.annotation_dir,
        concepts=["left object", "right object", "normalized object", "below threshold"],
        input_size=cfg.input_size,
        min_image_bytes=0,
        split="train",
        train_random_transforms=False,
    )


def test_resize_and_crop_geometry_matches_expected_crop_frame() -> None:
    assert resize_short_edge_size((500, 250)) == (512, PREPROCESS_RESIZE_SIZE)
    assert transform_box_for_model_input([0, 0, 500, 250], (500, 250), 224) == (0.0, 0.0, 1.0, 1.0)
    assert transform_box_for_model_input([0, 0, 10, 10], (500, 250), 224) is None

    transformed = transform_box_for_model_input([300, 50, 400, 200], (500, 250), 224)
    assert transformed is not None
    x1, y1, x2, y2 = transformed
    assert 0.7 < x1 < 1.0
    assert x2 == 1.0
    assert 0.15 < y1 < y2 < 0.9


def test_build_gdino_target_sample_drops_cropped_out_boxes_and_keeps_dense_global(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    dataset = make_dataset(tmp_path, cfg)
    annotations = dataset._load_annotation(0)

    global_target, concept_ids, masks = build_gdino_target_sample(
        annotations,
        image_size=(500, 250),
        concept_to_idx=dataset.concept_to_idx,
        n_concepts=len(dataset.concepts),
        cfg=cfg,
    )

    # Global labels are dense concept-presence targets, even when the box is
    # cropped out of the model input.
    assert global_target.tolist() == [1, 1, 1, 0]
    assert concept_ids.tolist() == [1, 2]
    assert masks.shape == (2, cfg.mask_h, cfg.mask_w)
    assert np.all(masks.sum(axis=(1, 2)) > 0)


def test_precompute_writes_crop_space_metadata_and_sparse_arrays(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    dataset = make_dataset(tmp_path, cfg)

    summary = precompute_target_store(dataset, Path(cfg.precomputed_target_dir), cfg)
    split_dir = Path(cfg.precomputed_target_dir) / "train"
    metadata = json.loads((split_dir / "metadata.json").read_text())

    assert summary["target_coordinate_frame"] == "resize_short_edge_then_center_crop"
    assert metadata["input_size"] == cfg.input_size
    assert metadata["preprocess_resize_size"] == PREPROCESS_RESIZE_SIZE
    assert metadata["total_entries"] == 3

    global_targets = np.load(split_dir / "global_targets.npy", mmap_mode="r")
    offsets = np.load(split_dir / "offsets.npy", mmap_mode="r")
    concept_ids = np.load(split_dir / "concept_ids.npy", mmap_mode="r")
    mask_targets = np.load(split_dir / "mask_targets.npy", mmap_mode="r")

    assert global_targets.shape == (2, 4)
    assert global_targets[0].tolist() == [1, 1, 1, 0]
    assert global_targets[1].tolist() == [0, 1, 0, 0]
    assert offsets.tolist() == [0, 2, 3]
    assert concept_ids.tolist() == [1, 2, 1]
    assert mask_targets.shape == (3, cfg.mask_h, cfg.mask_w)


def test_precomputed_store_rejects_old_original_image_frame_cache(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    dataset = make_dataset(tmp_path, cfg)
    precompute_target_store(dataset, Path(cfg.precomputed_target_dir), cfg)

    metadata_path = Path(cfg.precomputed_target_dir) / "train" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("target_coordinate_frame")
    metadata_path.write_text(json.dumps(metadata))

    store = PrecomputedTargetStore(Path(cfg.precomputed_target_dir) / "train")
    with pytest.raises(ValueError, match="target_coordinate_frame"):
        store.validate_target_geometry(cfg)


def test_precomputed_store_rejects_wrong_input_size(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    dataset = make_dataset(tmp_path, cfg)
    precompute_target_store(dataset, Path(cfg.precomputed_target_dir), cfg)

    store = PrecomputedTargetStore(Path(cfg.precomputed_target_dir) / "train")
    with pytest.raises(ValueError, match="input_size"):
        store.validate_target_geometry(replace(cfg, input_size=256))
