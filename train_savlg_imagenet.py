import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch


BASE_DEFAULTS = {
    "model_name": "savlg_cbm",
    "dataset": "imagenet",
    "backbone": "resnet50",
    "feature_layer": "layer4",
    "concept_set": "concept_files/imagenet_filtered.txt",
    "filter_set": None,
    "annotation_dir": "./annotations",
    "save_dir": "saved_models/imagenet",
    "activation_dir": "saved_activations",
    "device": "cuda",
    "seed": 6885,
    "val_split": 0.1,
    "allones_concept": False,
    "dense": False,
    "dense_lr": 1e-3,
    "cbl_type": "linear",
    "cbl_hidden_layers": 1,
    "cbl_use_batchnorm": False,
    "cbl_hidden_dim": 0,
    "cbl_batch_size": 256,
    "cbl_epochs": 7,
    "cbl_lr": 5e-4,
    "cbl_optimizer": "sgd",
    "cbl_scheduler": None,
    "cbl_weight_decay": 1e-5,
    "cbl_confidence_threshold": 0.15,
    "num_workers": 12,
    "spatial_num_workers": 12,
    "prefetch_factor": 4,
    "saga_batch_size": 512,
    "saga_lam": 5e-4,
    "saga_n_iters": 80,
    "saga_step_size": 0.1,
    "mask_h": 7,
    "mask_w": 7,
    "crop_to_concept_prob": 0.0,
    "use_activation_cache": False,
    "max_train_images": 0,
    "max_test_images": 0,
    "skip_train_val_eval": False,
    "skip_test_eval": False,
    "savlg_stream_supervision": True,
    "savlg_supervision_source": "gdino",
    "savlg_concept_filter_mode": "spatial_threshold",
    "savlg_global_target_mode": "binary_threshold",
    "savlg_spatial_stage": "conv5",
    "savlg_branch_arch": "shared",
    "savlg_spatial_branch_mode": "shared_stage",
    "savlg_global_head_mode": "spatial_pool",
    "savlg_residual_spatial_alpha": 0.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dedicated ImageNet SAVLG-CBM training entrypoint with A10-oriented defaults."
    )
    parser.add_argument(
        "--config",
        default="configs/unified/imagenet_savlg_cbm.json",
        help="Base JSON config to merge with the ImageNet SAVLG defaults.",
    )
    parser.add_argument("--annotation_dir", required=True, help="Directory containing ImageNet SAVLG annotations.")
    parser.add_argument("--save_dir", default=BASE_DEFAULTS["save_dir"])
    parser.add_argument("--activation_dir", default=BASE_DEFAULTS["activation_dir"])
    parser.add_argument("--device", default=BASE_DEFAULTS["device"])
    parser.add_argument("--seed", type=int, default=BASE_DEFAULTS["seed"])
    parser.add_argument("--cbl_epochs", type=int, default=BASE_DEFAULTS["cbl_epochs"])
    parser.add_argument("--cbl_batch_size", type=int, default=BASE_DEFAULTS["cbl_batch_size"])
    parser.add_argument("--cbl_lr", type=float, default=BASE_DEFAULTS["cbl_lr"])
    parser.add_argument("--num_workers", type=int, default=BASE_DEFAULTS["num_workers"])
    parser.add_argument(
        "--spatial_num_workers", type=int, default=BASE_DEFAULTS["spatial_num_workers"]
    )
    parser.add_argument("--prefetch_factor", type=int, default=BASE_DEFAULTS["prefetch_factor"])
    parser.add_argument("--saga_batch_size", type=int, default=BASE_DEFAULTS["saga_batch_size"])
    parser.add_argument("--saga_n_iters", type=int, default=BASE_DEFAULTS["saga_n_iters"])
    parser.add_argument("--saga_lam", type=float, default=BASE_DEFAULTS["saga_lam"])
    parser.add_argument("--saga_step_size", type=float, default=BASE_DEFAULTS["saga_step_size"])
    parser.add_argument("--val_split", type=float, default=BASE_DEFAULTS["val_split"])
    parser.add_argument("--max_train_images", type=int, default=BASE_DEFAULTS["max_train_images"])
    parser.add_argument("--max_test_images", type=int, default=BASE_DEFAULTS["max_test_images"])
    parser.add_argument(
        "--savlg_residual_spatial_alpha",
        type=float,
        default=BASE_DEFAULTS["savlg_residual_spatial_alpha"],
    )
    parser.add_argument(
        "--savlg_branch_arch",
        choices=["shared", "dual"],
        default=BASE_DEFAULTS["savlg_branch_arch"],
    )
    parser.add_argument(
        "--savlg_spatial_branch_mode",
        choices=["shared_stage", "multiscale_conv45"],
        default=BASE_DEFAULTS["savlg_spatial_branch_mode"],
    )
    parser.add_argument(
        "--savlg_global_head_mode",
        choices=["spatial_pool", "vlg_linear"],
        default=BASE_DEFAULTS["savlg_global_head_mode"],
    )
    parser.add_argument(
        "--savlg_spatial_stage",
        choices=["conv3", "conv4", "conv5"],
        default=BASE_DEFAULTS["savlg_spatial_stage"],
    )
    parser.add_argument(
        "--savlg_supervision_source",
        choices=["gdino", "groundedsam2"],
        default=BASE_DEFAULTS["savlg_supervision_source"],
    )
    parser.add_argument(
        "--disable_stream_supervision",
        action="store_true",
        help="Disable on-the-fly GDINO supervision streaming.",
    )
    parser.add_argument("--skip_train_val_eval", action="store_true")
    parser.add_argument("--skip_test_eval", action="store_true")
    parser.add_argument(
        "--enable_activation_cache",
        action="store_true",
        help="Enable backbone feature caching. Off by default for ImageNet SAVLG.",
    )
    parser.add_argument("--omp_threads", type=int, default=1)
    parser.add_argument("--mkl_threads", type=int, default=1)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--disable_cudnn_benchmark", action="store_true")
    parser.add_argument("--print_config", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def configure_runtime(cli_args: argparse.Namespace) -> None:
    os.environ["OMP_NUM_THREADS"] = str(cli_args.omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(cli_args.mkl_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    torch.set_num_threads(cli_args.torch_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    tf32_enabled = not cli_args.disable_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled
    torch.backends.cudnn.benchmark = not cli_args.disable_cudnn_benchmark
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli_args.seed)


def build_savlg_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    cfg = load_config(cli_args.config)
    merged = dict(BASE_DEFAULTS)
    merged.update(cfg)
    merged.update(
        {
            "annotation_dir": cli_args.annotation_dir,
            "save_dir": cli_args.save_dir,
            "activation_dir": cli_args.activation_dir,
            "device": cli_args.device,
            "seed": cli_args.seed,
            "cbl_epochs": cli_args.cbl_epochs,
            "cbl_batch_size": cli_args.cbl_batch_size,
            "cbl_lr": cli_args.cbl_lr,
            "num_workers": cli_args.num_workers,
            "spatial_num_workers": cli_args.spatial_num_workers,
            "prefetch_factor": cli_args.prefetch_factor,
            "saga_batch_size": cli_args.saga_batch_size,
            "saga_n_iters": cli_args.saga_n_iters,
            "saga_lam": cli_args.saga_lam,
            "saga_step_size": cli_args.saga_step_size,
            "val_split": cli_args.val_split,
            "max_train_images": cli_args.max_train_images,
            "max_test_images": cli_args.max_test_images,
            "skip_train_val_eval": cli_args.skip_train_val_eval,
            "skip_test_eval": cli_args.skip_test_eval,
            "use_activation_cache": cli_args.enable_activation_cache,
            "savlg_stream_supervision": not cli_args.disable_stream_supervision,
            "savlg_supervision_source": cli_args.savlg_supervision_source,
            "savlg_residual_spatial_alpha": cli_args.savlg_residual_spatial_alpha,
            "savlg_branch_arch": cli_args.savlg_branch_arch,
            "savlg_spatial_branch_mode": cli_args.savlg_spatial_branch_mode,
            "savlg_global_head_mode": cli_args.savlg_global_head_mode,
            "savlg_spatial_stage": cli_args.savlg_spatial_stage,
        }
    )
    merged["model_name"] = "savlg_cbm"
    merged["dataset"] = "imagenet"
    merged["backbone"] = "resnet50"
    merged["feature_layer"] = "layer4"
    return SimpleNamespace(**merged)


def main() -> None:
    cli_args = parse_args()
    configure_runtime(cli_args)
    args = build_savlg_args(cli_args)
    if cli_args.print_config:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
    from methods.savlg import train_savlg_cbm

    train_savlg_cbm(args)


if __name__ == "__main__":
    main()
