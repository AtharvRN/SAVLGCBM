import argparse
import json
import os
import random

import numpy as np
import torch
from loguru import logger

from data import utils as data_utils
from data.concept_dataset import get_concept_dataloader
from methods.common import load_run_info
from model.cbm import (
    Backbone,
    BackboneCLIP,
    ConceptLayer,
    FinalLayer,
    NormalizationLayer,
    load_cbm,
    test_model,
)
from model.utils import get_accuracy_cbm


def evaluate_vlg_cbm(load_dir: str, args) -> float:
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(
            args.backbone,
            use_penultimate=args.use_clip_penultimate,
            device=args.device,
        )
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)
    if os.path.exists(os.path.join(load_dir, "backbone.pt")):
        ckpt = torch.load(os.path.join(load_dir, "backbone.pt"))
        backbone.backbone.load_state_dict(ckpt)
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().split("\n")
    test_loader = get_concept_dataloader(
        args.dataset,
        "test",
        concepts,
        preprocess=backbone.preprocess,
        val_split=None,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
    cbl = ConceptLayer.from_pretrained(load_dir, args.device)
    normalization = NormalizationLayer.from_pretrained(load_dir, args.device)
    final_layer = FinalLayer.from_pretrained(load_dir, args.device)
    return float(test_model(test_loader, backbone, cbl, normalization, final_layer, args.device))


def evaluate_lf_cbm(load_dir: str, args) -> float:
    model = load_cbm(load_dir, args.device)
    dataset = data_utils.get_data(f"{args.dataset}_val", preprocess=model.preprocess)
    acc = get_accuracy_cbm(
        model,
        dataset,
        args.device,
        batch_size=getattr(args, "lf_batch_size", getattr(args, "batch_size", 64)),
        num_workers=args.num_workers,
    )
    return float(acc)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CBM checkpoint")
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    cli_args = parser.parse_args()

    run_info = load_run_info(cli_args.load_path)
    with open(os.path.join(cli_args.load_path, "args.txt"), "r") as f:
        args = argparse.Namespace(**json.load(f))
    if cli_args.device is not None:
        args.device = cli_args.device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = run_info.get("model_name", "vlg_cbm")
    if model_name == "vlg_cbm":
        metric = evaluate_vlg_cbm(cli_args.load_path, args)
    elif model_name == "lf_cbm":
        metric = evaluate_lf_cbm(cli_args.load_path, args)
    else:
        raise NotImplementedError(f"Evaluation for model_name={model_name} is not implemented yet.")
    logger.info(f"{model_name} test accuracy: {metric:.4f}")


if __name__ == "__main__":
    main()
