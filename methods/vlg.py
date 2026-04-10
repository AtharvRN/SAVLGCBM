import json
import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model.utils as utils
from data import utils as data_utils
from data.concept_dataset import (
    get_concept_dataloader,
    get_filtered_concepts_and_counts,
    get_final_layer_dataset,
    get_or_create_backbone_embedding_cache,
)
from loss import get_loss
from methods.common import build_run_dir, save_args, write_artifacts
from model.cbm import (
    Backbone,
    BackboneCLIP,
    ConceptLayer,
    FinalLayer,
    per_class_accuracy,
    test_model,
    train_cbl,
    train_dense_final,
    train_sparse_final,
)


def train_vlg_cbm(args):
    save_dir = build_run_dir(args.save_dir, args.dataset, "vlg_cbm")
    logger.add(
        os.path.join(save_dir, "train.log"),
        format="{time} {level} {message}",
        level="DEBUG",
    )
    logger.info("Saving model to {}", save_dir)
    save_args(args, save_dir)
    write_artifacts(
        save_dir,
        {
            "model_name": "vlg_cbm",
            "dataset": args.dataset,
            "backbone": args.backbone,
            "concept_layer_format": "cbl.pt",
            "normalization_format": [
                "train_concept_features_mean.pt",
                "train_concept_features_std.pt",
            ],
            "final_layer_format": ["final.pt"],
            "sparse_eval_style": "vlg_upstream",
        },
    )

    classes = data_utils.get_classes(args.dataset)
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(
            args.backbone, use_penultimate=args.use_clip_penultimate, device=args.device
        )
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)

    if args.load_dir is None:
        if args.skip_concept_filter:
            logger.info("Skipping concept filtering")
            concepts, concept_counts = data_utils.load_concept_and_count(
                os.path.dirname(args.concept_set), filter_file=args.filter_set
            )
        else:
            logger.info("Filtering concepts")
            raw_concepts = data_utils.get_concepts(args.concept_set, args.filter_set)
            concepts, concept_counts, filtered_concepts = get_filtered_concepts_and_counts(
                args.dataset,
                raw_concepts,
                preprocess=backbone.preprocess,
                val_split=args.val_split,
                batch_size=args.cbl_batch_size,
                num_workers=args.num_workers,
                confidence_threshold=args.cbl_confidence_threshold,
                label_dir=args.annotation_dir,
                use_allones=args.allones_concept,
                seed=args.seed,
            )
            data_utils.save_concept_count(concepts, concept_counts, save_dir)
            data_utils.save_filtered_concepts(filtered_concepts, save_dir)
    else:
        logger.info("Loading concepts from {}", args.load_dir)
        concepts, concept_counts = data_utils.load_concept_and_count(
            args.load_dir, filter_file=args.filter_set
        )

    with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)

    tb_writer = SummaryWriter(save_dir)
    activation_cache_dir = (
        args.activation_cache_dir
        if args.activation_cache_dir is not None
        else os.path.join(args.annotation_dir, "_cache", "backbone_embeddings")
    )

    augmented_train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=args.crop_to_concept_prob,
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
    train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
    val_cbl_loader = get_concept_dataloader(
        args.dataset,
        "val",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
    test_cbl_loader = get_concept_dataloader(
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

    loss_fn = get_loss(
        args.cbl_loss_type,
        len(concepts),
        len(train_cbl_loader.dataset),
        concept_counts,
        args.cbl_pos_weight,
        args.cbl_auto_weight,
        args.cbl_twoway_tp,
        args.device,
    )

    if args.load_dir is None:
        logger.info("Training CBL")
        cbl = ConceptLayer(
            backbone.output_dim,
            len(concepts),
            num_hidden=args.cbl_hidden_layers,
            device=args.device,
        )
        cached_val_embeddings = None
        cached_val_concepts = None
        use_activation_cache = args.use_activation_cache and not args.cbl_finetune
        if use_activation_cache:
            val_cached = get_or_create_backbone_embedding_cache(
                backbone,
                val_cbl_loader,
                device=args.device,
                cache_dir=activation_cache_dir,
                cache_tag="val",
            )
            cached_val_embeddings = val_cached["embeddings"]
            cached_val_concepts = val_cached["concept_one_hot"]
        cbl, backbone = train_cbl(
            backbone,
            cbl,
            augmented_train_cbl_loader,
            val_cbl_loader,
            args.cbl_epochs,
            loss_fn=loss_fn,
            lr=args.cbl_lr,
            weight_decay=args.cbl_weight_decay,
            concepts=concepts,
            tb_writer=tb_writer,
            device=args.device,
            finetune=args.cbl_finetune,
            optimizer=args.cbl_optimizer,
            scheduler=args.cbl_scheduler,
            backbone_lr=args.cbl_lr * args.cbl_bb_lr_rate,
            data_parallel=args.data_parallel,
            cached_val_embeddings=cached_val_embeddings,
            cached_val_concepts=cached_val_concepts,
        )
    else:
        logger.info("Loading CBL from {}", args.load_dir)
        cbl = ConceptLayer.from_pretrained(args.load_dir, args.device)
        if args.backbone.startswith("clip_"):
            raise NotImplementedError("Loading backbone from pretrained model is not supported yet")
        backbone = Backbone.from_pretrained(args.load_dir, args.device)

    cbl.save_model(save_dir)
    if args.cbl_finetune:
        backbone.save_model(save_dir)

    train_concept_loader, val_concept_loader, normalization_layer = get_final_layer_dataset(
        backbone,
        cbl,
        train_cbl_loader,
        val_cbl_loader,
        save_dir,
        load_dir=args.load_dir,
        batch_size=args.saga_batch_size,
        device=args.device,
        use_activation_cache=args.use_activation_cache and not args.cbl_finetune,
        activation_cache_dir=activation_cache_dir,
    )

    final_layer = FinalLayer(len(concepts), len(classes), device=args.device)
    if args.dense:
        logger.info("Training dense final layer with lr: {} ...", args.dense_lr)
        output_proj = train_dense_final(
            final_layer,
            train_concept_loader,
            val_concept_loader,
            args.saga_n_iters,
            args.dense_lr,
            device=args.device,
        )
    else:
        logger.info("Training sparse final layer ...")
        output_proj = train_sparse_final(
            final_layer,
            train_concept_loader,
            val_concept_loader,
            args.saga_n_iters,
            args.saga_lam,
            step_size=args.saga_step_size,
            device=args.device,
        )

    W_g = output_proj["path"][0]["weight"]
    b_g = output_proj["path"][0]["bias"]
    final_layer.load_state_dict({"weight": W_g, "bias": b_g})
    final_layer.save_model(save_dir)

    test_accuracy = test_model(
        test_cbl_loader, backbone, cbl, normalization_layer, final_layer, args.device
    )
    logger.info("Test accuracy: {}", test_accuracy)

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        out_dict = {}
        out_dict["per_class_accuracies"] = per_class_accuracy(
            nn.Sequential(backbone, cbl, normalization_layer, final_layer).to(args.device),
            test_cbl_loader,
            classes,
            device=args.device,
        )
        for key in ("lam", "lr", "alpha", "time"):
            out_dict[key] = float(output_proj["path"][0][key])
        out_dict["metrics"] = output_proj["path"][0]["metrics"]
        out_dict["metrics"]["test_accuracy"] = test_accuracy
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict["sparsity"] = {
            "Non-zero weights": nnz,
            "Total weights": total,
            "Percentage non-zero": nnz / total,
        }
        json.dump(out_dict, f, indent=2)

    utils.write_parameters_tensorboard(
        tb_writer, vars(args), test_accuracy * 100.0, (nnz / total) * 100.0
    )

    if args.visualize_concepts:
        target_layer = data_utils.BACKBONE_VISUALIZATION_TARGET_LAYER[args.backbone]
        os.mkdir(os.path.join(save_dir, "concept_visualization"))
        cbl_with_backbone = nn.Sequential(backbone, cbl).to(args.device)
        concepts_logits = []
        for images_tensor, _, _ in tqdm(test_cbl_loader):
            images_tensor = images_tensor.to(args.device)
            with torch.no_grad():
                concepts_logits.append(cbl_with_backbone(images_tensor).detach().cpu().numpy())
        concepts_logits = np.concatenate(concepts_logits, axis=0)
        for concept_idx, concept in enumerate(concepts):
            fig = utils.display_top_activated_images(
                concept_idx,
                concepts_logits,
                cbl_with_backbone,
                target_layer,
                test_cbl_loader.dataset,
                transform=backbone.preprocess,
                device=args.device,
                k=10,
            )
            fig.savefig(os.path.join(save_dir, "concept_visualization", f"{concept}.png"))

    return save_dir
