# ImageNet Localization Results

This note records the ImageNet val-tar localization comparison between VLG-CBM Grad-CAM maps and SAVLG native spatial maps.

## Artifacts

- VLG-CBM Grad-CAM JSON: `tmp/vlgcbm_gradcam_localization/vlgcbm_gradcam_localization_val_tar_full_20260429T093131Z.json`
- VLG-CBM Grad-CAM log: `tmp/vlgcbm_gradcam_localization/vlgcbm_gradcam_localization_val_tar_full_20260429T093131Z.log`
- SAVLG native JSON: `tmp/savlg_localization/localization_val_tar_alpha08_restart_20260429T082513Z.json`
- SAVLG native log: `tmp/savlg_localization/localization_val_tar_alpha08_restart_20260429T082513Z.log`

Remote source paths:

- VLG-CBM: `/workspace/SAVLGCBM-imagenet-test/results/vlgcbm_gradcam_localization/vlgcbm_gradcam_localization_val_tar_full_20260429T093131Z.json`
- SAVLG: `/workspace/savlg_imagenet_standalone_runs/savlg_imagenet_full_7ep_a100_dense_20260428T115004Z_savlg-imagenet-full-a100-7ep-dense-xzm2t/localization_val_tar_alpha08_restart_20260429T082513Z.json`

## Protocol

- Dataset: ImageNet validation tar, `50000` images.
- VLG-CBM uses Grad-CAM maps over concept logits.
- SAVLG uses native spatial concept maps from the trained spatial branch.
- For each image/concept pair, the evaluator produces one predicted heatmap. If multiple ground-truth boxes/masks exist for that concept in the same image, they are combined into one target mask and one tight/enclosing target box. This is not detection-style per-instance matching.

## Metric Definitions

- `soft_iou`: continuous overlap between normalized heatmap and binary GT mask:
  `sum(min(pred_heatmap, gt_mask)) / sum(max(pred_heatmap, gt_mask))`.
- `mass_in_gt`: fraction of heatmap mass falling inside the GT mask.
- `point_hit`: whether the heatmap argmax lies inside the GT mask.
- `mask_iou@t`: IoU between thresholded predicted mask and GT mask, where `t` is the heatmap threshold.
- `MeanIoU`: shorthand here for `mask_iou@mean`, i.e. mask IoU after thresholding the heatmap at its mean value.
- `LocAcc@0.5`: localization accuracy at IoU `0.5`, i.e. the fraction of concept/image predictions whose predicted tight box has IoU at least `0.5` with the combined GT box.
- `LocAcc@0.3`: the same localization-accuracy metric at IoU `0.3`.
- `LocAcc` is not mAP. It is a direct GT-known localization accuracy over one predicted box per target. CUB code computed AP-style mAP separately, but some older CUB docs/tables used `mAP@...` wording for logged MaxBoxAcc sweep values.

## Results

Headline localization metrics:

| Metric | VLG-CBM Grad-CAM | SALF-CBM Native | SAVLG Native |
|---|---:|---:|---:|
| `LocAcc@0.3`, heatmap `mean` | 0.5246 | 0.5985 | 0.6542 |
| `LocAcc@0.5`, heatmap `mean` | 0.3572 | 0.4375 | 0.4873 |
| `MeanIoU` | 0.2405 | 0.2691 | 0.3449 |

Other localization metrics:

| Metric | VLG-CBM Grad-CAM | SALF-CBM Native | SAVLG Native |
|---|---:|---:|---:|
| images seen | 50000 | 50000 | 50000 |
| images with targets | 49712 | 49712 | 49712 |
| instances | 313105 | 312884 | 313198 |
| `soft_iou` | 0.3298 | 0.2254 | 0.3230 |
| `mass_in_gt` | 0.4748 | 0.4251 | 0.4975 |
| `point_hit` | 0.4161 | 0.4114 | 0.5787 |
| `mask_iou@0.3` | 0.1270 | 0.3665 | 0.4421 |
| `LocAcc@0.5`, heatmap `0.3` | 0.2162 | 0.4397 | 0.4775 |
| `mask_iou@0.5` | 0.0851 | 0.2703 | 0.3670 |
| `LocAcc@0.5`, heatmap `0.5` | 0.1501 | 0.4098 | 0.4777 |

Relative to VLG-CBM Grad-CAM:

| Metric | SALF-CBM Delta | SAVLG Delta |
|---|---:|---:|
| `LocAcc@0.3`, heatmap `mean` | +0.0740 | +0.1296 |
| `LocAcc@0.5`, heatmap `mean` | +0.0803 | +0.1301 |
| `MeanIoU` | +0.0286 | +0.1044 |
| `soft_iou` | -0.1044 | -0.0068 |
| `mass_in_gt` | -0.0497 | +0.0226 |
| `point_hit` | -0.0047 | +0.1626 |
| `mask_iou@0.3` | +0.2395 | +0.3151 |
| `LocAcc@0.5`, heatmap `0.3` | +0.2235 | +0.2613 |
| `mask_iou@0.5` | +0.1852 | +0.2819 |
| `LocAcc@0.5`, heatmap `0.5` | +0.2597 | +0.3276 |

## Interpretation

VLG-CBM Grad-CAM and SAVLG native maps have similar continuous overlap by `soft_iou`, with VLG-CBM slightly higher. However, SAVLG is much stronger once maps are used as actual localization masks or boxes. SAVLG has substantially higher point-hit, thresholded mask IoU, recall, F1, and localization accuracy.

The practical takeaway is that SAVLG produces more usable direct localization maps than VLG-CBM post-hoc Grad-CAM under this ImageNet protocol, even though the continuous `soft_iou` score alone does not show the improvement.

## Top-Contributing Concept Renders

Qualitative top-concept render artifacts:

- Local directory: `tmp/top_concept_renders_20260429T113716Z`
- Remote directory: `/workspace/SAVLGCBM-imagenet-test/results/top_concept_renders_20260429T113716Z`
- Summary JSON: `tmp/top_concept_renders_20260429T113716Z/top_concept_render_summary.json`
- SAVLG page: `tmp/top_concept_renders_20260429T113716Z/savlg_top_concepts_page_001.png`
- SALF-CBM page: `tmp/top_concept_renders_20260429T113716Z/salfcbm_top_concepts_page_001.png`
- VLG-CBM page: `tmp/top_concept_renders_20260429T113716Z/vlgcbm_top_concepts_page_001.png`

Render protocol:

- Six ImageNet validation-tar images were rendered.
- For each model and image, the predicted class is selected.
- Concepts are ranked by positive class contribution:
  `normalized_concept_activation * predicted_class_final_layer_weight`.
- The top three concepts are shown as heatmap overlays on the model's center-cropped input image.

## SALF-CBM Pretrained Accuracy / NEC

This evaluates the original-author pretrained SALF-CBM ImageNet checkpoint:

- Remote checkpoint dir: `/root/salf-cbm_models/imagenet`
- Local result JSON: `tmp/salf_imagenet/salf_imagenet_full_nec_20260429T104918Z.json`
- Local result log: `tmp/salf_imagenet/salf_imagenet_full_nec_20260429T104918Z.log`
- Remote result JSON: `/workspace/SAVLGCBM-imagenet-test/results/salf_imagenet/salf_imagenet_full_nec_20260429T104918Z.json`

Protocol details:

- Backbone follows the original `show-and-tell` code: HuggingFace `microsoft/resnet-50` via `ResNetForImageClassification`.
- Feature path uses `target_model.resnet.children()[:-1]` and `.last_hidden_state`.
- `W_c.pt` is applied as a `1x1` convolution over spatial features.
- Feature maps are resized to `12x12`.
- Concept activations use softmax pooling over the `12x12` map.
- Class labels use the sorted-WNID order. Raw ILSVRC-ID order gives near-zero accuracy and is incorrect for these weights.
- NEC truncation is per class: keep top-`NEC` absolute final-layer weights in each class row of `W_g`.

Dense result:

| Metric | Value |
|---|---:|
| top1 | 0.75564 |
| top5 | 0.91632 |
| final-layer nnz | 29918 / 4741000 |
| final-layer sparsity | 0.99369 |

NEC result:

| NEC | nnz | top1 | top5 |
|---:|---:|---:|---:|
| 5 | 5000 | 0.53534 | 0.80406 |
| 10 | 9993 | 0.67546 | 0.87864 |
| 15 | 14826 | 0.72668 | 0.90076 |
| 20 | 19153 | 0.74510 | 0.90986 |
| 25 | 22736 | 0.75078 | 0.91412 |
| 30 | 25461 | 0.75366 | 0.91526 |

Summary:

- `ACC@NEC=5`: `0.53534`
- `avgACC` over NEC `[5,10,15,20,25,30]`: `0.69784`

## SALF-CBM Native Localization

This evaluates native SALF-CBM concept maps from the original-author pretrained ImageNet checkpoint.

- Local result JSON: `tmp/salf_imagenet/salf_native_localization_val_tar_20260429T105731Z.json`
- Local result log: `tmp/salf_imagenet/salf_native_localization_val_tar_20260429T105731Z.log`
- Remote result JSON: `/workspace/SAVLGCBM-imagenet-test/results/salf_imagenet/salf_native_localization_val_tar_20260429T105731Z.json`

Protocol details:

- Same checkpoint and HuggingFace ResNet-50 feature path as the SALF-CBM accuracy/NEC evaluation.
- `W_c.pt` is applied as a `1x1` convolution over spatial features.
- Spatial maps are resized to `12x12`.
- Maps are normalized with `proj_mean.pt` and `proj_std.pt`; the local spatial pattern is centered and shifted by the pooled normalized concept activation before localization metrics are computed.
- Ground-truth handling matches the SAVLG/VLG-CBM ImageNet localization protocol: multiple boxes for a concept/image are combined into one target mask and tight box.

Full-val result:

| Metric | Value |
|---|---:|
| images seen | 50000 |
| images with targets | 49712 |
| instances | 312884 |
| `soft_iou` | 0.2254 |
| `mass_in_gt` | 0.4251 |
| `point_hit` | 0.4114 |
| `mask_iou@0.3` | 0.3665 |
| `LocAcc@0.5`, heatmap `0.3` | 0.4397 |
| `mask_iou@0.5` | 0.2703 |
| `LocAcc@0.5`, heatmap `0.5` | 0.4098 |
| `mask_iou@mean` | 0.2691 |
| `LocAcc@0.5`, heatmap `mean` | 0.4375 |

Compared with SAVLG native maps, SALF-CBM is weaker on `point_hit`, thresholded `mask_iou`, and `LocAcc@0.5`, but it is substantially stronger than VLG-CBM Grad-CAM on thresholded mask and box localization.
