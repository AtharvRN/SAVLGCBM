# ImageNet Val Annotation Audit

Archive:
- `~/Downloads/imagenet_val.tar.gz`

Pod placement:
- extracted to `/root/imagenet_val`
- exposed as `/workspace/SAVLGCBM/annotations/imagenet_val`

Schema:
- `50000` JSON files
- each file is a list
- `payload[0]` contains `img_path`
- `payload[1:]` contains annotation dicts with `label`, `box`, `logit`, `logits`

Exact full-val concept coverage against `concept_files/imagenet_filtered.txt`:

- JSON files: `50000`
- total annotations: `1251359`
- mapped: `1117692`
- mapped rate: `0.893183`
- unmapped: `133667`
- unmapped rate: `0.106817`

Alias map applied before counting:
- `website -> a web page`
- `beer bottle -> a bottle with a long neck`
- `wine bottle -> a bottle with a long neck`
- `soda bottle -> a glass or plastic bottle`
- `ski -> a pair of skis`
- `metal nail -> nails`

Top unmapped labels after aliasing:

1. `strawberry` — `752`
2. `dumbbell` — `595`
3. `jellyfish` — `595`
4. `bubble` — `593`
5. `umbrella` — `576`
6. `banana` — `575`
7. `bee` — `517`
8. `corn` — `517`
9. `pretzel` — `503`
10. `fig` — `500`
11. `zucchini` — `472`
12. `candle` — `469`
13. `canoe` — `468`
14. `lipstick` — `462`
15. `gondola` — `460`
16. `broccoli` — `459`
17. `pomegranate` — `444`
18. `goldfish` — `442`
19. `orange` — `429`
20. `crate` — `423`

Smoke test status:
- `SpatialBackbone` now supports standard `resnet50`
- SAVLG ImageNet annotation path is valid
- current failure is missing ImageNet image root:
  - `datasets/imagenet/ILSVRC/Data/CLS-LOC/train`
