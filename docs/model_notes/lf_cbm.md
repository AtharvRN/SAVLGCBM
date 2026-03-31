# LF-CBM Notes

## Port plan

- Source reference: `Medical_CBM/label_free_cbm.py`
- Port target: general-domain VLG dataset stack

## What should be ported

- pseudo-label generation via CLIP image-text similarities
- projection-layer training
- optional linear / MLP concept bottleneck
- sparse final layer training
- LF checkpoint artifact handling

## Planned changes

- replace medical dataset handling with VLG general-domain loaders
- share backbone and concept-bank infrastructure with VLG-CBM
- integrate LF checkpoints into unified sparse evaluation

## Remaining work

- first unified training path implemented in `methods/lf.py`
- sparse evaluation currently assumes the linear LF checkpoint path
- MLP LF checkpoints need a dedicated sparse-eval loader path if they are to be used in NEC sweeps
