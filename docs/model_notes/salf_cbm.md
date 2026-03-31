# SALF-CBM Notes

## Port plan

- Source reference: `Medical_CBM/spatial_aware_cbm.py`
- Port target: general-domain VLG dataset stack

## What should be ported

- spatial pseudo-label generation
- spatial concept map head
- pooled concept activations
- sparse final layer training
- SALF spatial evaluation hooks

## Planned changes

- remove CheXpert / COVID-specific assumptions
- reuse VLG dataset and concept-bank conventions
- keep SALF checkpoint layout compatible with spatial evaluation

## Remaining work

- config examples added
- training port still pending
