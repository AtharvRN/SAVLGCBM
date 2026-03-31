# SAVLG-CBM Notes

## Port plan

- Source reference: `Medical_CBM_dev_clean/savlg_cbm.py`
- Port target: general-domain VLG dataset stack and annotation layout

## What should be ported

- annotation-driven concept presence targets
- spatial mask supervision
- spatial concept map head training
- pooled concept activations + sparse final layer

## Planned changes

- adapt supervision loading to VLG-style dataset/annotation organization
- preserve spatial checkpoint artifacts
- integrate with shared sparse evaluation and model registry

## Remaining work

- config examples added
- training port still pending
