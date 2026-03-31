# VLG-CBM Notes

## Port plan

- Source reference: local upstream `VLG-CBM` repo
- Port style: keep behavior as close as possible

## What should be ported directly

- dataset loading
- concept dataset wrapper
- concept filtering
- backbone wrappers
- concept bottleneck training
- sparse final layer training
- sparse / NEC evaluation

## Planned changes

- route through the new model registry
- add artifact metadata
- keep config compatibility with upstream JSON files

## Remaining work

- imported as the repo base
- unified dispatch added in `train_cbm.py`
- needs CLI/documentation cleanup and later comparison testing
