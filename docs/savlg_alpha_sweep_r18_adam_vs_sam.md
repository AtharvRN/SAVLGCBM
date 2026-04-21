# SAVLG `resnet18` Alpha Sweep: Adam vs SAM

This compares the same dual-branch SAVLG recipe across residual fusion weights `alpha` for:

- `Adam`
- `SAM` with `rho=0.05`

Saved CSV:
- [results/savlg_alpha_sweep_r18_adam_vs_sam.csv](/Users/atharvramesh/Projects/CBM/SAVLGCBM/results/savlg_alpha_sweep_r18_adam_vs_sam.csv)

## Summary Table

| alpha | Adam dense | SAM dense | Adam Acc@30 | SAM Acc@30 | Adam AVGACC | SAM AVGACC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.7594 | 0.7594 | 0.7560 | 0.7561 | 0.7556 | 0.7556 |
| 0.05 | 0.7594 | 0.7591 | 0.7566 | 0.7572 | 0.7560 | 0.7563 |
| 0.1 | 0.7589 | 0.7589 | 0.7572 | 0.7573 | 0.7566 | 0.7566 |
| 0.2 | 0.7591 | 0.7591 | 0.7570 | 0.7572 | 0.7570 | 0.7569 |
| 0.4 | 0.7599 | 0.7596 | 0.7610 | 0.7611 | 0.7602 | 0.7603 |
| 0.6 | 0.7615 | 0.7611 | 0.7613 | 0.7611 | 0.7613 | 0.7611 |
| 0.8 | 0.7623 | 0.7623 | 0.7615 | 0.7611 | 0.7607 | 0.7604 |
| 1.0 | 0.7630 | 0.7627 | 0.7618 | 0.7617 | 0.7613 | 0.7615 |

## Read

- Dense test accuracy increases with `alpha` for both optimizers.
- The best dense checkpoints are at the high end of the sweep:
  - `Adam`: best dense at `alpha=1.0` with `0.7630`
  - `SAM`: best dense at `alpha=1.0` with `0.7627`
- NEC metrics are also strongest in the `0.4-1.0` region for both optimizers.
- SAM is very close to Adam throughout the sweep.

## Main takeaways

- SAM does **not** materially change the best alpha region.
- The dominant effect is still `alpha`, not the optimizer.
- The strongest operating region remains roughly:
  - `alpha=0.4` to `alpha=1.0`
- On this sweep:
  - Adam is slightly better on dense accuracy at the top end.
  - SAM is essentially tied on NEC average accuracy.
