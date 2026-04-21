## SAVLG Alpha Sweep (`resnet18`, Adam) Localization Summary

Source JSONs:
- `/workspace/SAVLGCBM/results/alpha_sweep_r18_loc/loc_savlg_alpha_*.json`

Saved table:
- [savlg_alpha_sweep_r18_loc_summary.csv](/Users/atharvramesh/Projects/CBM/SAVLGCBM/results/savlg_alpha_sweep_r18_loc_summary.csv)

### Main result

Localization is effectively flat across `alpha` in this sweep.

- Best `mean IoU` is at `alpha=0.0`:
  - `0.0543497` at threshold `0.05`
- Best `mAP@0.1` is tied at high alpha:
  - `alpha=0.8` and `alpha=1.0`: `0.1944901` at threshold `0.05`
- Best `mAP@0.3` is essentially tied:
  - `0.0280506` for `alpha=0.0, 0.05, 0.2, 0.6, 0.8`
- Best `mAP@0.5` is at `alpha=0.0`:
  - `0.0034228` at threshold `0.2`
- `mAP@0.7` is identical across all alphas:
  - `0.0004731` at threshold `0.05`

### Takeaway

For this `resnet18` Adam sweep, changing `alpha` clearly affected sparse and dense classification metrics, but it did **not** materially change the localization metrics under the current native-map box protocol. The differences are at the fourth decimal place or smaller.
