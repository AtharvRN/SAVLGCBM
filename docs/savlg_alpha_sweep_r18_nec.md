**SAVLG Alpha Sweep NEC**

Source logs on `atharv-rwx-pod`:
- `/workspace/SAVLGCBM/logs/nec_savlg_alpha_0_2gpu_job.log`
- `/workspace/SAVLGCBM/logs/nec_savlg_alpha_0p05_2gpu_job.log`
- `/workspace/SAVLGCBM/logs/nec_savlg_alpha_0p1_2gpu_job.log`
- `/workspace/SAVLGCBM/logs/nec_savlg_alpha_0p2_2gpu_job.log`
- `/workspace/SAVLGCBM/logs/nec_savlg_alpha_0p4_2gpu_job.log`
- `/workspace/SAVLGCBM/logs/nec_savlg_alpha_0p6_2gpu_job.log`
- `/workspace/SAVLGCBM/logs/nec_savlg_alpha_0p8_2gpu_job.log`
- `/workspace/SAVLGCBM/logs/nec_savlg_alpha_1p0_2gpu_job.log`

Saved table:
- [savlg_alpha_sweep_r18_nec_summary.csv](/Users/atharvramesh/Projects/CBM/SAVLGCBM/results/savlg_alpha_sweep_r18_nec_summary.csv)

Setup:
- model: `SAVLG-CBM`
- backbone: `resnet18_cub`
- branch: `dual`
- sparse eval: `sparse_evaluation.py`
- `cbl_batch_size=128`
- `saga_batch_size=1024`
- `num_workers=24`

Results:

| alpha | Acc@5 | Acc@10 | Acc@15 | Acc@20 | Acc@25 | Acc@30 | AVGACC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.7546 | 0.7554 | 0.7556 | 0.7558 | 0.7560 | 0.7560 | 0.7556 |
| 0.05 | 0.7551 | 0.7560 | 0.7563 | 0.7558 | 0.7565 | 0.7566 | 0.7560 |
| 0.1 | 0.7553 | 0.7570 | 0.7566 | 0.7566 | 0.7570 | 0.7572 | 0.7566 |
| 0.2 | 0.7568 | 0.7572 | 0.7568 | 0.7572 | 0.7570 | 0.7570 | 0.7570 |
| 0.4 | 0.7592 | 0.7601 | 0.7601 | 0.7601 | 0.7606 | 0.7610 | 0.7602 |
| 0.6 | 0.7606 | 0.7610 | 0.7615 | 0.7617 | 0.7615 | 0.7613 | 0.7613 |
| 0.8 | 0.7596 | 0.7599 | 0.7604 | 0.7610 | 0.7615 | 0.7615 | 0.7607 |
| 1.0 | 0.7606 | 0.7608 | 0.7615 | 0.7617 | 0.7617 | 0.7618 | 0.7613 |

Key takeaways:
- Increasing `alpha` clearly helps over the `alpha=0` baseline.
- The biggest gain happens between `alpha=0.2` and `alpha=0.4`.
- Best `AVGACC` in this sweep is a tie:
  - `alpha=0.6`: `0.7613`
  - `alpha=1.0`: `0.7613`
- Best `Acc@30` is:
  - `alpha=1.0`: `0.7618`
- `alpha=0.4` already captures most of the gain:
  - `AVGACC = 0.7602`
- The sweep suggests the useful region is roughly:
  - `alpha ∈ [0.4, 1.0]`

Practical recommendation:
- If we want a conservative choice, use `alpha=0.6`.
- If we optimize purely for the best high-`k` sparse metric in this sweep, use `alpha=1.0`.

Notes:
- This is the completed NEC sweep from the earlier alpha job.
- The separate fresh SAM reruns for `alpha=0.2` and `alpha=0.5` on the restarted pod are different runs and should not be mixed into this table.
