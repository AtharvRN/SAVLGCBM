# Progress Notes

## 2026-04-07

### Residual sprint completed at 18:13 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint README and current residual artifact state, and then used direct pod checks on `atharv-rwx-pod-2` as the source of truth before deciding whether any more work was eligible.
- Verified the pod remains healthy:
  - phase: `Running`
  - restarts: `0`
- Verified residual-stage execution is fully complete:
  - both A10 GPUs are now idle at `0 MiB / 23028 MiB`
  - no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` processes remain
- Verified the final remaining native GT-present localization artifact is now present:
  - `Residual-Alpha-0.05`
    - output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha005_softalign_outside_v1_full.log`
    - mean IoU `0.13085882710717972`
    - `mAP@0.3 = 0.1188843234267777`
    - `mAP@0.5 = 0.0656205807861721`
    - `mAP@0.7 = 0.03112899065322384`
    - point hit `0.9889426671388659`
    - coverage `0.3656704605954214`
- Final residual-stage comparison across the exact three `TASK.md` runs:
  - `Residual-Alpha-0.05`
    - dense `0.7592400690846287`
    - `ACC@5 = 0.7560`
    - `AVGACC = 0.7568`
    - native mean IoU `0.13085882710717972`
    - native `mAP@0.5 = 0.0656205807861721`
    - native point hit / coverage `0.9889426671388659 / 0.3656704605954214`
  - `Residual-Alpha-0.10`
    - dense `0.7585492227979275`
    - `ACC@5 = 0.7572`
    - `AVGACC = 0.7581`
    - native mean IoU `0.13070872910587703`
    - native `mAP@0.5 = 0.06573367546834702`
    - native point hit / coverage `0.9889265447667087 / 0.36513806351353295`
  - `Residual-Alpha-0.05-PlusMIL`
    - dense `0.7590673575129534`
    - `ACC@5 = 0.7563`
    - `AVGACC = 0.7575`
    - native mean IoU `0.12157116163951714`
    - native `mAP@0.5 = 0.043425785090992264`
    - native point hit / coverage `0.9846741465743615 / 0.30137991567981354`
- Residual-stage interpretation after full dense + NEC + native evaluation:
  - all three exact residual runs preserve dense accuracy essentially at the `VLG` anchor level while also preserving sparse recovery near the frozen-global outside-penalty line
  - both plain residual variants materially improve over the frozen-global `soft-align + outside penalty` localization baseline on mean IoU, all reported `mAP` thresholds, point hit, and coverage
  - `Residual-Alpha-0.05` is the best dense run and also edges `Residual-Alpha-0.10` on native mean IoU, `mAP@0.3`, `mAP@0.7`, point hit, and coverage
  - `Residual-Alpha-0.10` remains slightly best on sparse recovery (`ACC@5`, `AVGACC`) and by a negligible margin on native `mAP@0.5`
  - `Residual-Alpha-0.05-PlusMIL` does not justify promotion because it trails the plain residual variants on sparse recovery and native localization
- Updated `docs/cub_results.md`, updated the sprint README, and added `savlg-research/2026-04-07--16-33-residual-spatial-coupling/DONE.txt`.

### Residual localization finalized for two runs; alpha=0.05 launched at 18:08 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint README, `runtime_state.json`, and recent loop logs, and then used a fresh direct pod check on `atharv-rwx-pod-2` before taking any action.
- Verified the pod remains healthy:
  - phase: `Running`
  - restarts: `0`
- Verified both GPUs were idle before launch:
  - GPU `0`: `0 MiB / 23028 MiB`
  - GPU `1`: `0 MiB / 23028 MiB`
- Verified the two earlier residual native GT-present localization jobs are now finalized:
  - `Residual-Alpha-0.10`
    - output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha010_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - mean IoU `0.13070872910587703`
    - `mAP@0.3 = 0.1188683369104664`
    - `mAP@0.5 = 0.06573367546834702`
    - `mAP@0.7 = 0.030992899959122733`
    - point hit `0.9889265447667087`
    - coverage `0.36513806351353295`
  - `Residual-Alpha-0.05-PlusMIL`
    - output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - mean IoU `0.12157116163951714`
    - `mAP@0.3 = 0.07900841114248819`
    - `mAP@0.5 = 0.043425785090992264`
    - `mAP@0.7 = 0.024953646221050316`
    - point hit `0.9846741465743615`
    - coverage `0.30137991567981354`
- Launched the remaining eligible residual native GT-present localization follow-up:
  - GPU `0`: `Residual-Alpha-0.05`
    - load path: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha005_softalign_outside_v1_full.log`
    - output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - launcher / main pid pair: `108706 / 108709`
    - startup evidence:
      - metadata and annotations loaded successfully
      - dataloader ready at `181` batches with `batch_size=32`
      - forward progress reached `26/181` at the verification check
- `Residual-Alpha-0.10` is currently the strongest finalized residual run on sparse recovery and on every finalized native GT-present localization metric.
- `DONE.txt` was not added because `Residual-Alpha-0.05` native localization is still in flight.

### Residual localization jobs still healthy at 18:04 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint README, `runtime_state.json`, and recent loop logs, and used a fresh live pod check on `atharv-rwx-pod-2` before taking any action.
- Verified the pod remains healthy:
  - phase: `Running`
  - restarts: `0`
  - age at the first successful check: about `86m`
- Confirmed both visible GPUs are still occupied by the same two native GT-present localization jobs:
  - GPU `0`: `Residual-Alpha-0.10`
    - launcher / main pid pair: `106453 / 106458`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha010_softalign_outside_v1_full.log`
    - forward progress over a 20s recheck: `169/181 -> 172/181`
  - GPU `1`: `Residual-Alpha-0.05-PlusMIL`
    - launcher / main pid pair: `106443 / 106456`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha005_localmil_outside_v1_full.log`
    - forward progress over a 20s recheck: `169/181 -> 172/181`
- GPU memory stayed allocated on both devices across the successful `nvidia-smi` checks:
  - GPU `0`: `709 MiB / 23028 MiB`
  - GPU `1`: `709 MiB / 23028 MiB`
- No residual native-localization JSON has finalized yet:
  - `/workspace/SAVLGCBM/results/native_savlg_residual_alpha010_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still absent
  - `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still absent
- The remaining unlaunched residual native-localization follow-up is still `Residual-Alpha-0.05`, but no launch was safe in this iteration because both visible GPUs are already occupied by healthy eligible jobs.
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no new finalized residual localization artifact landed in this iteration.

### Residual localization jobs still healthy at 18:01 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint README, `runtime_state.json`, and recent progress notes, and used a fresh live pod check on `atharv-rwx-pod-2` before taking any action.
- Verified the pod remains healthy:
  - phase: `Running`
  - restarts: `0`
  - age at the successful check: about `84m`
- Confirmed both visible GPUs are still occupied by the same two native GT-present localization jobs:
  - GPU `0`: `Residual-Alpha-0.10`
    - launcher / main pid pair: `106453 / 106458`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha010_softalign_outside_v1_full.log`
    - forward progress over a 15s recheck: `157/181 -> 159/181`
  - GPU `1`: `Residual-Alpha-0.05-PlusMIL`
    - launcher / main pid pair: `106443 / 106456`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha005_localmil_outside_v1_full.log`
    - forward progress over a 15s recheck: `157/181 -> 159/181`
- GPU memory stayed allocated on both devices across the two successful `nvidia-smi` checks:
  - GPU `0`: `709 MiB / 23028 MiB`
  - GPU `1`: `709 MiB / 23028 MiB`
- The remaining unlaunched residual native-localization follow-up is still `Residual-Alpha-0.05`, but no launch was safe in this iteration because both visible GPUs are already occupied by healthy eligible jobs.
- A later results-file existence query hit the intermittent OIDC/DNS auth failure again, so the iteration decision relied only on the earlier successful pod checks and the confirmed log forward progress.
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no new finalized residual localization artifact landed in this iteration.

### Residual localization jobs still healthy at 17:41 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint notes and `docs/progress_notes.md`, and used a fresh live pod check on `atharv-rwx-pod-2` before taking any new action.
- Verified the pod remains healthy:
  - phase: `Running`
  - restarts: `0`
- Confirmed both visible GPUs are still occupied by the two native GT-present localization jobs launched at `17:29 PDT`:
  - GPU `0`: `Residual-Alpha-0.10`
    - launcher / main pid pair: `106453 / 106458`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha010_softalign_outside_v1_full.log`
    - forward progress over a 15s recheck: `58/181 -> 63/181`
  - GPU `1`: `Residual-Alpha-0.05-PlusMIL`
    - launcher / main pid pair: `106443 / 106456`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha005_localmil_outside_v1_full.log`
    - forward progress over a 15s recheck: `58/181 -> 63/181`
- GPU memory remained allocated on both devices throughout the check:
  - GPU `0`: `709 MiB / 23028 MiB`
  - GPU `1`: `709 MiB / 23028 MiB`
- No residual native-localization JSON has finalized yet:
  - `/workspace/SAVLGCBM/results/native_savlg_residual_alpha010_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still absent
  - `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still absent
- The remaining unlaunched residual native-localization follow-up is still `Residual-Alpha-0.05`, but no launch was safe in this iteration because both visible GPUs are already occupied by healthy eligible jobs.
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no new finalized residual localization artifact landed in this iteration.

### Residual localization jobs still healthy at 17:33 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint notes, and used a fresh live pod check on `atharv-rwx-pod-2` before taking any new action.
- Verified live pod visibility is available again in this session:
  - `kubectl get pod atharv-rwx-pod-2 -o wide` succeeded
  - pod phase remains `Running`
  - restarts remain `0`
- Confirmed both visible GPUs are still occupied by the two native GT-present localization jobs launched at `17:29 PDT`:
  - GPU `0`: `Residual-Alpha-0.10`
    - launcher / main pid pair: `106453 / 106458`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha010_softalign_outside_v1_full.log`
    - forward progress over a 15s recheck: `13/181 -> 16/181`
  - GPU `1`: `Residual-Alpha-0.05-PlusMIL`
    - launcher / main pid pair: `106443 / 106456`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha005_localmil_outside_v1_full.log`
    - forward progress over a 15s recheck: `13/181 -> 16/181`
- GPU memory remained allocated on both devices throughout the check:
  - GPU `0`: `709 MiB / 23028 MiB`
  - GPU `1`: `709 MiB / 23028 MiB`
- No residual native-localization JSON has finalized yet:
  - `/workspace/SAVLGCBM/results/native_savlg_residual_alpha010_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still absent
  - `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still absent
- The remaining unlaunched residual native-localization follow-up is still `Residual-Alpha-0.05`, but no launch was safe in this iteration because both visible GPUs are already occupied by healthy eligible jobs.
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no new finalized residual localization artifact landed in this iteration.

### Residual NEC finalized and native localization launched at 17:29 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint notes, and used a fresh live pod check on `atharv-rwx-pod-2` before taking any new action.
- Local static validation still passes:
  - `python3 -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py`
- Pod-side static validation passed again immediately before launch:
  - `/opt/conda/envs/cbm/bin/python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py`
- Verified the pod was healthy and idle before relaunch:
  - phase: `Running`
  - restarts: `0`
  - both GPUs sampled at `0 MiB / 23028 MiB`
  - no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` process was visible
- Verified all three residual dense runs are now finalized and dense-safe:
  - `Residual-Alpha-0.05`: dense accuracy `0.7592400690846287`
  - `Residual-Alpha-0.10`: dense accuracy `0.7585492227979275`
  - `Residual-Alpha-0.05-PlusMIL`: dense accuracy `0.7590673575129534`
- Verified the remaining two residual sparse follow-ups are also finalized:
  - `Residual-Alpha-0.10` NEC:
    - `ACC@5 = 0.7572`
    - `ACC@10 = 0.7584`
    - `ACC@15 = 0.7580`
    - `ACC@20 = 0.7579`
    - `ACC@25 = 0.7584`
    - `ACC@30 = 0.7585`
    - `AVGACC = 0.7581`
  - `Residual-Alpha-0.05-PlusMIL` NEC:
    - `ACC@5 = 0.7563`
    - `ACC@10 = 0.7572`
    - `ACC@15 = 0.7572`
    - `ACC@20 = 0.7573`
    - `ACC@25 = 0.7584`
    - `ACC@30 = 0.7584`
    - `AVGACC = 0.7575`
- Because both GPUs were free and no residual localization JSON existed yet, the next eligible work was native GT-present localization:
  - GPU `0`: launched `Residual-Alpha-0.10` native eval
    - load path: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_48_21`
    - output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha010_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha010_softalign_outside_v1_full.log`
    - launcher / main pid pair: `106453 / 106458`
  - GPU `1`: launched `Residual-Alpha-0.05-PlusMIL` native eval
    - load path: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_00_03_22`
    - output: `/workspace/SAVLGCBM/results/native_savlg_residual_alpha005_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_residual_alpha005_localmil_outside_v1_full.log`
    - launcher / main pid pair: `106443 / 106456`
- Verified both localization jobs started cleanly:
  - both logs loaded CUB metadata and annotations successfully
  - both dataloaders initialized at `181` batches with `batch_size=32`
  - both advanced from `0/181` to `1/181`
- `Residual-Alpha-0.05` native localization remains outstanding for a later iteration because only two visible GPUs were available and both were occupied by eligible localization jobs after launch.
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no finalized residual native-localization output has landed yet.

### Residual NEC verification status at 17:21 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint notes, and used a fresh live pod check on `atharv-rwx-pod-2` instead of stale local state.
- Local static validation still passes:
  - `python3 -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py`
- Verified all three residual dense runs are finalized and dense-safe:
  - `Residual-Alpha-0.05`: dense accuracy `0.7592400690846287`
  - `Residual-Alpha-0.10`: dense accuracy `0.7585492227979275`
  - `Residual-Alpha-0.05-PlusMIL`: dense accuracy `0.7590673575129534`
- Verified the first residual sparse follow-up is finalized:
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1`
  - `Residual-Alpha-0.05` NEC:
    - `ACC@5 = 0.7560`
    - `ACC@10 = 0.7568`
    - `ACC@15 = 0.7568`
    - `ACC@20 = 0.7570`
    - `ACC@25 = 0.7570`
    - `ACC@30 = 0.7572`
    - `AVGACC = 0.7568`
- Verified both GPUs remain occupied by the two remaining eligible NEC follow-ups, so no new launch was safe or necessary:
  - GPU `0`: `Residual-Alpha-0.10` NEC
  - GPU `1`: `Residual-Alpha-0.05-PlusMIL` NEC
- Localization evaluation was not launched in this iteration because the dense-first policy still prioritizes the two unfinished NEC jobs already keeping both GPUs busy.

### Residual spatial coupling recovery and dense launches at 16:50 PDT

- Re-read `TASK.md`, inspected `savlg-research/`, and implemented the residual SAVLG concept path:
  - `c_final = c_global + alpha * c_spatial`
  - `c_spatial` uses fixed `lse` pooling
  - dense concept extraction, dense evaluation, sparse extraction, and eval helpers now use `c_final`
- Added the exact three allowed configs for this stage:
  - `Residual-Alpha-0.05`
  - `Residual-Alpha-0.10`
  - `Residual-Alpha-0.05-PlusMIL`
- Direct pod inspection then showed the working pod was not healthy:
  - `atharv-rwx-pod-2`: `Failed`
  - `Reason: DeadlineExceeded`
- Recovered the pod with the documented RWX spec:
  - deleted `atharv-rwx-pod-2`
  - reapplied `/Users/atharvramesh/UCSD/Fall2025/DSC190/Project/pod_scripts/pod-using-rwx-pvc-2.yaml`
  - verified the recreated pod reached `Running`
- Rebuilt the required `cbm` env:
  - `conda create -y -n cbm --clone base`
  - `pip install -r requirements.txt`
  - re-pinned the task-required versions after `pip` drifted them:
    - `numpy==1.26.4`
    - `opencv-python==4.11.0.86`
- Verified pod-side imports and versions:
  - `torch 2.2.2`
  - `torchvision 0.17.2`
  - `numpy 1.26.4`
  - `cv2 4.11.0`
  - `loguru 0.7.3`
  - `clip`
  - `pytorchcv 0.0.74`
- Synced the residual-stage code/config/doc files into `/workspace/SAVLGCBM` and verified pod-side `py_compile` for:
  - `methods/savlg.py`
  - `train_cbm.py`
  - `evaluations/sparse_utils.py`
  - `scripts/evaluate_native_spatial_maps.py`
- Verified the anchor checkpoints still exist on the PVC:
  - `/workspace/SAVLGCBM/saved_models/cub/cub_cbm_2026_04_06_06_16_10`
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13`
- Verified both GPUs were idle and there were no pre-existing residual-stage logs or checkpoints, so new dense launches were eligible.
- Launched the first two exact dense runs:
  - GPU `0`
    - config: `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_v1.json`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_v1.log`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1`
    - launcher pid: `1715`
    - latest completed epoch summary:
      - `[SAVLG CBL] epoch=7 train_loss=0.453151 val_loss=1.150138 best_val=1.042034`
  - GPU `1`
    - config: `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_vlgwarm_residual_alpha010_softalign_outside_v1.json`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha010_softalign_outside_v1.log`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_48_21`
    - launcher pid: `5117`
    - latest completed epoch summary:
      - `[SAVLG CBL] epoch=4 train_loss=0.493503 val_loss=1.103325 best_val=1.041177`
- One recoverable startup race occurred on the first `alpha=0.10` launch:
  - `pytorchcv` tried to remove `/root/.torch/models/resnet18_cub-2333-200d8b9c.pth.zip` after a concurrent backbone download
  - once the `alpha=0.05` run had cached `resnet18_cub`, the `alpha=0.10` run was relaunched successfully
- Live GPU allocation at the final check:
  - GPU `0`: `831 MiB`
  - GPU `1`: `831 MiB`
- `Residual-Alpha-0.05-PlusMIL` was not launched in this iteration because both eligible GPUs are already occupied by the first two exact dense runs.
- NEC remains ineligible until one of these dense checkpoints finishes and clears the dense gate.

### Residual spatial coupling implementation hold at 16:33 PDT

- Re-read `TASK.md`, inspected the existing `savlg-research/` sprint state, and confirmed the new active stage is the residual-coupling follow-up to the frozen-global `soft_align` line.
- Implemented the required concept-logit coupling in `SAVLG`:
  - `c_final = c_global + alpha * c_spatial`
  - `c_spatial` uses fixed `lse` pooling over the spatial concept maps
  - dense training, dense evaluation, and sparse concept extraction now all use `c_final`
- Added the exact three allowed dense configs for this stage:
  - `Residual-Alpha-0.05`
  - `Residual-Alpha-0.10`
  - `Residual-Alpha-0.05-PlusMIL`
- This sandbox still cannot verify live pod state safely:
  - `/workspace/SAVLGCBM` is not mounted here
  - `ps` is blocked
  - `nvidia-smi` is unavailable
- No launch was taken because doing so without fresh pod/job visibility would risk duplicating healthy work and violating the active-job rule in `TASK.md`.
- Validation for the code/config changes was deferred to local static checks in this iteration; no new dense / NEC / localization metrics were produced, so no sprint was marked done and `docs/cub_results.md` was left unchanged.

## 2026-04-06

### VLG-warm no-outside localization hold at 22:49 PDT

- Re-read `TASK.md`, inspected the existing `savlg-research/` sprint state, and resolved the stale local hold by checking `atharv-rwx-pod-2` directly.
- Pod state at the live check:
  - `Running`
  - restarts `0`
  - both GPUs occupied only by the active no-outside native `gt_present` localization jobs
  - GPU `0`: pid `207289`, `709 MiB / 23028 MiB`
  - GPU `1`: pid `207864`, `709 MiB / 23028 MiB`
- Verified the active workers are the intended `evaluate_native_spatial_maps.py` runs for:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57`
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57-1`
- The expected output JSONs are still missing:
  - `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_only_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_v1_full_meanthr_bs32_nw8_gtpresent.json`
- Health check over the live logs showed continued forward progress and no traceback / OOM text:
  - `soft-align only`: about `46/181` to `56/181`
  - `soft-align + local-mil`: about `46/181` to `58/181`
- No new action was eligible in this iteration:
  - no duplicate localization launch because both GPUs are already busy with the only eligible jobs
  - no dense or NEC launch because the sprint is already in the final localization-comparison phase
  - no kill or restart because both jobs are healthy
- Validation passed locally:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no new finalized localization artifact landed in this iteration.

### VLG-warm no-outside localization launch

- Re-read `TASK.md`, inspected the existing `savlg-research/` sprint state, and replaced the stale hold assumption with a direct pod check on `atharv-rwx-pod-2`.
- Pod state at the live check:
  - `Running`
  - restarts `0`
  - both GPUs idle
  - no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` jobs
- Recovered the finished no-outside soft-align sparse results directly from pod artifacts:
  - `soft-align only`
    - dense checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57`
    - dense test accuracy `0.7590673575129534`
    - `ACC@5 = 0.7549`
    - `AVGACC = 0.7560`
  - `soft-align + local-mil`
    - dense checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57-1`
    - dense test accuracy `0.7590673575129534`
    - `ACC@5 = 0.7549`
    - `AVGACC = 0.7560`
- Since those two checkpoints had already cleared the dense and NEC gates and the outside-penalty pair already had completed localization outputs, the next eligible action was the missing native `gt_present` localization for the two no-outside soft-align variants.
- Launched native localization on both GPUs:
  - GPU `0`
    - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_only_v1_full_meanthr_bs32_nw8_gtpresent.log`
    - output: `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_only_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - GPU `1`
    - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57-1`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_v1_full_meanthr_bs32_nw8_gtpresent.log`
    - output: `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_v1_full_meanthr_bs32_nw8_gtpresent.json`
- Startup evidence after launch:
  - both logs reached `Evaluating 1 checkpoints on 5790 images with 671 shared annotated concepts`
  - both logs reached `dataloader ready: batches=181 batch_size=32 num_workers=8`
  - visible progress at check:
    - `soft-align only`: `12/181`
    - `soft-align + local-mil`: `1/181`
  - GPU memory after launch:
    - GPU `0`: `709 MiB / 23028 MiB`
    - GPU `1`: `709 MiB / 23028 MiB`
- Validation passed locally and on pod:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
  - `/opt/conda/envs/cbm/bin/python -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`

## 2026-04-05

### Localization eval recovery

- Re-read `TASK.md`, inspected the existing sprint directories, and recovered the current localization state directly from the live pods.
- `atharv-rwx-pod` is healthy and already running the missing paper-style Grad-CAM point-localization comparisons required by the current `TASK.md` gate:
  - `GPU 0`: `SAVLG` point rerun writing `/workspace/SAVLGCBM/results/gradcam_point_savlg_best_full.json`
  - `GPU 1`: `SALF` point rerun writing `/workspace/SAVLGCBM/results/gradcam_point_salf_best_full.json`
- `atharv-rwx-pod-2` is effectively fresh:
  - pod is `Running`
  - `/workspace/SAVLGCBM`, annotations, and CUB metadata are present
  - required conda env `cbm` is missing and only `base` exists
  - no relaunch was taken there because that would require the full recovery checklist and would have duplicated the healthy point jobs already running on `atharv-rwx-pod`
- Recovered completed localization artifacts that were not yet summarized in the main docs:
  - native-map bbox comparison:
    - `SAVLG`: mean IoU `0.0233`, `mAP@0.3 = 0.00620`, `mAP@0.5 = 0.00505`, `mAP@0.7 = 0.00458`
    - `SALF`: mean IoU `0.0778`, `mAP@0.3 = 0.00645`, `mAP@0.5 = 0.00500`, `mAP@0.7 = 0.00456`
  - Grad-CAM bbox comparison:
    - `SAVLG`: mean IoU `0.0797`, `mAP@0.3 = 0.0266`, `mAP@0.5 = 0.0109`, `mAP@0.7 = 0.00648`
    - `SALF`: mean IoU `0.0789`, `mAP@0.3 = 0.00661`, `mAP@0.5 = 0.00512`, `mAP@0.7 = 0.00456`
- Current interpretation:
  - `SAVLG` already has a clear Grad-CAM bbox-mAP advantage over `SALF`
  - native-map localization is not yet a clear `SAVLG` improvement
  - the next model-change decision remains blocked on the point hit / coverage outputs from the in-flight Grad-CAM reruns

### Localization eval recovery follow-up

- Rechecked live pods before taking any new action and confirmed the point-localization gate is still the only eligible unfinished work:
  - `atharv-rwx-pod` still has the healthy `SAVLG` and `SALF` Grad-CAM point evals running
  - the two output files are still not present:
    - `/workspace/SAVLGCBM/results/gradcam_point_savlg_best_full.json`
    - `/workspace/SAVLGCBM/results/gradcam_point_salf_best_full.json`
- Recovered `atharv-rwx-pod-2` to a ready state after the restart instead of launching duplicate work:
  - rebuilt conda env `cbm`
  - installed repo requirements with `numpy==1.26.4` and `opencv-python==4.11.0.86`
  - verified imports for `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, and `pytorchcv`
  - verified `/workspace/SAVLGCBM`, annotations, and `datasets/CUB_200_2011`
  - preloaded `resnet18_cub` successfully, which cached weights under `/root/.torch/models/`
- No relaunch was taken after recovery because there is still no additional safe eligible job:
  - duplicating the two active point evals would violate `TASK.md`
  - the next dense-first model-change sprint is still blocked on the missing point hit / coverage metrics

### Localization eval recovery status at 12:14 PDT

- Rechecked both live pods directly after recovery:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Required static checks still pass:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
- One of the two Grad-CAM point jobs is now complete:
  - `SALF` output landed at `/workspace/SAVLGCBM/results/gradcam_point_salf_best_full.json`
  - point hit rate `0.9427`
  - point coverage `0.9513`
  - matched part hits / total `62328 / 66114`
- The remaining gate item is still running healthily:
  - `SAVLG` Grad-CAM point eval on `atharv-rwx-pod` GPU `0`
  - latest visible progress from the log is about `483/724`
  - output `/workspace/SAVLGCBM/results/gradcam_point_savlg_best_full.json` is still missing
- No new launch was taken even though other GPUs were idle:
  - duplicating the active `SAVLG` point job would violate the non-duplication rule
  - starting the next dense sprint before the point comparison lands would violate the current `TASK.md` decision order

### Localization eval recovery status at 12:16 PDT

- Executed one additional `TASK.md` iteration and rechecked the persisted local loop state first:
  - `savlg-research/` sprint directories
  - `savlg-research/runtime_state.json`
  - newest `savlg-research/loop-logs/`
- Required static checks still pass:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
- This sandbox could not safely verify live pod state in this iteration:
  - `kubectl get pods -o wide` failed with auth-provider DNS/OIDC errors for `authentik.nrp-nautilus.io`
  - `/workspace/SAVLGCBM` is not mounted here
  - `ps` is blocked and `nvidia-smi` is unavailable
- Because the latest verified pod-visible state still leaves the gate blocked only on `/workspace/SAVLGCBM/results/gradcam_point_savlg_best_full.json`, no relaunch or next-sprint launch was taken without fresh verification.

### Localization gate complete and next sprint launched

- Rechecked live pods directly once the missing artifact landed:
  - `/workspace/SAVLGCBM/results/gradcam_point_savlg_best_full.json`
  - `/workspace/SAVLGCBM/results/gradcam_point_salf_best_full.json`
- Final Grad-CAM point-localization comparison is now complete:
  - `SAVLG`: hit rate `0.7374`, coverage `0.0114`, matched part hits / total `584 / 792`
  - `SALF`: hit rate `0.9427`, coverage `0.9513`, matched part hits / total `62328 / 66114`
- Combined interpretation across completed localization artifacts:
  - `SAVLG` is clearly ahead of `SALF` on Grad-CAM bbox mAP
  - `SAVLG` is not ahead on native-map localization
  - `SAVLG` fails badly on the paper-style point metric, so the current clean-global model still lacks usable localized concept evidence
- Both pods were idle after the point eval completed, so the next single dense-first sprint became eligible.
- Chosen next change from `TASK.md`:
  - soft containment loss on top of the clean global-concept `SAVLG` path
- Implemented that change locally:
  - added `savlg_local_loss_mode="containment"` to `SAVLG`
  - created `configs/unified/cub_savlg_cbm_clean_global_softcontain_v1.json`
- Static checks passed locally and on `atharv-rwx-pod`.
- Launched the new dense run on `atharv-rwx-pod` GPU `0`:
  - config: `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_softcontain_v1.json`
  - log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_softcontain_v1.log`
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13`
  - pid: `2999`

### Soft containment status at 12:34 PDT

- Re-read `TASK.md`, the active soft-containment sprint record, and the current loop state before taking any new action.
- Required static checks still pass:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
- Verified live pod state directly:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dense soft-containment job is still the only active eligible job for the current sprint:
  - pod: `atharv-rwx-pod`
  - GPU `0`: active (`31%`, `1209 MiB / 81920 MiB`)
  - pid chain still healthy:
    - launcher `2998`
    - main `2999`
  - command:
    - `/opt/conda/envs/cbm/bin/python -u train_cbm.py --config /workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_softcontain_v1.json`
  - run dir still has no final dense or NEC artifacts:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13/test_metrics.json` is absent
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13/metrics.csv` is absent
  - latest `train.log` tail shows healthy progress around `3313/4000` in the current solver stage
- `atharv-rwx-pod-2` is recovered and ready but still correctly idle for this sprint:
  - both visible GPUs are idle
  - no active `train_cbm.py`, `sparse_evaluation.py`, or localization-eval processes
  - `cbm` env and required data roots are present
- No new launch was taken:
  - NEC is still blocked by the dense gate because no `test_metrics.json` has landed yet
  - duplicating the dense run on the recovered second pod would violate the one-idea-at-a-time and non-duplication rules

### Soft containment status at 12:54 PDT

- Re-read `TASK.md`, rechecked the active soft-containment sprint record, and reran the required static checks:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The three in-flight follow-up jobs for the soft-containment checkpoint all remain healthy:
  - native-map localization on `atharv-rwx-pod`
    - latest visible progress: `101/181`
    - output still absent:
      - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
  - Grad-CAM localization on `atharv-rwx-pod`
    - latest visible progress: `390/724`
    - output still absent:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
  - tuned NEC on `atharv-rwx-pod-2`
    - latest visible sparse-path checkpoint:
      - `(74) lambda 0.0001 ... [test acc 0.7444], sparsity 2067/134200`
    - `metrics.csv` still absent for:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13`
- `atharv-rwx-pod-2` GPU `1` remains intentionally idle:
  - `scripts/evaluate_native_spatial_maps.py` already emits both bbox and paper-style point metrics
  - there is no additional non-duplicative eligible launch for this one-idea sprint
- No new launch, relaunch, or kill was taken in this iteration because all eligible follow-up work is already running healthily and no result finalized during this check.

### Dual-branch status hold at 13:31 PDT

- Re-read `TASK.md`, re-inspected the current sprint directories plus `savlg-research/runtime_state.json`, and checked the newest loop logs before deciding whether anything safe remained to do.
- The active unresolved sprint is still the dual-branch follow-up launched at `13:22 PDT`:
  - dense run already completed at `0.744559585492228`
  - NEC plus native-map and Grad-CAM localization evals were already launched in prior iterations
- This local sandbox does not currently have the pod-side artifact view needed to advance that sprint safely:
  - `saved_models/` is absent here
  - `results/` is absent here
  - local `logs/` only contains older `2026-04-02` files
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
  - `/workspace/SAVLGCBM` is not mounted
- Because of that visibility gap, no new action was taken:
  - no duplicate launch
  - no speculative relaunch
  - no kill
  - no `DONE.txt`
- `docs/cub_results.md` was not updated in this iteration because no new finalized NEC or localization artifacts are present in this filesystem.

### Dual-branch live pod check at 13:26 PDT

- Re-read `TASK.md`, rechecked the active sprint README, and then verified the actual pod state directly with `kubectl` before deciding whether any action was still eligible.
- Both pods remain healthy:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dual-branch dense checkpoint is still the same completed run:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12`
  - dense `test_metrics.json` present
  - sparse `metrics.csv` still absent
- All currently eligible follow-up jobs are still active and writing fresh logs:
  - native-map localization on `atharv-rwx-pod` GPU `0`
    - main pid `8270`
    - latest visible progress `26/181`
    - output `/workspace/SAVLGCBM/results/native_savlg_dualbranch_softcontain_v1_full_meanthr_bs32_nw8_nocache.json` still absent
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`
    - main pid `8271`
    - latest visible progress `100/724`
    - output `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_softcontain_v1_full.json` still absent
  - NEC on `atharv-rwx-pod-2` GPU `0`
    - main pid `2634`
    - latest visible sparse-solver progress about `1837/4000`
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12/metrics.csv` still absent
- `atharv-rwx-pod-2` GPU `1` remains idle, but there is still no additional non-duplicative eligible job for this single-idea sprint.
- No new launch, relaunch, or kill was taken in this iteration because NEC plus both localization evals are already running healthily and no finalized dual-branch result landed during this check.

### Dual-branch sparse recovery at 13:30 PDT

- Re-read `TASK.md`, rechecked the active sprint state, and verified both pods live with `kubectl` before touching the sparse path.
- Reran `py_compile` for `sparse_evaluation.py` locally and on `atharv-rwx-pod-2`; both passed.
- The dual-branch dense checkpoint is still the same completed run:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12`
  - dense `test accuracy = 0.744559585492228`
- The original tuned NEC follow-up for that run finished this iteration, but it collapsed:
  - run dir `metrics.csv` now exists with only:
    - `NEC = 655.445014834404`
    - `Accuracy = 0.01295336801558733`
  - terminal NEC log summary:
    - repeated `lambda = 0.010101`, sparsity `0.9768`
    - target-NEC accuracies only `0.0055` to `0.0062`
    - `Average acc = 0.0056`

## 2026-04-06

### Soft-align ablation prep at 20:45 PDT

- Re-read `TASK.md`, inspected the existing sprint directories under `savlg-research/`, the persisted loop state, and the latest loop summaries so this iteration would not duplicate the stale consistency sprint hold.
- Confirmed the repo did not yet contain the new `soft_align` or outside-penalty local loss work required by the current task.
- Implemented the new loss path locally:
  - added `soft_align` to `methods/savlg.py` as a normalized `KL(Q_box || P_map)` spatial loss on the soft-box targets
  - added a separate normalized outside-mass penalty term controlled by `savlg_outside_penalty_w`
  - wired both through SAVLG train/val loss computation and saved run metadata
  - exposed `--savlg_outside_penalty_w` in `train_cbm.py`
- Added the six approved warm-start frozen-global multiscale configs for the current loss study:
  - `local_mil` only
  - `containment` only
  - `soft_align` only
  - `soft_align + local_mil`
  - `soft_align + outside penalty`
  - `soft_align + local_mil + outside penalty`
- Added `scripts/run_savlg_vlgwarm_loss_ablation_queue.sh` so a later launch-capable iteration can fill both GPUs with distinct dense-first ablations from this exact approved list.
- Validation from this checkout:
  - `python3 -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
    - passed
  - JSON parse check for the new ablation configs
    - passed
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
    - passed
  - direct local runtime import smoke test for `methods.savlg`
    - blocked because this shell is missing `loguru`
- No pod launch was taken in this iteration because live cluster visibility is still blocked here:
  - `kubectl get pods -o wide` fails with an OIDC discovery / DNS error for `authentik.nrp-nautilus.io`
  - `ps` is blocked
  - `pgrep` cannot read the process list
  - `nvidia-smi` is unavailable
- Because active pod jobs and actual GPU idleness cannot be re-verified safely from this environment, launching dense ablations would risk duplicating healthy work and violating `TASK.md`.
- Recorded the active sprint under `savlg-research/2026-04-06--20-45-softalign-outside-ablation/README.md`.
- Left `docs/cub_results.md` unchanged because there are no new dense / NEC / localization results yet.
- Per `TASK.md`, that sparse collapse is not treated as meaningful; the recovery move is to increase `max_glm_steps`.
- Took exactly one new action:
  - cloned the checkpoint to preserve the collapsed artifacts:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s400`
  - launched one higher-step NEC rerun on `atharv-rwx-pod-2` GPU `0`:
    - `CUDA_VISIBLE_DEVICES=0 nohup /opt/conda/envs/cbm/bin/python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s400 --lam 0.01 --max_glm_steps 400 > /workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec400.log 2>&1 < /dev/null &`
  - launcher/main pid pair at startup check:
    - `2900` / `2902`
  - startup evidence:
    - pod2 GPU `0` shows the new NEC worker resident
    - log already advanced to about `44/169`
- Existing localization evals on `atharv-rwx-pod` were left untouched and remain healthy:
  - native-map progress `41/181`
  - Grad-CAM progress `158/724`
- No `DONE.txt` was added and `docs/cub_results.md` was not updated because the replacement NEC rerun plus both localization outputs are still in flight.

### Dual-branch live status at 13:33 PDT

- Re-read `TASK.md`, rechecked the dual-branch sprint record, and verified both pods again with `kubectl` before deciding whether anything new was still safe.
- Pod health remains unchanged:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The two required localization evals are still the active A100 follow-up jobs:
  - native-map eval output still absent at `/workspace/SAVLGCBM/results/native_savlg_dualbranch_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
  - native-map progress now `74/181`
  - Grad-CAM eval output still absent at `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_softcontain_v1_full.json`
  - Grad-CAM progress now `286/724`
- The `nec400` sparse recovery rerun is still actively running on `atharv-rwx-pod-2` GPU `0`:
  - main pid `2902`
  - latest visible solver progress `1727/4000`
  - sampled GPU state:
    - GPU `0`: `761 MiB`, `63%`
    - GPU `1`: `1 MiB`, `0%`
- Important filesystem caveat for the next loop:
  - the cloned path `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s400` already contains copied `metrics.csv` and `metrics.txt` from the collapsed source run
  - those files are not valid `nec400` completion evidence yet; the live process and `logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec400.log` remain the source of truth until the rerun exits or rewrites them
- No new action was taken:
  - no pod restart evidence was present
  - no additional distinct non-duplicative job exists for pod2 GPU `1`
  - no `DONE.txt`
  - no `docs/cub_results.md` update because no finalized meaningful sparse or localization result has landed

### Multiscale local-state hold at 15:34 PDT

- Re-read `TASK.md`, re-inspected the sprint directories, `savlg-research/runtime_state.json`, and the newest persisted loop logs before deciding whether any follow-up was eligible.
- Local static checks still pass:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py`
- This sandbox still cannot safely verify live pod state directly:
  - `/workspace/SAVLGCBM` is not mounted here
  - `ps` is blocked
  - `nvidia-smi` is unavailable
  - local pod-side `saved_models/` and `results/` artifacts are absent
- The newest persisted pod-visible evidence remains the multiscale hold recorded at `15:33 PDT`:
  - active run: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
  - `test_metrics.json` still absent at that check
  - `metrics.csv` still absent at that check
  - `atharv-rwx-pod-2` healthy but idle
- No new launch, relaunch, recovery action, or kill was taken in this iteration:
  - NEC is still blocked by the dense-first gate
  - localization for the multiscale checkpoint is still blocked on dense completion
  - no `DONE.txt` was added because the sprint is still in progress

### Multi-scale spatial branch launched

- Re-read `TASK.md`, re-inspected the finished `mlp(h=1)` sprint, and verified that both research pods were healthy and idle before taking the next dense-first action.
- Chosen next single change from the ranked `TASK.md` list:
  - add a multi-scale spatial branch that keeps the global concept path on `conv5` and fuses `conv4 + conv5` only for localization
- Implemented that change locally:
  - `methods/salf.py`: expose `conv4` and `conv5` stage features from `resnet18_cub`
  - `methods/savlg.py`: add the fused multi-scale spatial branch and SAVLG reload helpers
  - `evaluations/sparse_utils.py`: reload multi-scale SAVLG checkpoints correctly for NEC
  - `scripts/evaluate_native_spatial_maps.py`: reload multi-scale SAVLG checkpoints correctly for native-map and Grad-CAM eval
  - `train_cbm.py`: add `--savlg_spatial_branch_mode`
  - new config: `/Users/atharvramesh/Projects/CBM/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1.json`
- Static checks passed:
  - local: `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py`
  - pod: `/opt/conda/envs/cbm/bin/python -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py`
- Copied the changed files and new config into `/workspace/SAVLGCBM/` on `atharv-rwx-pod`.
- Launched the new dense run on `atharv-rwx-pod` GPU `0`:
  - config:
    - `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1.json`
  - log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1.log`
  - run dir:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
  - launcher/main pid pair:
    - `14216` / `14221`
- Startup evidence:
  - saved args confirm:
    - `implementation_status=dual_branch_multiscale_conv45_localmil_lse_v1`
    - `savlg_spatial_branch_mode=multiscale_conv45`
    - `savlg_spatial_stage=conv5`
    - `mask_h=14`
    - `mask_w=14`
    - `savlg_use_local_mil=true`
    - `savlg_local_pooling=lse`
  - log confirms:
    - `Saving SAVLG-CBM model to saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
    - the new `14x14` train-supervision cache build is actively progressing
- No NEC or localization follow-up was launched in this iteration because the dense gate for this new checkpoint is not known yet.

### Dual-branch NEC recovery at 13:38 PDT

- Re-read `TASK.md`, rechecked the active dual-branch sprint record, and verified both pods live with `kubectl` before taking any new action.
- Reran required sparse static checks:
  - local: `python3 -m py_compile sparse_evaluation.py`
  - `atharv-rwx-pod-2`: `/opt/conda/envs/cbm/bin/python -m py_compile /workspace/SAVLGCBM/sparse_evaluation.py`
  - result: passed in both places
- Pod health remains unchanged:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The two required localization evals remain the active A100 follow-up jobs and were left untouched:
  - native-map eval output still absent at `/workspace/SAVLGCBM/results/native_savlg_dualbranch_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
  - native-map progress now `92/181`
  - Grad-CAM eval output still absent at `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_softcontain_v1_full.json`
  - Grad-CAM progress now `412/724`
- The prior sparse recovery rerun is now complete and still unusable:
  - cloned path: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s400`
  - rewritten `metrics.csv` now reports:
    - `NEC = 645.5450026392937`
    - `Accuracy = 0.0321243517100811`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec400.log` ends with:
    - `did not converge at 4000 iterations`
    - target `ACC@5/10/15/20/25/30` all still `0.0050`
    - `Average acc = 0.0050`
- Per `TASK.md`, that still counts as sparse collapse, so the eligible next recovery action was to increase `max_glm_steps` again.
- Took exactly one new action:
  - created a fresh recovery clone at `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s800`
  - removed copied sparse-path files from that clone before launch:
    - `metrics.csv`
    - `W_g@NEC=*.pt`
    - `b_g@NEC=*.pt`
  - launched one `nec800` rerun on `atharv-rwx-pod-2` GPU `0`:
    - `CUDA_VISIBLE_DEVICES=0 nohup /opt/conda/envs/cbm/bin/python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s800 --lam 0.01 --max_glm_steps 800 > /workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec800.log 2>&1 < /dev/null &`
  - launcher/main pid pair at startup check:
    - `3157` / `3168`
  - startup evidence:
    - pod2 process tree shows the new sparse worker on the cloned load path
    - GPU `0` now has the worker resident (`233 MiB`)
    - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec800.log` has started and reached `0/169`
- No additional action was taken:
  - both localization evals are already the required A100 jobs
  - `nec800` is the only new distinct non-duplicative sparse recovery action
  - pod2 GPU `1` still has no separate eligible job for this sprint
- No `DONE.txt` and no `docs/cub_results.md` update yet because the localization outputs plus `nec800` remain in flight.

## 2026-04-04

### SAVLG stop summary

- The external autoresearch loop was stopped after broad follow-up coverage.
- The only model-change direction that clearly paid off was teacher distillation.
- Best sparse `SAVLG-CBM` result remains the earlier distillation checkpoint:
  - dense `0.7444`
  - `ACC@5 = 0.7069`
  - `AVGACC = 0.7193`
- Later MIL distillation ablations improved dense accuracy to about `0.7484`, but tuned NEC underperformed that earlier sparse-best run:
  - `w010`: dense `0.7483592400690846`, `AVGACC = 0.6866`
  - `w025`: dense `0.7481865284974093`, `AVGACC = 0.6862`
- Main conclusion:
  - dense training is no longer the limiting factor
  - sparse evaluation / NEC path selection is the main bottleneck
- Important interpretation note:
  - `metrics.csv` records the full GLM path
  - reported `ACC@5` / `AVGACC` come from the targeted NEC selection procedure, not from reading the first row of `metrics.csv`

### SAVLG external-loop status at 00:02 UTC

- Rechecked current policy and sprint state before acting:
  - `TASK.md`
  - all `savlg-research/` sprint directories
  - `savlg-research/runtime_state.json`
- Required static checks pass:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
- Verified live pod health:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Recovered dense-control completion for NEC-path sprint:
  - run: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_23_47_35`
  - dense test accuracy: `0.7483592400690846`
  - gate check: passes `>0.73`, so NEC launch is eligible
- Respected active healthy jobs already running on `atharv-rwx-pod-2`:
  - NEC `w010` on GPU1:
    - `python sparse_evaluation.py --load_path saved_models/cub/savlg_cbm_cub_2026_04_03_20_56_45 --lam 0.01 --max_glm_steps 40`
  - NEC `w035` on GPU0:
    - `python sparse_evaluation.py --load_path saved_models/cub/savlg_cbm_cub_2026_04_03_23_06_30 --lam 0.01 --max_glm_steps 40`
- Filled newly free GPU capacity:
  - `atharv-rwx-pod` GPU0 was idle, so launched tuned NEC for dense-control run:
    - `CUDA_VISIBLE_DEVICES=0 python sparse_evaluation.py --load_path saved_models/cub/savlg_cbm_cub_2026_04_03_23_47_35 --lam 0.01 --max_glm_steps 40`
  - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_necpath_dense_control_nec_tuned.log`
  - main PID: `3760`
- No job kills were performed, no pod restart/env rebuild was needed, and sprint remains in progress pending final NEC metrics.

## 2026-04-03

### SAVLG external-loop status at 23:29 UTC

- Verified live pod health and left active healthy jobs untouched:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Verified both GPUs on `atharv-rwx-pod-2` are occupied by eligible jobs:
  - `GPU 0`: dense `w035` run (`python train_cbm.py --config configs/unified/cub_savlg_cbm_mil_distill_w035.json`)
  - `GPU 1`: tuned NEC for `w010` (`--load_path saved_models/cub/savlg_cbm_cub_2026_04_03_20_56_45 --lam 0.01 --max_glm_steps 40`)
- Verified `atharv-rwx-pod` A100 is occupied by tuned NEC for `w025`:
  - `python sparse_evaluation.py --load_path saved_models/cub/savlg_cbm_cub_2026_04_03_22_57_19 --lam 0.01 --max_glm_steps 40`
- Recovered newly completed dense metrics for `w025` from pod artifacts:
  - run: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_22_57_19`
  - train accuracy: `1.0`
  - val accuracy: `1.0`
  - test accuracy: `0.7481865284974093`
  - gate check: passes `>0.73`, so NEC promotion is valid
- Sample tuned-NEC progress checkpoints:
  - `w010`: `(10) lambda 0.0050 ... [test acc 0.2668]`
  - `w025`: `(10) lambda 0.0050 ... [test acc 0.2661]`
- Required static checks pass:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
- Iteration action decision:
  - no new launch and no kill were taken because all eligible slots were already occupied by healthy jobs
  - no pod restart/env rebuild flow was needed

### SAVLG external-loop status at 22:49 UTC

- Restored direct pod visibility and verified health:
  - `atharv-rwx-pod`: `Running`, restarts `0`, `savlgcbm` env present
  - `atharv-rwx-pod-2`: `Running`, restarts `0`, `savlgcbm` env present
- Confirmed active healthy jobs on `atharv-rwx-pod-2` and left them untouched:
  - `GPU 0`: dense `w025` distillation run
  - `GPU 1`: tuned NEC for `w010` distillation run
- Recovered selective-local-weighting dense completion from pod artifacts:
  - run: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_20_24_18`
  - dense test accuracy: `0.7369602763385147` (log-rounded `0.7370`)
  - decision: close sprint as a dense regression; NEC not promoted
- Updated sprint logs accordingly:
  - closed `savlg-research/2026-04-03--20-19-selective-local-weighting/` with `DONE.txt`
  - updated `savlg-research/2026-04-03--20-50-mil-distill-ablation/README.md` with current active dense+NEC status
- Required static checks remain passing:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`

### SAVLG external-loop status at 21:11 UTC

- Re-inspected local persistent state before taking any new action:
  - `TASK.md`
  - all `savlg-research/` sprint directories
  - `savlg-research/runtime_state.json`
  - newest `savlg-research/loop-logs/`
- Confirmed the newest visible loop log is now `iter-0007-20260403T211127Z.log`, but it still does not provide any newer pod-visible completion state for the two in-progress SAVLG sprints:
  - selective localization weighting on `atharv-rwx-pod-2`
  - MIL distillation weight ablation on `atharv-rwx-pod`
- Confirmed local visibility is still insufficient to distinguish a finished job from a healthy active pod job:
  - `ps` remains blocked with `operation not permitted`
  - `nvidia-smi` remains unavailable
  - local `saved_models/` is still absent
  - local `logs/` still contains only older non-`SAVLG` files
- Reran the required static checks locally:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
  - result: passed
- No new launch, relaunch, NEC promotion, or sprint closeout was taken in this iteration because doing so from the current sandbox would risk duplicating the last documented healthy pod jobs and would violate `TASK.md`.

### SAVLG external-loop status at 21:05 UTC

- Re-inspected local persistent state before taking any new action:
  - `TASK.md`
  - all `savlg-research/` sprint directories
  - `savlg-research/runtime_state.json`
  - newest `savlg-research/loop-logs/`
- Confirmed local `saved_models/` is still absent in this sandbox, so there are still no newly visible dense or NEC artifacts for the two in-progress SAVLG sprints:
  - selective localization weighting on `atharv-rwx-pod-2`
  - MIL distillation weight ablation on `atharv-rwx-pod`
- Confirmed recent automatic loop attempts `iter-0029` through `iter-0034` did not advance research state:
  - each exited with a usage-limit error before any pod inspection, relaunch, or documentation work
- Reran the required static checks locally:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
  - result: passed
- No new launch or relaunch was taken in this iteration because local sandbox visibility is still insufficient to distinguish a finished job from a healthy active pod job, and duplicating either in-progress sprint would violate the scheduling and safety rules in `TASK.md`.

### SAVLG external-loop status at 20:24 UTC

- Rechecked pod `atharv-rwx-pod-2` directly after the earlier in-flight SAVLG jobs finished:
  - status: `Running`
  - restarts: `0`
  - conda env: `savlgcbm` exists
  - active `train_cbm.py` / `sparse_evaluation.py` jobs were clear before launching the next sprint
- Closed the finished distillation sprint with final dense + tuned NEC metrics:
  - dense `0.7444`
  - `ACC@5 = 0.7069`
  - `AVGACC = 0.7193`
  - sparse performance improved slightly over the current `SAVLG` baseline, but dense accuracy still stayed below `0.7459`
- Closed the finished OICR self-refinement sprint as a negative dense result:
  - dense `0.7380`
  - NEC skipped because the dense run regressed materially against the current `SAVLG` baseline and the stronger soft-box / distillation follow-ups
- Started the next prioritized dense sprint for selective localization weighting:
  - pod: `atharv-rwx-pod-2`
  - GPU: `0`
  - log: `/workspace/SAVLGCBM/logs/cub_savlg_mil_softbox_selective_local_weighting.log`
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_20_24_18`
  - config: `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_mil_softbox_selective_local_weighting.json`
- Current visible launch health for the new sprint:
  - cached soft-box supervision reloaded successfully
  - `GPU 0` sample after launch: `1993 MiB`, `30%` util
  - concept-head training has advanced through epoch `2`
  - best visible validation loss so far: `0.565416`

### SAVLG external-loop status at 20:07 UTC

- Recovered direct pod visibility in the external research loop and confirmed the pod is healthy:
  - pod: `atharv-rwx-pod-2`
  - status: `Running`
  - restarts: `0`
  - conda env: `savlgcbm` exists
- Confirmed the scheduler is already in the correct dense-plus-NEC state from `TASK.md`:
  - `GPU 0`: active OICR dense run `savlg_cbm_cub_2026_04_03_19_59_53`
  - `GPU 1`: active tuned NEC sweep for the dense distillation run `savlg_cbm_cub_2026_04_03_18_58_51`
- Recorded the completed WILDCAT auxiliary local top-k dense result:
  - train `1.0000`
  - val `1.0000`
  - test `0.7380`
  - NEC intentionally skipped because the dense result regressed against both the `SAVLG-CBM` baseline and the soft-box follow-up
- Recorded the dense distillation checkpoint outcome before sparse completion:
  - train `1.0000`
  - val `1.0000`
  - test `0.7443868739205527`
  - latest visible NEC checkpoint is `0.7162`, still slightly below the current sparse baseline `0.7170`
- Recorded the latest OICR dense progress:
  - best concept-head validation loss so far: `0.601160`
  - final sparse linear solver is active around step `879/4000`
- No new launch was taken in this loop iteration because both GPUs already had healthy higher-priority jobs and launching another job would have violated the pod scheduling policy in `TASK.md`.

## 2026-03-31

### Initial inspection

- Confirmed target repo started nearly empty.
- Confirmed local references exist:
  - `/Users/atharvramesh/Projects/CBM/VLG-CBM`
  - `/Users/atharvramesh/Projects/CBM/Medical_CBM`
  - `/Users/atharvramesh/Projects/CBM/Medical_CBM_dev_clean`

### VLG-CBM findings

- Upstream structure is compact and suitable as the base.
- Key reusable components identified:
  - `train_cbm.py`
  - `data/utils.py`
  - `data/concept_dataset.py`
  - `model/cbm.py`
  - `evaluations/sparse_utils.py`
  - `sparse_evaluation.py`
  - `configs/*.json`
  - local `glm_saga/`

### Medical_CBM findings

- LF-CBM and SALF-CBM implementations are present in the active checkout.
- NEC evaluator is richer than upstream VLG and already supports:
  - upstream-style GLM-SAGA sweep
  - post-hoc NEC
  - concept caching
  - SALF delegation
- The active `Medical_CBM` checkout does not contain:
  - `savlg_cbm.py`
  - `vlg_cbm.py`
  - `utils/vlg_core.py`
  - `utils/vlg_annotations.py`

### Medical_CBM dev-clean findings

- `Medical_CBM_dev_clean` contains:
  - `savlg_cbm.py`
  - `utils/vlg_core.py`
  - `utils/vlg_annotations.py`
- This snapshot will be used as the SAVLG reference where the active repo is incomplete.

### Architecture decision

- Base the new codebase on VLG-CBM structure and evaluation behavior.
- Add the smallest possible shared layer:
  - config normalization
  - model registry
  - artifact metadata
  - per-model pipeline modules
- Port order:
  1. VLG-CBM
  2. LF-CBM
  3. SALF-CBM
  4. SAVLG-CBM

### Documentation created

- `docs/reference_audit.md`
- `docs/architecture_plan.md`
- `docs/progress_notes.md`
- `docs/final_technical_overview.md`
- `docs/model_notes/`
- `docs/experiment_guide.md`

### Immediate next steps

1. Bootstrap this repo by copying the VLG-CBM code structure.
2. Preserve local `glm_saga`.
3. Add unified registry/config plumbing.
4. Rewire VLG training through the new registry without changing behavior.
5. Port LF-CBM next.

### First implementation pass completed

- Copied the upstream VLG-CBM code structure into this repo.
- Added a small shared method layer:
  - `methods/common.py`
  - `methods/registry.py`
  - `methods/lf.py`
- Updated `train_cbm.py` to dispatch by `model_name`.
- Added `artifacts.json` metadata writing for saved runs.
- Added `eval_cbm.py` as a unified checkpoint evaluator for current runnable models.
- Updated `sparse_evaluation.py` to route by saved `model_name`.
- Added unified config examples under `configs/unified/`.

### Verification boundary

- Compilation check completed successfully with:
  - `python -m compileall /Users/atharvramesh/Projects/CBM/SAVLGCBM`
- Runtime validation was intentionally skipped per user instruction.

### 2026-04-01

### CIFAR10 LF sparse-eval fix

- While running NEC / ANEC evaluation for the finished `lf_cbm` CIFAR10 checkpoint on the pod, sparse evaluation failed with a CLIP feature-shape mismatch:
  - checkpoint projection expected `2048`-dim CLIP-RN50 penultimate features
  - loader rebuilt the model with `1024`-dim final CLIP-RN50 features
- Fixed `model/cbm.py` so checkpoint loading respects `args.use_clip_penultimate` for CLIP backbones in both:
  - `load_cbm`
  - `load_std`
- This preserves VLG behavior for non-CLIP backbones and makes LF sparse evaluation consistent with LF training-time backbone reconstruction.

### CIFAR10 VLG throughput adjustment

- The first CIFAR10 `vlg_cbm` run used `cbl_batch_size=32` and `num_workers=6`, which produced `1407` CBL steps per epoch and underutilized the A100 during frozen-backbone CBL training.
- Updated `configs/unified/cifar10_vlg_cbm.json` to:
  - `cbl_batch_size=128`
  - `num_workers=12`
- Plan for pod execution:
  - preserve the completed LF-CBM NEC outputs
  - restart the in-flight CIFAR10 VLG-CBM run from scratch with the larger batch configuration

### LF-CBM CIFAR10 protocol alignment

- Audited the active LF-CBM port against the original `Label-free-CBM` CIFAR10 training script.
- Found that the previous unified CIFAR10 LF config was not faithful to the original protocol:
  - it used `clip_RN50` instead of `ViT-B/16` for CLIP pseudo-labeling
  - it used CLIP penultimate features instead of the original final `clip_RN50` image embedding
  - it used an internal train/val split instead of the original official train/test split behavior
  - it used a different CIFAR10 concept set file
  - it used different LF cutoffs and iteration counts
- Patched `methods/lf.py` to support an explicit `lf_original_protocol` mode that:
  - uses the original official train/test split semantics
  - loads the CLIP pseudo-label model name separately via `lf_clip_name`
  - writes `metrics.txt` in the original LF repo layout for compatibility
- Updated `configs/unified/cifar10_lf_cbm.json` to match the original CIFAR10 LF protocol:
  - vendored original CIFAR10 concept list
  - `lf_clip_name=ViT-B/16`
  - `use_clip_penultimate=false`
  - `clip_cutoff=0.25`
  - `interpretability_cutoff=0.45`
  - `proj_steps=1000`
  - `proj_batch_size=50000`
  - `saga_batch_size=256`
  - `saga_n_iters=1000`

### CIFAR10 LF / VLG reproduction follow-up

- Fresh CIFAR10 `lf_cbm` training with the original LF protocol reached:
  - train accuracy `0.8621`
  - test accuracy `0.8550`
- Fresh NEC sweep for that checkpoint reached:
  - `ACC@5 = 0.8257`
  - `AVGACC = 0.8464`
- Added `configs/unified/cifar10_lf_cbm_proj2000.json` for an exploratory higher-budget LF projection run while preserving the original-protocol `1000`-step config.
- Audited the `vlg_cbm` path against the local upstream `VLG-CBM` repo:
  - `train_cbm.py` differs only by model-registry dispatch and extra artifact metadata on the `vlg_cbm` path
  - `sparse_evaluation.py` differs only by model-name routing
  - `model/cbm.py` differs by a CLIP penultimate/final loader fix that is neutral for standard VLG runs when `use_clip_penultimate` follows the saved args
- Reproduction plan on the pod:
  - run exploratory LF `proj_steps=2000` on one GPU
  - run upstream-style CIFAR10 VLG training on the other GPU using `configs/cifar10.json` with only throughput overrides for `cbl_batch_size` and `num_workers`

### VLG annotation cache

- Identified the main VLG throughput bottleneck in `data/concept_dataset.py`:
  - each `__getitem__` reparsed `annotations/<dataset>_<split>/<idx>.json`
  - this cost was paid during concept filtering, every CBL epoch, and every validation pass
- Patched `ConceptDataset` to persist cached annotation artifacts under `annotations/_cache`:
  - a filtered parsed-annotation cache keyed by dataset split and confidence threshold
  - a dense concept matrix cache keyed by dataset split, confidence threshold, and concept-bank hash
- Kept the external training API unchanged:
  - VLG still uses the same dataloader construction and model code
  - the cache only replaces repeated JSON parsing with one-time preprocessing plus cache loads
- Kept memory use tighter for standard VLG loaders:
  - loaders that only need concept targets can load the cached concept matrix directly
  - bounding-box annotations are only loaded into memory when the cropped-concept path needs them

### SALF-CBM CIFAR10 first pass

- Ported a first-pass `salf_cbm` trainer into `methods/salf.py`.
- The port follows the `Medical_CBM/spatial_aware_cbm.py` structure where it matters:
  - prompt-grid spatial CLIP target tensor generation
  - cached `P_train` / `P_val` tensors
  - spatial concept bottleneck training against those targets
  - pooled concept features followed by the same sparse final-layer path used elsewhere in this repo
- For the first general-domain port, the active SALF scope is intentionally narrower than the medical reference:
  - `clip_RN50` spatial backbone only
  - `prompt_grid` spatial target source only
  - CIFAR10 config aligned as closely as practical with the current LF-CBM setup
- Added a runnable CIFAR10 SALF config at `configs/unified/cifar10_salf_cbm.json` with:

- Ported a first-pass `savlg_cbm` trainer into `methods/savlg.py`.
- The current SAVLG port:
  - reuses the SALF spatial backbone and concept-layer stack
  - swaps prompt-grid pseudo-targets for annotation-driven concept presence and box masks
  - saves SALF-compatible spatial artifacts so the existing sparse evaluator can be reused
- Verified from live `CUB` annotations that the VLG JSON contains:
  - concept labels
  - concept logits
  - per-concept boxes
  - but `img_path` is `null`, so alignment must be index-based
  - the same CIFAR10 LF concept bank
  - `lf_original_protocol=true`
  - `lf_clip_name=ViT-B/16`
  - aggressive spatial extraction defaults for the 80GB A100 target pod

### CUB LF-CBM CLIP track

- Added `configs/unified/cub_lf_cbm_clip_rn50.json` for the paper-oriented CUB LF run.
- This keeps the stale `resnet18_cub` unified config untouched and records the CLIP-based assumptions explicitly:
  - shared VLG `CUB` concept bank: `concept_files/cub_filtered.txt`
  - `backbone=clip_RN50`
  - `lf_clip_name=ViT-B/16`
  - `lf_original_protocol=true`
  - original LF CUB regularization settings where applicable:
    - `clip_cutoff=0.26`
    - `proj_steps=5000`
    - `saga_lam=0.0002`
    - `saga_n_iters=5000`
- Throughput-only changes for the A100 pod:
  - `lf_batch_size=256`
  - `num_workers=12`

### CUB ImageFolder startup fix

- Identified a CUB-specific startup stall in the LF and SALF subset wrappers:
  - `TransformedSubset`, `DualTransformSubset`, and `RawSubset` were building `targets` via `base_dataset[idx][1]`
  - for `ImageFolder` datasets like `CUB`, that triggers image file opens during subset construction
  - on the PVC-backed pod storage, this caused the run to stall before the first `LF features` batch
- Patched the subset wrappers to read labels from dataset metadata when available:
  - prefer `dataset.targets`
  - fall back to `dataset.samples` / `dataset.imgs`
  - only fall back to `__getitem__` when metadata is unavailable

### CUB LF-CBM result capture

- Recorded the completed `CUB` `LF-CBM` run and NEC sweep in `docs/cub_results.md`.
- Saved result snapshot:
  - full accuracy:
    - train `0.8626`
    - test `0.6634`
  - NEC:
    - `ACC@5 = 0.3402`
    - `ACC@10 = 0.5161`
    - `ACC@15 = 0.5900`
    - `ACC@20 = 0.6290`
    - `ACC@25 = 0.6402`
    - `ACC@30 = 0.6516`
    - `AVGACC = 0.5612`

### CUB SALF preparation

- Audited the current SALF port before the first serious `CUB` run.
- Found a semantics mismatch in the prompt-grid path:
  - prompts were being drawn on raw PIL images
  - this makes `prompt_radius` depend on dataset-native resolution instead of CLIP input resolution
- Patched `methods/salf.py` so prompt-grid circles are drawn after resize and center crop into CLIP input space.
- Planned first `CUB` SALF config choices:
  - `backbone=clip_RN50`
  - `lf_clip_name=ViT-B/16`
  - shared `VLG-CBM` `CUB` concept bank
  - `grid_h=7`
  - `grid_w=7`
  - `prompt_radius=32`
- This change is behavior-preserving for CIFAR datasets and removes unnecessary image I/O during startup for `CUB` and similar `ImageFolder` datasets.

## 2026-04-03

### SAVLG external-loop status at 21:03 UTC

- Re-read `TASK.md`, current `savlg-research/` sprint directories, `savlg-research/runtime_state.json`, and the newest loop log before taking action.
- Reran the required local static checks:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
  - result: passed
- Local workspace visibility is still incomplete for pod recovery or rescheduling:
  - `logs/` only contains older non-`SAVLG` files
  - `saved_models/` is absent locally
  - `ps` is blocked in this sandbox
  - `nvidia-smi` is unavailable in this sandbox
- The latest persisted pod-visible scheduler state still indicates:
  - `atharv-rwx-pod-2` had the selective-local-weighting dense sprint in progress
  - `atharv-rwx-pod` had the `MIL` distillation ablation `w010` dense run in progress
- No new dense or NEC launch was taken in this iteration because launching from this sandbox would risk duplicating or colliding with those last documented healthy jobs and would violate `TASK.md`.

### CUB LF-CBM root-cause fix

- Rechecked the active LF port against the untouched `Label-free-CBM` repo on the same pod.
- Found the main mismatch in `methods/lf.py`:
  - the LF similarity function was not the upstream cubed-cosine objective
  - projection training and interpretability filtering were calling it on the wrong tensor orientation
- Patched the LF port to match upstream behavior exactly:
  - raw LF concept loading for `lf_original_protocol=true`
  - exact cubed-cosine similarity
  - corrected projection-loss and interpretability-filter call order
- Result:
  - faithful fixed-sim `LF-CBM` rerun reached `train=1.0000`, `val=0.7402`, `test=0.7402`
  - this matches the untouched original LF repo run on the same pod, which reached `val acc 0.7401`

### CUB LF-CBM tuned NEC follow-up

- Ran a shorter NEC sweep for the corrected LF checkpoint to avoid spending time in the all-zero lambda prefix.
- Sweep settings:
  - `--lam 0.01`
  - `--max_glm_steps 40`
- Recorded sparse results:
  - `ACC@5 = 0.5147`
  - `ACC@10 = 0.6808`
  - `ACC@15 = 0.7155`
  - `ACC@20 = 0.7261`
  - `ACC@25 = 0.7266`
  - `ACC@30 = 0.7266`
  - `AVGACC = 0.6817`

### CUB SALF-CBM fixed-sim follow-up

- Ported the same LF fixes into `methods/salf.py`:
  - exact upstream LF cubed-cosine similarity
  - raw LF concept loading for `lf_original_protocol=true`
- Identified a severe SALF prompt-grid bottleneck:
  - the old path rendered `49` prompted PIL images per sample and re-ran CLIP preprocessing on each one
- Replaced that path with a tensorized prompt-grid overlay implementation.
- Prompt-grid throughput improved from roughly `140-170 s/it` to roughly `17-22 s/it`.
- The corrected `CUB SALF-CBM` dense rerun finished at:
  - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/salf_cbm_cub_2026_04_03_05_30_10`
  - train accuracy `1.0000`
  - val accuracy `0.7320`
  - test accuracy `0.7320`
- Tuned NEC follow-up on the same checkpoint used:
  - `--lam 0.01`
  - `--max_glm_steps 40`
- Recorded sparse results:
  - `ACC@5 = 0.5335`
  - `ACC@10 = 0.6938`
  - `ACC@15 = 0.7159`
  - `ACC@20 = 0.7174`
  - `ACC@25 = 0.7174`
  - `ACC@30 = 0.7174`
  - `AVGACC = 0.6826`
- This moves corrected `SALF-CBM` into essentially the same sparse regime as corrected `LF-CBM` on `CUB`.

### CUB SAVLG loop follow-up

- Recovered pod visibility for the external `SAVLG` sprint loop on `atharv-rwx-pod-2`.
- Verified the unfinished `MIL + soft-box` dense run had already completed at:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_17_17_13`
- Recorded dense metrics for that run:
  - train accuracy `1.0000`
  - val accuracy `1.0000`
  - test accuracy `0.7442`
- This is slightly below the prior `MIL` rerun (`0.7447`), so soft-box supervision did not improve dense accuracy.
- Because the dense result still passed the `> 0.73` gate, a tuned NEC follow-up was launched on:
  - `GPU 1`
  - log: `/workspace/SAVLGCBM/logs/cub_savlg_mil_softbox_nec_tuned.log`
- Started the next prioritized dense sprint: `VLG` teacher distillation into the `MIL + soft-box` `SAVLG` variant.
- Added optional teacher loading and pooled-logit distillation hooks in `methods/savlg.py`, exposed the new knobs in `train_cbm.py`, and created:
  - `configs/unified/cub_savlg_cbm_mil_softbox_distill.json`
- The new dense job was launched on:
  - `GPU 0`
  - log: `/workspace/SAVLGCBM/logs/cub_savlg_mil_softbox_distill.log`
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_03_18_58_51`
- Both jobs were still running when this iteration ended.

### SAVLG external-loop status at 21:06 UTC

- Re-read `TASK.md`, inspected all current `savlg-research/` sprint directories, rechecked `savlg-research/runtime_state.json`, and reran the required static checks:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
  - result: passed
- Rechecked the newest local runner logs and confirmed `iter-0029` through `iter-0034` are usage-limit failures from the external runner, not new research work or pod-state updates.
- Local recovery and scheduling visibility is still insufficient:
  - `saved_models/` is absent in this sandbox
  - `ps` is blocked in this sandbox
  - `nvidia-smi` is unavailable in this sandbox
- Because the last persisted healthy pod-visible state still points to unfinished dense jobs on `atharv-rwx-pod-2` and `atharv-rwx-pod`, no dense or NEC relaunch was taken in this iteration.

### SAVLG external-loop status at 22:35 UTC

- Re-inspected local persistent state before taking action:
  - `TASK.md`
  - all `savlg-research/` sprint directories
  - `savlg-research/runtime_state.json`
  - newest `savlg-research/loop-logs/`
- Reran required static checks locally:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
  - result: passed
- Local visibility remains insufficient to safely schedule jobs without duplication:
  - `logs/` contains only older non-`SAVLG` logs
  - `saved_models/` is absent in this sandbox
  - `nvidia-smi` is unavailable
  - `ps` is blocked with `operation not permitted`
- No launch, relaunch, NEC promotion, or sprint closeout was taken in this iteration because doing so from this sandbox risks colliding with already-running healthy pod jobs and would violate `TASK.md`.

### SAVLG external-loop status at 22:38 UTC

- Executed exactly one new loop iteration and re-read `TASK.md` before taking action.
- Re-inspected sprint and runner state:
  - `savlg-research/runtime_state.json`
  - `savlg-research/loop-logs/` newest entries
  - active unfinished sprint READMEs in `savlg-research/`
- Reran required static checks locally:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
  - result: passed
- Local visibility is still insufficient for safe pod-side scheduling decisions:
  - `logs/` still contains only older non-`SAVLG` files
  - `saved_models/` and `saved_models/cub/` are absent in this sandbox
  - `ps` is blocked with `operation not permitted`
- `nvidia-smi` is unavailable
- No eligible safe launch/relaunch/closeout action was taken in this iteration; both active SAVLG sprints remain in-progress.

### SAVLG external-loop status at 22:40 UTC

- Executed exactly one new loop iteration and re-read `TASK.md` before taking action.
- Re-inspected local sprint/runtime state:
  - `savlg-research/runtime_state.json`
  - current unfinished sprint `README.md` files
  - newest `savlg-research/loop-logs/` entries
- Reran required static checks locally:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
  - result: passed
- Attempted direct pod visibility using `kubectl`:
  - `kubectl` is installed locally
  - `kubectl get pod atharv-rwx-pod{,-2}` failed due to auth-provider DNS/OIDC failure:
    - `lookup authentik.nrp-nautilus.io: no such host`
    - `Unable to connect to the server: getting credentials`
- Local artifacts still do not expose current pod-side SAVLG run completion:
  - `saved_models/` is absent in this sandbox
  - `logs/` still only contains older non-`SAVLG` files
- No eligible safe launch/relaunch/closeout action was taken in this iteration; active SAVLG sprints remain in-progress.

### SAVLG external-loop status at 22:42 UTC

- Executed one additional loop iteration under `TASK.md` policy and rechecked local persistent state:
  - `TASK.md`
  - `savlg-research/` sprint directories
  - `savlg-research/runtime_state.json`
- Reran required static checks locally:
  - `python -m py_compile methods/savlg.py`
  - `python -m py_compile train_cbm.py`
  - `python -m py_compile sparse_evaluation.py`
  - result: passed
- This sandbox still cannot safely verify pod-side process/GPU state:
  - `ps` is blocked with `operation not permitted`
  - `nvidia-smi` is unavailable
  - local `saved_models/` is absent
- No new launch, relaunch, NEC promotion, or sprint closeout was taken in this iteration because doing so would risk duplicating or colliding with the last documented healthy pod jobs.

### Soft-containment dense status at 12:32 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, and the newest loop log before taking action.
- Reran required static checks locally:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Direct pod visibility was available again through `kubectl`:
  - `atharv-rwx-pod`: `Running`, restarts `0`, age about `137m`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`, age about `42m`
- Verified the active soft-containment dense run is still healthy on `atharv-rwx-pod`:
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13`
  - active launcher/main pid pair:
    - `2998`
    - `2999`
  - `nvidia-smi` shows pod GPU `0` active and pod GPU `1` idle
  - latest log tail reached `SAVLG CBL epoch 21`
  - last completed validation summary at check:
    - `epoch=19 train_loss=0.221969 val_loss=0.290982 best_val=0.281373`
  - dense completion artifact is still absent:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13/test_metrics.json`
  - NEC artifact is still absent:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13/metrics.csv`
- Treated the newer `atharv-rwx-pod-2` as a restart/recovery case and re-verified the full launch prerequisites:
  - `/workspace/SAVLGCBM` present
  - conda env `cbm` present
  - imports pass for `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, `pytorchcv`
  - compatibility pins still match:
    - `numpy==1.26.4`
    - `opencv-python==4.11.0`
  - `resnet18_cub` weights cached
  - annotations and `datasets/CUB_200_2011` present
  - no active `train_cbm.py`, `sparse_evaluation.py`, or localization-eval jobs
- No new launch or relaunch was taken in this iteration:
  - the only eligible dense-first experiment is already running healthily
  - NEC remains blocked until dense test accuracy lands and clears the `0.74` gate
  - duplicating the dense run on an idle GPU would violate the one-idea-at-a-time and non-duplication rules

### Soft containment dense completion and follow-up launches at 12:36 PDT

- Rechecked `TASK.md`, the active sprint README, and live pod state before taking any new action.
- Required static checks still pass:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
- Re-verified the pod recovery checklist before relaunching follow-up work:
  - both pods are `Running`
  - both pods have the `cbm` env
  - both pods pass imports for `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, and `pytorchcv`
  - `atharv-rwx-pod-2` still has cached `resnet18_cub` weights
- The active soft-containment dense run is now complete:
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13`
  - dense test accuracy: `0.7450777202072539`
  - gate decision: passes the `0.74` NEC threshold, but does not beat the current clean-global `SAVLG` dense best `0.7492`
- Because the dense gate is satisfied and the cluster was idle, this iteration launched the next eligible jobs for the same checkpoint:
  - `atharv-rwx-pod-2` GPU `0`: tuned NEC
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_softcontain_v1_nec150.log`
    - command: `python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13 --lam 0.01 --max_glm_steps 150`
  - `atharv-rwx-pod` GPU `0`: native-map localization eval
    - log: `/workspace/SAVLGCBM/logs/native_savlg_softcontain_v1_full.log`
    - output: `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
  - `atharv-rwx-pod` GPU `1`: Grad-CAM localization eval
    - log: `/workspace/SAVLGCBM/logs/gradcam_savlg_softcontain_v1_full.log`
    - output: `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
- Startup health was verified from the live logs:
  - NEC progressed into the eval dataloader (`71/169` visible at check)
  - native-map eval progressed to `5/181`
  - Grad-CAM eval progressed to `19/724`
- No `DONE.txt` was added because the sprint is now in the NEC/localization phase and still in progress.

### Soft-containment follow-up status at 12:43 PDT

- Re-read `TASK.md`, re-inspected the active soft-containment sprint README plus the main docs, and reran the required static checks:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The two localization evals for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` are still healthy on `atharv-rwx-pod`:
  - native-map eval main pid `5268`
    - latest visible progress: `27/181`
  - Grad-CAM eval main pid `5269`
    - latest visible progress: `105/724`
  - neither output JSON has landed yet:
    - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
- The tuned NEC job remains healthy on `atharv-rwx-pod-2` GPU `0`:
  - main pid `2009`
  - first sparse-path checkpoint is now visible in the log:
    - `(0) lambda 0.0101 ... [test acc 0.7121], sparsity 403/134200`
  - `metrics.csv` is still absent for the run dir, so no sparse summary is finalized yet
- `atharv-rwx-pod-2` GPU `1` is idle, but there is no additional safe eligible launch:
  - the current checkpoint already has its single tuned NEC run plus both localization modes active
  - `scripts/evaluate_native_spatial_maps.py` already computes both bbox and paper-style point metrics, so there is no separate missing point-localization job to add
  - launching another NEC or localization job would duplicate active healthy work and violate the one-idea-at-a-time loop rule

### Soft-containment follow-up status at 12:46 PDT

- Re-read `TASK.md`, re-inspected the active sprint docs, and reran the required static checks:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The two localization evals for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` are still healthy on `atharv-rwx-pod`:
  - native-map eval main pid `5268`
    - latest visible progress: `48/181`
  - Grad-CAM eval main pid `5269`
    - latest visible progress: `186/724`
  - neither output JSON has landed yet:
    - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
- The tuned NEC job remains healthy on `atharv-rwx-pod-2` GPU `0`:
  - main pid `2009`
  - latest visible sparse-path checkpoint:
    - `(19) lambda 0.0027 ... [test acc 0.7301], sparsity 520/134200`
  - `metrics.csv` is still absent for the run dir, so no sparse summary is finalized yet
- `atharv-rwx-pod-2` GPU `1` is still idle, but there is no additional safe eligible launch:
  - the current checkpoint already has its single tuned NEC run plus both localization modes active
  - no dense, NEC, or localization artifact finalized during this iteration
  - launching another NEC or localization job would duplicate active healthy work and violate the one-idea-at-a-time loop rule

### Soft-containment follow-up status at 12:47 PDT

- Re-read `TASK.md`, re-inspected the active sprint README plus recent loop logs, and reran the required static checks:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The two localization evals for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` are still healthy on `atharv-rwx-pod`:
  - native-map eval main pid `5268`
    - latest visible progress: `60/181`
    - sampled GPU `0` utilization was low at that instant, but the process is still alive and the log advanced beyond the prior check
  - Grad-CAM eval main pid `5269`
    - latest visible progress: `233/724`
  - neither output JSON has landed yet:
    - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
- The tuned NEC job remains healthy on `atharv-rwx-pod-2` GPU `0`:
  - main pid `2009`
  - latest visible sparse-path checkpoint:
    - `(32) lambda 0.0011, loss 0.5892, acc 1.0000 [val acc 1.0000] [test acc 0.7371], sparsity 0.004567809239940388 [613/134200], time 9.241639614105225, lr 0.1000`
  - `metrics.csv` is still absent for the run dir, so no sparse summary is finalized yet
- `atharv-rwx-pod-2` GPU `1` is still idle, but there is no additional safe eligible launch:
  - the current checkpoint already has its single tuned NEC run plus both localization modes active
  - no dense, NEC, or localization artifact finalized during this iteration
  - launching another NEC or localization job would duplicate active healthy work and violate the one-idea-at-a-time loop rule

### Soft-containment follow-up status at 12:51 PDT

- Re-read `TASK.md`, re-inspected the active sprint docs, and reran the required static checks:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Direct pod visibility is still healthy:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The two localization evals for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` remain healthy on `atharv-rwx-pod`:
  - native-map eval launcher/main pid pair `5251` / `5268`
    - latest visible progress: `82/181`
  - Grad-CAM eval launcher/main pid pair `5263` / `5269`
    - latest visible progress: `316/724`
  - both output JSON files are still absent:
    - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
  - instantaneous GPU utilization was low at the sample, but both processes are still resident and the logs advanced beyond the prior iteration
- The tuned NEC job remains healthy on `atharv-rwx-pod-2` GPU `0`:
  - launcher/main pid pair `2004` / `2009`
  - `metrics.csv` is still absent for the run dir
  - latest visible checkpoints:
    - `(49) lambda 0.0003 ... [test acc 0.7402], sparsity 825/134200`
    - `(50) lambda 0.0003 ... [test acc 0.7404], sparsity 845/134200`
    - `(51) lambda 0.0003 ... [test acc 0.7402], sparsity 865/134200`
- `atharv-rwx-pod-2` GPU `1` remains intentionally idle because the current checkpoint already has every eligible non-duplicative follow-up job running.

### Soft-containment NEC completion at 12:57 PDT

- Re-read `TASK.md`, rechecked the active sprint state, and reran the required static checks:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The tuned NEC follow-up for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` is now complete on `atharv-rwx-pod-2`:
  - finalized sparse artifacts are present:
    - `metrics.csv`
    - `W_g@NEC=5,10,15,20,25,30`
    - `b_g@NEC=5,10,15,20,25,30`
  - finalized sparse metrics from `logs/cub_savlg_clean_global_softcontain_v1_nec150.log`:
    - `ACC@5 = 0.7413`
    - `ACC@10 = 0.7444`
    - `ACC@15 = 0.7447`
    - `ACC@20 = 0.7442`
    - `ACC@25 = 0.7447`
    - `ACC@30 = 0.7449`
    - `AVGACC = 0.7440`
  - interpretation:
    - this is slightly below the clean-global `SAVLG` checkpoint on both `ACC@5` (`0.7440`) and `AVGACC` (`0.7479`)
    - soft containment has not yet produced a new sparse best even though it cleared the dense gate
- The only unfinished work for this sprint is now the pair of localization evals still running on `atharv-rwx-pod`:
  - native-map eval main pid `5268`
    - latest visible progress: `119/181`
  - Grad-CAM eval main pid `5269`
    - latest visible progress: `459/724`
  - both output JSONs are still absent:
    - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
- No new launch or relaunch was taken:
  - NEC is already complete
  - the two remaining localization jobs are healthy and already occupy the only remaining eligible work for this sprint

### Soft-containment localization hold at 13:00 PDT

- Re-read `TASK.md`, rechecked the active sprint docs, and verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Dense and NEC for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` remain finalized:
  - dense test accuracy `0.7450777202072539`
  - tuned NEC `metrics.csv` present
- The only unfinished work is still the active localization pair on `atharv-rwx-pod`, and both logs advanced since the previous iteration:
  - native-map eval launcher/main pid pair `5251` / `5268`
    - latest visible progress: `139/181`
  - Grad-CAM eval launcher/main pid pair `5263` / `5269`
    - latest visible progress: `535/724`
  - both result files are still absent:
    - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
  - sampled GPU utilization on `atharv-rwx-pod` was near zero, but both processes remain resident and their logs continue to grow, so they were treated as healthy
- `atharv-rwx-pod-2` is now fully idle, but no additional safe launch is eligible:
  - dense and NEC are already complete
  - duplicating the two in-flight localization evals would violate the non-duplication and one-idea-at-a-time rules
  - the next sprint decision remains blocked on these localization outputs landing

### Soft-containment localization hold at 13:02 PDT

- Re-read `TASK.md`, rechecked the active sprint docs, and reran the required static checks:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Dense and NEC for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` remain finalized:
  - dense test accuracy `0.7450777202072539`
  - tuned NEC artifacts remain present:
    - `metrics.csv`
    - `W_g@NEC=5,10,15,20,25,30`
    - `b_g@NEC=5,10,15,20,25,30`
- The only unfinished work is still the active localization pair on `atharv-rwx-pod`, and both jobs were treated as healthy:
  - native-map eval launcher/main pid pair `5251` / `5268`
    - elapsed about `23:30`
    - latest visible progress: `151/181`
    - log mtime: `2026-04-05 20:02:17 UTC`
  - Grad-CAM eval launcher/main pid pair `5263` / `5269`
    - elapsed about `23:30`
    - latest visible progress: `583/724`
    - log mtime: `2026-04-05 20:02:26 UTC`
  - both result files are still absent:
    - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
  - `nvidia-smi pmon -c 1` showed only low instantaneous SM activity (`0` and `5`), but both processes remain resident and both logs advanced this iteration
- `atharv-rwx-pod-2` is idle, but no additional safe launch is eligible:
  - dense and NEC are already complete
  - duplicating the two remaining localization evals would violate the non-duplication and one-idea-at-a-time rules
  - the current jobs were not killed because `TASK.md` does not justify interrupting healthy in-flight work

### Soft-containment localization hold at 13:04 PDT

- Re-read `TASK.md`, rechecked the active sprint docs, and reran the required static checks:
  - `python -m py_compile methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Dense and NEC for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` remain finalized:
  - dense test accuracy `0.7450777202072539`
  - tuned NEC artifacts remain present:
    - `metrics.csv`
    - `W_g@NEC=5,10,15,20,25,30`
    - `b_g@NEC=5,10,15,20,25,30`
- The only unfinished work is still the active localization pair on `atharv-rwx-pod`, and both jobs were treated as healthy:
  - native-map eval launcher/main pid pair `5251` / `5268`
    - latest visible progress: `165/181`
    - log mtime: `2026-04-05 20:04:27 UTC`
  - Grad-CAM eval launcher/main pid pair `5263` / `5269`
    - latest visible progress: `637/724`
    - log mtime: `2026-04-05 20:04:34 UTC`
  - both result files are still absent:
    - `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`
  - `nvidia-smi pmon -c 1` showed only low instantaneous SM activity (`0` and `3`), but both processes remain resident and both logs advanced this iteration
- `atharv-rwx-pod-2` is idle, but no additional safe launch is eligible:
  - dense and NEC are already complete
  - duplicating the two remaining localization evals would violate the non-duplication and one-idea-at-a-time rules
  - the current jobs were not killed because `TASK.md` does not justify interrupting healthy in-flight work

### Soft-containment finalized and dual-branch sprint launched at 13:14 PDT

- Re-read `TASK.md`, inspected the active sprint docs plus the completed pod artifacts, and reran the required static checks:
  - local:
    - `python -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - pod:
    - `/opt/conda/envs/cbm/bin/python -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed in both places
- Verified live pod state directly before taking the next action:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
  - the soft-containment localization processes are now gone
  - both pods were idle before the next launch
- Soft-containment sprint `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_19_26_13` is now fully finalized:
  - dense test accuracy `0.7450777202072539`
  - tuned NEC:
    - `ACC@5 = 0.7413`
    - `ACC@10 = 0.7444`
    - `ACC@15 = 0.7447`
    - `ACC@20 = 0.7442`
    - `ACC@25 = 0.7447`
    - `ACC@30 = 0.7449`
    - `AVGACC = 0.7440`
  - native localization from `/workspace/SAVLGCBM/results/native_savlg_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`:
    - mean IoU `0.0798541042960401`
    - `mAP@0.3 = 0.9499911012709035`
    - `mAP@0.5 = 0.968202961291218`
    - `mAP@0.7 = 0.9733652412116287`
    - point hit rate `0.8518411967779056`
    - point coverage `0.050016547476869504`
  - Grad-CAM localization from `/workspace/SAVLGCBM/results/gradcam_savlg_softcontain_v1_full.json`:
    - mean IoU `0.08616969404445114`
    - `mAP@0.3 = 0.03324174849464784`
    - `mAP@0.5 = 0.01265974443866138`
    - `mAP@0.7 = 0.0064152962499236695`
    - point hit rate `0.7765486725663717`
    - point coverage `0.006503877865231593`
- Interpretation from the completed soft-containment sprint:
  - bbox localization improved over the clean-global `SAVLG` checkpoint, especially on native maps
  - classification/sparse metrics still remain below the clean-global best (`dense 0.7492`, `ACC@5 0.7440`, `AVGACC 0.7479`)
  - paper-style point evidence is still not good enough:
    - Grad-CAM hit rate improved slightly over clean-global `SAVLG` (`0.7765` vs `0.7374`)
    - Grad-CAM coverage regressed further (`0.00650` vs `0.0114`)
    - corrected `SALF` remains far stronger on point localization (`0.9427` hit, `0.9513` coverage)
- Chosen next ranked change from `TASK.md`:
  - dual-branch global + spatial architecture
  - keep the strong global concept path for classification
  - move localization pressure onto a separate spatial head
- Implemented that next sprint locally and synced the launch-critical files to `atharv-rwx-pod`:
  - added `savlg_branch_arch="dual"` plumbing
  - added a dual-head spatial concept layer with separate global and spatial branches
  - updated native / Grad-CAM evaluation to use the spatial branch for native maps and the global branch for pooled logits
  - created `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_softcontain_v1.json`
- Launched the new dense-first run on `atharv-rwx-pod` GPU `0`:
  - config: `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_softcontain_v1.json`
  - log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_softcontain_v1.log`
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12`
  - launcher/main pid pair:
    - `6092` / `6093`
  - startup args confirm:
    - `implementation_status=dual_branch_clean_global_softcontain_v1`
    - `savlg_branch_arch=dual`
    - `savlg_global_target_mode=binary_threshold`
    - `savlg_target_mode=soft_box`
    - `savlg_local_loss_mode=containment`
    - `loss_mask_w=0.25`
    - `cbl_early_stop_patience=8`
    - `cbl_min_epochs=15`
    - `cbl_min_delta=0.001`
  - startup health check:
    - run dir currently contains `args.txt` and `train.log`
    - log reached `SAVLG CBL epoch 2`
    - first validation summary:
      - `epoch=0 train_loss=0.376909 val_loss=0.286996 best_val=0.286996`
    - sampled GPU state:
      - `GPU 0`: active (`925 MiB / 81920 MiB`, `26%`)
      - `GPU 1`: idle

### Dual-branch status hold at 13:18 PDT

- Re-read `TASK.md`, rechecked the active sprint record, and verified live pod state directly before taking any new action.
- `kubectl get pods -n wenglab-interpretable-ai` shows both relevant pods still healthy:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dense dual-branch run on `atharv-rwx-pod` remains the only active eligible job for the current sprint:
  - launcher/main pid pair `6092` / `6093`
  - elapsed about `278s`
  - command:
    - `/opt/conda/envs/cbm/bin/python -u train_cbm.py --config /workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_softcontain_v1.json`
  - run dir still has only startup artifacts:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12/args.txt`
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12/train.log`
  - no dense completion artifact yet:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12/test_metrics.json` is absent
  - no NEC artifact yet:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12/metrics.csv` is absent
  - sampled GPU state:
    - GPU `0`: `1207 MiB`, `31%`
    - GPU `1`: idle
  - latest log tail shows healthy progress around `1822/4000` in the current solver stage
- `atharv-rwx-pod-2` is fully idle at this check:
  - no active `train_cbm.py`, `sparse_evaluation.py`, or localization-eval processes
  - both visible GPUs sampled at `1 MiB`, `0%`
- No new launch was taken:
  - NEC is still blocked by the dense gate because `test_metrics.json` has not landed yet
  - localization eval is also blocked until the dense checkpoint exists
  - duplicating the dense run or launching speculative follow-up work on idle GPUs would violate the one-idea-at-a-time and non-duplication rules

### Dual-branch dense gate passed and follow-up jobs launched at 13:22 PDT

- Re-read `TASK.md`, rechecked the active sprint plus recent loop state, and verified the dense run directory on the pod before launching anything new.
- Reran lightweight static checks before the follow-up launches:
  - local:
    - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `atharv-rwx-pod`:
    - `/opt/conda/envs/cbm/bin/python -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `atharv-rwx-pod-2`:
    - `/opt/conda/envs/cbm/bin/python -m py_compile sparse_evaluation.py`
    - import probe for `loguru`, `torch`, `torchvision`, `numpy`, `cv2`, `clip`, `pytorchcv`
  - result: passed in all three checks
- The dense dual-branch run `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12` is now complete:
  - `val accuracy = 1.0000`
  - `test accuracy = 0.744559585492228`
  - this satisfies the `TASK.md` dense gate, so NEC is eligible
- Launched the required follow-up jobs without starting any new model-change sprint:
  - NEC on `atharv-rwx-pod-2` GPU `0`:
    - command:
      - `CUDA_VISIBLE_DEVICES=0 nohup /opt/conda/envs/cbm/bin/python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12 --lam 0.01 --max_glm_steps 150 > /workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec150.log 2>&1 < /dev/null &`
    - launcher/main pid pair:
      - `2633` / `2634`
  - native-map localization on `atharv-rwx-pod` GPU `0`:
    - output:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - log:
      - `/workspace/SAVLGCBM/logs/native_savlg_dualbranch_softcontain_v1_full.log`
    - launcher/main pid pair:
      - `8260` / `8270`
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`:
    - output:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_softcontain_v1_full.json`
    - log:
      - `/workspace/SAVLGCBM/logs/gradcam_savlg_dualbranch_softcontain_v1_full.log`
    - launcher/main pid pair:
      - `8269` / `8271`
- Startup evidence at launch check:
  - `atharv-rwx-pod` `nvidia-smi` shows both A100s occupied by the two localization eval workers
  - `atharv-rwx-pod-2` `nvidia-smi` shows GPU `0` occupied by NEC
  - native eval log reached:
    - `SAVLG_DualBranch_Native dataloader ready: batches=181 batch_size=32 num_workers=8`
  - Grad-CAM eval log reached:
    - `SAVLG_DualBranch_GradCAM dataloader ready: batches=724 batch_size=8 num_workers=8`
    - progress `1/724`
  - NEC log is live and started at `0/169`
- No further launch was taken after that:
  - NEC plus the two full localization evals are the only distinct required follow-up jobs for this checkpoint
  - launching more work would either duplicate in-flight evaluation or jump ahead to a new idea before this sprint is fully measured

### Dual-branch status hold at 13:41 PDT

- Re-read `TASK.md`, rechecked the active dual-branch sprint record, and queried both live pods directly before deciding whether any new action was eligible.
- `kubectl get pods -n wenglab-interpretable-ai -o wide` still shows both relevant pods healthy:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The completed dense checkpoint is unchanged:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12`
  - `test accuracy = 0.744559585492228`
- The active follow-up jobs remain healthy and are still the only distinct jobs warranted for this sprint:
  - `atharv-rwx-pod` GPU `0`:
    - native-map localization eval main pid `8270`
    - result file still absent
    - latest visible progress `125/181`
  - `atharv-rwx-pod` GPU `1`:
    - Grad-CAM localization eval main pid `8271`
    - result file still absent
    - latest visible progress `482/724`
  - `atharv-rwx-pod-2` GPU `0`:
    - `nec800` sparse recovery rerun main pid `3168`
    - `metrics.csv` still absent
    - latest visible sparse-solver progress `1195/4000`
- `atharv-rwx-pod-2` GPU `1` remains idle, but no second distinct non-duplicative job is eligible without violating the one-idea-at-a-time rule.
- No launch, relaunch, or kill was taken in this status-hold iteration:
  - all required follow-up jobs for the dual-branch checkpoint are already in flight
  - there is still no finalized meaningful sparse or localization result to record in `docs/cub_results.md`

### Dual-branch local status hold at 13:43 PDT

- Re-read `TASK.md`, rechecked the active dual-branch sprint record, and then inspected this local workspace for any newly synced sprint artifacts before taking action.
- The local visibility gap remains unchanged in this session:
  - `saved_models/` is absent
  - `results/` is absent
  - local `logs/` still only contains older pulled files from the earlier baseline work
- Because of that, there is still no local artifact that supersedes the latest verified pod-visible state already recorded at `13:41 PDT`:
  - native-map localization eval still in flight
  - Grad-CAM localization eval still in flight
  - `nec800` sparse recovery rerun still in flight
- No launch, relaunch, or kill was taken in this iteration:
  - launching anything new from this filesystem would risk duplicating healthy work
  - there is still no finalized meaningful sparse or localization result to add to `docs/cub_results.md`

### Dual-branch local status hold at 13:44 PDT

- Re-read `TASK.md`, rechecked the active dual-branch sprint record plus the newest local loop logs, and verified this filesystem still has no synced dual-branch outputs.
- The local visibility gap is unchanged:
  - `saved_models/` is absent
  - `results/` is absent
  - local `logs/` still only contains the older pulled baseline logs
- Because there is still no local artifact newer than the already recorded `13:43 PDT` pod-visible status, no safe new action is eligible from this workspace.
- No launch, relaunch, or kill was taken in this iteration:
  - the only eligible follow-up jobs remain the already-recorded native-map eval, Grad-CAM eval, and `nec800` sparse recovery rerun
  - starting anything new here would risk duplicating healthy work or violating the one-idea-at-a-time rule

### Dual-branch NEC recovery at 13:47 PDT

- Re-read `TASK.md`, rechecked the active dual-branch sprint record, and verified both live pods directly again before taking any new action.
- Reran the sparse static check locally and on `atharv-rwx-pod-2`:
  - `python3 -m py_compile sparse_evaluation.py`
  - `/opt/conda/envs/cbm/bin/python -m py_compile /workspace/SAVLGCBM/sparse_evaluation.py`
  - both passed
- `atharv-rwx-pod` localization evals remain healthy and were left untouched:
  - native-map output still absent; latest visible progress about `162/181`
  - Grad-CAM output still absent; latest visible progress about `624/724`
  - both logs still had fresh `20:47 UTC` mtimes at check
- `atharv-rwx-pod-2` `nec800` finished and remained collapsed:
  - run dir:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s800`
  - `metrics.csv`:
    - `NEC = 655.3799834251404`
    - `Accuracy = 0.024352332577109337`
  - log summary:
    - `did not converge at 4000 iterations`
    - target-NEC accuracies again collapsed to about `0.0052`
    - `Average acc = 0.0052`
- Per `TASK.md`, that sparse result is not meaningful, so the eligible safe recovery move remained to increase `max_glm_steps`.
- Launched exactly one higher-step NEC rerun on `atharv-rwx-pod-2` GPU `0`:
  - clone:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s1600`
  - command:
    - `CUDA_VISIBLE_DEVICES=0 nohup /opt/conda/envs/cbm/bin/python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s1600 --lam 0.01 --max_glm_steps 1600 > /workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec1600.log 2>&1 < /dev/null &`
  - main pid:
    - `3441`
- Cleaned copied sparse summary artifacts from the fresh clone after launch so future iterations do not misread stale `W_g@NEC=*`, `b_g@NEC=*`, or `metrics.*` files before the rerun finishes.
- Startup check passed:
  - pod2 GPU `0` shows the new sparse worker resident (`477 MiB`, `10%`)
  - the `nec1600` log had already progressed to about `112/169`
- No additional launch was taken:
  - the two A100 localization jobs are already the required active follow-up work
  - `nec1600` is the only new non-duplicative recovery job made eligible by the failed `nec800` run

### Dual-branch localization landed at 13:53 PDT

- Re-read `TASK.md`, rechecked the active dual-branch sprint record, and queried both live pods directly again before deciding whether any new launch was still eligible.
- `atharv-rwx-pod` localization outputs are now complete:
  - native-map output:
    - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_softcontain_v1_full_meanthr_bs32_nw8_nocache.json`
    - mean IoU `0.10386529624037594`
    - `mAP@0.3 = 0.22793726036719067`
    - `mAP@0.5 = 0.22849189875236445`
    - `mAP@0.7 = 0.22686268117877612`
    - point hit rate `0.9317171436510142`
    - point coverage `0.41429126437112396`
    - matched part hits / total `26826 / 28792`
  - Grad-CAM output:
    - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_softcontain_v1_full.json`
    - mean IoU `0.07861747122510306`
    - `mAP@0.3 = 0.015886130176933487`
    - `mAP@0.5 = 0.007540785230536763`
    - `mAP@0.7 = 0.005351927302307943`
    - point hit rate `0.8132530120481928`
    - point coverage `0.0023885923133372664`
    - matched part hits / total `135 / 166`
- `atharv-rwx-pod` now has no remaining active follow-up workers for this sprint; both A100s are idle.
- `atharv-rwx-pod-2` still has the only active eligible job:
  - `nec1600` on GPU `0`
  - pid `3441`
  - `metrics.csv` is still absent for `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s1600`
- No new launch, relaunch, or kill was taken in this iteration:
  - launching a new dense run would skip ahead before the current sprint's sparse recovery is resolved
  - there is no second distinct non-duplicative job left for the idle GPUs

### Dual-branch NEC recovery at 13:58 PDT

- Re-read `TASK.md`, rechecked the active dual-branch sprint record, and queried both live pods directly again before taking any new action.
- Reran the sparse static check locally and on `atharv-rwx-pod-2`:
  - `python3 -m py_compile sparse_evaluation.py`
  - `/opt/conda/envs/cbm/bin/python -m py_compile /workspace/SAVLGCBM/sparse_evaluation.py`
  - both passed
- Pod health remains stable:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dual-branch dense checkpoint is unchanged and still clears the NEC gate:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12`
  - dense `test accuracy = 0.744559585492228`
- Dual-branch localization is now fully landed and remains the strongest part of this checkpoint:
  - native-map output:
    - mean IoU `0.10386529624037594`
    - `mAP@0.3 / 0.5 / 0.7 = 0.22793726036719067 / 0.22849189875236445 / 0.22686268117877612`
    - point hit / coverage `0.9317171436510142 / 0.41429126437112396`
  - Grad-CAM output:
    - mean IoU `0.07861747122510306`
    - `mAP@0.3 / 0.5 / 0.7 = 0.015886130176933487 / 0.007540785230536763 / 0.005351927302307943`
    - point hit / coverage `0.8132530120481928 / 0.0023885923133372664`
- Every NEC attempt through `max_glm_steps=1600` is now complete and still collapsed:
  - base `nec150`:
    - `NEC = 655.445014834404`
    - `Accuracy = 0.01295336801558733`
  - `nec400`:
    - `NEC = 645.5450026392937`
    - `Accuracy = 0.0321243517100811`
  - `nec800`:
    - `NEC = 655.3799834251404`
    - `Accuracy = 0.024352332577109337`
  - `nec1600`:
    - `NEC = 647.7299939990044`
    - `Accuracy = 0.029879102483391762`
    - log still ends with:
      - `did not converge at 4000 iterations`
      - target-NEC accuracies collapsing to about `0.0052`
      - `Average acc = 0.0052`
- Per `TASK.md`, these sparse results are not meaningful and the next eligible recovery move remained to increase `max_glm_steps`.
- Launched exactly one new sparse recovery rerun on `atharv-rwx-pod-2` GPU `0`:
  - fresh clone:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s3200`
  - command:
    - `CUDA_VISIBLE_DEVICES=0 nohup /opt/conda/envs/cbm/bin/python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s3200 --lam 0.01 --max_glm_steps 3200 > /workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec3200.log 2>&1 < /dev/null &`
  - launcher / main pid:
    - `3778 / 3782`
- Cleared copied sparse summary artifacts from the fresh clone before using it, so future iterations will not misread stale `metrics.*`, `W_g@NEC=*`, or `b_g@NEC=*` outputs.
- Startup check passed:
  - pod2 GPU `0` shows the new sparse worker resident (`477 MiB`)
  - `logs/cub_savlg_clean_global_dualbranch_softcontain_v1_nec3200.log` advanced to about `35/169`
- No other launch, relaunch, or kill was taken:
  - both localization jobs are already complete
  - `nec3200` is the only distinct non-duplicative eligible follow-up for this checkpoint
  - starting a new dense idea now would violate the one-idea-at-a-time rule

### Dual-branch live hold at 14:01 PDT

- Re-read `TASK.md`, rechecked the active sprint README plus `savlg-research/` state, and verified both live pods directly again before touching anything.
- Pod state remains stable:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dual-branch localization outputs remain complete and unchanged on `atharv-rwx-pod`:
  - native:
    - mean IoU `0.10386529624037594`
    - `mAP@0.3 / 0.5 / 0.7 = 0.22793726036719067 / 0.22849189875236445 / 0.22686268117877612`
    - point hit / coverage `0.9317171436510142 / 0.41429126437112396`
  - Grad-CAM:
    - mean IoU `0.07861747122510306`
    - `mAP@0.3 / 0.5 / 0.7 = 0.015886130176933487 / 0.007540785230536763 / 0.005351927302307943`
    - point hit / coverage `0.8132530120481928 / 0.0023885923133372664`
- Both A100s on `atharv-rwx-pod` are now idle, but there is still no second distinct eligible job for this sprint because localization is already complete.
- The only active eligible follow-up job is still the sparse recovery rerun on `atharv-rwx-pod-2` GPU `0`:
  - checkpoint clone:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s3200`
  - command:
    - `CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cbm/bin/python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_20_14_12_necpath_s3200 --lam 0.01 --max_glm_steps 3200`
  - launcher / main pid:
    - `3778 / 3782`
  - `metrics.csv` is still absent for the fresh clone
  - log progress at check:
    - about `1421/4000`
- No new launch, relaunch, or kill was taken in this iteration:
  - launching anything on the idle A100s would duplicate completed localization work or skip ahead of the current sparse-recovery gate
  - `nec3200` is still the only safe active job for this one-idea sprint

### Dual-branch no-safe-action hold at 14:03 PDT

- Re-read `TASK.md`, re-inspected the active dual-branch sprint record, `savlg-research/runtime_state.json`, and the newest persisted loop log before taking any action.
- Local persistent state in this sandbox is still insufficient to advance the sprint safely:
  - `runtime_state.json` still points to iteration `36` with `status="running_iteration"`
  - the referenced `iter-0036-20260405T210237Z.last.txt` is missing locally
  - the referenced loop log exists and already records the latest verified pod-visible state at `14:01 PDT`
  - local `logs/` contains only older copied files
  - local `results/` is absent
  - `nvidia-smi` is unavailable here
- Because there is no newer local artifact proving that the active `nec3200` rerun has finished or failed, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse outcome is visible in the persistent filesystem.

### Dual-branch no-safe-action hold at 14:04 PDT

- Re-read `TASK.md`, re-inspected the active dual-branch sprint record, recent loop logs, and `savlg-research/runtime_state.json` before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state is still insufficient to advance the sprint safely:
  - `runtime_state.json` now points at iteration `37` with `status="running_iteration"`
  - the active log path is `savlg-research/loop-logs/iter-0037-20260405T210409Z.log`
  - the corresponding `iter-0037-20260405T210409Z.last.txt` is still not present locally
  - local `logs/` still contains only older copied files
  - local `saved_models/cub/` is absent
  - local `results/` is absent
  - `ps` is blocked and `nvidia-smi` is unavailable here
- The latest persisted pod-visible evidence still shows only one unresolved eligible follow-up for this sprint:
  - the `nec3200` sparse recovery rerun
- Because there is still no newer local artifact proving that `nec3200` has finished or failed, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse outcome is visible in the persistent filesystem.

### Dual-branch no-safe-action hold at 14:06 PDT

- Re-read `TASK.md`, re-inspected the active dual-branch sprint record, `docs/cub_results.md`, recent loop logs, and `savlg-research/runtime_state.json` before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state is still insufficient to advance the sprint safely:
  - `runtime_state.json` now points at iteration `38` with `status="running_iteration"`
  - the active log path `savlg-research/loop-logs/iter-0038-20260405T210548Z.log` exists locally
  - the corresponding `iter-0038-20260405T210548Z.last.txt` is still missing locally
  - local `saved_models/cub/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files
  - `ps` is blocked and `nvidia-smi` is unavailable here
- The latest persisted pod-visible evidence still shows only one unresolved eligible follow-up for this sprint:
  - the `nec3200` sparse recovery rerun
- Because there is still no newer local artifact proving that `nec3200` has finished, failed, or freed a new safe next step, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse outcome is visible in the persistent filesystem.

### Dual-branch no-safe-action hold at 14:08 PDT

- Re-read `TASK.md`, re-inspected the active dual-branch sprint record, recent `savlg-research/loop-logs/`, and `savlg-research/runtime_state.json` before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state remains insufficient to advance the sprint safely:
  - wall-clock check was `2026-04-05 14:08:08 PDT`
  - `runtime_state.json` still points at iteration `39` with `status="running_iteration"`
  - the latest local loop summary still says the only unresolved eligible follow-up is the pod-side `nec3200` sparse recovery rerun
  - local `saved_models/cub/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files
  - `ps` is blocked and `nvidia-smi` is unavailable here
- Because there is still no newer local artifact proving that `nec3200` has finished, failed, or freed a new safe next step, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse outcome is visible in the persistent filesystem.

### Dual-branch no-safe-action hold at 14:09 PDT

- Re-read `TASK.md`, re-inspected the existing `savlg-research/` sprint directories, the active dual-branch sprint README, current loop logs, and `savlg-research/runtime_state.json` before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state remains insufficient to advance the sprint safely:
  - wall-clock check was `2026-04-05 14:09:44 PDT`
  - `savlg-research/runtime_state.json` still shows an active loop iteration in progress
  - the newest persisted local loop log still says the only unresolved eligible follow-up is the pod-side `nec3200` sparse recovery rerun
  - local `saved_models/cub/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files
  - `ps` is blocked and `nvidia-smi` is unavailable here
- Because there is still no newer local artifact proving that the active `nec3200` rerun has finished, failed, or freed a new safe next step, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse outcome is visible in the persistent filesystem.

### Dual-branch no-safe-action hold at 14:12 PDT

- Re-read `TASK.md`, re-inspected the existing `savlg-research/` sprint directories, the active dual-branch sprint README, recent loop-log summaries, and `savlg-research/runtime_state.json` before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state remains insufficient to advance the sprint safely:
  - wall-clock check was `2026-04-05 14:12:15 PDT`
  - `savlg-research/runtime_state.json` now points at iteration `41` with `status="running_iteration"`
  - the active log path `savlg-research/loop-logs/iter-0041-20260405T211144Z.log` exists locally
  - the corresponding `iter-0041-20260405T211144Z.last.txt` is still absent locally
  - local `saved_models/cub/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files from earlier work
  - `ps` is blocked and `nvidia-smi` is unavailable here
- Because there is still no newer local artifact proving that the active `nec3200` rerun has finished, failed, or freed a new safe next step, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse outcome is visible in the persistent filesystem.

### Dual-branch no-safe-action hold at 14:14 PDT

- Re-read `TASK.md`, re-inspected the existing `savlg-research/` sprint directories, the active dual-branch sprint README, recent loop logs, and `savlg-research/runtime_state.json` before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state remains insufficient to advance the sprint safely:
  - wall-clock check was `2026-04-05 14:14:49 PDT`
  - `savlg-research/runtime_state.json` now points at iteration `42` with `status="running_iteration"`
  - the active log path `savlg-research/loop-logs/iter-0042-20260405T211412Z.log` exists locally
  - the corresponding `iter-0042-20260405T211412Z.last.txt` is still absent locally
  - local `saved_models/cub/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files:
    - `cub_lf_resnet18_cub_nec_fast.log`
    - `cub_salf_resnet18_cub_nec_fast.log`
    - `cub_salf_resnet18_cub_pod2_small.log`
  - `cub_vlg_clip_rn50_nec_fast.log`
  - `ps` is blocked and `nvidia-smi` is unavailable here
- Because there is still no newer local artifact proving that the active `nec3200` rerun has finished, failed, or freed a new safe next step, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse outcome is visible in the persistent filesystem.

### Dual-branch no-safe-action hold at 14:16 PDT

- Re-read `TASK.md`, re-inspected the existing `savlg-research/` sprint directories, the active dual-branch sprint README, recent loop logs, and `savlg-research/runtime_state.json` before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state remains insufficient to advance the sprint safely:
  - wall-clock check was `2026-04-05 14:16:45 PDT`
  - `savlg-research/runtime_state.json` now points at iteration `43` with `status="running_iteration"`
  - the active log path `savlg-research/loop-logs/iter-0043-20260405T211556Z.log` exists locally
  - the corresponding `iter-0043-20260405T211556Z.last.txt` is still absent locally
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files from earlier work
  - `ps` is blocked and `nvidia-smi` is unavailable here
- Because there is still no newer local artifact proving that the active `nec3200` rerun has finished, failed, or freed a new safe next step, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse outcome is visible in the persistent filesystem.

### Dual-branch no-safe-action hold at 14:19 PDT

- Re-read `TASK.md`, re-inspected the existing `savlg-research/` sprint directories, the active dual-branch sprint README, `savlg-research/runtime_state.json`, and the newest local loop-log artifacts before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state remains insufficient to advance the sprint safely:
  - wall-clock check was `2026-04-05 14:19 PDT`
  - `savlg-research/runtime_state.json` now points at iteration `44` with `status="running_iteration"`
  - the active log path `savlg-research/loop-logs/iter-0044-20260405T211900Z.log` exists locally
  - the corresponding `iter-0044-20260405T211900Z.last.txt` is still absent locally
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files from earlier work
  - `ps` is blocked and `nvidia-smi` is unavailable here
- Because there is still no newer local artifact proving that the unresolved dual-branch sparse recovery or localization follow-up has finished, failed, or freed a new safe next step, no relaunch, new launch, or kill was safe in this iteration.
- The dual-branch sprint remains in progress and still should not receive `DONE.txt` until a meaningful sparse or localization outcome is visible in the persistent filesystem.

### Dual-branch completion and next launch at 14:24 PDT

- Re-read `TASK.md`, re-inspected the active sprint records, and reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`; the local compile check passed.
- Recovered live pod visibility with `kubectl` and verified both pods are `Running` with `0` restarts.
- Verified the dual-branch checkpoint is fully terminal now:
  - dense test accuracy `0.744559585492228`
  - native localization finished at mean IoU `0.1039`, point hit `0.9317`, coverage `0.4143`
  - Grad-CAM localization finished at mean IoU `0.0786`, point hit `0.8133`, coverage `0.00239`
  - all NEC recovery paths are complete and collapsed:
    - base `0.0130`
    - `nec400` `0.0321`
    - `nec800` `0.0244`
    - `nec1600` `0.0299`
    - `nec3200` `0.0266`
- Closed the dual-branch sprint with `DONE.txt` because there is no remaining eligible follow-up on that checkpoint.
- Chose the next single ranked non-duplicative change:
  - keep the dual-branch clean-global soft-containment setup
  - add auxiliary local MIL pooling on the spatial branch only
  - keep the main global path on `avg` pooling to respect the archived warning against retrying pure top-k replacement of the main pooling path
- Added the new dense config:
  - `configs/unified/cub_savlg_cbm_clean_global_dualbranch_localmil_lse_v1.json`
- Opened the next sprint:
  - `savlg-research/2026-04-05--14-24-dual-branch-local-mil/README.md`
- Copied that new config into the pod workspace because `/workspace/SAVLGCBM` does not auto-sync newly added local files.
- Launched exactly one new dense run on `atharv-rwx-pod` GPU `0`:
  - config: `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_localmil_lse_v1.json`
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_v1.log`
  - launcher/main PIDs: `9135 / 9136`
- Startup verification after launch:
  - pod-side `py_compile` passed
  - saved args confirm `savlg_branch_arch=dual`, `savlg_pooling=avg`, `savlg_use_local_mil=true`, `savlg_local_pooling=lse`, `loss_local_mil_w=0.1`
  - A100 GPU `0` is active at about `925 MiB`, `26%`
  - the training log advanced into `SAVLG CBL epoch 2`
- No NEC or second launch was taken because the new dense gate has not landed yet, so `docs/cub_results.md` does not need another update in this iteration.

### Dual-branch local-MIL no-safe-action hold at 14:30 PDT

- Re-read `TASK.md`, re-inspected the existing `savlg-research/` sprint directories, the active local-MIL sprint README, `savlg-research/runtime_state.json`, and the newest local loop-log artifacts before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state remains insufficient to advance the active local-MIL sprint safely:
  - wall-clock check was `2026-04-05 14:30 PDT`
  - `savlg-research/runtime_state.json` now points at iteration `46` with `status="running_iteration"`
  - the active log path `savlg-research/loop-logs/iter-0046-20260405T212936Z.log` exists locally
  - the corresponding `iter-0046-20260405T212936Z.last.txt` is still absent locally
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files from earlier work
  - `ps` is blocked, `pgrep` cannot access the process table here, and `nvidia-smi` is unavailable
- The latest verified experiment state is still just the previous iteration's dense launch:
  - local-MIL dense run on `atharv-rwx-pod` GPU `0`
  - run dir `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - log `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_v1.log`
- Because there is still no newer local artifact proving that the dense run has finished, failed, or cleared the `0.74` dense gate, no NEC launch, new dense launch, relaunch, or kill was safe in this iteration.
- The active local-MIL sprint remains in progress and should not receive `DONE.txt` until a meaningful dense result or downstream artifact is visible in the persistent filesystem.

### Dual-branch local-MIL no-safe-action hold at 14:33 PDT

- Re-read `TASK.md`, re-inspected the active local-MIL sprint README, `savlg-research/runtime_state.json`, the newest loop logs, and the local `logs/` directory before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state is still insufficient to advance the sprint safely:
  - wall-clock check was `2026-04-05 14:33 PDT`
  - `savlg-research/runtime_state.json` now points at iteration `47` with `status="running_iteration"`
  - the active log path `savlg-research/loop-logs/iter-0047-20260405T213113Z.log` exists locally
  - the corresponding `iter-0047-20260405T213113Z.last.txt` is still absent locally
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files last updated on `2026-04-02`
  - `ps` is blocked and `nvidia-smi` is unavailable
- The latest verified experiment state is unchanged:
  - local-MIL dense run on `atharv-rwx-pod` GPU `0`
  - run dir `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - log `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_v1.log`
- Because there is still no newer persistent artifact proving that the dense run has finished, failed, or cleared the `0.74` dense gate, no NEC launch, new dense launch, relaunch, or kill was safe in this iteration either.

### Dual-branch local-MIL no-safe-action hold at 14:34 PDT

- Re-read `TASK.md`, re-inspected the existing sprint directories, the active local-MIL sprint README, `savlg-research/runtime_state.json`, the newest loop logs, and the local `logs/` tree before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state is still insufficient to advance the sprint safely:
  - wall-clock check was `2026-04-05 14:34 PDT`
  - `savlg-research/runtime_state.json` now points at iteration `48` with `status="running_iteration"`
  - the active log path `savlg-research/loop-logs/iter-0048-20260405T213419Z.log` exists locally
  - the corresponding `iter-0048-20260405T213419Z.last.txt` is still absent locally
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older copied files last updated on `2026-04-02`
  - `ps` is blocked and `nvidia-smi` is unavailable
- The latest verified experiment state is unchanged:
  - local-MIL dense run on `atharv-rwx-pod` GPU `0`
  - run dir `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - log `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_v1.log`
- Because there is still no newer persistent artifact proving that the dense run has finished, failed, or cleared the `0.74` dense gate, no NEC launch, new dense launch, relaunch, or kill was safe in this iteration either.

### Dual-branch local-MIL dense gate cleared at 14:38 PDT

- Re-read `TASK.md`, re-inspected the active local-MIL sprint state, reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`, and then recovered direct pod visibility with `kubectl`.
- Verified both research pods are healthy and unrestarted:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Re-verified `atharv-rwx-pod-2` is safe for relaunch after its earlier rebuild:
  - imports passed for `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, and `pytorchcv`
  - required roots are present:
    - `/workspace/SAVLGCBM`
    - `/workspace/SAVLGCBM/annotations`
    - `/workspace/SAVLGCBM/datasets/CUB_200_2011`
- Pod-side compile checks passed on both pods for:
  - `/workspace/SAVLGCBM/methods/salf.py`
  - `/workspace/SAVLGCBM/methods/savlg.py`
  - `/workspace/SAVLGCBM/train_cbm.py`
  - `/workspace/SAVLGCBM/sparse_evaluation.py`
  - `/workspace/SAVLGCBM/scripts/evaluate_native_spatial_maps.py`
- The dense local-MIL checkpoint is now complete:
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - dense log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_v1.log`
  - `test_metrics.json` accuracy: `0.7500863557858376`
- Dense-gate interpretation:
  - clears the `0.74` NEC threshold in `TASK.md`
  - slightly exceeds the prior clean-global dense best `0.7492`
  - this is now the best dense `SAVLG` checkpoint recovered so far in this repo, but sparse and localization metrics are still pending
- Because both A100s were idle and three downstream jobs were now eligible, this iteration launched all non-duplicative follow-up work:
  - NEC on `atharv-rwx-pod-2` GPU `0`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_v1_nec150.log`
    - pid pair: `4166 / 4167`
  - native-map localization on `atharv-rwx-pod` GPU `0`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_dualbranch_localmil_lse_v1_full.log`
    - output: `/workspace/SAVLGCBM/results/native_savlg_dualbranch_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
    - pid pair: `11270 / 11271`
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`
    - log: `/workspace/SAVLGCBM/logs/gradcam_savlg_dualbranch_localmil_lse_v1_full.log`
    - output: `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_localmil_lse_v1_full.json`
    - pid pair: `11274 / 11275`
- Startup verification:
  - native eval progressed to about `3/181`
  - Grad-CAM eval progressed to about `13/724`
  - NEC progressed to about `30/169`
  - sampled resident GPU memory after launch:
    - `atharv-rwx-pod` GPU `0`: about `805 MiB`
    - `atharv-rwx-pod` GPU `1`: about `1055 MiB`
    - `atharv-rwx-pod-2` GPU `0`: about `477 MiB`
- No `DONE.txt` was added because the local-MIL sprint is now in the NEC/localization phase and remains in progress.

### Dual-branch local-MIL in-flight status at 14:43 PDT

- Re-read `TASK.md`, re-inspected the existing sprint directories to avoid duplication, reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`, and then checked live pod state with `kubectl`.
- Both pods remain healthy and unrestarted:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dense checkpoint is unchanged and remains the best dense-only `SAVLG` result recovered so far:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - dense `test_metrics.json` accuracy `0.7500863557858376`
- All previously eligible downstream jobs are still running healthily:
  - native-map localization on `atharv-rwx-pod` GPU `0`
    - progress about `35/181`
    - output still absent:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`
    - progress about `134/724`
    - output still absent:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_localmil_lse_v1_full.json`
  - tuned NEC on `atharv-rwx-pod-2` GPU `0`
    - `metrics.csv` still absent
    - the log is progressing through the sparse path; the latest visible intermediate checkpoints are:
      - `(0) test acc 0.6858`
      - `(1) test acc 0.6886`
      - `(2) test acc 0.6895`
    - per prior NEC notes, these intermediate path rows are not the final reported `ACC@5` or `AVGACC`
- GPU occupancy at check:
  - `atharv-rwx-pod` GPU `0`: about `805 MiB`
  - `atharv-rwx-pod` GPU `1`: about `1057 MiB`
  - `atharv-rwx-pod-2` GPU `0`: about `761 MiB`
  - `atharv-rwx-pod-2` GPU `1`: idle
- No new launch, relaunch, or kill was taken in this iteration because the current sprint already has all non-duplicative eligible follow-up work in flight, and no finalized sparse or localization artifact landed yet.

### Dual-branch local-MIL local hold at 14:44 PDT

- Re-read `TASK.md`, re-inspected all current sprint directories, `savlg-research/runtime_state.json`, the newest loop logs, and the active local-MIL sprint README before taking any action.
- Reran the standard local compile check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Local persistent state is still incomplete for a safe advance:
  - wall-clock check was `2026-04-05 14:44:26 PDT`
  - `savlg-research/runtime_state.json` points at iteration `51` with `status="running_iteration"`
  - the current log `savlg-research/loop-logs/iter-0051-20260405T214346Z.log` exists locally
  - the corresponding `iter-0051-20260405T214346Z.last.txt` is still absent locally
  - local `saved_models/` is absent
  - local `results/` is absent
  - this filesystem exposes only `./logs`
  - `ps` is blocked and `nvidia-smi` is unavailable
  - `/workspace/SAVLGCBM` is not mounted here
- The newest completed verified state remains the prior finished iteration message in `iter-0050-20260405T214210Z.last.txt`, which says:
  - dense checkpoint `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29` completed at `0.7500863557858376`
  - NEC is already running on `atharv-rwx-pod-2` GPU `0`
  - native localization is already running on `atharv-rwx-pod` GPU `0`
  - Grad-CAM localization is already running on `atharv-rwx-pod` GPU `1`
- Because no newer finalized artifact is visible and this session cannot safely re-verify the pod-side jobs, no launch, relaunch, recovery action, or kill was safe in this iteration either.

### Dual-branch local-MIL live hold at 14:47 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/runtime_state.json` plus the active sprint record, reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`, and then rechecked the live pods directly with `kubectl`.
- Both pods remain healthy with no restart recovery needed:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dense checkpoint is unchanged and is still the best dense-only `SAVLG` result in the repo so far:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - dense `test_metrics.json` accuracy `0.7500863557858376`
- All eligible downstream jobs are still active and progressing:
  - native-map localization on `atharv-rwx-pod` GPU `0`
    - progress about `59/181`
    - output still absent:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`
    - progress about `225/724`
    - output still absent:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_localmil_lse_v1_full.json`
  - tuned NEC on `atharv-rwx-pod-2` GPU `0`
    - `metrics.csv` still absent
    - latest visible intermediate path rows are:
      - `(19) test acc 0.7073`
      - `(20) test acc 0.7079`
      - `(21) test acc 0.7083`
      - `(22) test acc 0.7083`
    - these remain sparse-path checkpoints, not final `ACC@5` / `AVGACC` metrics
- GPU occupancy at check:
  - `atharv-rwx-pod` GPU `0`: about `805 MiB`
  - `atharv-rwx-pod` GPU `1`: about `1057 MiB`
  - `atharv-rwx-pod-2` GPU `0`: about `761 MiB`
  - `atharv-rwx-pod-2` GPU `1`: idle
- No new launch, relaunch, recovery action, or kill was taken in this iteration:
  - all non-duplicative eligible follow-up jobs are already in flight
  - the idle secondary GPU does not have a safe distinct job under the current dense-first single-idea sprint
  - no finalized sparse or localization artifact landed yet

### Dual-branch local-MIL live hold at 14:50 PDT

- Re-read `TASK.md`, re-inspected the active local-MIL sprint record plus `savlg-research/runtime_state.json`, reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`, and then rechecked the live pods directly with `kubectl`.
- Both pods remain healthy with no restart recovery needed:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dense checkpoint is unchanged and remains the current best dense-only `SAVLG` result in the repo:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - dense `test_metrics.json` accuracy `0.7500863557858376`
- All eligible downstream jobs are still active and their final artifacts are still absent:
  - native-map localization on `atharv-rwx-pod` GPU `0`
    - progress about `87/181`
    - output still absent:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`
    - progress about `337/724`
    - output still absent:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_localmil_lse_v1_full.json`
  - tuned NEC on `atharv-rwx-pod-2` GPU `0`
    - `metrics.csv` is still absent:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29/metrics.csv`
    - latest intermediate sparse-path rows are:
      - `(51) test acc 0.7188`, sparsity `1078/134200`
      - `(52) test acc 0.7190`, sparsity `1115/134200`
      - `(53) test acc 0.7190`, sparsity `1148/134200`
    - these remain path checkpoints, not final `ACC@5` or `AVGACC`
- GPU occupancy at check:
  - `atharv-rwx-pod` GPU `0`: about `805 MiB`
  - `atharv-rwx-pod` GPU `1`: about `1057 MiB`
  - `atharv-rwx-pod-2` GPU `0`: about `761 MiB`
  - `atharv-rwx-pod-2` GPU `1`: idle
- No new launch, relaunch, recovery action, or kill was taken in this iteration because the current sprint already has all non-duplicative eligible follow-up work in flight, and no finalized sparse or localization artifact landed yet.

### Dual-branch local-MIL local hold at 14:54 PDT

- Re-read `TASK.md`, re-inspected all sprint directories plus `savlg-research/runtime_state.json`, checked the newest completed loop messages, and reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`; the compile check passed.
- Local persistent state still does not expose the pod-side artifact roots needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - `/workspace/SAVLGCBM` is not mounted here
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- `savlg-research/runtime_state.json` still reports an active loop invocation:
  - `iteration = 54`
  - `status = "running_iteration"`
- The newest verified experiment state available in the persistent filesystem remains unchanged from the prior live hold:
  - dense checkpoint `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29` completed at `0.7500863557858376`
  - NEC is already in flight on `atharv-rwx-pod-2` GPU `0`
  - native localization is already in flight on `atharv-rwx-pod` GPU `0`
  - Grad-CAM localization is already in flight on `atharv-rwx-pod` GPU `1`
- No launch, relaunch, recovery action, or kill was taken in this iteration because no newer finalized sparse or localization artifact is visible here, and acting without fresh pod-side visibility would risk duplicating healthy work.

### Dual-branch local-MIL local hold at 14:59 PDT

- Re-read `TASK.md`, re-inspected the sprint directories plus `savlg-research/runtime_state.json`, checked the newest loop logs, and reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`; the compile check passed.
- Local persistent state still cannot safely advance the sprint on its own:
  - wall-clock check was `2026-04-05 14:59:44 PDT`
  - `savlg-research/runtime_state.json` points at iteration `56` with `status="running_iteration"`
  - local `saved_models/` is absent
  - local `results/` is absent
  - `/workspace/SAVLGCBM` is not mounted here
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- A fresh control-plane recheck was attempted but failed:
  - `kubectl get pods -o wide`
  - failed with the same auth-provider DNS/OIDC error for `authentik.nrp-nautilus.io`
- The newest verified experiment state therefore remains the prior completed live-pod check:
  - dense checkpoint `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29` is complete at `0.7500863557858376`
  - tuned NEC is complete for that checkpoint at `ACC@5 = 0.7187` and `AVGACC = 0.7268`
  - the only unfinished eligible work is still the two localization evals already running on `atharv-rwx-pod`
- No launch, relaunch, recovery action, or kill was taken in this iteration because no newer finalized localization artifact is visible here, and the failed `kubectl` recheck means there is no safe basis for further pod-side action.

### Dual-branch local-MIL live hold at 15:01 PDT

- Re-read `TASK.md`, rechecked the active local-MIL sprint record plus `savlg-research/runtime_state.json`, reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`, and then re-verified the live pods directly with `kubectl`.
- Both research pods remain healthy with no restart recovery required:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The dense and sparse state is unchanged:
  - dense checkpoint `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29` remains complete at `0.7500863557858376`
  - tuned NEC remains complete at `ACC@5 = 0.7187`, `ACC@10 = 0.7249`, `ACC@15 = 0.7271`, `ACC@20 = 0.7288`, `ACC@25 = 0.7301`, `ACC@30 = 0.7314`, `AVGACC = 0.7268`
  - `atharv-rwx-pod-2` has no active research processes and both visible GPUs are idle
- The only unfinished eligible work is still the two localization evals already running on `atharv-rwx-pod`:
  - native-map localization on GPU `0`
    - main pid `11271`
    - output still absent:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
    - latest visible progress `159/181`
  - Grad-CAM localization on GPU `1`
    - main pid `11275`
    - output still absent:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_localmil_lse_v1_full.json`
    - latest visible progress `614/724`
- Sampled `nvidia-smi` utilization on `atharv-rwx-pod` was `0%` for both GPUs at the instant of the check, but the live log mtimes and counters were still advancing, so the two eval jobs were treated as healthy and left untouched.
- No launch, relaunch, recovery action, or kill was taken in this iteration because finishing those two in-flight localization artifacts remains the only safe next step under `TASK.md`.

### Dual-branch local-MIL complete and MLP rerun launched at 15:11 PDT

- Re-read `TASK.md`, rechecked the active local-MIL sprint, reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`, and then verified both pods directly with `kubectl`.
- The completed local-MIL checkpoint is now fully evaluated:
  - dense checkpoint:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_21_27_29`
  - dense `test accuracy = 0.7500863557858376`
  - tuned NEC complete:
    - `ACC@5 = 0.7187`
    - `ACC@10 = 0.7249`
    - `ACC@15 = 0.7271`
    - `ACC@20 = 0.7288`
    - `ACC@25 = 0.7301`
    - `ACC@30 = 0.7314`
    - `AVGACC = 0.7268`
  - native localization complete:
    - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
    - mean IoU `0.09073396545708716`
    - `mAP@0.3 = 0.9134870013023054`
    - `mAP@0.5 = 0.9319806619771949`
    - `mAP@0.7 = 0.9366440387587317`
    - point hit rate `0.9529931699477702`
    - point coverage `0.035814495589737684`
  - Grad-CAM localization complete:
    - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_localmil_lse_v1_full.json`
    - mean IoU `0.07988204531850018`
    - `mAP@0.3 = 0.0266030624693556`
    - `mAP@0.5 = 0.010985647975612694`
    - `mAP@0.7 = 0.006500362319831989`
    - point hit rate `0.729794933655006`
    - point coverage `0.011928572456365022`
- Interpretation:
  - this is the best dense-only `SAVLG` checkpoint so far
  - sparse performance still regresses versus the clean-global best complete `SAVLG` run
  - localization is mixed rather than clearly better than the best dual-branch soft-containment checkpoint, so local-MIL does not become the new default
- With both pods now idle, the next ranked controlled rerun became eligible:
  - dual-branch local-MIL with `cbl_type="mlp"` and `cbl_hidden_layers=1`
- Created:
  - `configs/unified/cub_savlg_cbm_clean_global_dualbranch_localmil_lse_mlp_h1_v1.json`
  - `savlg-research/2026-04-05--15-08-dual-branch-localmil-mlp/README.md`
- Verified the new config on `atharv-rwx-pod` and launched exactly one new dense run on A100 GPU `0`:
  - config:
    - `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_localmil_lse_mlp_h1_v1.json`
  - log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_mlp_h1_v1.log`
  - run dir:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_11_21`
  - launcher / main pid:
    - `11947 / 11948`
- Startup checks on the new run:
  - pod `py_compile` still passes
  - `args.txt` confirms `cbl_type=mlp`, `cbl_hidden_layers=1`, `savlg_branch_arch=dual`, and `savlg_use_local_mil=true`
  - sampled GPU state right after launch:
    - `GPU 0`: active (`937 MiB`, `27%`)
    - `GPU 1`: idle
  - startup log already advanced through `SAVLG CBL epoch 3`
- NEC remains deferred until the new dense result lands and the `0.74` gate can be rechecked.

### Dual-branch local-MIL MLP rerun complete at 15:18 PDT

- Re-read `TASK.md`, rechecked the active MLP sprint record plus `savlg-research/runtime_state.json`, reran `python3 -m py_compile methods/salf.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`, and then re-verified the live pods directly with `kubectl`.
- Both research pods are healthy and no restart recovery is pending:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- While the dense rerun was still active, `atharv-rwx-pod-2` was rechecked against the `TASK.md` restart checklist so it would be ready if the gate cleared:
  - `/workspace/SAVLGCBM`, annotations, and `datasets/CUB_200_2011` are present
  - conda env `cbm` is present
  - imports verified: `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, `pytorchcv`
  - version pins still match: `numpy 1.26.4`, `cv2 4.11.0`
  - `resnet18_cub` weights resolve successfully
- The dense MLP rerun then finished on `atharv-rwx-pod`:
  - checkpoint:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_11_21`
  - dense log:
    - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_localmil_lse_mlp_h1_v1.log`
  - final `test_metrics.json` accuracy:
    - `0.7347150259067358`
  - final log summary:
    - `SAVLG-CBM train accuracy=1.0000 val accuracy=1.0000 test accuracy=0.7347`
- Gate outcome:
  - `0.7347 < 0.74`, so NEC is not eligible for this sprint under `TASK.md`
  - no localization follow-up was launched because the controlled dense rerun already resolved negatively
- Interpretation:
  - the `mlp(h=1)` head is a clear regression relative to the matched dual-branch local-MIL linear-head run (`0.7501`)
  - this sprint is complete negative evidence and should not be promoted

### Multiscale spatial-branch live hold at 15:29 PDT

- Re-read `TASK.md`, re-inspected the sprint directories plus `savlg-research/runtime_state.json`, and rechecked both pods directly with `kubectl` before taking any action.
- Both research pods remain healthy and no restart recovery is needed:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The active multiscale dense run is still healthy on `atharv-rwx-pod`:
  - config:
    - `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1.json`
  - run dir:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
  - main pid:
    - `14221`
  - current visible state:
    - only `args.txt` and `train.log` are present
    - no final `test_metrics.json` yet
    - no sparse `metrics.csv` yet
  - current log progress:
    - concept-head training is advancing around `SAVLG CBL epoch 8`
    - latest completed summary seen:
      - `epoch=6 train_loss=0.163067 val_loss=0.267558 best_val=0.263469`
- Sampled GPU/process state:
  - `atharv-rwx-pod` GPU `0`: `NVIDIA A100-SXM4-80GB`, about `1053 MiB`, `37%` utilization
  - `atharv-rwx-pod` GPU `1`: idle
  - `atharv-rwx-pod-2`: no active `train_cbm.py`, `sparse_evaluation.py`, or localization-eval processes
  - `atharv-rwx-pod-2` currently exposes `2 x NVIDIA RTX 4000 Ada Generation`, both idle at this check
- No new launch, relaunch, recovery action, or kill was taken in this iteration:
  - NEC is still blocked until this checkpoint finishes and the dense `0.74` gate can be evaluated
  - localization follow-up for this checkpoint is not yet eligible because dense training is still in progress
  - launching anything else on the idle GPUs would have duplicated work or violated the current dense-first single-idea sprint

### Multiscale spatial-branch local-state hold at 15:36 PDT

- Re-read `TASK.md`, re-inspected the active multiscale sprint record plus `savlg-research/runtime_state.json`, and reran `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py`.
- Local static checks still pass.
- This sandbox still cannot refresh pod-visible state directly:
  - `/workspace/SAVLGCBM` is not mounted here
  - `ps` is blocked
  - `nvidia-smi` is unavailable
  - local `saved_models/` and `results/` pod artifacts for the active run are absent in this checkout
- The newest verified source of truth therefore remains the persisted multiscale hold already recorded in `savlg-research/2026-04-05--15-23-multiscale-spatial-branch/README.md`:
  - active run:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
  - latest verified artifact state:
    - no `test_metrics.json`
    - no `metrics.csv`
    - dense gate not yet cleared
- No new launch, relaunch, recovery action, or kill was taken in this iteration:
  - NEC remains blocked until a final dense result lands
  - localization remains blocked for this checkpoint until dense completion is known
  - no eligible safe non-duplicative action exists from this sandbox without fresher pod-visible evidence

### Multiscale dense completion and follow-up launches at 15:39 PDT

- Re-read `TASK.md`, re-inspected the active multiscale sprint record and `savlg-research/runtime_state.json`, and verified both pods directly with `kubectl` before taking any new action.
- The multiscale dense checkpoint is now complete:
  - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
  - dense test accuracy: `0.7506044905008635`
  - gate decision:
    - clears the `0.74` NEC threshold
    - slightly exceeds the previous best dense-only `SAVLG` checkpoint (`0.7500863557858376`)
    - exceeds the best complete clean-global `SAVLG` dense checkpoint (`0.7492`)
- Local and pod-side static checks passed again:
  - local:
    - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - pod:
    - `/opt/conda/envs/cbm/bin/python -m py_compile /workspace/SAVLGCBM/methods/salf.py /workspace/SAVLGCBM/methods/savlg.py /workspace/SAVLGCBM/evaluations/sparse_utils.py /workspace/SAVLGCBM/scripts/evaluate_native_spatial_maps.py /workspace/SAVLGCBM/train_cbm.py /workspace/SAVLGCBM/sparse_evaluation.py`
- `atharv-rwx-pod-2` launch prerequisites were re-verified:
  - `/workspace/SAVLGCBM`, `/workspace/SAVLGCBM/annotations`, and `/workspace/SAVLGCBM/datasets/CUB_200_2011` present
  - conda env `cbm` present
  - imports still pass for `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, and `pytorchcv`
- Before relaunching follow-ups, both pods were idle for research work:
  - `atharv-rwx-pod`: no active `train_cbm.py`, `sparse_evaluation.py`, or localization-eval processes; both A100s idle
  - `atharv-rwx-pod-2`: no active research processes; both RTX 4000 Ada GPUs idle
- Follow-up jobs launched from the dense-cleared checkpoint:
  - native-map localization on `atharv-rwx-pod` GPU `0`:
    - output `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
    - log `/workspace/SAVLGCBM/logs/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.log`
    - launcher/main pid pair `16456 / 16458`
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`:
    - output `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
    - log `/workspace/SAVLGCBM/logs/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.log`
    - launcher/main pid pair `16457 / 16459`
  - NEC on `atharv-rwx-pod-2` GPU `0`:
    - log `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_v1_nec150.log`
    - relaunched launcher/main pid pair `4855 / 4856`
- The first NEC launch exposed a real reload bug instead of a model failure:
  - `sparse_evaluation.py` crashed with unexpected `conv4_proj.weight` and `conv5_proj.weight` keys while reconstructing the multiscale SAVLG concept layer
  - fixed [evaluations/sparse_utils.py](/Users/atharvramesh/Projects/CBM/SAVLGCBM/evaluations/sparse_utils.py) so `sparsity_acc_test_savlg_cbm(...)` now rebuilds checkpoints with `build_savlg_concept_layer(args, backbone, len(concepts))`
  - copied the patched file to both pods and reran pod-side `py_compile`
- Startup evidence after the fix:
  - `atharv-rwx-pod` GPUs now show resident localization jobs:
    - GPU `0`: about `925 MiB`
    - GPU `1`: about `1055 MiB`
  - native log reached:
    - `SAVLG_Multiscale_Native dataloader ready: batches=181 batch_size=32 num_workers=8`
    - progress about `4/181`
  - Grad-CAM log reached:
    - `SAVLG_Multiscale_GradCAM dataloader ready: batches=724 batch_size=8 num_workers=8`
    - progress about `17/724`
  - `atharv-rwx-pod-2` GPU `0` now shows about `675 MiB` resident for NEC
  - NEC log progressed through about `21/169` concept-extraction batches after the loader fix
- The multiscale sprint is therefore now the best dense-only `SAVLG` checkpoint in the repo, but it remains in progress until NEC plus both localization outputs land.

### Multiscale live hold at 15:43 PDT

- Re-read `TASK.md`, rechecked the active multiscale sprint record, and reran:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Verified both pods directly again with `kubectl`; both remain healthy with no restart recovery needed:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- All currently eligible multiscale follow-up jobs are still active and writing forward progress:
  - native-map localization on `atharv-rwx-pod` GPU `0`
    - pid pair `16456 / 16458`
    - latest visible progress:
      - `SAVLG_Multiscale_Native map eval:  10%|█         | 19/181 [05:11<43:35, 16.15s/it]`
    - output still absent:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`
    - pid pair `16457 / 16459`
    - latest visible progress:
      - `SAVLG_Multiscale_GradCAM map eval:  11%|█         | 79/724 [05:23<44:49,  4.17s/it]`
    - output still absent:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
  - NEC on `atharv-rwx-pod-2` GPU `0`
    - pid pair `4855 / 4856`
    - concept extraction and the intermediate evaluation passes are complete
    - the `4000`-step sparse solver is the current active stage
    - final sparse artifact still absent:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28/metrics.csv`
- `atharv-rwx-pod-2` GPU `1` remains idle, but there is still no additional distinct non-duplicative launch that fits this dense-first single-idea sprint.
- No new launch, relaunch, recovery action, or kill was taken in this iteration, and `docs/cub_results.md` was not updated because no new finalized NEC or localization metrics landed during this check.

### Multiscale live hold and stale-sprint closure at 15:53 PDT

- Re-read `TASK.md`, re-inspected all sprint directories in `savlg-research/`, and reran:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Verified both pods directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The active multiscale follow-up jobs are still the only live research work:
  - native-map localization on `atharv-rwx-pod` GPU `0`
    - pid `16458`
    - latest visible progress:
      - `49/181`
    - output still absent:
      - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
  - Grad-CAM localization on `atharv-rwx-pod` GPU `1`
    - pid `16459`
    - latest visible progress:
      - `198/724`
    - output still absent:
      - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
  - NEC on `atharv-rwx-pod-2` GPU `0`
    - pid `4856`
    - latest visible checkpoint:
      - `(39) lambda 0.0007 ... [test acc 0.7420]`
    - `metrics.csv` still absent for:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
- `atharv-rwx-pod-2` GPU `1` is still idle, so I explicitly checked the two older stale “unfinished” NEC sprints for eligible backlog work.
- Recovered final completion evidence for those stale sprints:
  - `2026-04-03--20-50-mil-distill-ablation`
    - `w035` tuned NEC is complete
    - final summary:
      - `ACC@5 = 0.4770`
      - `AVGACC = 0.6815`
  - `2026-04-03--23-47-nec-path-scheduling`
    - `necpath_lam0005_s30` is complete
    - final summary:
      - `ACC@5 = 0.7033`
      - `AVGACC = 0.7145`
- Those older sprints have no remaining active jobs and no missing eligible launches, so the idle second Ada GPU does not have a policy-aligned backlog task to run in this iteration.
- No new launch, relaunch, recovery action, or kill was taken:
  - the current multiscale sprint already has all distinct eligible follow-up jobs in flight
  - older stale NEC sprints were closed instead of being left falsely “in progress”

### Multiscale sparse finalization at 16:02 PDT

- Re-read `TASK.md`, rechecked the active multiscale sprint, and reran:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Verified finalized multiscale classification artifacts:
  - dense checkpoint:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
  - dense test accuracy:
    - `0.7506044905008635`
  - sparse curve:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28/metrics.csv`
- Used the idle `atharv-rwx-pod-2` GPU `1` for an exact post-hoc sparse readout from the saved `W_g@NEC=*` heads so the finalized checkpoint could be documented without disturbing the still-running localization jobs.
- Exact sparse metrics for the multiscale checkpoint:
  - `ACC@5 = 0.7445595860481262`
  - `ACC@10 = 0.7481865286827087`
  - `ACC@15 = 0.7490500807762146`
  - `ACC@20 = 0.7485319375991821`
  - `ACC@25 = 0.7487046718597412`
  - `ACC@30 = 0.7488774061203003`
  - `AVGACC = 0.7479850351810455`
- Classification interpretation:
  - the multiscale checkpoint is now the best fully measured `SAVLG` classification result in this repo so far on all three main summary metrics:
    - dense `0.7506`
    - `ACC@5 0.7446`
    - `AVGACC 0.7480`
  - this is only a marginal sparse improvement over the earlier clean-global `SAVLG` frontier, so localization still decides whether the multiscale branch is a real promotion
  - the checkpoint still remains below the current `VLG-CBM` sparse frontier
- Live unfinished work remains only the two localization jobs on `atharv-rwx-pod`:
  - native main pid `16458`
    - latest visible progress:
      - `83/181`
  - Grad-CAM main pid `16459`
    - latest visible progress:
      - `334/724`
  - both output JSON files are still absent
- No launch, relaunch, or kill was taken after the exact sparse readout:
  - the localization jobs are healthy and already occupy the only unfinished eligible follow-up slots for this sprint

### Multiscale localization hold at 16:07 PDT

- Re-read `TASK.md`, rechecked the active multiscale sprint record, and reran:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Dense and sparse remain finalized for the multiscale checkpoint:
  - dense `0.7506044905008635`
  - `ACC@5 0.7445595860481262`
  - `AVGACC 0.7479850351810455`
- Verified live pod state directly again:
  - `atharv-rwx-pod`: `Running`, restarts `0`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- The only unfinished eligible work remains the two healthy localization jobs on `atharv-rwx-pod`:
  - native main pid `16458`
    - latest visible progress:
      - `112/181`
  - Grad-CAM main pid `16459`
    - latest visible progress:
      - `450/724`
  - both output JSON files are still absent
- `atharv-rwx-pod-2` is now fully idle, but there is still no additional distinct non-duplicative dense-first action eligible under `TASK.md`.
- No launch, relaunch, recovery action, or kill was taken in this iteration, and `DONE.txt` was not added because localization metrics are still missing.

### Multiscale localization relaunch at 16:18 PDT

- Re-read `TASK.md`, re-inspected the active sprint plus current docs, and reran:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Verified that the multiscale checkpoint is still classification-complete:
  - dense `0.7506044905008635`
  - `ACC@5 0.7445595860481262`
  - `AVGACC 0.7479850351810455`
- Verified live cluster state directly:
  - `atharv-rwx-pod`: `Completed`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Re-verified `atharv-rwx-pod-2` launch prerequisites before recovery:
  - `/workspace/SAVLGCBM` present
  - conda env `cbm` usable
  - imports pass for `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, and `pytorchcv`
  - `/workspace/SAVLGCBM/annotations` present
  - `/workspace/SAVLGCBM/datasets/CUB_200_2011` present
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28/test_metrics.json` present
- Confirmed the unfinished localization outputs were still absent on the shared PVC:
  - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
  - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
- Relaunched the two interrupted localization evals on `atharv-rwx-pod-2`, one per GPU:
  - native on GPU `0`
    - launcher/main pid pair `5484 / 5485`
    - log `/workspace/SAVLGCBM/logs/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_relaunch_pod2.log`
  - Grad-CAM on GPU `1`
    - launcher/main pid pair `5494 / 5495`
    - log `/workspace/SAVLGCBM/logs/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_relaunch_pod2.log`
- Startup evidence after relaunch:
  - GPU `0`: about `577 MiB / 20475 MiB`
  - GPU `1`: about `727 MiB / 20475 MiB`
  - native log reached `1/181`
  - Grad-CAM log reached `4/724`
- No additional launch was taken because these are the only distinct unfinished eligible follow-up jobs for the active multiscale sprint.

### Multiscale localization hold at 16:29 PDT

- Re-read `TASK.md`, rechecked the active multiscale sprint record, and verified live cluster state directly with `kubectl`.
- Current pod state:
  - `atharv-rwx-pod`: `Completed`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Dense and sparse remain finalized for the multiscale checkpoint:
  - dense `0.7506044905008635`
  - `ACC@5 0.7445595860481262`
  - `AVGACC 0.7479850351810455`
- The only unfinished eligible work is already running on `atharv-rwx-pod-2`, one localization eval per GPU:
  - native main pid `5485`
    - latest visible progress:
      - `63/181`
  - Grad-CAM main pid `5495`
    - latest visible progress:
      - `255/724`
  - both output JSON files are still absent
- Sampled pod-2 GPU memory at the same check:
  - GPU `0`: `577 MiB / 20475 MiB`
  - GPU `1`: `729 MiB / 20475 MiB`
- No new launch, relaunch, recovery action, or kill was taken in this iteration because both eligible GPUs were already occupied by the recovered localization jobs and no finalized localization artifact had landed yet.

### Multiscale local-sandbox hold at 16:30 PDT

- Re-read `TASK.md`, rechecked the active multiscale sprint, the persisted loop state, and the newest loop summaries before taking any action.
- Reran the required local static check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py`
  - result: passed
- This sandbox still cannot safely re-verify live pod state in the current iteration:
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
  - pod-mounted `saved_models/` and `results/` are not present locally
  - local `logs/` still only contain older pulled files from `2026-04-02`
- Because the latest verified persisted state already has the only unfinished eligible multiscale localization jobs recovered onto `atharv-rwx-pod-2`, no new launch, relaunch, recovery action, or kill was taken from this local-only view.

### Multiscale local-sandbox hold at 16:32 PDT

- Re-read `TASK.md`, re-inspected the existing sprint directories plus `savlg-research/runtime_state.json`, and rechecked the newest loop logs before taking any action.
- Reran the required local static check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py`
  - result: passed
- The local artifact view is still unchanged:
  - no local `saved_models/`
  - no local `results/`
  - local `logs/` still only contain older pulled files from `2026-04-02`
- This sandbox still cannot safely re-verify live pod state in the current iteration:
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because the latest persisted verified state already shows the only unfinished eligible multiscale localization jobs recovered onto `atharv-rwx-pod-2`, no new launch, relaunch, recovery action, or kill was taken from this local-only view.

### Multiscale local-sandbox hold at 16:33 PDT

- Re-read `TASK.md`, rechecked the active multiscale sprint record plus `savlg-research/runtime_state.json`, and rechecked the local artifact roots before taking any action.
- Reran the required local static check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- The local artifact view is still unchanged:
  - no local `saved_models/`
  - no local `results/`
  - local `logs/` still only contain older pulled files from `2026-04-02`
- This sandbox still cannot safely re-verify live pod state in the current iteration:
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because the latest persisted verified state already shows the only unfinished eligible multiscale localization jobs recovered onto `atharv-rwx-pod-2`, no new launch, relaunch, recovery action, or kill was taken from this local-only view.

### Multiscale local-sandbox hold at 16:36 PDT

- Re-read `TASK.md`, re-inspected the active multiscale sprint record plus `savlg-research/runtime_state.json`, and rechecked the local artifact roots before taking any action.
- Reran the required local static check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- The local artifact view is still unchanged:
  - no local `/workspace/SAVLGCBM`
  - no local `saved_models/`
  - no local `results/`
  - local `logs/` still only contain older pulled files from `2026-04-02`
- `savlg-research/runtime_state.json` still points to `iter-0094-20260405T233600Z` with status `running_iteration`
- This sandbox still cannot safely re-verify live pod state in the current iteration:
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because the latest persisted verified state already shows the only unfinished eligible multiscale localization jobs recovered onto `atharv-rwx-pod-2`, no new launch, relaunch, recovery action, or kill was taken from this local-only view.

### Multiscale local-sandbox hold at 16:37 PDT

- Re-read `TASK.md`, re-inspected the active multiscale sprint record plus `savlg-research/runtime_state.json`, and rechecked the local artifact roots before taking any action.
- Reran the required local static check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- The local artifact view is still unchanged:
  - no local `/workspace/SAVLGCBM`
  - no local `saved_models/`
  - no local `results/`
  - local `logs/` still only contain older pulled files from `2026-04-02`
- `savlg-research/runtime_state.json` now points to `iter-0095-20260405T233714Z` with status `running_iteration`
- This sandbox still cannot safely re-verify live pod state in the current iteration:
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because the latest persisted verified state already shows the only unfinished eligible multiscale localization jobs recovered onto `atharv-rwx-pod-2`, no new launch, relaunch, recovery action, or kill was taken from this local-only view.

### Multiscale local-sandbox hold at 16:39 PDT

- Re-read `TASK.md`, re-inspected the active multiscale sprint record plus `savlg-research/runtime_state.json`, and rechecked the local artifact roots before taking any action.
- Reran the required local static check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- The local artifact view is still unchanged:
  - no local `/workspace/SAVLGCBM`
  - no local `saved_models/`
  - no local `results/`
  - local `logs/` still only contain older pulled files from `2026-04-02`
- `savlg-research/runtime_state.json` now points to `iter-0096-20260405T233835Z` with status `running_iteration`
- This sandbox still cannot safely re-verify live pod state in the current iteration:
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
  - `kubectl` credential refresh fails because OIDC discovery cannot resolve `authentik.nrp-nautilus.io`
- Because the latest persisted verified state already shows the only unfinished eligible multiscale localization jobs recovered onto `atharv-rwx-pod-2`, no new launch, relaunch, recovery action, or kill was taken from this local-only view.

### Multiscale local-sandbox hold at 16:40 PDT

- Re-read `TASK.md`, re-inspected the active multiscale sprint record plus `savlg-research/runtime_state.json`, and rechecked the local artifact roots before taking any action.
- Reran the required local static check:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- The local artifact view is still unchanged:
  - no local `/workspace/SAVLGCBM`
  - no local `saved_models/`
  - no local `results/`
  - local `logs/` still only contain older pulled files from `2026-04-02`
- `savlg-research/runtime_state.json` now points to `iter-0097-20260405T234000Z` with status `running_iteration`
- This sandbox still cannot safely re-verify live pod state in the current iteration:
  - `ps` is sandbox-blocked
  - `pgrep` cannot read the process table
  - `nvidia-smi` is unavailable
  - `kubectl` credential refresh fails because OIDC discovery cannot resolve `authentik.nrp-nautilus.io`
- Because the latest persisted verified state already shows the only unfinished eligible multiscale localization jobs recovered onto `atharv-rwx-pod-2`, no new launch, relaunch, recovery action, or kill was taken from this local-only view.

### Multiscale live pod hold at 16:43 PDT

- Re-read `TASK.md`, re-inspected the active multiscale sprint record, and reran:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Verified live cluster state directly:
  - `atharv-rwx-pod`: `Completed`
  - `kubectl describe pod atharv-rwx-pod` reports `Reason: DeadlineExceeded`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Re-verified `atharv-rwx-pod-2` recovery prerequisites:
  - `/workspace/SAVLGCBM` present
  - conda env `cbm` usable
  - imports pass for `torch 2.11.0+cu130`, `torchvision 0.26.0+cu130`, `numpy 1.26.4`, `cv2 4.11.0`, `loguru 0.7.3`, `clip`, and `pytorchcv 0.0.74`
  - `/workspace/SAVLGCBM/annotations` present
  - `/workspace/SAVLGCBM/datasets/CUB_200_2011` present
- Verified the multiscale checkpoint is still classification-complete:
  - dense `0.7506044905008635`
  - `ACC@5 0.7445595860481262`
  - `AVGACC 0.7479850351810455`
- Confirmed the only unfinished artifacts remain the two localization outputs:
  - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
  - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
- Verified those are already the only active eligible jobs on `atharv-rwx-pod-2`, one per GPU:
  - native on GPU `0`
    - main pid `5485`
    - latest visible progress `147/181`
    - sampled memory about `577 MiB / 20475 MiB`
  - Grad-CAM on GPU `1`
    - main pid `5495`
    - latest visible progress `591/724`
    - sampled memory about `729 MiB / 20475 MiB`
- No new launch, relaunch, recovery action, or kill was taken in this iteration because both visible GPUs on the healthy pod already hold the only unfinished eligible multiscale follow-up jobs, and no finalized localization artifact has landed yet.

### Multiscale live pod hold at 16:45 PDT

- Re-read `TASK.md`, re-inspected the open multiscale sprint, and reran:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Verified live cluster state directly again:
  - `atharv-rwx-pod`: still `Completed`, with `kubectl describe` reporting `Reason: DeadlineExceeded`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
- Re-verified `atharv-rwx-pod-2` recovery prerequisites:
  - `/workspace/SAVLGCBM` present
  - conda env `cbm` usable
  - imports pass for `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, and `pytorchcv`
  - annotations, `CUB_200_2011`, and cached `resnet18_cub` weights present
- Confirmed the multiscale checkpoint remains classification-complete with no new finalized localization artifact:
  - dense `0.7506044905008635`
  - `ACC@5 0.7445595860481262`
  - `AVGACC 0.7479850351810455`
  - localization outputs still absent:
    - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
    - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
- Verified those are still the only active eligible jobs on `atharv-rwx-pod-2`, one per GPU:
  - native on GPU `0`
    - launcher/main pid pair `5484 / 5485`
    - latest visible progress `167/181`
    - sampled memory about `577 MiB / 20475 MiB`
  - Grad-CAM on GPU `1`
    - launcher/main pid pair `5494 / 5495`
    - latest visible progress `668/724`
    - sampled memory about `729 MiB / 20475 MiB`
- No new launch, relaunch, recovery action, or kill was taken in this iteration because both visible GPUs on the only healthy pod already hold the only unfinished eligible multiscale follow-up jobs, and no finalized localization artifact has landed yet.

### Multiscale sprint closed at 16:51 PDT

- Re-read `TASK.md`, reran the standard local static check, and re-verified live pod state with `kubectl`.
- Final pod state at closeout:
  - `atharv-rwx-pod`: `Completed` from `DeadlineExceeded`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
  - both recovered multiscale localization relaunches finished during this iteration
  - no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` processes remained afterward
- The multiscale checkpoint is now fully measured:
  - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_22_26_28`
  - dense `0.7506044905008635`
  - `ACC@5 0.7445595860481262`
  - `AVGACC 0.7479850351810455`
- Final localization outputs landed:
  - native:
    - `/workspace/SAVLGCBM/results/native_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full_meanthr_bs32_nw8_nocache.json`
    - mean IoU `0.1194`
    - `mAP@0.3 = 0.4773`
    - `mAP@0.5 = 0.4915`
    - `mAP@0.7 = 0.4975`
    - point hit / coverage `0.9708 / 0.2525`
  - Grad-CAM:
    - `/workspace/SAVLGCBM/results/gradcam_savlg_dualbranch_multiscale_conv45_localmil_lse_v1_full.json`
    - mean IoU `0.0796`
    - `mAP@0.3 = 0.0265`
    - `mAP@0.5 = 0.0110`
    - `mAP@0.7 = 0.00655`
    - point hit / coverage `0.7190 / 0.0110`
- Interpretation:
  - this becomes the best fully measured `SAVLG` classification checkpoint in the repo so far by a small margin over the earlier clean-global best
  - native-map localization improves materially and is the strongest recovered `SAVLG` localization evidence so far
  - Grad-CAM point localization remains weak, so the multiscale result is not yet a full `VLG`-level win
- Next-step decision:
  - carry this multiscale checkpoint forward as the new `SAVLG` reference
  - the next sprint should attack the remaining Grad-CAM / paper-style point-localization gap rather than re-running the same multiscale recipe

### Global-spatial consistency sprint launched at 16:58 PDT

- Re-read `TASK.md`, inspected all existing sprint directories plus the newest loop summaries, and verified live pod state directly before taking the next action.
- Verified the prior multiscale sprint is already complete, so no healthy research jobs remained to preserve:
  - `atharv-rwx-pod`: `Completed`
  - `atharv-rwx-pod-2`: `Running`, restarts `0`
  - `atharv-rwx-pod-2` had no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` processes at check time
  - both visible pod2 GPUs were idle
- Re-verified the healthy pod2 environment before relaunch:
  - `/workspace/SAVLGCBM`, annotations, and `datasets/CUB_200_2011` present
  - conda env `cbm` present
  - imports passed for `torch`, `torchvision`, `numpy`, `cv2`, `loguru`, `clip`, and `pytorchcv`
  - cached `resnet18_cub` weights present under `/root/.torch/models/`
- Chosen next single change:
  - weak detached spatial-to-global concept consistency on top of the completed multiscale dual-branch local-MIL recipe
  - goal: keep the strong native spatial localization while nudging the classifier-facing global branch away from the remaining Grad-CAM / point-localization diffusion
- Implemented locally:
  - added the new consistency loss and config support in `methods/savlg.py` and `train_cbm.py`
  - created `configs/unified/cub_savlg_cbm_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.json`
- Static checks passed locally and on pod:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
- Synced the changed files to `/workspace/SAVLGCBM/` on `atharv-rwx-pod-2`.
- Launched exactly one dense-first run on `atharv-rwx-pod-2` GPU `0`:
  - config: `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.json`
  - log: `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
  - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - launcher/main pid pair: `6538 / 6543`
- Startup evidence:
  - pod-side config values confirm the new consistency settings:
    - `savlg_global_spatial_consistency_w = 0.05`
    - `savlg_global_spatial_consistency_warmup_epochs = 5`
  - log confirms the new run directory and cached supervision reuse
  - concept-head training was already advancing through epoch `1` at roughly `59/169` train batches at the check
- No NEC or localization follow-up was launched yet because the dense gate is still pending for this new checkpoint.

### Global-spatial consistency local hold at 16:59 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the open consistency sprint README, and the newest loop logs before taking any action.
- Confirmed from persisted state that the current sprint already has exactly one dense-first run launched:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This local checkout does not expose the pod-side artifact tree needed to advance the sprint safely:
  - `saved_models/` is absent locally
  - `results/` is absent locally
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because the dense-first run was already launched in the prior iteration and no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Soft-align ablation launch recovery at 20:55 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, the active `softalign-outside-ablation` sprint, and the persisted loop state before taking action.
- Live pod access was restored on `atharv-rwx-pod-2`:
  - pod status `Running`, restarts `0`
  - both GPUs idle before launch
  - no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` processes
- The pod had restarted into a degraded state:
  - only conda `base` existed
  - `cbm` was missing
  - pod-side repo still lacked the current `soft_align` codepath and six ablation configs
- Rebuilt the environment and repo state on the pod:
  - `conda create -y -n cbm --clone base`
  - `pip install -r requirements.txt` inside `cbm`
  - synced `methods/common.py`, `methods/savlg.py`, `train_cbm.py`, `scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`, and the six warm-start frozen-global ablation configs into `/workspace/SAVLGCBM`
- Verified pod imports in `cbm` for:
  - `torch 2.2.2`
  - `torchvision 0.17.2`
  - `numpy 1.26.4`
  - `cv2 4.11.0`
  - `loguru 0.7.3`
  - `clip`
  - `pytorchcv 0.0.74`
  - `tensorboard`
- Validation passed:
  - local `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - pod `/opt/conda/envs/cbm/bin/python -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - local and pod `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
- Parallel dense launch exposed an operational bug:
  - both jobs initially chose the same second-stamped run directory
  - one launch failed with `FileExistsError`
- Patched the launch path:
  - `methods/common.py`
    - `build_run_dir()` now retries atomically with `-1`, `-2`, ... suffixes instead of racing on `os.path.exists()`
  - `scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
    - now uses `/opt/conda/envs/${ENV_NAME}/bin/python` directly instead of relying on shell activation
- Relaunched the dense-first ablations and kept both GPUs busy:
  - GPU `1`: `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_containment_only_v1.json`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_containment_only_v1.log`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_03`
  - GPU `0`: `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_localmil_only_v1.json`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_localmil_only_v1.log`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_15`
- Startup checks after relaunch:
  - both logs show VLG warm-start init for `671/671` concepts and frozen global head
  - both runs reached `SAVLG CBL epoch 1`
  - both had visible progress at check: `9/169` train batches
  - both GPUs were occupied at the last check: `733 MiB / 733 MiB`
- No NEC or localization jobs were launched because the new dense runs have not finished yet.
- `docs/cub_results.md` remained unchanged and no `DONE.txt` was added.

### Soft-align ablation local hold at 21:14 PDT

- Re-read `TASK.md`, inspected the existing `savlg-research/` sprint directories, the active `softalign-outside-ablation` README, `savlg-research/runtime_state.json`, and the newest persisted loop summaries before taking any action.
- Checked the visible local artifact roots to avoid duplicating work:
  - local `saved_models/` is absent
  - local `logs/` does not contain the active soft-align dense logs
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- The newest persisted safe state is `savlg-research/loop-logs/iter-0007-20260407T040600Z.last.txt`, which already records:
  - dense `containment_only_v1` and `localmil_only_v1` finished at test accuracy `0.7590673575129534`
  - the next two allowed dense-first ablations were launched to keep both GPUs busy:
    - `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_only_v1.json`
    - `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_v1.json`
- Because this checkout cannot directly verify whether those pod-side jobs are still running or finished, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because there are still no newly visible finalized `ACC@5` / `AVGACC` / localization artifacts in this filesystem.

### Global-spatial consistency local hold at 17:01 PDT

- Re-read `TASK.md`, re-inspected the existing sprint directories in `savlg-research/`, `savlg-research/runtime_state.json`, the active consistency sprint README, and the newest loop logs before taking any action.
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` does not contain the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.
### Soft-align outside dense queue advanced at 21:23 PDT

- Latest sprint state this iteration:
  - verified the first four dense ablations are complete on `atharv-rwx-pod-2`, each at dense test accuracy `0.7590673575129534`
  - launched the final two allowed dense ablations:
    - `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_outside_v1.json`
    - `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1.json`
  - both new runs initialized from `saved_models/cub/cub_cbm_2026_04_06_06_16_10` and advanced into `SAVLG CBL epoch 1`
- NEC and localization were intentionally not launched in this iteration because `TASK.md` still required finishing the dense-first queue.
- The detailed run-dir / PID / log record for this 21:23 PDT step is captured in `savlg-research/2026-04-06--20-45-softalign-outside-ablation/README.md`.

### Soft-align outside local hold at 21:32 PDT

- Re-read `TASK.md`, re-inspected the active sprint under `savlg-research/`, `savlg-research/runtime_state.json`, the newest visible loop summaries, and the visible local artifact roots before taking any action.
- Confirmed that the newest visible completed loop summary is still `savlg-research/loop-logs/iter-0011-20260407T042845Z.last.txt`, which already records the final two allowed dense ablations as healthy in progress.
- This checkout still cannot verify live pod/process/GPU state or pod-side artifact completion directly:
  - local `saved_models/` is absent
  - local `logs/` does not contain the active soft-align-outside dense logs
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Reran the required local static validation:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Because no newer dense completion artifact is visible here, no new safe action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no newly visible finalized dense / sparse / localization result is present in this filesystem.

### Soft-align outside direct pod hold at 21:30 PDT

- Re-read `TASK.md`, re-inspected the active soft-align sprint state, and directly verified pod health on `atharv-rwx-pod-2`.
- Pod status remains healthy:
  - `Running`
  - restarts `0`
  - `cbm` env present at `/opt/conda/envs/cbm`
- Confirmed the first four dense ablations remain complete at dense test accuracy `0.7590673575129534`:
  - `containment_only_v1`
  - `localmil_only_v1`
  - `softalign_only_v1`
  - `softalign_localmil_v1`
- Confirmed both GPUs are still occupied by the final two allowed dense jobs:
  - GPU `0`: `softalign_outside_v1`, `35%` util, `831 MiB / 23028 MiB`
  - GPU `1`: `softalign_localmil_outside_v1`, `47%` util, `831 MiB / 23028 MiB`
- Current pod-side progress is healthy and advancing:
  - `softalign_outside_v1` finished concept extraction and entered sparse readout optimization (`17/4000`)
  - `softalign_localmil_outside_v1` completed dense training through epoch `22` and entered concept extraction
- No new action was eligible under `TASK.md`:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because the final two dense runs have not yet produced finalized dense / sparse / localization outputs.

### Soft-align outside local hold at 21:27 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the active soft-align sprint README, and the newest persisted loop summaries before taking any action.
- Reran the required local static check:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Confirmed the newest visible persisted summary is still `savlg-research/loop-logs/iter-0009-20260407T042137Z.last.txt`, which records:
  - the first four dense ablations completed at dense test accuracy `0.7590673575129534`
  - the remaining two dense jobs were launched:
    - `cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_outside_v1`
    - `cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1`
- This checkout still does not expose the live pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `logs/` does not contain the active soft-align-outside logs
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because the currently running-or-recently-launched final dense jobs cannot be verified from this filesystem, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no newly visible finalized dense / sparse / localization result is present in this checkout.

### Global-spatial consistency local hold at 17:03 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the active consistency sprint README, and the newest loop logs before taking any action.
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` does not contain the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Soft-align outside dense queue advanced at 21:23 PDT

- Re-read `TASK.md`, re-inspected the active `savlg-research/2026-04-06--20-45-softalign-outside-ablation/README.md`, `savlg-research/runtime_state.json`, and the existing sprint tree before taking action.
- Verified live pod state directly on `atharv-rwx-pod-2`:
  - pod `Running`, restarts `0`
  - both GPUs idle before launch
  - no healthy active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` jobs
- Verified that the first four dense ablations are now complete on the pod, all at dense test accuracy `0.7590673575129534`:
  - `containment_only_v1`: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_03`
  - `localmil_only_v1`: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_15`
  - `softalign_only_v1`: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57`
  - `softalign_localmil_v1`: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_07_57-1`
- These dense-only results preserve the `TASK.md` baseline closely enough to keep the sprint moving, but `ACC@5` / `AVGACC` remain unavailable because NEC has not run for these checkpoints yet.
- Launched the remaining two allowed dense ablations to keep both GPUs busy:
  - GPU `0`: `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_outside_v1.json`
    - launcher/main pid pair: `131841 / 131842`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_outside_v1.log`
  - GPU `1`: `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1.json`
    - launcher/main pid pair: `131845 / 131846`
    - run dir: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13-1`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1.log`
- Verified startup from the pod logs:
  - both runs initialized from `saved_models/cub/cub_cbm_2026_04_06_06_16_10`
  - both runs froze the global head
  - both logs advanced into `SAVLG CBL epoch 1` and reached `51/169`
- No NEC or localization action was taken in this iteration because `TASK.md` says dense first and the remaining dense queue was still eligible at launch time.
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because this iteration did not produce a full `dense + ACC@5 + AVGACC + localization` report.

### Global-spatial consistency local hold at 17:06 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the active consistency sprint README, and the newest loop logs before taking any action.
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still only contains older pulled files and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:07 PDT

- Re-read `TASK.md`, confirmed the only unfinished sprint under `savlg-research/` is the active consistency sprint, and re-inspected `savlg-research/runtime_state.json` plus the newest loop summaries before taking any action.
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older pulled files and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:09 PDT

- Re-read `TASK.md`, re-inspected the existing sprint directories in `savlg-research`, `savlg-research/runtime_state.json`, and the newest loop logs before taking any action.
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older pulled files and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:13 PDT

- Re-read `TASK.md`, re-inspected the existing sprint directories in `savlg-research/`, `savlg-research/runtime_state.json`, and the newest loop log for the in-progress consistency sprint before taking any action.
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local filesystem only exposes `./logs/` and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:16 PDT

- Re-read `TASK.md`, re-inspected the active consistency sprint README, `savlg-research/runtime_state.json`, the newest loop logs (`iter-0105` through `iter-0109`), and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still only contains older pulled files and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:18 PDT

- Re-read `TASK.md`, confirmed the only unfinished sprint under `savlg-research/` is the active consistency sprint, and re-inspected `savlg-research/runtime_state.json`, loop summaries `iter-0107` through `iter-0109`, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older pulled files and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:20 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the active consistency sprint README, and the newest visible loop logs before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older pulled files and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:21 PDT

- Re-read `TASK.md`, re-inspected the existing sprint directories under `savlg-research/`, `savlg-research/runtime_state.json`, the active consistency sprint README, and the newest loop summaries before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older pulled files from `2026-04-02` and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:22 PDT

- Re-read `TASK.md`, confirmed the only unfinished sprint under `savlg-research/` is the active consistency sprint, and re-inspected `savlg-research/runtime_state.json`, the newest loop log, the sprint README, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older pulled files and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:24 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the active consistency sprint README, and the newest loop logs before taking any action.
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is empty
  - local `logs/` still contains only older pulled files from `2026-04-02` and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Global-spatial consistency local hold at 17:25 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the active consistency sprint README, the newest loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/salf.py methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py sparse_evaluation.py`
  - result: passed
- Confirmed from persisted state that the current sprint still has only the previously launched dense-first run recorded:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_05_23_57_50`
  - `/workspace/SAVLGCBM/logs/cub_savlg_clean_global_dualbranch_multiscale_conv45_localmil_lse_consistency_v1.log`
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still contains only older pulled files from `2026-04-02` and not the active consistency-run log
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.
 
### Soft-align outside dense queue advanced at 21:23 PDT

- Latest sprint state this iteration:
  - verified the first four dense ablations are complete on `atharv-rwx-pod-2`, each at dense test accuracy `0.7590673575129534`
  - launched the final two allowed dense ablations:
    - `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_outside_v1.json`
    - `configs/unified/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1.json`
  - both new runs initialized from `saved_models/cub/cub_cbm_2026_04_06_06_16_10` and advanced into `SAVLG CBL epoch 1`
- NEC and localization were intentionally not launched in this iteration because `TASK.md` still required finishing the dense-first queue.
- The detailed run-dir / PID / log record for this 21:23 PDT step is captured in `savlg-research/2026-04-06--20-45-softalign-outside-ablation/README.md`.

### Soft-align outside local hold at 21:34 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the active soft-align sprint README, and the newest persisted loop summaries before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Confirmed from persisted state that the latest trustworthy dense-queue status is still:
  - iteration `0009` launched the final two allowed dense ablations
  - iteration `0011` recorded both as healthy and occupying the two GPUs
  - iteration `0012` held because this sandbox could not directly verify pod state
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `logs/` still contains only older pulled files from `2026-04-02` and not the active soft-align-outside logs
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no new finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Soft-align outside local hold at 21:36 PDT

- Re-read `TASK.md`, re-inspected the active soft-align sprint README, `savlg-research/runtime_state.json`, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Confirmed from persisted state that the latest trustworthy dense-queue status is still:
  - iteration `0009` launched the final two allowed dense ablations
  - iteration `0011` recorded both as healthy and occupying the two GPUs
  - iterations `0012` and `0013` held because this sandbox could not directly verify pod state
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still does not contain the active soft-align-outside logs
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no new finalized dense artifact is visible here, no new action was eligible:
  - no duplicate dense launch
  - no NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized dense / sparse / localization result is present in this filesystem.

### Soft-align outside NEC launched at 21:39 PDT

- Re-read `TASK.md`, rechecked the active soft-align sprint record, the persisted loop state, and the existing sprint tree before taking action.
- Reran the required local static validation:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Direct pod verification on `atharv-rwx-pod-2` showed:
  - pod `Running`, restarts `0`
  - no active `train_cbm.py`, `sparse_evaluation.py`, or localization jobs
  - both GPUs idle before launch
- Fully verified the dense queue is now complete for all six allowed ablations:
  - every dense checkpoint currently reports test accuracy `0.7590673575129534`
  - the final two dense runs are:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13`
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13-1`
  - no ablation checkpoint has `metrics.csv` yet, so `ACC@5` and `AVGACC` are still missing
- Because the dense-first gate is now satisfied and the pod was idle, this iteration launched the next two sparse follow-ups on the outside-penalty variants:
  - GPU `0`:
    - `/opt/conda/envs/cbm/bin/python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13 --lam 0.01 --max_glm_steps 150`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_outside_v1_nec150.log`
    - launcher / main pid pair: `196821 / 196822`
  - GPU `1`:
    - `/opt/conda/envs/cbm/bin/python -u sparse_evaluation.py --load_path /workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13-1 --lam 0.01 --max_glm_steps 150`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1_nec150.log`
    - launcher / main pid pair: `196831 / 196832`
- Immediate post-launch verification:
  - both main NEC workers are `Rl`
  - sampled GPU usage is back up:
    - GPU `0`: `709 MiB`, `40%`
    - GPU `1`: `709 MiB`, `20%`
  - both logs are already progressing through the initial `169`-batch pass
- No localization follow-up was launched in this iteration because the new outside-penalty checkpoints still need their first `ACC@5` / `AVGACC` results before promotion.
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized sparse or localization artifact landed in this iteration.

### Soft-align outside local hold at 21:42 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, `savlg-research/runtime_state.json`, the active soft-align sprint README, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Confirmed from persisted state that the latest trustworthy status is now:
  - iteration `0015` verified all six allowed dense ablations complete at dense test accuracy `0.7590673575129534`
  - iteration `0015` launched `NEC150` on:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13`
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13-1`
  - that summary also recorded both NEC workers active and both GPUs occupied at launch verification time
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still does not contain the active soft-align-outside NEC logs
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no newer finalized sparse or localization artifact is visible here, no new action was eligible:
  - no duplicate NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized `ACC@5` / `AVGACC` / localization result is present in this filesystem.

### Soft-align outside local hold at 21:59 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, the active soft-align sprint README, `savlg-research/runtime_state.json`, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
  - result: passed
- Confirmed from persisted state that the newest trustworthy completion summary is now `iter-0020-20260407T045003Z.last.txt`, which means:
  - all six allowed dense warm-start local-loss ablations are already complete at dense test accuracy `0.7590673575129534`
  - the two outside-penalty checkpoints already have tuned `NEC150` results at `ACC@5 = 0.7549` and `AVGACC = 0.7560`
  - the current active pod-side follow-up is native `gt_present` localization on the two outside-penalty dense checkpoints
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still does not contain the active native-localization logs
  - `/workspace/SAVLGCBM` is not present here
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no newer finalized localization artifact is visible here, no new action was eligible:
  - no duplicate localization launch
  - no duplicate NEC launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized native GT-present result is present in this filesystem.

### Soft-align outside local hold at 21:44 PDT

- Re-read `TASK.md`, re-inspected the full `savlg-research/` sprint tree, the active soft-align sprint README, `savlg-research/runtime_state.json`, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Confirmed from persisted state that the newest visible completion summary is still `iter-0016-20260407T044159Z.last.txt`, which means:
  - the last verified pod-side action remains iteration `0015` launching `NEC150` on the two outside-penalty checkpoints
  - subsequent iterations correctly held because this sandbox cannot directly verify live pod state
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still does not contain the active soft-align-outside NEC logs
  - `/workspace/SAVLGCBM` is not present here
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no newer finalized sparse or localization artifact is visible here, no new action was eligible:
  - no duplicate NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized `ACC@5` / `AVGACC` / localization result is present in this filesystem.

### Soft-align outside local hold at 21:46 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, the active soft-align sprint README, `savlg-research/runtime_state.json`, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Confirmed from persisted state that the newest trustworthy completion summary is still `iter-0017-20260407T044355Z.last.txt`, which means:
  - the last verified pod-side action remains iteration `0015` launching `NEC150` on the two outside-penalty checkpoints
  - later iterations correctly held because this sandbox cannot directly verify live pod/process/GPU state or read pod-side completion artifacts
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still does not contain the active soft-align-outside NEC logs
  - `/workspace/SAVLGCBM` is not present here
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no newer finalized sparse or localization artifact is visible here, no new action was eligible:
  - no duplicate NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized `ACC@5` / `AVGACC` / localization result is present in this filesystem.

### Soft-align outside local hold at 21:49 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, the active soft-align sprint README, `savlg-research/runtime_state.json`, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - result: passed
- Confirmed from persisted state that the newest visible completion summary is now `iter-0018-20260407T044521Z.last.txt`, which means:
  - the last verified pod-side action still remains iteration `0015` launching `NEC150` on the two outside-penalty checkpoints
  - later iterations correctly held because this sandbox cannot directly verify live pod/process/GPU state or read pod-side completion artifacts
  - `savlg-research/runtime_state.json` now points at iteration `19`, but the corresponding `.last.txt` completion artifact is not yet present in this checkout
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` still does not contain the active soft-align-outside NEC logs
  - `/workspace/SAVLGCBM` is not present here
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no newer finalized sparse or localization artifact is visible here, no new action was eligible:
  - no duplicate NEC launch
  - no localization launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized `ACC@5` / `AVGACC` / localization result is present in this filesystem.

### Soft-align outside local hold at 22:01 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, the active soft-align sprint README, `savlg-research/runtime_state.json`, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
  - result: passed
- Confirmed from persisted state that the newest trustworthy pod-side summary remains `iter-0020-20260407T045003Z.last.txt`, which means:
  - all six allowed dense warm-start local-loss ablations are already complete at dense test accuracy `0.7590673575129534`
  - the two outside-penalty checkpoints already have tuned `NEC150` results at `ACC@5 = 0.7549` and `AVGACC = 0.7560`
  - native `gt_present` localization remains the active pod-side follow-up for the two outside-penalty dense checkpoints
- This checkout still does not expose the pod-side artifact tree needed to advance the sprint safely:
  - local `saved_models/` is absent
  - local `results/` is absent
  - local `logs/` does not contain the active native-localization logs
  - `/workspace/SAVLGCBM` is not present here
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
- Because no newer finalized localization artifact is visible here, no new action was eligible:
  - no duplicate localization launch
  - no duplicate NEC launch
  - no kill or restart
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no new finalized native GT-present result is present in this filesystem.

### Soft-align outside localization health check at 22:03 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, the active soft-align sprint README, `savlg-research/runtime_state.json`, and the newest persisted loop summaries before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
  - result: passed
- Switched from inferred local state to a direct pod check on `atharv-rwx-pod-2`:
  - pod status remains `Running`
  - restarts remain `0`
  - `nvidia-smi` shows both GPUs occupied only by the two native localization workers:
    - GPU `0`: pid `201170`, `709 MiB / 23028 MiB`
    - GPU `1`: pid `201171`, `709 MiB / 23028 MiB`
- Verified both active workers are the expected `evaluate_native_spatial_maps.py` jobs for the two dense+NEC-qualified outside-penalty checkpoints:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13`
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13-1`
- Checked the live pod logs for both localization jobs:
  - both logs advanced to `69/181`
  - neither job has emitted a final output JSON yet
- Verified dense artifacts are still present for both checkpoints (`args.txt`, `metrics.csv`, `test_metrics.json`, `W_g.pt`), but the localization outputs are still missing:
  - `results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
  - `results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
- No new action was eligible in this iteration:
  - no duplicate localization launch because both GPUs are already busy with the only eligible follow-up jobs
  - no NEC launch because the decisive phase is already localization
  - no kill or restart because both jobs are healthy
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no finalized native GT-present localization artifact is available yet.

### Soft-align outside localization health check at 22:10 PDT

- Re-read `TASK.md`, re-inspected the active soft-align sprint README, `savlg-research/runtime_state.json`, the newest loop summaries, and the current pod-visible artifact roots before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
  - result: passed
- Re-verified the live pod state on `atharv-rwx-pod-2`:
  - pod status remains `Running`
  - restarts remain `0`
  - the only active eligible jobs are still the two native `gt_present` localization evals for:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13`
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13-1`
  - active evaluator roots remain visible as pid `201170` and pid `201171`
- Verified that the expected localization outputs are still not finalized:
  - `results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still missing
  - `results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still missing
- Checked live log progress for both jobs:
  - outside-only native eval advanced from `69/181` to `119/181`
  - `soft_align + local_mil + outside penalty` native eval advanced from `69/181` to `119/181`
  - no traceback, OOM, or explicit error text is present in either log
- No new action was eligible in this iteration:
  - no duplicate localization launch because both GPUs remain occupied by the only eligible follow-up jobs
  - no dense or NEC launch because the dense gate is already resolved and localization is still in progress
  - no kill or restart because both jobs remain healthy
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no finalized native GT-present localization artifact is available yet.

### Soft-align outside localization health check at 22:13 PDT

- Re-read `TASK.md`, re-inspected `savlg-research/`, the active soft-align sprint README, `savlg-research/runtime_state.json`, and the newest persisted loop summaries before taking any action.
- Reran the required local static checks:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
  - result: passed
- Re-verified the live pod state on `atharv-rwx-pod-2`:
  - pod status remains `Running`
  - restarts remain `0`
  - the only active eligible jobs are still the two native `gt_present` localization evals for:
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13`
    - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_04_23_13-1`
  - active evaluator roots remain visible as pid `201170` and pid `201171`
  - `nvidia-smi` still reports both GPUs reserved by those workers at `709 MiB / 23028 MiB`
- Verified that the expected localization outputs are still not finalized:
  - `results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still missing
  - `results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json` is still missing
- Checked live health rather than trusting the instantaneous `0%` GPU-util snapshot:
  - both evaluator roots stayed in `R` state
  - both logs advanced from `137/181` to `141/181` over a 25-second recheck window
  - no traceback or OOM text is present in either log
- No new action was eligible in this iteration:
  - no duplicate localization launch because both GPUs remain occupied by the only eligible follow-up jobs
  - no dense or NEC launch because the dense gate is already resolved and localization is still in progress
  - no kill or restart because the jobs are still making forward progress
- `docs/cub_results.md` was left unchanged and no `DONE.txt` was added because no finalized native GT-present localization artifact is available yet.

### Soft-align outside localization completed at 22:22 PDT

- Re-read `TASK.md`, rechecked the active soft-align sprint state, and verified the live pod directly on `atharv-rwx-pod-2`.
- The two outside-penalty native `gt_present` evals both completed and freed the GPUs:
  - `soft-align + outside penalty`
    - output: `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - mean IoU `0.12455332332837685`
    - `mAP@0.3 = 0.04907096879397959`
    - `mAP@0.5 = 0.033569865976427096`
    - `mAP@0.7 = 0.023689953026913697`
    - point hit `0.9750421585160203`
    - coverage `0.08532742420536138`
  - `soft-align + local-mil + outside penalty`
    - output: `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_softalign_localmil_outside_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - mean IoU `0.1255463541960814`
    - `mAP@0.3 = 0.031103271337829037`
    - `mAP@0.5 = 0.025754280184576945`
    - `mAP@0.7 = 0.022404031878906265`
    - point hit `0.9838308457711443`
    - coverage `0.0694130681899938`
- Both outside-penalty variants still preserve dense `0.7590673575129534` and sparse `ACC@5 = 0.7549`, `AVGACC = 0.7560`, and both improve on the earlier warm-start proof-of-concept native localization metrics.
- With both GPUs idle again, the next eligible non-duplicative work was the missing sparse follow-up for the no-outside `soft_align` pair, so both were launched immediately:
  - GPU `0`: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_only_v1_nec150.log`
    - launcher / main pid pair: `202872 / 202874`
  - GPU `1`: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_softalign_localmil_v1_nec150.log`
    - launcher / main pid pair: `202883 / 202885`
- Post-launch verification:
  - both GPUs are occupied again at about `983 MiB / 23028 MiB`
  - both logs advanced through feature extraction and into the sparse solver
- `DONE.txt` was not added because the sprint still has active sparse comparisons in flight.

### Soft-align outside hold at 22:34 PDT

- Re-read `TASK.md`, re-inspected the active sprint README, `savlg-research/runtime_state.json`, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local validation:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
  - result: passed
- The newest persisted completion summary is now `savlg-research/loop-logs/iter-0030-20260407T053230Z.last.txt`.
- The newest persisted direct pod evidence still remains the `2026-04-06 22:31 PDT` check for the active sprint:
  - pod `atharv-rwx-pod-2` was `Running` with `0` restarts
  - both GPUs were occupied by healthy eligible no-outside sparse jobs:
    - `softalign_only_v1_nec150`
    - `softalign_localmil_v1_nec150`
- This shell still cannot obtain a fresher trustworthy live view:
  - `ps` is blocked
  - `nvidia-smi` is unavailable
  - local `logs/` still does not include the active `softalign*_nec150` logs
  - local `results/` and `saved_models/cub/` are absent
- No new action was eligible in this iteration:
  - no duplicate dense / NEC / localization launch
  - no kill or restart
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no newly visible finalized sparse or localization artifact is present in this filesystem snapshot.

### Soft-align outside hold at 22:36 PDT

- Re-read `TASK.md`, re-inspected the active sprint README, `savlg-research/runtime_state.json`, the newest persisted loop summaries, and the visible local artifact roots before taking any action.
- Reran the required local validation:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
  - result: passed
- The newest trustworthy direct pod evidence available in this checkout still remains the `2026-04-06 22:31 PDT` note already captured in the active sprint:
  - pod `atharv-rwx-pod-2` was `Running` with `0` restarts
  - both GPUs were occupied by healthy eligible no-outside sparse jobs:
    - `softalign_only_v1_nec150`
    - `softalign_localmil_v1_nec150`
- The newest persisted completion summary remains `savlg-research/loop-logs/iter-0031-20260407T053402Z.last.txt`, and it reaches the same hold decision from the same evidence.
- This shell still cannot obtain a fresher trustworthy live view:
  - `ps` is blocked with `operation not permitted`
  - `pgrep` cannot obtain the process list
  - `nvidia-smi` is unavailable
  - local `logs/` still does not include the active `softalign*_nec150` logs
  - local `results/` and `saved_models/cub/` are absent
- No new action was eligible in this iteration:
  - no duplicate dense / NEC / localization launch
  - no kill or restart
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no newly visible finalized sparse or localization artifact is present in this filesystem snapshot.

### Soft-align controls sparse launch at 23:04 PDT

- Re-read `TASK.md`, re-inspected the active soft-align sprint state, and checked the live pod directly on `atharv-rwx-pod-2` before taking any new action.
- Verified the current safe live state:
  - pod `atharv-rwx-pod-2` is `Running`
  - restarts remain `0`
  - no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` jobs were present before launch
  - both GPUs were idle at `0 MiB / 23028 MiB`
  - env / path checks passed for:
    - `/opt/conda/envs/cbm/bin/python`
    - `/workspace/SAVLGCBM/annotations`
    - `/workspace/SAVLGCBM/datasets/CUB_200_2011`
- Confirmed the remaining gap in the allowed six-run ablation set:
  - `containment_only_v1` and `localmil_only_v1` each had only dense artifacts
  - neither had an `NEC150` log yet
  - neither had a native `gt_present` localization result yet
- Also confirmed that the no-outside soft-align localization pair is now finalized on pod:
  - `soft-align only`
    - mean IoU `0.124099317393116`
    - `mAP@0.3 = 0.04719790371291147`
    - `mAP@0.5 = 0.032559621861637134`
    - `mAP@0.7 = 0.023469499207264884`
    - point hit `0.9763696831787152`
    - coverage `0.08220498726563737`
  - `soft-align + local-mil`
    - mean IoU `0.1250153356970898`
    - `mAP@0.3 = 0.0296853607373322`
    - `mAP@0.5 = 0.02518812710654583`
    - `mAP@0.7 = 0.02234245708810698`
    - point hit `0.9831700042607584`
    - coverage `0.06754248384822366`
- With both GPUs idle and dense already preserved at `0.7590673575129534` for the two control ablations, the next eligible dense-gated action was sparse follow-up, so both `NEC150` jobs were launched:
  - GPU `0`
    - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_03`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_containment_only_v1_nec150.log`
    - launcher / main pid pair: `208774 / 208778`
  - GPU `1`
    - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_15`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_frozenglobal_multiscale_localmil_only_v1_nec150.log`
    - launcher / main pid pair: `208775 / 208779`
- Post-launch verification:
  - GPU `0`: `52%`, `711 MiB / 23028 MiB`
  - GPU `1`: `47%`, `711 MiB / 23028 MiB`
  - both logs advanced through feature extraction and into the test pass
- `DONE.txt` was not added because the sprint still needs finalized sparse outputs for these two controls, and then any required native localization follow-up.

### Soft-align controls localization launch at 23:16 PDT

- Re-read `TASK.md`, re-inspected the active soft-align sprint state, and checked the live pod directly on `atharv-rwx-pod-2` before taking any new action.
- Verified the current safe live state:
  - pod `atharv-rwx-pod-2` is `Running`
  - restarts remain `0`
  - no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` jobs were present before launch
  - both GPUs were idle at `0 MiB / 23028 MiB`
  - env / path checks still pass for:
    - `/opt/conda/envs/cbm/bin/python`
    - `/workspace/SAVLGCBM/annotations`
    - `/workspace/SAVLGCBM/datasets/CUB_200_2011`
- Confirmed the remaining gap in the allowed six-run ablation set:
  - `containment_only_v1` now has dense `0.7590673575129534`, `metrics.csv`, and finalized sparse `ACC@5 = 0.7549`, `AVGACC = 0.7560`
  - `localmil_only_v1` now has dense `0.7590673575129534`, `metrics.csv`, and finalized sparse `ACC@5 = 0.7549`, `AVGACC = 0.7560`
  - neither control checkpoint had a native `gt_present` localization result yet
- With both GPUs idle and every dense-qualified checkpoint now through sparse recovery, the next eligible dense-gated action was the missing control native-localization pair, so both jobs were launched:
  - GPU `0`
    - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_03`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_containment_only_v1_full_meanthr_bs32_nw8_gtpresent.log`
    - launcher / main pid pair: `213056 / 213058`
  - GPU `1`
    - checkpoint: `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_03_54_15`
    - log: `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_localmil_only_v1_full_meanthr_bs32_nw8_gtpresent.log`
    - launcher / main pid pair: `213067 / 213069`
- Post-launch verification:
  - both main eval processes are healthy with worker subprocesses attached in the pod process table
  - both GPUs are occupied again at `709 MiB / 23028 MiB`
  - both logs advanced into `savlg_cbm map eval` and reached `3/181` without traceback or OOM text
- Local validation passed again:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
- `docs/cub_results.md` was updated to reflect that sparse follow-up is now complete for all six variants and that the active work is the two control native `gt_present` evals.
- `DONE.txt` was not added because the sprint still needs finalized native-localization outputs for those two controls.

### Local-mil control localization completed at 23:27 PDT

- Re-read `TASK.md`, re-inspected the active soft-align sprint state, and verified the live pod directly on `atharv-rwx-pod-2` before taking any action.
- Local validation passed again:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
- Verified the live pod state:
  - pod `atharv-rwx-pod-2` is still `Running`
  - restarts remain `0`
  - GPU `0` is still occupied by the remaining containment-control native `gt_present` eval:
    - pid `213058`
    - `/workspace/SAVLGCBM/logs/native_savlg_vlgwarm_frozenglobal_multiscale_containment_only_v1_full_meanthr_bs32_nw8_gtpresent.log`
  - GPU `1` is now idle because the matching `localmil_only_v1` control native eval has finished and there is no second eligible non-duplicate launch left under the six-run `TASK.md` ablation set
- New finalized control-localization evidence landed:
  - `local-mil only`
    - output: `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_localmil_only_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - mean IoU `0.06317313391705029`
    - `mAP@0.3 = 0.024668750019905216`
    - `mAP@0.5 = 0.02298420380038701`
    - `mAP@0.7 = 0.022159182919480994`
    - point hit `0.8277669902912621`
    - coverage `0.07410391815474049`
- Live hold reason after this completion:
  - the remaining containment-control native eval is still progressing and has advanced to `145/181`
  - all six allowed dense runs are already complete
  - all six dense-qualified sparse `NEC150` follow-ups are already complete
  - five of the six native `gt_present` localization outputs are now finalized
  - launching anything else now would duplicate work or step outside `TASK.md`
- `DONE.txt` was not added because the sprint still lacks the final containment-control native-localization output.

### Containment control localization completed at 23:31 PDT

- Re-read `TASK.md`, re-inspected the active soft-align sprint README and persisted loop state, and verified the live pod directly on `atharv-rwx-pod-2` before taking any action.
- Local validation passed again:
  - `python3 -m py_compile methods/common.py methods/savlg.py train_cbm.py sparse_evaluation.py scripts/evaluate_native_spatial_maps.py`
  - `bash -n scripts/run_savlg_vlgwarm_loss_ablation_queue.sh`
- Verified the live pod state at closeout:
  - pod `atharv-rwx-pod-2` is still `Running`
  - restarts remain `0`
  - both GPUs are now idle at `0 MiB / 23028 MiB`
  - no active `train_cbm.py`, `sparse_evaluation.py`, or `evaluate_native_spatial_maps.py` processes remain on the pod
- New finalized control-localization evidence landed:
  - `containment only`
    - output: `/workspace/SAVLGCBM/results/native_savlg_vlgwarm_frozenglobal_multiscale_containment_only_v1_full_meanthr_bs32_nw8_gtpresent.json`
    - mean IoU `0.12015226059184847`
    - `mAP@0.3 = 0.043528346568986145`
    - `mAP@0.5 = 0.032490715726793476`
    - `mAP@0.7 = 0.024530855580414665`
    - point hit `0.9673325010403662`
    - coverage `0.06915406420421025`
- Closeout interpretation:
  - all six allowed dense runs are complete at dense `0.7590673575129534`
  - all six dense-qualified sparse `NEC150` follow-ups are complete at `ACC@5 = 0.7549` and `AVGACC = 0.7560`
  - all six native `gt_present` localization outputs are now finalized
  - `soft-align + outside penalty` remains best on `mAP@0.3`, `mAP@0.5`, and coverage
  - `containment only` is best on `mAP@0.7`
  - `soft-align + local-mil + outside penalty` remains best on mean IoU and point hit
- With the final containment-control localization output landed and no active eligible jobs left under `TASK.md`, the sprint is complete and `DONE.txt` was added.

### Warm-start ablation no-op iteration at 23:36 PDT

- Re-read `TASK.md`, re-inspected all `savlg-research/` sprint directories, the active `softalign-outside-ablation` sprint log, and the current local docs/results snapshot before taking any action.
- Confirmed the current `TASK.md`-allowed line is already fully complete in persistent state:
  - `savlg-research/2026-04-06--20-45-softalign-outside-ablation/DONE.txt` is present
  - the sprint README already records finalized dense, `NEC150`, and native `gt_present` localization outputs for all six allowed variants
  - `docs/cub_results.md` already contains the final six-run comparison and closeout interpretation
- This shell still cannot directly inspect live pod jobs:
  - `ps` is sandbox-blocked
  - `nvidia-smi` is unavailable
  - local `logs/` only shows older pulled files and no active warm-start ablation logs
- No eligible safe action remained for this iteration:
  - the only older non-`DONE` sprint is outside current `TASK.md` policy
  - launching anything in the completed warm-start line would duplicate finished work
- No code, launch, kill, or doc-result changes were taken beyond recording this no-op iteration.

### Residual dense runs still healthy at 16:54 PDT

- Re-read `TASK.md`, re-inspected the active residual-coupling sprint state, and checked the live pod directly on `atharv-rwx-pod-2` before taking any new action.
- Verified the current safe live state:
  - pod `atharv-rwx-pod-2` is `Running`
  - restarts remain `0`
  - pod age at check time was about `16m`
  - both GPUs are occupied by the two exact dense residual runs
  - GPU `0`: `831 MiB / 23028 MiB`, sampled `33%` utilization on both checks
  - GPU `1`: `831 MiB / 23028 MiB`, sampled `37%` then `30%` utilization
- Confirmed the active processes are the expected dense jobs:
  - `Residual-Alpha-0.05`
    - launcher / main pid pair: `1715 / 1717`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_v1.log`
    - progressed over the recheck window from `epoch 19` batch `79/169` to `epoch 21` batch `23/169`
  - `Residual-Alpha-0.10`
    - launcher / main pid pair: `5117 / 5118`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha010_softalign_outside_v1.log`
    - progressed over the recheck window from `epoch 16` batch `146/169` to `epoch 18` batch `93/169`
- Confirmed no dense-finished artifact is available yet:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1` currently exposes only `train.log`
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_48_21` currently exposes only `train.log`
  - neither run has `metrics.csv` or `test_metrics.json` yet
- No new action was eligible in this iteration:
  - no duplicate dense launch
  - no `Residual-Alpha-0.05-PlusMIL` launch because both eligible GPUs are still occupied
  - no NEC launch because no residual dense checkpoint has finished and cleared the dense gate
  - no kill or restart because both jobs are healthy and still making forward progress
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no finalized dense, sparse, or localization artifact landed in this iteration.

### Residual dense gate passed and follow-up jobs launched at 17:03 PDT

- Re-read `TASK.md`, re-inspected the active residual-coupling sprint state, and checked the live pod directly on `atharv-rwx-pod-2` before taking any launch action.
- Verified the first two exact residual dense runs are now complete:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1/test_metrics.json`
    - `accuracy = 0.7592400690846287`
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_48_21/test_metrics.json`
    - `accuracy = 0.7585492227979275`
  - both therefore clear the loop's current `> 0.74` dense gate for NEC promotion
  - the pod was idle immediately before relaunch:
    - both GPUs sampled at `0 MiB / 23028 MiB`
    - no active `train_cbm.py`, `sparse_evaluation.py`, or localization-eval processes remained
- Kept both GPUs busy with the next distinct eligible jobs:
  - GPU `0`: launched the pending exact dense run `Residual-Alpha-0.05-PlusMIL`
    - config:
      - `/workspace/SAVLGCBM/configs/unified/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_localmil_outside_v1.json`
    - log:
      - `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_localmil_outside_v1.log`
    - run dir:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_00_03_22`
    - launcher / main pid pair:
      - `67146 / 67148`
    - startup check:
      - VLG warm start succeeded
      - global-head freeze succeeded
      - latest completed epoch summary:
        - `[SAVLG CBL] epoch=0 train_loss=0.852276 val_loss=1.066138 best_val=1.066138`
  - GPU `1`: launched the first residual NEC follow-up for the slightly stronger dense checkpoint `Residual-Alpha-0.05`
    - load path:
      - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_07_23_45_15-1`
    - log:
      - `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_v1_nec150.log`
    - launcher / main pid pair:
      - `67150 / 67151`
    - startup check:
      - evaluation reload completed cleanly
      - sparse solver advanced to about `211/4000`
- Final live check for this iteration:
  - GPU `0`: `733 MiB / 23028 MiB`, sampled `30%`
  - GPU `1`: `711 MiB / 23028 MiB`, sampled `49%`
- No `DONE.txt` was added and `docs/cub_results.md` was left unchanged because residual-stage sparse `ACC@5` / `AVGACC` and localization artifacts have not landed yet.

### Residual +MIL dense and alpha=0.05 NEC still healthy at 17:06 PDT

- Re-read `TASK.md`, re-inspected the active residual-coupling sprint state, and queried `atharv-rwx-pod-2` directly before taking any new action.
- Verified the pod is still healthy:
  - phase: `Running`
  - restarts: `0`
  - creation timestamp unchanged at `2026-04-07T23:36:34Z`
- Verified both GPUs remain occupied by the current eligible residual-stage jobs:
  - GPU `0`: `Residual-Alpha-0.05-PlusMIL`
    - memory/utilization sample: `831 MiB / 23028 MiB`, `38%`
    - launcher / main pid pair: `67146 / 67148`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_localmil_outside_v1.log`
    - forward progress over the recheck window: from `epoch 10` batch `34/169` to `epoch 11` batch `29/169`
    - latest completed epoch summary in the log: `[SAVLG CBL] epoch=8 train_loss=0.439252 val_loss=1.189472 best_val=1.066138`
  - GPU `1`: `Residual-Alpha-0.05` NEC
    - memory/utilization sample: `983 MiB / 23028 MiB`, `39%`
    - launcher / main pid pair: `67150 / 67151`
    - log: `/workspace/SAVLGCBM/logs/cub_savlg_cbm_vlgwarm_residual_alpha005_softalign_outside_v1_nec150.log`
    - forward progress over the recheck window: from completed lambda line `(3)` to completed lambda line `(7)`
    - latest visible NEC checkpoint: `(7) lambda 0.0062 ... [test acc 0.7518] ... sparsity 0.002451564828614009 [329/134200]`
- Verified no new finalized residual-stage artifacts landed in this iteration:
  - `/workspace/SAVLGCBM/saved_models/cub/savlg_cbm_cub_2026_04_08_00_03_22` still contains only `args.txt` and `train.log`
  - no completed residual NEC summary artifact is present yet
  - no native localization evaluation job is active yet
- No eligible safe action remained in this iteration:
  - no new launch because both visible GPUs are already occupied by healthy distinct jobs
  - no kill or restart because both jobs are advancing
  - no `Residual-Alpha-0.10` NEC launch yet
  - no localization eval launch yet
- `DONE.txt` was not added and `docs/cub_results.md` was left unchanged because no new finalized dense, sparse, or localization result landed in this iteration.

### Residual dense and NEC finalized; localization still in flight at 17:38 PDT

- Re-read `TASK.md`, re-inspected the active residual sprint README and latest loop summaries, and verified local static validity with:
  - `python3 -m py_compile methods/savlg.py evaluations/sparse_utils.py scripts/evaluate_native_spatial_maps.py train_cbm.py`
- Verified live pod state directly on `atharv-rwx-pod-2`:
  - pod phase: `Running`
  - restarts: `0`
  - both A10 GPUs still allocated to active residual native-localization jobs
  - GPU `0`: `709 MiB / 23028 MiB`
  - GPU `1`: `709 MiB / 23028 MiB`
- Confirmed all three exact residual dense runs are now finalized and all three dense-qualified `NEC150` follow-ups are complete:
  - `Residual-Alpha-0.05`
    - dense `0.7592400690846287`
    - `ACC@5 = 0.7560`
    - `ACC@10 = 0.7568`
    - `ACC@15 = 0.7568`
    - `ACC@20 = 0.7570`
    - `ACC@25 = 0.7570`
    - `ACC@30 = 0.7572`
    - `AVGACC = 0.7568`
  - `Residual-Alpha-0.10`
    - dense `0.7585492227979275`
    - `ACC@5 = 0.7572`
    - `ACC@10 = 0.7584`
    - `ACC@15 = 0.7580`
    - `ACC@20 = 0.7579`
    - `ACC@25 = 0.7584`
    - `ACC@30 = 0.7585`
    - `AVGACC = 0.7581`
  - `Residual-Alpha-0.05-PlusMIL`
    - dense `0.7590673575129534`
    - `ACC@5 = 0.7563`
    - `ACC@10 = 0.7572`
    - `ACC@15 = 0.7572`
    - `ACC@20 = 0.7573`
    - `ACC@25 = 0.7584`
    - `ACC@30 = 0.7584`
    - `AVGACC = 0.7575`
- Current residual interpretation from finalized dense/sparse artifacts only:
  - residual coupling preserves dense performance very close to the `VLG` anchor for all three exact runs
  - `Residual-Alpha-0.05` is the strongest dense checkpoint by a small margin
  - `Residual-Alpha-0.10` is the strongest sparse checkpoint so far on `ACC@5` and `AVGACC`
  - `+local_mil` did not beat the plain `alpha=0.10` residual variant on sparse metrics
- Confirmed the only unfinished residual-stage work is native GT-present localization:
  - active healthy jobs:
    - `Residual-Alpha-0.10` native localization
    - `Residual-Alpha-0.05-PlusMIL` native localization
  - both logs advanced from `38/181` to `40/181` during the recheck window
  - remaining unlaunched follow-up:
    - `Residual-Alpha-0.05` native localization
- No new launch or restart was taken because both visible GPUs are already occupied by healthy eligible localization jobs.
- `DONE.txt` was not added because residual native localization is still incomplete.

### Sparse-eval refactor freeze and 4-CBM smoke verification on `refactor/sparse-eval-cleanup`

- Centralized eval loading in `model/cbm.py` with shared wrappers:
  - `VLGCBMEval`
  - `LFCBMEval`
  - `SALFCBMEval`
  - `SAVLGCBMEval`
  - plus `load_eval_cbm(...)`
- Simplified `sparse_evaluation.py` to:
  - load the checkpoint through the shared eval API
  - extract concept activations through `get_concept_activations(...)`
  - run the shared GLM / NEC path
  - support `--max_images` for small smoke tests
- Unified training dispatch in `train_cbm.py` through `methods/registry.py`:
  - added `methods/vlg.py`
  - removed the active `vlg_cbm` special-case in `train_cbm.py`
- Fixed branch-local code/runtime issues encountered during the smoke checks:
  - `methods/savlg.py`: removed stray top-level `return save_dir`
  - `methods/lf.py`: removed stray top-level `return save_dir`
  - `methods/savlg.py`: fixed SALF eval loading for old no-bias checkpoints
  - `clip/clip.py`: fixed `tokenize()` to use imported `version.parse(...)` instead of undefined `packaging.version.parse(...)`
- Real pod smoke verification on `atharv-rwx-pod`:
  - eval / NEC smoke path passed for all four CBMs on real checkpoints:
    - `VLG`
    - `LF`
    - `SALF`
    - `SAVLG`
  - training smoke path status:
    - `VLG`: verified through epoch 0 and validation; then entered post-epoch final-layer dataset / backbone embedding cache creation
    - `SALF`: originally failed in vendored CLIP; after the `clip/clip.py` fix it now runs and has entered `Computing SALF prompt-grid similarities for train`
    - `LF`: CLIP checkpoint corruption was fixed by deleting `/root/.cache/clip/ViT-B-16.pt` and explicitly re-running `clip.load("ViT-B/16")`; LF rerun remains queued after SALF on GPU1
    - `SAVLG`: remains queued after the active VLG smoke run on GPU0
- Important operational note:
  - the visible long `VLG` runtime on the pod is not epoch 0 training anymore
  - epoch 0 already finished cleanly
  - the slow step is the post-epoch backbone embedding cache build in `data.concept_dataset:get_or_create_backbone_embedding_cache(...)`
  - this should be decoupled from `cbl_batch_size` and given a much larger A100-specific embedding-cache batch size in later methodology cleanup
