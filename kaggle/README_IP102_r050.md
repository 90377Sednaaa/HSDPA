# IP102 — CR-HSDPA r=0.50 — 4-Stage Training (200 epochs total)

Trains `cfg/cr_hsdpa_050.yaml` from scratch on `dmrashidpferrer/ip102-yolo-dataset`
(102 classes). IP102 cannot fit 200 epochs in one 12-hour Kaggle session
(~10 h per 50 epochs on Dual T4), so training is split into **4 chained stages
of 50 epochs**. Each stage's `best.pt` seeds the next stage.

Branch: `HSDPA-DM-Testing`. Repo: `https://github.com/90377Sednaaa/HSDPA.git`.
Account-agnostic: works on any Kaggle account linked via Kaggle CLI.

## Files

| Stage | Notebook | Kernel metadata | Epochs | `lr0` | Output dir |
|---|---|---|---|---|---|
| 1 | `train_r050_stage1.ipynb` | `kernel-metadata-r050-s1.json` | 1–50 | 0.01 | `dataset_r050_stage1/ip102_cr050_stage1/` |
| 2 | `train_r050_stage2.ipynb` | `kernel-metadata-r050-s2.json` | 51–100 | 0.006 | `dataset_r050_stage2/ip102_cr050_stage2/` |
| 3 | `train_r050_stage3.ipynb` | `kernel-metadata-r050-s3.json` | 101–150 | 0.003 | `dataset_r050_stage3/ip102_cr050_stage3/` |
| 4 | `train_r050_stage4.ipynb` | `kernel-metadata-r050-s4.json` | 151–200 | 0.0015 | `dataset_r050_stage4/ip102_cr050_stage4/` |

Common hyperparams: SGD + momentum 0.9, `batch=32`, `imgsz=640`, `amp=True`,
`save_period=10` (+ automatic `best.pt` / `last.pt`), Dual T4 (`device=[0,1]`),
Section 2.7 eval cell after training.

## Prerequisites

1. Kaggle CLI installed and linked (`kaggle.json` for the training account).
2. Each kernel needs **GPU on** and **internet on** (clones GitHub + pip-installs repo).
3. In every `kernel-metadata-r050-sN.json`, replace `YOUR_KAGGLE_USERNAME`
   with the training account's username.

## Run order

### Stage 1 (from scratch)

1. Attach dataset `dmrashidpferrer/ip102-yolo-dataset` (already in s1 metadata).
2. Push and run:
   `kaggle kernels push -p kaggle --kernel-metadata kernel-metadata-r050-s1.json`
   (or push `train_r050_stage1.ipynb` from the Kaggle UI with the dataset attached).
3. Run all cells top to bottom, exactly once. Takes ~10 h.
4. When finished: kernel Output → **Save Version as Dataset**, e.g.
   `<user>/cr-r050-stage1-output`.

### Stages 2–4 (chained)

1. In the next stage's metadata file, replace the placeholder
   `YOUR_KAGGLE_USERNAME/cr-r050-stage<N-1>-output` with the real output
   dataset slug from the previous step.
2. Push and run that stage's notebook (attaches both the IP102 dataset
   and the prior stage output). The notebook auto-finds
   `/kaggle/input/**/best.pt` and continues from it with the stage LR.
   Do **not** set `resume=True` across stages — the notebook handles this
   (`resume=True` is only for restarting an interrupted run in the same
   `project/name` folder via `last.pt`).
3. Save output as a dataset, repeat until stage 4 (~200 cumulative epochs).

## After each stage

Per `AGENTS.md` hygiene rules:

1. Download the kernel output.
2. Copy only essentials into `dataset_r050_stage<N>/`:
   `weights/best.pt`, `results.csv`, `results.png`, `confusion_matrix*.png`,
   `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png`,
   `thesis_metrics_summary_cr050_stage<N>.csv`, `training_time.txt`.
3. Delete the raw download folder (`Remove-Item -Recurse -Force ...`).
4. Never commit `.pt` files to git — commit only csv/png/json artifacts.

## Troubleshooting

- `Could not locate IP102 dataset` → dataset not attached; add
  `dmrashidpferrer/ip102-yolo-dataset` as kernel input.
- `prior checkpoint not found` on stage 2+ → prior stage output dataset
  not attached; check `dataset_sources` in that stage's metadata.
- Kernel timed out mid-stage → re-push the **same** stage; it resumes from
  its own `last.pt` automatically.
