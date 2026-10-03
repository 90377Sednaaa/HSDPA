# R2000 — CR-HSDPA r=0.50 — Single 200-Epoch Run

Trains `cfg/cr_hsdpa_050.yaml` (Channel-Reduced HSDPA, r=0.50) from scratch on
`dmrashidpferrer/r2000-pestnet` (16 rice-pest classes). R2000 is small
(~1,225 train images), so the full **200 epochs fit in one Kaggle 12-hour
Dual-T4 session** — no stage splitting needed.

Branch: `HSDPA-DM-Testing`. Repo: `https://github.com/90377Sednaaa/HSDPA.git`.
Training account: `dmrashidferrer` (linked via Kaggle CLI).

## Files

- Notebook: `train_r2000_cr050_200ep.ipynb`
- Kernel metadata: `kernel-metadata-r2000-cr050.json`
- Output dir: `dataset_r2000_r050/r2000_cr050_200ep/`

Hyperparams: SGD `lr0=0.01` + momentum 0.9, `batch=32`, `imgsz=640`,
`amp=True`, `save_period=25` (+ automatic `best.pt` / `last.pt`), Dual T4,
Section 2.7 eval cell after training.

### R2000 class handling (automatic)

Source model/dataset configs are `nc=102` (IP102). The notebook handles the
16-class mismatch at runtime — no manual yaml editing:

1. Finds the dataset under `/kaggle/input` (`images/train`).
2. Prefers a `data.yaml` shipped with the dataset for `nc`/`names`;
   falls back to `nc=16` with generic names.
3. Writes `cfg/r2000.yaml` and a patched copy of the model yaml
   (`nc=16`) to `/tmp`, then trains from that.

## API token setup (Terminal A — R2000 account `dmrashidferrer`)

The token is only used for local `push`/`pull`/`status` API calls — never for
running the kernel, and never written to any file. Set it per terminal session
(PowerShell syntax):

    $env:KAGGLE_API_TOKEN = "<PASTE-R2000-KGAT-TOKEN>"
    kaggle kernels list --mine   # expect dmrashidferrer/* kernels

Keep one terminal per account; the token lives only in that shell's memory.

## Prerequisites

1. Kaggle CLI installed and linked as `dmrashidferrer` (`KAGGLE_API_TOKEN`
   env var set in the terminal you push from — see Token setup above).
2. Kernel needs **GPU on** and **internet on**.
3. `kernel-metadata-r2000-cr050.json` already uses
   `id: dmrashidferrer/r2000-cr050-200ep` — no placeholder step remains.

## Run

1. Attach dataset `dmrashidpferrer/r2000-pestnet` (already in metadata).
2. Push and run from the `kaggle/` folder (Kaggle CLI 2.x only accepts a
   folder containing the notebook + a file named exactly
   `kernel-metadata.json` — `push.ps1` stages that for you):
   `.\push.ps1 r2000-cr050`
   (or upload `train_r2000_cr050_200ep.ipynb` from the Kaggle UI).
3. Run all cells top to bottom, exactly once (single `model.train()` cell).
4. If the kernel is interrupted: re-push/re-run the **same** notebook — it
   resumes automatically from its own `last.pt`.

## After the run

Per `AGENTS.md` hygiene rules:

1. Download the kernel output.
2. Copy only essentials into `dataset_r2000_r050/`:
   `weights/best.pt`, `results.csv`, `results.png`, `confusion_matrix*.png`,
   `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png`,
   `thesis_metrics_summary_r2000_cr050_200ep.csv`, `training_time.txt`.
3. Delete the raw download folder (`Remove-Item -Recurse -Force ...`).
4. Never commit `.pt` files to git — commit only csv/png/json artifacts.

## Troubleshooting

- `Could not locate R2000 dataset` → dataset not attached; add
  `dmrashidpferrer/r2000-pestnet` as kernel input.
- Class-count errors → check the auto-discover cell output (`Using nc=...`);
  the dataset layout must contain `images/train`, `images/val`.

## k=3 ablation variant (depth study at fixed r=0.50)

- Notebook: `train_r2000_cr050_k3_200ep.ipynb`
- Kernel metadata: `kernel-metadata-r2000-cr050-k3.json`
  (`id: dmrashidferrer/r2000-cr050-k3-200ep`)
- Model config: `cfg/cr_hsdpa_050_k3.yaml` — same as r050 but `CR_HSDPA`
  entries pass `k=3` (3 TDA levels instead of 4; see `CR_HSDPA(..., k=4)`
  default in `ultralytics/nn/ext/Blocks.py`).
- Output dir: `dataset_r2000_r050_k3/r2000_cr050_k3_200ep/`
- Push from the `kaggle/` folder: `.\push.ps1 r2000-cr050-k3`
- Expected params at nc=16: ≈67.18M (−1.38M vs k4's 68.56M).
