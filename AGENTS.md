# AGENTS.md — Guidelines & Operational Playbook for AI Agents

This document defines the architectural context, dataset conventions, execution protocols, and mandatory rules for AI agents and human contributors interacting with the **HSDPA / CR-HSDPA** research codebase.

---

## 1. Project & Thesis Context

* **Thesis Title:** *A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection*
* **Baseline Reference:** *Attention-PestNet: an attention-augmented framework for robust insect pest detection in diverse agricultural environments* (Doan et al., *Plant Methods* 2026, DOI: `10.1186/s13007-025-01489-z`).
* **Core Hypothesis:** By introducing a Channel-Reduction (CR) bottleneck ($r = 0.25$) into the Hierarchical Scaled Dot-Product Attention (HSDPA) blocks, we significantly reduce parameter overhead (~19.66% parameter reduction, saving 15.6M parameters) and inference latency while retaining high pest detection accuracy (mAP@50 and Recall) on challenging agricultural benchmarks.
* **Current Phase Directive (Thesis Panel Defense):** Per panelist instruction, the original **Baseline Attention-PestNet architecture** (`cfg/hsdpa.yaml`, $r = 1.00$) must be retrained from scratch on the newly established standard dataset `leanadrianmurillo/ip102-yolo` to serve as the rigorous experimental anchor before comparative ablation analysis.

---

## 2. Model Architecture Specifications

* **Base Framework:** Ultralytics YOLOv8 (v8.0.0 codebase in `ultralytics/`).
* **Base Architecture:** YOLOv8-Large (`YOLOv8-L` scale: `ex: [0.33, 1.00, 1024]`).
* **Backbone Components:**
  * **SDC (Spatial Downsampling Convolution):** Two-layer convolution without residual skip connections, implemented as `CBS` (`Conv-BatchNorm-SiLU`).
  * **MSPA (Multi-Scale Spatial Attention):** Implemented as `Spatial_Attention`, extracting horizontal and vertical spatial relationships combined with multi-scale pooling.
  * **C2f:** Standard YOLOv8 Cross-Stage Partial with 2 Convolutions.
  * **SPPF:** Spatial Pyramid Pooling - Fast.
* **Neck / Head Components:**
  * **CR-HSDPA:** Lightweight Hierarchical Scaled Dot-Product Attention featuring channel compression ($r=0.25$), stacked hierarchical self-attention passes ($k=3$), and $1 \times 1$ projection back to channel space.
  * **PANet:** Path Aggregation Network feature pyramid.
  * **Detect Head:** Decoupled anchor-free multi-scale detection head.
* **Key Configuration Files:**
  * `cfg/cr_hsdpa_025.yaml`: Primary proposed model ($r = 0.25$).
  * `cfg/hsdpa.yaml`: Baseline Attention-PestNet model ($r = 1.0$).
  * `cfg/l_hsdpa.yaml`: Lightweight HSDPA variant ($r = 0.5$).
  * `cfg/ip102.yaml`: Dataset configuration for 102 insect pest categories.

---

## 3. Training & Curriculum Protocol

Due to Kaggle 12-hour session constraints, training on Kaggle Dual NVIDIA T4 GPUs is partitioned into a **Two-Stage Curriculum**:

1. **Stage 1 (Epochs 1 – 50):**
   * Trains from scratch with SGD (`lr0=0.01`, `momentum=0.9`, `batch=32`, `imgsz=640`).
   * Saves best checkpoint at `dataset_r025_stage1/best.pt` (mAP@50: ~65.34%).
2. **Stage 2 (Epochs 51 – 100):**
   * Resumes/fine-tunes from Stage 1 `best.pt` with a smoother learning rate (`lr0=0.006`).
   * Reaches the full 100 cumulative epochs.
   * Saves best checkpoint at `dataset_r025_stage2/best.pt` (mAP@50: ~66.69%, Recall: ~67.54%).

---

## 4. Mandatory File Hygiene & Folder Naming Rules (CRITICAL)

Whenever an agent runs training, downloads results, or manages artifacts, the following rules **MUST** be strictly enforced:

### Rule 1: Explicit Reduction-Ratio Folder Naming (`dataset_r<ratio>_stage<stage>/`)
All experiment artifact directories **must explicitly identify the reduction ratio ($r$) and training stage**. Do NOT use vague generic names like `dataset_stage1`.

* **Naming Standard:** `dataset_r<ratio>_stage<stage_number>/`
  * **$r = 0.25$ (Proposed CR-HSDPA):** `dataset_r025_stage1/` and `dataset_r025_stage2/`
  * **$r = 0.50$ (L-HSDPA Ablation):** `dataset_r050_stage1/` and `dataset_r050_stage2/`
  * **$r = 0.75$ (Ablation Variant):** `dataset_r075_stage1/` and `dataset_r075_stage2/`
  * **$r = 1.00$ (Baseline Attention-PestNet):** `dataset_r100_stage1/` and `dataset_r100_stage2/`

### Rule 2: Immediate Post-Download Cleanup of Raw Kaggle Outputs
* Downloading Kaggle kernel outputs generates complete repository duplicates (`kaggle_stage1_output/`, `kaggle_stage2_output/`), consuming **3 to 5+ GB** of redundant disk space.
* **Mandatory Procedure:**
  1. Extract only the essential target artifacts into the corresponding ratio folder:
     * Checkpoint (`weights/best.pt`) $\rightarrow$ copy to `dataset_r<ratio>_stage<X>/best.pt`
     * Metrics and curves (`results.csv`, `results.png`, `confusion_matrix.png`, `F1_curve.png`, `PR_curve.png`) $\rightarrow$ copy to `dataset_r<ratio>_stage<X>/`
  2. **Immediately delete** the raw temporary download folder (e.g. `Remove-Item -Recurse -Force kaggle_stageX_output`).

### Rule 3: Git Hygiene for Model Weights
* Never commit `.pt` files directly into Git unless Git LFS is explicitly tracking them.
* Keep `.gitignore` updated:
  ```gitignore
  kaggle_stage*_output/
  dataset_r*_stage1/
  dataset_r*_stage*/*.pt
  dataset_*/*.pt
  ```
* Commit only lightweight, verifiable artifacts: `results.csv`, `results.png`, evaluation curves, and `dataset-metadata.json`.

### Rule 4: Single-Flow Notebook Integrity
* Never leave duplicate or legacy `model.train()` or evaluation cells inside `kaggle/train_kaggle.ipynb`.
* Jupyter executes sequentially from top to bottom. Duplicate training cells will cause Kaggle to start a second training run automatically upon completion of the first.
* Always verify the cell sequence of `kaggle/train_kaggle.ipynb` before pushing to Kaggle.

---

## 5. Standard Evaluation Protocol (Section 2.7)

When evaluating checkpoints for the thesis benchmark:
1. **Detection Performance:**
   * Dataset: `cfg/ip102.yaml` (split: `val`, `imgsz=640`, `batch=32`).
   * Record: `Precision`, `Recall`, `mAP@50`, `mAP@50:95`.
2. **Efficiency Benchmark:**
   * Precision: `FP32`.
   * Batch size: `1` (single dummy input `1 x 3 x 640 x 640`).
   * Warmup: 20 iterations.
   * Timed benchmark: 200 iterations with `torch.cuda.synchronize()`.
   * Record: `Latency (ms)`, `Throughput (FPS)`, `Total Parameters`, `Model File Size (MB)`.
3. **Artifact Output:**
   * Save structured metrics to `thesis_metrics_summary_stageX.csv`.

---

## 6. Directory Layout Reference

```text
HSDPA/
├── cfg/                             # Model and dataset YAML configurations
│   ├── cr_hsdpa_025.yaml            # Proposed model configuration (r=0.25)
│   ├── cr_hsdpa_050.yaml            # Ablation configuration (r=0.50)
│   ├── cr_hsdpa_075.yaml            # Ablation configuration (r=0.75)
│   ├── cr_hsdpa_100.yaml            # Uncompressed configuration (r=1.00)
│   ├── hsdpa.yaml                   # Baseline model configuration
│   └── ip102.yaml                   # IP102 dataset path & 102 class definitions
├── dataset_r025_stage1/             # Proposed r=0.25 Stage 1 (Epochs 1-50) weights & metadata
│   ├── best.pt
│   └── dataset-metadata.json
├── dataset_r025_stage2/             # Proposed r=0.25 Stage 2 (Epochs 51-100) final weights & curves
│   ├── best.pt                      # Final 100-epoch trained weights
│   ├── results.csv                  # Metric history across all epochs
│   ├── results.png                  # Loss and validation curves
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   ├── F1_curve.png
│   └── dataset-metadata.json
├── kaggle/                          # Kaggle notebook & kernel metadata
│   ├── kernel-metadata.json         # Kaggle CLI configuration
│   ├── train_baseline.ipynb         # Dedicated notebook for Baseline Attention-PestNet (r=1.00)
│   ├── train_cr_hsdpa.ipynb         # Dedicated notebook for Proposed CR-HSDPA (r=0.25)
│   └── train_kaggle.ipynb           # Default execution entrypoint for Kaggle
├── ultralytics/                     # Custom Ultralytics framework source
│   ├── cfg/default.yaml             # Mandatory default config for Ultralytics
│   ├── nn/ext/Blocks.py             # Custom attention modules (CR_HSDPA, SDC, MSPA)
│   └── nn/tasks.py                  # Model parsing & layer integration
├── AGENTS.md                        # Agent guidelines and operational playbook
├── README.md                        # Project overview & documentation
└── setup.py                         # Editable install setup for Ultralytics
```
