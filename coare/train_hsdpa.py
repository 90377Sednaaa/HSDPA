#!/usr/bin/env python3
"""
Training & Evaluation Script for Attention-PestNet (HSDPA) on DOST-ASTI COARE HPC Cluster.
Thesis: A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention
Author: Lean Adrian Murillo et al. (University of Mindanao)
"""

import argparse
import os
import sys
import time
from pathlib import Path
import torch

os.environ["WANDB_MODE"] = "disabled"

# Ensure local repository takes precedence
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train HSDPA Baseline / CR-HSDPA on COARE HPC")
    parser.add_argument("--model", type=str, default="cfg/hsdpa.yaml", help="Model YAML config or checkpoint .pt")
    parser.add_argument("--data", type=str, default="cfg/ip102_coare.yaml", help="Dataset configuration file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (e.g. 32 for A100/P40)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"], help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--project", type=str, default="dataset_r100_coare", help="Artifacts root directory")
    parser.add_argument("--name", type=str, default="baseline_hsdpa", help="Experiment run name")
    parser.add_argument("--resume", action="store_true", help="Resume training from weights/last.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("COARE HPC Training Launcher - HSDPA Pest Detection")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Model Config: {args.model}")
    print(f"Dataset Config: {args.data}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print(f"Optimizer: {args.optimizer} (lr0={args.lr0}, momentum={args.momentum})")
    print(f"Destination: {args.project}/{args.name}")
    print("=" * 70)

    # Check for resume
    last_pt = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume and last_pt.exists():
        print(f"Resuming training from checkpoint: {last_pt}")
        model = YOLO(str(last_pt))
        resume_flag = True
    else:
        model = YOLO(args.model)
        resume_flag = False

    start_time = time.time()

    # Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        project=args.project,
        name=args.name,
        resume=resume_flag,
        save=True,
        exist_ok=True,
    )

    elapsed_sec = time.time() - start_time
    elapsed_hours = elapsed_sec / 3600.0

    print("\n" + "=" * 70)
    print(f"Training Complete! Wall-clock time: {elapsed_hours:.2f} hours ({elapsed_sec:.1f} seconds)")
    print("=" * 70)

    # Save training duration log
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_time.txt", "w") as f:
        f.write(f"Total training time: {elapsed_hours:.2f} hours ({elapsed_sec:.1f} seconds)\n")

    # Run Section 2.7 Validation Benchmark
    best_pt = out_dir / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\nRunning validation benchmark on best checkpoint: {best_pt}")
        eval_model = YOLO(str(best_pt))
        metrics = eval_model.val(data=args.data, imgsz=args.imgsz, batch=args.batch)

        mp = metrics.box.mp
        mr = metrics.box.mr
        map50 = metrics.box.map50
        map = metrics.box.map

        print(f"Validation Results: Precision={mp:.4f}, Recall={mr:.4f}, mAP@50={map50:.4f}, mAP@50:95={map:.4f}")

        summary_file = out_dir / "coare_metrics_summary.csv"
        with open(summary_file, "w") as f:
            f.write("Model,Epochs,Batch,Precision,Recall,mAP50,mAP50_95,Training_Hours\n")
            f.write(f"{args.name},{args.epochs},{args.batch},{mp:.4f},{mr:.4f},{map50:.4f},{map:.4f},{elapsed_hours:.2f}\n")
        print(f"Saved evaluation metrics to: {summary_file}")


if __name__ == "__main__":
    main()
