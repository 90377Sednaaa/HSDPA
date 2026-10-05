import pandas as pd
import matplotlib.pyplot as plt

# Load unified 200 epochs
df = pd.read_csv('results_200epochs.csv')
x = df['cumulative_epoch']

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
fig.suptitle('Baseline Attention-PestNet (HSDPA, r=1.00) — 200 Cumulative Epochs on IP102-YOLO', fontsize=16, fontweight='bold', y=0.98)

def add_stage_lines(ax):
    for epoch, label in [(50, 'Stage 1 (lr=0.01)'), (100, 'Stage 2 (lr=0.006)'), (150, 'Stage 3 (lr=0.004)'), (190, 'Mosaic Off')]:
        ax.axvline(epoch, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# 1. Box Loss
axs[0, 0].plot(x, df['train/box_loss'], label='Train Box Loss', color='#1f77b4', lw=1.5)
axs[0, 0].plot(x, df['val/box_loss'], label='Val Box Loss', color='#ff7f0e', lw=1.5)
axs[0, 0].set_title('Box Loss (Train vs. Val)', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Cumulative Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()
add_stage_lines(axs[0, 0])

# 2. Classification Loss
axs[0, 1].plot(x, df['train/cls_loss'], label='Train Cls Loss', color='#2ca02c', lw=1.5)
axs[0, 1].plot(x, df['val/cls_loss'], label='Val Cls Loss', color='#d62728', lw=1.5)
axs[0, 1].set_title('Classification Loss (Train vs. Val)', fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel('Cumulative Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
add_stage_lines(axs[0, 1])

# 3. DFL Loss
axs[0, 2].plot(x, df['train/dfl_loss'], label='Train DFL Loss', color='#9467bd', lw=1.5)
axs[0, 2].plot(x, df['val/dfl_loss'], label='Val DFL Loss', color='#8c564b', lw=1.5)
axs[0, 2].set_title('DFL Loss (Train vs. Val)', fontsize=12, fontweight='bold')
axs[0, 2].set_xlabel('Cumulative Epoch')
axs[0, 2].set_ylabel('Loss')
axs[0, 2].legend()
add_stage_lines(axs[0, 2])

# 4. Precision & Recall
axs[1, 0].plot(x, df['metrics/precision(B)'] * 100, label='Precision (%)', color='#0072B2', lw=1.5)
axs[1, 0].plot(x, df['metrics/recall(B)'] * 100, label='Recall (%)', color='#D55E00', lw=1.5)
axs[1, 0].set_title('Precision & Recall Curves', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Cumulative Epoch')
axs[1, 0].set_ylabel('Percentage (%)')
axs[1, 0].legend()
add_stage_lines(axs[1, 0])

# 5. mAP@50 & mAP@50:95
axs[1, 1].plot(x, df['metrics/mAP50(B)'] * 100, label='mAP@50 (%)', color='#009E73', lw=1.8)
axs[1, 1].plot(x, df['metrics/mAP50-95(B)'] * 100, label='mAP@50:95 (%)', color='#CC79A7', lw=1.5)
axs[1, 1].set_title('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Cumulative Epoch')
axs[1, 1].set_ylabel('mAP (%)')
axs[1, 1].legend()
add_stage_lines(axs[1, 1])

# 6. Learning Rate Schedule
axs[1, 2].plot(x, df['lr/pg0'], label='Learning Rate (lr0)', color='#E69F00', lw=1.5)
axs[1, 2].set_title('Learning Rate Schedule across Stages', fontsize=12, fontweight='bold')
axs[1, 2].set_xlabel('Cumulative Epoch')
axs[1, 2].set_ylabel('Learning Rate')
axs[1, 2].set_yscale('log')
axs[1, 2].legend()
add_stage_lines(axs[1, 2])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results_200epochs.png', dpi=300)
plt.close()
print('Successfully generated results_200epochs.png')
