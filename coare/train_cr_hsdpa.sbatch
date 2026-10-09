#!/bin/bash
# ==============================================================================
# SLURM Job Submission: Proposed CR-HSDPA (r=0.25 Primary / r=0.50 Ablation)
# DOST-ASTI COARE HPC Cluster (Saliksik)
# ==============================================================================
#SBATCH --account=<YOUR_PROJECT_ACCOUNT>    # Replace with your assigned project account
#SBATCH --partition=gpu_a100                # Primary: gpu_a100 (8x A100). Fallback: gpu (1x P40)
#SBATCH --qos=gpu-a100_default              # Primary: gpu-a100_default. Fallback: gpu-p40_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14                  # A100 partition allows up to 14 CPUs
#SBATCH --gres=gpu:1                        # Request 1 dedicated GPU
#SBATCH --mem=64G                           # RAM allocated to job
#SBATCH --time=72:00:00                     # Max runtime (up to 3 days on COARE)
#SBATCH --job-name="cr_hsdpa_025"
#SBATCH --output="logs_%x_%j.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leanmurillo0201@gmail.com
#SBATCH --requeue

ulimit -s unlimited

echo "========================================================================"
echo "Job Started      : $(date)"
echo "SLURM_JOB_ID     : ${SLURM_JOB_ID}"
echo "Node Allocated   : ${SLURM_JOB_NODELIST}"
echo "CPUs Allocated   : ${SLURM_CPUS_PER_TASK}"
echo "GPU Device Info  : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
echo "Working Directory: $(pwd)"
echo "========================================================================"

# 1. Faster local scratch I/O for temporary files
export JOB_TMPDIR=/tmp/${USER}/SLURM_${SLURM_JOB_ID}
mkdir -p ${JOB_TMPDIR}
export TMPDIR=${JOB_TMPDIR}

# 2. Reset and Load Environment Modules
module purge
module load anaconda/3-2023.07-2

# 3. Activate Conda Environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pestnet

# Verify PyTorch CUDA recognition
python -c "import torch; print('PyTorch CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

# 4. Execute Proposed CR-HSDPA Training
# Select Model: cfg/cr_hsdpa_025.yaml (r=0.25) or cfg/cr_hsdpa_050.yaml (r=0.50)
python coare/train_hsdpa.py \
    --model cfg/cr_hsdpa_025.yaml \
    --data cfg/ip102_coare.yaml \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --workers 8 \
    --optimizer SGD \
    --lr0 0.01 \
    --momentum 0.9 \
    --project dataset_r025_coare \
    --name cr_hsdpa_025

# 5. Cleanup Temporary Node Storage
rm -rf ${JOB_TMPDIR}

echo "========================================================================"
echo "Job Completed    : $(date)"
echo "========================================================================"
