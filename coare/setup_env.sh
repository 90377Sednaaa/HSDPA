#!/bin/bash
# ==============================================================================
# Environment Setup Script for DOST-ASTI COARE HPC Cluster (Saliksik)
# Run this inside an interactive compute node session (salloc), NOT on login node!
# ==============================================================================

set -e

echo "=== Setting up COARE Environment for User: ${USER} ==="

# 1. Load Anaconda Module
module purge
module load anaconda/3-2023.07-2

# 2. Redirect Conda Cache & Envs to High-Speed Scratch (Preserves 100GB /home quota)
mkdir -p /scratch3/${USER}/conda/envs
mkdir -p /scratch3/${USER}/conda/pkgs

conda config --add envs_dirs /scratch3/${USER}/conda/envs
conda config --add pkgs_dirs /scratch3/${USER}/conda/pkgs

echo "Conda directory configuration verified (~/.condarc):"
cat ~/.condarc

# 3. Create PyTorch Virtual Environment (pestnet)
if conda info --envs | grep -q "pestnet"; then
    echo "Environment 'pestnet' already exists. Activating..."
else
    echo "Creating environment 'pestnet' with PyTorch and CUDA 11.8..."
    mamba create -y -n pestnet python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge
fi

# 4. Activate and Install Repository Dependencies
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pestnet

# Install requirements
pip install pyyaml tqdm matplotlib pandas seaborn psutil thop

# Install ultralytics in editable mode
cd /scratch3/${USER}/HSDPA
pip install -e .

echo "=== Verification ==="
python -c "import torch, ultralytics; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Ultralytics:', ultralytics.__version__)"

echo "=== Setup Completed Successfully! ==="
