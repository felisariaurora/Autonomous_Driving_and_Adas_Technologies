#!/bin/bash
#SBATCH --job-name=deeplab_pretrained
#SBATCH --output=log_deeplab_pretrained_%j.txt
#SBATCH --error=err_deeplab_pretrained_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aurora.felisari@studenti.unipr.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# 1. Pulizia e caricamento moduli
module purge
module load gnu8/8.3.0
module load python/3.9.10
module load cuda

# 2. Attivazione venv
source ~/project/venv_hpc/bin/activate

# 3. Info di debug
echo "================================================"
echo "Job: DeepLabV3+ TRUE Fine-Tuning (ImageNet backbone)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Data: $(date)"
echo "================================================"
which python3

# 4. Lancio training
~/project/venv_hpc/bin/python3 train_deeplab_pretrained.py
