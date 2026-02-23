#!/bin/bash
#SBATCH --job-name=cityscapes_coarse
#SBATCH --output=log_coarse_%j.txt
#SBATCH --error=err_coarse_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aurora.felisari@studenti.unipr.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu

# 1. Pulizia e caricamento moduli
module purge
module load gnu8/8.3.0
module load python/3.9.10
module load cuda

# 2. direttamente venv
source ~/project/venv_hpc/bin/activate

# 3. Debug
echo "Lancio addestramento su: $(hostname)"
which python3                         # Check versione python

# 4. Lancio del training con venv
~/project/venv_hpc/bin/python3 train_coarse.py
