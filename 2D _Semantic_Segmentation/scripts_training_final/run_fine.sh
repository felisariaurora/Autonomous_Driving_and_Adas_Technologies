#!/bin/bash
#SBATCH --job-name=cityscapes_fine
#SBATCH --output=log_fine_%j.txt
#SBATCH --error=err_fine_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aurora.felisari@studenti.unipr.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# 1. Caricamento moduli corretti (quelli che funzionano!)
module purge
module load gnu8/8.3.0
module load python/3.9.10
module load cuda

# 2. Attivazione ambiente
source ~/project/venv_hpc/bin/activate

# 3. Lancio del training usando il percorso assoluto del venv
~/project/venv_hpc/bin/python3 train_fine.py
