#!/bin/bash
#SBATCH --job-name=deeplab_fine
#SBATCH --output=log_deeplab_fine_%j.txt
#SBATCH --error=err_deeplab_fine_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aurora.felisari@studenti.unipr.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

module purge
module load gnu8/8.3.0
module load python/3.9.10
module load cuda

source ~/project/venv_hpc/bin/activate
~/project/venv_hpc/bin/python3 train_deeplab_fine.py
