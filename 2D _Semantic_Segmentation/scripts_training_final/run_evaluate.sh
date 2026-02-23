#!/bin/bash
#SBATCH --job-name=evaluate_metrics
#SBATCH --output=log_evaluate_%j.txt
#SBATCH --error=err_evaluate_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aurora.felisari@studenti.unipr.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

module purge
module load gnu8/8.3.0
module load python/3.9.10
module load cuda

source ~/project/venv_hpc/bin/activate

echo "================================================"
echo "Benchmark completo — $(date)"
echo "Host: $(hostname)"
echo "================================================"

~/project/venv_hpc/bin/python3 ~/project/evaluate_metrics.py
