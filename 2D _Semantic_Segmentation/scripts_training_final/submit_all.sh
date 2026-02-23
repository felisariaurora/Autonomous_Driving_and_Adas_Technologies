#!/bin/bash
# ============================================================
# submit_all.sh — Lancia tutti i job SLURM con le giuste
# dipendenze tra coarse e fine-tuning.
#
# USO:
#   cd ~/project/scripts_training_final
#   bash submit_all.sh
#
# Dipendenze gestite automaticamente:
#   run_coarse.sh       ---> run_fine.sh         (U-Net)
#   run_deeplab_coarse  ---> run_deeplab_fine     (DeepLab)
#
# Job indipendenti (partono subito):
#   run_unet_scratch.sh
#   run_deeplab_scratch.sh
#   run_deeplab_pretrained.sh
# ============================================================

set -e  # Fermati se sbatch fallisce

echo "================================================"
echo " Invio job SLURM — $(date)"
echo "================================================"

# ── BLOCCO 1: job indipendenti (partono subito in parallelo) ──

echo ""
echo "[1/5] Lancio U-Net da scratch (indipendente)..."
JID_UNET_SCRATCH=$(sbatch --parsable run_unet_scratch.sh)
echo "      Job ID: $JID_UNET_SCRATCH"

echo "[2/5] Lancio DeepLab da scratch (indipendente)..."
JID_DL_SCRATCH=$(sbatch --parsable run_deeplab_scratch.sh)
echo "      Job ID: $JID_DL_SCRATCH"

echo "[3/5] Lancio DeepLab pretrained fine-tuning (indipendente)..."
JID_DL_PRETRAINED=$(sbatch --parsable run_deeplab_pretrained.sh)
echo "      Job ID: $JID_DL_PRETRAINED"

# ── BLOCCO 2: pipeline U-Net coarse → fine ──

echo "[4/5] Lancio U-Net Coarse..."
JID_COARSE=$(sbatch --parsable run_coarse.sh)
echo "      Job ID: $JID_COARSE"

echo "      Lancio U-Net Fine (attende la fine di Coarse: $JID_COARSE)..."
JID_FINE=$(sbatch --parsable --dependency=afterok:$JID_COARSE run_fine.sh)
echo "      Job ID: $JID_FINE"

# ── BLOCCO 3: pipeline DeepLab coarse → fine ──

echo "[5/5] Lancio DeepLab Coarse..."
JID_DL_COARSE=$(sbatch --parsable run_deeplab_coarse.sh)
echo "      Job ID: $JID_DL_COARSE"

echo "      Lancio DeepLab Fine (attende la fine di DeepLab Coarse: $JID_DL_COARSE)..."
JID_DL_FINE=$(sbatch --parsable --dependency=afterok:$JID_DL_COARSE run_deeplab_fine.sh)
echo "      Job ID: $JID_DL_FINE"

# ── Riepilogo finale ──

echo ""
echo "================================================"
echo " RIEPILOGO JOB INVIATI"
echo "================================================"
echo " U-Net Scratch:        $JID_UNET_SCRATCH   (parte subito)"
echo " DeepLab Scratch:      $JID_DL_SCRATCH     (parte subito)"
echo " DeepLab Pretrained:   $JID_DL_PRETRAINED  (parte subito)"
echo " U-Net Coarse:         $JID_COARSE         (parte subito)"
echo " U-Net Fine:           $JID_FINE           (attende $JID_COARSE)"
echo " DeepLab Coarse:       $JID_DL_COARSE      (parte subito)"
echo " DeepLab Fine:         $JID_DL_FINE        (attende $JID_DL_COARSE)"
echo "================================================"
echo ""
echo "Monitora con:"
echo "  squeue -u \$USER"
echo "  squeue -j $JID_UNET_SCRATCH,$JID_DL_SCRATCH,$JID_DL_PRETRAINED,$JID_COARSE,$JID_FINE,$JID_DL_COARSE,$JID_DL_FINE"
echo ""
echo "Log in tempo reale (es. U-Net Scratch):"
echo "  tail -f log_unet_scratch_${JID_UNET_SCRATCH}.txt"
