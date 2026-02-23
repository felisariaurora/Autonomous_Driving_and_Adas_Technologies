import torch
import os

# --- Rilevamento Dispositivo (Fondamentale per Mac) ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Usa l'accelerazione Apple Silicon (M1/M2/M3)
else:
    DEVICE = "cpu"

print(f"[CONFIG] Dispositivo rilevato: {DEVICE}")

# --- Percorsi ---
# Basati sulla posizione di config.py, funziona indipendentemente da dove
# viene lanciato lo script (root del progetto o scripts_training_final/)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT      = os.path.join(_PROJECT_ROOT, "data", "cityscapes")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")

# --- Parametri Dataset (DEVONO essere uguali al training) ---
NUM_CLASSES = 19
IMG_HEIGHT = 512  # Altezza usata in training
IMG_WIDTH = 1024  # Larghezza usata in training
IGNORE_INDEX = 255

# --- Parametri Training ---
BATCH_SIZE = 2    # Basso per non bloccare il Mac
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
ACCUMULATION_STEPS = 8

# --- Debug ---
DEBUG_MODE = False