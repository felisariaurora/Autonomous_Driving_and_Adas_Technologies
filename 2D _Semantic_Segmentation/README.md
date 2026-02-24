# Semantic Segmentation for ADAS: U-Net vs DeepLabV3+

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-HPC%20%7C%20SLURM-green)
![Dataset](https://img.shields.io/badge/Dataset-Cityscapes-orange)
![Classes](https://img.shields.io/badge/Classes-19-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Course](https://img.shields.io/badge/Course-ADAS-informational)
![University](https://img.shields.io/badge/University-UniPR-darkblue)

<p align="center">
  <a href="#-project-overview"> <b>English Version</b></a> &nbsp;|&nbsp;
  <a href="#-descrizione-del-progetto"> <b>Versione Italiana</b></a>
</p>

---

<a name="-project-overview"></a>
## Project Overview

This repository contains the implementation and experimental results for the **2D Semantic Segmentation** project.

The project performs a systematic comparison between two major deep learning architectures — **U-Net** (implemented from scratch) and **DeepLabV3+** (with multiple training strategies) — applied to the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) for pixel-level urban scene understanding.

---

### Key Goals

1. **Architecture Comparison** — Evaluate the trade-off between U-Net (lightweight, ~31M params) and DeepLabV3+ (state-of-the-art, ~63M params) on the same benchmark.
2. **Training Strategy Study** — Investigate three strategies: training from scratch, Coarse-to-Fine transfer, and ImageNet pretrained fine-tuning.
3. **HPC Pipeline** — Full SLURM-based training pipeline with automatic job dependency management.
4. **Rigorous Evaluation** — Val mIoU monitored every epoch; best model saved on validation performance, not final epoch.

---

### Architectures

#### U-Net (from scratch)
A fully custom encoder-decoder network with skip connections, implemented from scratch in PyTorch.
- **Encoder**: 5 levels of DoubleConv (Conv→BN→ReLU×2) + MaxPool 2×2; channels: 64→128→256→512→1024
- **Decoder**: ConvTranspose2d upsampling + concatenation of skip connections
- **Output**: 1×1 Conv → 19 classes
- ~**31M parameters** | ~124 MB

#### DeepLabV3+ (ResNet50 backbone)
State-of-the-art segmentation model using Atrous Spatial Pyramid Pooling (ASPP).
- **Backbone**: ResNet50 — deep feature extractor with residual connections
- **ASPP**: Parallel atrous convolutions (rates 6, 12, 18) + Global Average Pooling — captures multi-scale context
- **Decoder**: Lightweight upsampling fusing high- and low-level features
- ~**63M parameters** | ~159 MB

---

### Training Strategies

| Strategy | Dataset | Starting Point | Epochs | Notes |
|---|---|---|---|---|
| **Scratch** | gtFine | Random weights | 50 | Baseline for both models |
| **Coarse → Fine** | gtCoarse → gtFine | Random → internal transfer | 50 + 30 | Official Cityscapes benchmark approach |
| **Pretrained** | gtFine | **ImageNet backbone** | 40 | True fine-tuning; classifier head replaced for 19 classes |

**Common training details:**
- **Augmentation**: Random horizontal flip (50%), random scale+crop (75–125%), color jitter
- **Loss**: CrossEntropyLoss with `ignore_index=255`; ENet class weights for coarse phase
- **Optimizer**: Adam | **Scheduler**: ReduceLROnPlateau (monitors val mIoU)
- **Gradient Accumulation** (U-Net): ×8 steps → effective batch size = 16
- **Best model**: saved at maximum val mIoU, evaluated every epoch on 100 val images

---

### 📊 Results

#### Quantitative — mIoU & Pixel Accuracy

| Model | mIoU | Pixel Accuracy |
|---|---|---|
| U-Net — Scratch | 57.44% | 93.05% |
| U-Net — Coarse→Fine | 57.70% | 93.06% |
| DeepLabV3+ — Coarse | 63.26% | 92.11% |
| DeepLabV3+ — Coarse→Fine | 70.34% | 94.83% |
| DeepLabV3+ — Pretrained | 73.00% | 95.02% |
| **DeepLabV3+ — Scratch** | **74.20%** | **95.33%** |

> Results computed on the Cityscapes validation set (500 images, 19 classes, `ignore_index=255`).
> Evaluated with `evaluate_metrics.py` using the best checkpoint (max val mIoU) for each model.

---

### 📂 Repository Structure

```
📦 HPC_Submission/
│
├── 📄 config.py                        # Paths, hyperparameters, device detection
├── 📄 requirements.txt                 # Python dependencies
│
├── 📁 models/
│   └── unet_model.py                   # U-Net custom implementation
│
├── 📁 utils/
│   ├── dataset.py                      # Cityscapes DataLoader (with augmentation)
│   ├── metrics.py                      # mIoU + validate_model()
│   └── class_weights.py               # ENet class weights for imbalance handling
│
├── 📁 scripts_training_final/
│   ├── train_coarse.py                 # U-Net — Coarse phase
│   ├── train_fine.py                   # U-Net — Fine-tuning
│   ├── train_unet_scratch_base.py      # U-Net — Scratch
│   ├── train_deeplab_coarse.py         # DeepLab — Coarse phase
│   ├── train_deeplab_fine.py           # DeepLab — Fine-tuning
│   ├── train_deeplab_scratch.py        # DeepLab — Scratch
│   ├── train_deeplab_pretrained.py     # DeepLab — ImageNet pretrained
│   ├── submit_all.sh                   # Submit all jobs with dependencies
│   ├── run_coarse.sh                   # SLURM — U-Net Coarse
│   ├── run_fine.sh                     # SLURM — U-Net Fine
│   ├── run_unet_scratch.sh             # SLURM — U-Net Scratch
│   ├── run_deeplab_coarse.sh           # SLURM — DeepLab Coarse
│   ├── run_deeplab_fine.sh             # SLURM — DeepLab Fine
│   ├── run_deeplab_scratch.sh          # SLURM — DeepLab Scratch
│   └── run_deeplab_pretrained.sh       # SLURM — DeepLab Pretrained
│
├── 📁 checkpoints/                     # Trained model weights (.pth) — not tracked by git
├── 📁 data/                            # Cityscapes dataset — not tracked by git
├── 📁 results_comparison/              # Visual segmentation outputs
├── 📁 thesis_plots/                    # Training curves and logs
├── 📁 presentation/                    # Slide content 
│
├── 📄 evaluate_metrics.py              # Full benchmark on val set
├── 📄 predict_all.py                   # Inference + visualization
├── 📄 model_complexity.py              # Parameter count
└── 📄 plot_real_loss.py                # Training curve plots
```

---

### Usage

#### 1. Local Inference (Requires `.pth` checkpoints)

```bash
# Generate visual comparisons for all trained models
python3 predict_all.py

# Run full benchmark (mIoU + Pixel Accuracy on 500 val images)
python3 evaluate_metrics.py
```

#### 2. HPC Training — All jobs at once (recommended)

```bash
# Upload code to HPC (excludes checkpoints and data)
rsync -avz --progress \
  --exclude='checkpoints/' --exclude='data/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  ./ aurora.felisari@login.hpc.unipr.it:~/project/

# Connect and submit
ssh aurora.felisari@login.hpc.unipr.it
cd ~/project/scripts_training_final
sed -i 's/\r//' *.sh
bash submit_all.sh
```

`submit_all.sh` automatically handles dependencies:
- 5 independent jobs start immediately in parallel
- Fine-tuning jobs start automatically once their Coarse phase completes

#### 3. HPC Training — Single job

```bash
cd ~/project/scripts_training_final
sbatch run_deeplab_pretrained.sh
```

#### 4. Monitor jobs

```bash
squeue -u $USER
tail -f log_unet_scratch_<JOBID>.txt
```

---

### ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Download Cityscapes dataset
# Place it under: data/cityscapes/leftImg8bit/ and data/cityscapes/gtFine/
```

---

### Key Implementation Details

- **Class imbalance** handled via ENet-derived weights passed to `CrossEntropyLoss`
- **Augmentation** applied synchronously on image and mask using `torchvision.transforms.functional`
- **Pretrained fine-tuning**: classifier head replaced (`Conv2d(256→19)`), differential learning rates for backbone vs classifier
- **Paths**: `config.py` uses `os.path.abspath(__file__)` — works correctly regardless of launch directory
- **Memory**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in all training scripts

---

### Author

**Aurora Felisari**
Università di Parma — MS in Computer Science

---

---

<a name="-descrizione-del-progetto"></a>
## Descrizione del Progetto

Questa repository contiene l'implementazione e i risultati sperimentali del progetto di **Segmentazione Semantica 2D**.

Il progetto confronta sistematicamente due architetture di deep learning — **U-Net** (implementata da zero) e **DeepLabV3+** (con diverse strategie di training) — applicate al [dataset Cityscapes](https://www.cityscapes-dataset.com/) per la classificazione a livello di pixel di scene urbane.

---

### Obiettivi Principali

1. **Confronto Architetturale** — Analisi del compromesso tra U-Net (leggera, ~31M parametri) e DeepLabV3+ (stato dell'arte, ~63M parametri) sullo stesso benchmark.
2. **Studio delle Strategie di Training** — Confronto tra training da zero, transfer Coarse→Fine e fine-tuning con backbone ImageNet pretrained.
3. **Pipeline HPC** — Training completo su cluster SLURM con gestione automatica delle dipendenze tra job.
4. **Valutazione Rigorosa** — Val mIoU monitorato ogni epoca; miglior modello salvato sulla performance di validazione, non sull'ultima epoca.

---

### Architetture

#### U-Net (da zero)
Rete encoder-decoder con skip connections, implementata completamente da zero in PyTorch.
- **Encoder**: 5 livelli DoubleConv (Conv→BN→ReLU×2) + MaxPool 2×2; canali: 64→128→256→512→1024
- **Decoder**: Upsampling con ConvTranspose2d + concatenazione delle skip connections
- **Output**: Conv 1×1 → 19 classi
- ~**31M parametri** | ~124 MB

#### DeepLabV3+ (backbone ResNet50)
Modello all'avanguardia per la segmentazione che utilizza l'Atrous Spatial Pyramid Pooling (ASPP).
- **Backbone**: ResNet50 — feature extractor profondo con connessioni residuali
- **ASPP**: Convoluzioni dilatate parallele (tassi 6, 12, 18) + Global Average Pooling — cattura contesto multi-scala
- **Decoder**: Upsampling leggero che fonde feature di alto e basso livello
- ~**63M parametri** | ~159 MB

---

### Strategie di Training

| Strategia | Dataset | Punto di partenza | Epoche | Note |
|---|---|---|---|---|
| **Scratch** | gtFine | Pesi random | 50 | Baseline per entrambi i modelli |
| **Coarse → Fine** | gtCoarse → gtFine | Random → transfer interno | 50 + 30 | Approccio ufficiale del paper Cityscapes |
| **Pretrained** | gtFine | **Backbone ImageNet** | 40 | Vero fine-tuning; head sostituita per 19 classi |

**Dettagli comuni:**
- **Augmentation**: Random horizontal flip (50%), random scale+crop (75–125%), color jitter
- **Loss**: CrossEntropyLoss con `ignore_index=255`; class weights ENet per la fase coarse
- **Optimizer**: Adam | **Scheduler**: ReduceLROnPlateau (monitora val mIoU)
- **Gradient Accumulation** (U-Net): ×8 step → batch effettivo = 16
- **Best model**: salvato al massimo val mIoU, calcolato ogni epoca su 100 immagini di validation

---

### Risultati

#### Quantitativi — mIoU e Pixel Accuracy

| Modello | mIoU | Pixel Accuracy |
|---|---|---|
| U-Net — Scratch | 57.44% | 93.05% |
| U-Net — Coarse→Fine | 57.70% | 93.06% |
| DeepLabV3+ — Coarse | 63.26% | 92.11% |
| DeepLabV3+ — Coarse→Fine | 70.34% | 94.83% |
| DeepLabV3+ — Pretrained | 73.00% | 95.02% |
| **DeepLabV3+ — Scratch** | **74.20%** | **95.33%** |

> Risultati calcolati sul validation set di Cityscapes (500 immagini, 19 classi, `ignore_index=255`).
> Valutato con `evaluate_metrics.py` usando il miglior checkpoint (max val mIoU) per ogni modello.

---

### Utilizzo

#### 1. Inferenza in locale (richiede i checkpoint `.pth`)

```bash
# Genera le visualizzazioni comparative per tutti i modelli
python3 predict_all.py

# Esegui il benchmark completo (mIoU + Pixel Accuracy su 500 immagini)
python3 evaluate_metrics.py
```

#### 2. Training su HPC — Tutti i job in una volta (raccomandato)

```bash
# Carica il codice sull'HPC (escludi checkpoint e dati)
rsync -avz --progress \
  --exclude='checkpoints/' --exclude='data/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  ./ aurora.felisari@login.hpc.unipr.it:~/project/

# Connettiti e lancia
ssh aurora.felisari@login.hpc.unipr.it
cd ~/project/scripts_training_final
sed -i 's/\r//' *.sh   
bash submit_all.sh
```

`submit_all.sh` gestisce automaticamente le dipendenze:
- 5 job indipendenti partono subito in parallelo
- I fine-tuning partono automaticamente al completamento della fase coarse

#### 3. Training su HPC — Singolo job

```bash
cd ~/project/scripts_training_final
sbatch run_deeplab_pretrained.sh
```

#### 4. Monitoraggio

```bash
squeue -u $USER
tail -f log_unet_scratch_<JOBID>.txt
```

---

### ⚙️ Setup

```bash
# Clona la repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Installa le dipendenze
pip install -r requirements.txt

# Scarica il dataset Cityscapes
# Posizionalo in: data/cityscapes/leftImg8bit/ e data/cityscapes/gtFine/
```

---

### Dettagli Implementativi

- **Sbilanciamento delle classi** gestito tramite class weights ENet nella `CrossEntropyLoss`
- **Augmentation** applicata sincronizzata su immagine e maschera con `torchvision.transforms.functional`
- **Fine-tuning pretrained**: head classificatore sostituita (`Conv2d(256→19)`), learning rate differenziato backbone vs classificatore
- **Path assoluti**: `config.py` usa `os.path.abspath(__file__)` — funziona indipendentemente dalla directory di lancio
- **Memoria GPU**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` impostato in tutti gli script

---

### Autore

**Aurora Felisari**
Università di Parma — Studentessa Magistrale in Scienze Informatiche
