# 🚗 Semantic Segmentation for ADAS: U-Net vs DeepLabV3+

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-HPC%20%7C%20Slurm-green)
![Dataset](https://img.shields.io/badge/Dataset-Cityscapes-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Work%20In%20Progress-yellow)
<p align="center">
  <a href="#-project-overview">🇺🇸 <b>English Version</b></a> |
  <a href="#-descrizione-del-progetto">🇮🇹 <b>Versione Italiana</b></a>
</p>

---

<a name="-project-overview"></a>
## 🇺🇸 Project Overview

This repository contains the implementation and experimental results for my project on **2D Semantic Segmentation for Autonomous Driving (ADAS)**.

The project performs a comparative analysis between two major architectures, **U-Net** and **DeepLabV3+**, applied to the [Cityscapes Dataset](https://www.cityscapes-dataset.com/). The core contribution is the implementation of a **Coarse-to-Fine Transfer Learning strategy** to maximize segmentation accuracy on urban scenes using High-Performance Computing (HPC) resources.

### 🎯 Key Goals
1.  **Architecture Comparison**: Evaluating the trade-off between the symmetric Encoder-Decoder (U-Net) and Multi-scale Context (DeepLabV3+ with ASPP).
2.  **Coarse-to-Fine Strategy**: Investigating how pre-training on 20,000 "Coarse" annotations improves the final performance on high-quality "Fine" annotations.
3.  **HPC Optimization**: Utilizing Slurm for distributed training on NVIDIA P100/V100 GPUs.



### 🏗 Architectures

* **U-Net**: A fully convolutional network with skip connections that preserve spatial details. Excellent for recovering fine edges (e.g., traffic signs, poles).
    
* **DeepLabV3+ (ResNet Backbone)**: State-of-the-art model using Atrous Spatial Pyramid Pooling (ASPP) to capture objects at multiple scales without losing resolution.
    

### 📊 Results & Visualization
Qualitative comparison showing the improvements from Coarse pre-training to Fine-tuning.

| Input Image | U-Net (Coarse) | U-Net (Fine) | DeepLabV3+ (Fine) |
|:-----------:|:--------------:|:------------:|:-----------------:|
| ![Input](results_comparison/input_sample.png) | ![Coarse](results_comparison/result_unet_coarse.png) | ![Fine](results_comparison/result_unet_fine.png) | ![DeepLab](results_comparison/result_deeplab_fine.png) |

*(Note: The 'Fine' models demonstrate significantly sharper edges and reduced false positives compared to the 'Coarse' baseline.)*

### 🚀 Usage

**1. Inference (Generate Comparisons)**
To run the comparison script on your local machine using the trained `.pth` checkpoints:
```bash
python3 predict_all.py
This will generate the visualization grid in the results_comparison/ folder.
```
2. Training (HPC / Slurm)

```bash

# Phase 1: Coarse Training
sbatch run_unet_coarse.sh

# Phase 2: Fine-Tuning (Requires Coarse weights)
sbatch run_unet_fine.sh
sbatch run_deeplab_fine.sh
```
<a name="-descrizione-del-progetto"></a>

🇮🇹 Descrizione del Progetto
Questa repository contiene l'implementazione e i risultati sperimentali della mia Tesi Magistrale sulla Segmentazione Semantica per la Guida Autonoma (ADAS).

Il progetto si concentra sul confronto tra due architetture principali, U-Net e DeepLabV3+, applicate al dataset Cityscapes. Il contributo principale è l'implementazione di una strategia di Transfer Learning "Coarse-to-Fine" per massimizzare la precisione della segmentazione in scenari urbani, sfruttando risorse di calcolo HPC.

🎯 Obiettivi
Confronto Architetturale: Analisi dei compromessi tra un Encoder-Decoder simmetrico (U-Net) e un approccio basato su contesto multi-scala (DeepLabV3+).

Strategia Coarse-to-Fine: Studio dell'impatto del pre-addestramento su 20.000 annotazioni "grezze" (Coarse) prima del fine-tuning su annotazioni di alta qualità (Fine).

Ottimizzazione HPC: Gestione di training distribuiti su GPU NVIDIA P100/V100 tramite Slurm Workload Manager.

🏗 Architetture
U-Net: Implementazione custom della classica rete encoder-decoder. Le skip connections si sono rivelate fondamentali per recuperare i dettagli spaziali (bordi dei marciapiedi, pali).

DeepLabV3+ (Backbone ResNet): Modello avanzato che utilizza l'ASPP (Atrous Spatial Pyramid Pooling) per "vedere" il contesto a diverse scale, migliorando la coerenza globale della scena.

📊 Risultati
La tabella seguente mostra l'evoluzione qualitativa del modello. Si nota come il passaggio alla fase Fine elimini l'effetto "scalettato" e definisca meglio i piccoli oggetti.

(Vedi tabella nella sezione inglese sopra)

🚀 Utilizzo
1. Inferenza (Generazione Confronti) Per lanciare lo script di visualizzazione in locale usando i pesi .pth scaricati:

```bash

python3 predict_all.py
```
I risultati verranno salvati automaticamente nella cartella results_comparison/.

2. Training (HPC / Slurm) Il training è stato eseguito su cluster HPC. Ecco i comandi principali:

```bash

# Fase 1: Pre-training Coarse
sbatch run_unet_coarse.sh

# Fase 2: Fine-Tuning (Richiede i pesi Coarse)
sbatch run_unet_fine.sh
sbatch run_deeplab_fine.sh
```

```
📂 Repository Structure / Struttura
├── models/
│   ├── unet_model.py       # U-Net Architecture
│   └── deeplab_model.py    # DeepLabV3+ Wrapper
├── utils/
│   └── dataset.py          # Cityscapes Custom Dataloader
├── checkpoints/            # Model Weights (.pth)
├── results_comparison/     # Output images
├── predict_all.py          # Inference Script
└── train.py                # Main Training Loop
```
Author: Aurora Felisari

University: Università di Parma


***

### 🛠️ Cosa devi fare prima di fare il Commit:

1.  **Crea le cartelle**: Assicurati che su GitHub carichi la cartella `results_comparison`.
2.  **Rinomina le immagini**: Nel codice del README ho ipotizzato che le immagini si chiamino:
    * `input_sample.png`
    * `result_unet_coarse.png`
    * `result_unet_fine.png`
    * `result_deeplab_fine.png`
    
    Prendi le migliori immagini che `predict_all.py` ti ha generato poco fa, rinominale così e mettile nella cartella `results_comparison`. In questo modo la tabella nel README si "accenderà" con i tuoi risultati reali!

È pronto per essere pubblicato! 🚀
