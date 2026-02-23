"""
plot_real_loss.py
Genera grafici di training loss e val mIoU per tutti i modelli.
Output: thesis_plots/Graph_Training_Loss.png
         thesis_plots/Graph_Val_mIoU.png
         thesis_plots/Graph_Combined.png
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import re

LOG_DIR    = "./thesis_plots"
OUTPUT_DIR = "./thesis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Configurazione modelli ────────────────────────────────────────────────────
# (label, file_log, colore, stile)
MODELS = [
    ("U-Net Scratch",       "log_unet_scratch_2410416.txt",        "#3498db", "-"),
    ("U-Net Coarse",        "log_coarse_2410419.txt",              "#85c1e9", "--"),
    ("U-Net Coarse→Fine",   "log_fine_2410420.txt",                "#1a5276", "-."),
    ("DeepLab Scratch",     "log_deeplab_scratch_2410417.txt",     "#e74c3c", "-"),
    ("DeepLab Coarse",      "log_deeplab_2410421.txt",             "#f1948a", "--"),
    ("DeepLab Pretrained",  "log_deeplab_pretrained_2410418.txt",  "#922b21", "-."),
    ("DeepLab Coarse→Fine", "log_deeplab_fine_2410422.txt",        "#8e44ad", "-"),
]


def parse_log(filepath):
    """
    Estrae train_loss e val_miou da un log di training.
    Supporta entrambi i formati (U-Net e DeepLab).
    Restituisce (epochs, losses, mious).
    """
    losses, mious = [], []

    if not os.path.exists(filepath):
        print(f"  [SKIP] {os.path.basename(filepath)} non trovato")
        return [], [], []

    with open(filepath, 'r') as f:
        for line in f:
            # Train Loss (formato comune: "Train Loss: X.XXXX")
            m = re.search(r'Train Loss:\s*([\d.]+)', line)
            if m:
                losses.append(float(m.group(1)))

            # Val mIoU (formato: "Val mIoU: X.XXXX")
            m = re.search(r'Val mIoU:\s*([\d.]+)', line)
            if m:
                mious.append(float(m.group(1)))

    epochs = list(range(1, len(losses) + 1))
    loss_str = f"{losses[-1]:.4f}" if losses else "N/A"
    miou_str = f"{max(mious)*100:.1f}%" if mious else "N/A"
    print(f"  {os.path.basename(filepath)}: {len(losses)} epoche, loss finale={loss_str}, best mIoU={miou_str}")
    return epochs, losses, mious


def plot_training_loss(data):
    fig, ax = plt.subplots(figsize=(11, 6))

    for label, epochs, losses, mious, color, ls in data:
        if losses:
            ax.plot(epochs, losses, label=label, color=color,
                    linestyle=ls, linewidth=2, alpha=0.9)

    ax.set_title("Training Loss — Confronto Tutti i Modelli", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)

    path = os.path.join(OUTPUT_DIR, "Graph_Training_Loss.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Salvato: {path}")


def plot_val_miou(data):
    fig, ax = plt.subplots(figsize=(11, 6))

    for label, epochs, losses, mious, color, ls in data:
        if mious:
            ax.plot(range(1, len(mious) + 1), [m * 100 for m in mious],
                    label=label, color=color, linestyle=ls, linewidth=2, alpha=0.9)

    ax.set_title("Val mIoU per Epoca — Confronto Tutti i Modelli", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoca", fontsize=12)
    ax.set_ylabel("Val mIoU (%)", fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0, top=80)

    path = os.path.join(OUTPUT_DIR, "Graph_Val_mIoU.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Salvato: {path}")


def plot_combined(data):
    """Grafico 2×1: loss sopra, val mIoU sotto — ideale per la presentazione."""
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 1, hspace=0.38)

    ax_loss = fig.add_subplot(gs[0])
    ax_miou = fig.add_subplot(gs[1])

    for label, epochs, losses, mious, color, ls in data:
        if losses:
            ax_loss.plot(epochs, losses, label=label, color=color,
                         linestyle=ls, linewidth=2, alpha=0.9)
        if mious:
            ax_miou.plot(range(1, len(mious) + 1), [m * 100 for m in mious],
                         label=label, color=color, linestyle=ls, linewidth=2, alpha=0.9)

    ax_loss.set_title("Training Loss", fontsize=13, fontweight='bold')
    ax_loss.set_xlabel("Epoca", fontsize=11)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax_loss.legend(fontsize=9, loc='upper right', ncol=2)
    ax_loss.grid(True, linestyle='--', alpha=0.4)
    ax_loss.set_xlim(left=1)
    ax_loss.set_ylim(bottom=0)

    ax_miou.set_title("Val mIoU (100 immagini, calcolato ogni epoca)", fontsize=13, fontweight='bold')
    ax_miou.set_xlabel("Epoca", fontsize=11)
    ax_miou.set_ylabel("Val mIoU (%)", fontsize=11)
    ax_miou.legend(fontsize=9, loc='lower right', ncol=2)
    ax_miou.grid(True, linestyle='--', alpha=0.4)
    ax_miou.set_xlim(left=1)
    ax_miou.set_ylim(bottom=0, top=80)

    fig.suptitle("Semantic Segmentation su Cityscapes — Confronto Training",
                 fontsize=14, fontweight='bold', y=0.98)

    path = os.path.join(OUTPUT_DIR, "Graph_Combined.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Salvato: {path}")


def plot_unet_vs_deeplab(data):
    """Due grafici affiancati: U-Net a sinistra, DeepLab a destra."""
    unet_data   = [d for d in data if "U-Net" in d[0]]
    deeplab_data = [d for d in data if "DeepLab" in d[0]]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, group, title in zip(axes, [unet_data, deeplab_data],
                                 ["U-Net — Val mIoU", "DeepLabV3+ — Val mIoU"]):
        for label, epochs, losses, mious, color, ls in group:
            if mious:
                ax.plot(range(1, len(mious) + 1), [m * 100 for m in mious],
                        label=label, color=color, linestyle=ls, linewidth=2.2)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Epoca", fontsize=11)
        ax.set_ylabel("Val mIoU (%)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(left=1)
        ax.set_ylim(0, 80)

    fig.suptitle("Confronto Strategie di Training — Val mIoU per Epoca",
                 fontsize=14, fontweight='bold')

    path = os.path.join(OUTPUT_DIR, "Graph_UNet_vs_DeepLab.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Salvato: {path}")


if __name__ == "__main__":
    print("Parsing log files...")
    data = []
    for label, logfile, color, ls in MODELS:
        filepath = os.path.join(LOG_DIR, logfile)
        epochs, losses, mious = parse_log(filepath)
        if epochs:
            data.append((label, epochs, losses, mious, color, ls))

    if not data:
        print("Nessun log trovato in", LOG_DIR)
    else:
        print(f"\nGenerazione grafici ({len(data)} modelli)...")
        plot_training_loss(data)
        plot_val_miou(data)
        plot_combined(data)
        plot_unet_vs_deeplab(data)
        print("\nFatto! Grafici in:", OUTPUT_DIR)
