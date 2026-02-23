import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import os
import sys

# Fix per certificati SSL
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --- Local Imports ---
from models.unet_model import UNet
from config import DEVICE, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH

# Palette Cityscapes
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
])

def decode_segmap(mask):
    r, g, b = np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)
    for l in range(0, NUM_CLASSES):
        idx = mask == l
        r[idx], g[idx], b[idx] = CITYSCAPES_PALETTE[l]
    return np.stack([r, g, b], axis=2).astype(np.uint8)

def get_model(ckpt_name):
    """Riconosce l'architettura dal nome del file"""
    if "deeplab" in ckpt_name.lower():
        print(f"   [ARCHITETTURA] DeepLabV3+ (ResNet50)")
        model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    else:
        print(f"   [ARCHITETTURA] U-Net Custom")
        model = UNet(n_channels=3, n_classes=NUM_CLASSES)
    return model.to(DEVICE)

def predict_and_save(image_path, ckpt_path, output_dir):
    ckpt_name = os.path.basename(ckpt_path).replace(".pth", "")
    save_path = os.path.join(output_dir, f"VIS_{ckpt_name}.png")

    # Inizializza il modello corretto
    model = get_model(ckpt_name)

    # Caricamento pesi
    try:
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"ERR- Impossibile caricare {ckpt_name}: {e}")
        return

    # Preprocessing
    try:
        img = Image.open(image_path).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    except FileNotFoundError:
        print(f"ERR- Immagine non trovata: {image_path}")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Inferenza
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict): output = output['out']
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    pred_color = decode_segmap(pred_mask)
    plt.figure(figsize=(14, 6))

    # Input
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Originale")
    plt.axis('off')

    # Output
    plt.subplot(1, 2, 2)
    plt.imshow(pred_color)
    plt.title(f"Predizione: {ckpt_name}")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Salvato: {save_path}")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    TEST_IMAGE = "./data/cityscapes/leftImg8bit/val/munster/munster_000008_000019_leftImg8bit.png"

    CKPT_DIR = "./checkpoints/"
    OUT_DIR = "./results_comparison/"

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CKPT_DIR):
        print(f"ERR- Cartella checkpoint non trovata: {CKPT_DIR}")
        sys.exit(1)

    all_ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")]
    all_ckpts.sort()


    for ckpt_file in all_ckpts:
        full_path = os.path.join(CKPT_DIR, ckpt_file)
        print(f"\nProcessing: {ckpt_file}...")
        predict_and_save(TEST_IMAGE, full_path, OUT_DIR)

    print("\nLe immagini sono nella cartella 'results_comparison'")