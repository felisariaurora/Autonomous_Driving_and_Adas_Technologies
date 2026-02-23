"""
train_deeplab_pretrained.py
============================
VERO Fine-Tuning di DeepLabV3+ con backbone ResNet50 pre-addestrato su ImageNet.

Strategia:
- Il backbone ResNet50 viene caricato con i pesi ImageNet (DeepLabV3_ResNet50_Weights.DEFAULT)
- Il classificatore (ultimo layer Conv2d 256->21) viene SOSTITUITO con uno nuovo (256->19)
  per adattarlo alle 19 classi Cityscapes
- Si usano due learning rate distinti:
    * Backbone: lr basso (1e-5) — già pre-addestrato, non va "rotto"
    * Classifier+AuxClassifier: lr alto (1e-4) — nuovo, deve imparare velocemente
- Questo è il vero approccio "fine-tuning" richiesto dall'assignment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from tqdm import tqdm
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import CityscapesDataset
from utils.metrics import validate_model
from config import DEVICE, DATA_ROOT, NUM_CLASSES, CHECKPOINT_DIR

BATCH_SIZE      = 4
EPOCHS          = 40
LR_BACKBONE     = 1e-5   # Basso: il backbone è già pre-addestrato
LR_CLASSIFIER   = 1e-4   # Alto: la head è nuova e deve imparare
VAL_IMAGES      = 100


def build_pretrained_deeplab(num_classes):
    """
    Carica DeepLabV3+ con backbone ResNet50 pre-addestrato su ImageNet,
    poi sostituisce il classificatore per adattarsi alle classi target.
    """
    # Carica il modello completo con pesi COCO/ImageNet
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # Sostituisci il classificatore principale (256 -> num_classes invece di 256 -> 21)
    in_channels_main = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels_main, num_classes, kernel_size=1)

    # Sostituisci anche il classificatore ausiliario (usato durante il training)
    in_channels_aux = model.aux_classifier[4].in_channels
    model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)

    return model


def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, leave=True, desc="DeepLab Pretrained")
    mean_loss = []

    for images, targets in loop:
        images  = images.to(DEVICE)
        targets = targets.long().to(DEVICE)

        optimizer.zero_grad()
        output_dict = model(images)

        # Loss sul classificatore principale
        loss_main = loss_fn(output_dict['out'], targets)

        # Loss sul classificatore ausiliario (contribuisce al training del backbone)
        loss_aux  = loss_fn(output_dict['aux'], targets)

        # Loss totale: pesatura standard DeepLab (aux weight = 0.4)
        loss = loss_main + 0.4 * loss_aux
        loss.backward()
        optimizer.step()

        mean_loss.append(loss_main.item())  # Log solo loss principale per confronto
        loop.set_postfix(loss=f"{loss_main.item():.4f}")

    return sum(mean_loss) / len(mean_loss)


def main():
    print(f"[INFO] DeepLabV3+ TRUE Fine-Tuning (ImageNet backbone) — {DEVICE}")
    print(f"[INFO] Backbone LR: {LR_BACKBONE} | Classifier LR: {LR_CLASSIFIER}")

    train_ds = CityscapesDataset(root_dir=DATA_ROOT, split='train', mode='fine', augment=True)
    val_ds   = CityscapesDataset(root_dir=DATA_ROOT, split='val',   mode='fine', augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"[INFO] Train: {len(train_ds)} immagini | Val: {len(val_ds)} immagini")

    model = build_pretrained_deeplab(NUM_CLASSES).to(DEVICE)
    print(f"[INFO] Backbone pre-addestrato su ImageNet caricato. Head sostituita per {NUM_CLASSES} classi.")

    # Parametri del backbone (lr basso)
    backbone_params = [p for name, p in model.named_parameters()
                       if 'backbone' in name and p.requires_grad]

    # Parametri del classificatore (lr alto)
    classifier_params = [p for name, p in model.named_parameters()
                         if 'backbone' not in name and p.requires_grad]

    optimizer = optim.Adam([
        {'params': backbone_params,    'lr': LR_BACKBONE},
        {'params': classifier_params,  'lr': LR_CLASSIFIER},
    ])

    # Scheduler sul val mIoU
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoca {epoch}/{EPOCHS} ---")

        avg_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_miou = validate_model(model, val_loader, NUM_CLASSES, DEVICE,
                                  is_deeplab=True, max_images=VAL_IMAGES)

        scheduler.step(val_miou)
        lr_bb  = optimizer.param_groups[0]['lr']
        lr_cls = optimizer.param_groups[1]['lr']
        print(f"[INFO] Train Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} ({val_miou*100:.2f}%)")
        print(f"       LR backbone: {lr_bb:.2e} | LR classifier: {lr_cls:.2e}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(CHECKPOINT_DIR, "deeplab_pretrained_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] mIoU={best_miou:.4f} -> {best_path}")

        # Recovery checkpoint
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "deeplab_pretrained_latest.pth"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, f"deeplab_pretrained_epoch_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "deeplab_pretrained_final.pth"))
    print(f"\n[DONE] Fine-tuning completato. Best Val mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()
