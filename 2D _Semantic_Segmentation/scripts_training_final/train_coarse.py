import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_model import UNet
from utils.dataset import CityscapesDataset
from utils.class_weights import get_class_weights
from utils.metrics import validate_model
from config import (
    DEVICE, DATA_ROOT, BATCH_SIZE, ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_EPOCHS, NUM_CLASSES, CHECKPOINT_DIR
)

VAL_IMAGES = 100  # Immagini usate per la validation rapida ogni epoca


def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, leave=True, desc="Training Coarse")
    mean_loss = []
    optimizer.zero_grad()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        predictions = model(data)
        loss = loss_fn(predictions, targets)
        loss = loss / ACCUMULATION_STEPS
        loss.backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        current_loss = loss.item() * ACCUMULATION_STEPS
        mean_loss.append(current_loss)
        loop.set_postfix(loss=f"{current_loss:.4f}")

    # Flush dei gradienti residui a fine epoca
    optimizer.step()
    optimizer.zero_grad()

    return sum(mean_loss) / len(mean_loss)


def main():
    print(f"--- CONFIGURAZIONE U-NET COARSE ---")
    print(f"[DEVICE] {DEVICE}")
    print(f"[BATCH SIZE EFFETTIVO] {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"-----------------------------------")

    train_ds = CityscapesDataset(root_dir=DATA_ROOT, split='train', mode='coarse', augment=True)
    val_ds   = CityscapesDataset(root_dir=DATA_ROOT, split='val',   mode='fine',   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)

    weights = get_class_weights(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=255)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_miou = 0.0
    print(f"\n[START] Training Coarse — {NUM_EPOCHS} epoche")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoca {epoch+1}/{NUM_EPOCHS} ---")

        avg_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)

        val_miou = validate_model(model, val_loader, NUM_CLASSES, DEVICE,
                                  is_deeplab=False, max_images=VAL_IMAGES)

        scheduler.step(val_miou)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[INFO] Train Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} ({val_miou*100:.2f}%) | LR: {current_lr:.2e}")

        # Salva il miglior modello basandosi su val mIoU
        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(CHECKPOINT_DIR, "unet_coarse_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] Nuovo miglior modello! mIoU={best_miou:.4f} -> {best_path}")

        # Checkpoint storico ogni 10 epoche
        if (epoch + 1) % 10 == 0:
            ckpt = os.path.join(CHECKPOINT_DIR, f"unet_coarse_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)

    # Salvataggio finale
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "unet_coarse_final.pth"))
    print(f"\n[DONE] Training Coarse completato. Best Val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
