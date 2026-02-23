import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.unet_model import UNet
    from utils.dataset import CityscapesDataset
    from utils.metrics import validate_model
    from config import DEVICE, DATA_ROOT, BATCH_SIZE, ACCUMULATION_STEPS, NUM_CLASSES, CHECKPOINT_DIR
    print("[INFO] Import completati.")
except Exception as e:
    print(f"[ERROR] Errore negli import: {e}")
    sys.exit(1)

FINE_LR      = 1e-5
FINE_EPOCHS  = 30
VAL_IMAGES   = 100


def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, leave=True, desc="Fine-Tuning")
    mean_loss = []
    optimizer.zero_grad()

    for batch_idx, (data, targets) in enumerate(loop):
        data    = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        predictions = model(data)
        loss = loss_fn(predictions, targets) / ACCUMULATION_STEPS
        loss.backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        current_loss = loss.item() * ACCUMULATION_STEPS
        mean_loss.append(current_loss)
        loop.set_postfix(loss=f"{current_loss:.4f}")

    optimizer.step()
    optimizer.zero_grad()
    return sum(mean_loss) / len(mean_loss)


def train_fine():
    print("[INFO] Avvio Fase 2: Fine-Tuning U-Net su gtFine")

    train_ds = CityscapesDataset(root_dir=DATA_ROOT, split='train', mode='fine', augment=True)
    val_ds   = CityscapesDataset(root_dir=DATA_ROOT, split='val',   mode='fine', augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"[INFO] Train set: {len(train_ds)} immagini | Val set: {len(val_ds)} immagini")

    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)

    coarse_weights = os.path.join(CHECKPOINT_DIR, "unet_coarse_best.pth")
    if not os.path.exists(coarse_weights):
        # Fallback al checkpoint finale se il best non esiste
        coarse_weights = os.path.join(CHECKPOINT_DIR, "unet_coarse_final.pth")
    if os.path.exists(coarse_weights):
        print(f"[INFO] Caricamento pesi coarse da: {coarse_weights}")
        model.load_state_dict(torch.load(coarse_weights, map_location=DEVICE))
    else:
        print(f"[ERROR] Pesi coarse NON trovati in {CHECKPOINT_DIR}!")
        sys.exit(1)

    loss_fn   = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=FINE_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, FINE_EPOCHS + 1):
        print(f"\n--- Epoca {epoch}/{FINE_EPOCHS} ---")

        avg_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_miou = validate_model(model, val_loader, NUM_CLASSES, DEVICE,
                                  is_deeplab=False, max_images=VAL_IMAGES)

        scheduler.step(val_miou)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[INFO] Train Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} ({val_miou*100:.2f}%) | LR: {current_lr:.2e}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(CHECKPOINT_DIR, "unet_fine_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] Nuovo best! mIoU={best_miou:.4f} -> {best_path}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "unet_fine_final.pth"))
    print(f"\n[DONE] Fine-tuning completato. Best Val mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    train_fine()
