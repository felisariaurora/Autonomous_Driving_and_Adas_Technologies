import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import CityscapesDataset
from utils.class_weights import get_class_weights
from utils.metrics import validate_model
from config import DEVICE, DATA_ROOT, NUM_CLASSES, CHECKPOINT_DIR

BATCH_SIZE = 2
EPOCHS     = 50
LR         = 1e-4
VAL_IMAGES = 100


def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, leave=True, desc="DeepLab Coarse")
    mean_loss = []

    for images, targets in loop:
        images  = images.to(DEVICE)
        targets = targets.long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss    = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        mean_loss.append(loss.item())
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return sum(mean_loss) / len(mean_loss)


def main():
    print(f"[INFO] DeepLabV3+ Coarse — {DEVICE}")

    train_ds = CityscapesDataset(root_dir=DATA_ROOT, split='train', mode='coarse', augment=True)
    val_ds   = CityscapesDataset(root_dir=DATA_ROOT, split='val',   mode='fine',   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"[INFO] Train Coarse: {len(train_ds)} immagini | Val: {len(val_ds)} immagini")

    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES).to(DEVICE)

    # Class weights ENet per gestire lo sbilanciamento Cityscapes
    weights = get_class_weights(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=255)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoca {epoch}/{EPOCHS} ---")

        avg_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_miou = validate_model(model, val_loader, NUM_CLASSES, DEVICE,
                                  is_deeplab=True, max_images=VAL_IMAGES)

        scheduler.step(val_miou)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[INFO] Train Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} ({val_miou*100:.2f}%) | LR: {current_lr:.2e}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(CHECKPOINT_DIR, "deeplab_coarse_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] mIoU={best_miou:.4f} -> {best_path}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, f"deeplab_coarse_epoch_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "deeplab_coarse_final.pth"))
    print(f"\n[DONE] Training Coarse completato. Best Val mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()
