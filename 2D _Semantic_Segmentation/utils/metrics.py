import torch
import numpy as np


def calculate_iou(pred_mask, true_mask, num_classes=19, ignore_index=255):
    """
    Calcola mIoU su un singolo batch.
    Usata durante la valutazione finale con evaluate_metrics.py.
    """
    iou_list = []

    pred_mask = pred_mask.view(-1).cpu().numpy()
    true_mask = true_mask.view(-1).cpu().numpy()

    valid_pixels = true_mask != ignore_index
    pred_mask = pred_mask[valid_pixels]
    true_mask = true_mask[valid_pixels]

    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = true_mask == cls

        intersection = (pred_inds & target_inds).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection

        if union == 0:
            continue

        iou_list.append(intersection / union)

    return np.mean(iou_list) if iou_list else 0.0


def validate_model(model, loader, num_classes, device, is_deeplab=False, max_images=100):
    """
    Esegue una validation rapida durante il training.

    Args:
        model: il modello PyTorch (già in modalità eval viene rimesso in train alla fine)
        loader: DataLoader del validation set
        num_classes: numero di classi (19 per Cityscapes)
        device: dispositivo CUDA/CPU
        is_deeplab: se True, il modello restituisce un dict e si usa ['out']
        max_images: numero massimo di immagini da valutare (default 100 per velocità)

    Returns:
        float: mIoU sul subset di validation
    """
    model.eval()
    total_inter = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i >= max_images:
                break

            images = images.to(device)
            targets_np = targets.numpy()  # [B, H, W]

            outputs = model(images)
            if is_deeplab:
                outputs = outputs['out']

            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # [B, H, W]

            # Maschera pixel validi (escludi ignore_index=255)
            valid = (targets_np != 255)

            for cls in range(num_classes):
                pred_cls = (preds == cls) & valid
                true_cls = (targets_np == cls) & valid

                inter = (pred_cls & true_cls).sum()
                union = (pred_cls | true_cls).sum()

                total_inter[cls] += inter
                total_union[cls] += union

    # Calcolo mIoU: ignora classi mai presenti nel subset
    valid_classes = total_union > 0
    if valid_classes.sum() == 0:
        model.train()
        return 0.0

    iou = total_inter[valid_classes] / total_union[valid_classes]
    miou = float(np.mean(iou))

    model.train()
    return miou
