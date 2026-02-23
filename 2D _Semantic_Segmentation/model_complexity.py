import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from models.unet_model import UNet
from config import NUM_CLASSES

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_stats(name, model):
    total_params = count_parameters(model)
    print(f"\n📊 {name} Statistics:")
    print(f"   - Parametri Totali: {total_params:,}")
    print(f"   - Peso Stimato (MB): {total_params * 4 / 1024 / 1024:.2f} MB")

    # Conto i layer convoluzionali (approssimazione)
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    print(f"   - Numero Layer Conv2d: {len(conv_layers)}")

print("-" * 40)
print("   CONFRONTO COMPLESSITÀ MODELLI")
print("-" * 40)

# 1. U-Net
unet = UNet(n_channels=3, n_classes=NUM_CLASSES)
print_model_stats("U-Net Custom", unet)

# 2. DeepLabV3+ (ResNet50 Backbone)
deeplab = deeplabv3_resnet50(num_classes=NUM_CLASSES)
print_model_stats("DeepLabV3+ (ResNet50)", deeplab)

print("-" * 40)