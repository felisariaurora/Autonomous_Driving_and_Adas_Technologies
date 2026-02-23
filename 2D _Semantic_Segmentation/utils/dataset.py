import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from config import IMG_HEIGHT, IMG_WIDTH, DEBUG_MODE

class CityscapesDataset(Dataset):
    """
    Custom Dataset per Cityscapes.
    Supporta augmentation sincronizzata su immagine e maschera.
    """

    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    def __init__(self, root_dir, split='train', mode='fine', augment=False):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.augment = augment

        mask_suffix = '_gtCoarse_labelIds.png' if mode == 'coarse' else '_gtFine_labelIds.png'

        if mode == 'coarse':
            self.images_dir = os.path.join(self.root_dir, 'leftImg8bit', split)
            self.targets_dir = os.path.join(self.root_dir, 'gtCoarse', split)
        else:
            self.images_dir = os.path.join(self.root_dir, 'leftImg8bit', split)
            self.targets_dir = os.path.join(self.root_dir, 'gtFine', split)

        self.images = []
        self.targets = []

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory immagini non trovata: {self.images_dir}")

        for city in sorted(os.listdir(self.images_dir)):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            if not os.path.isdir(img_dir):
                continue

            for file_name in sorted(os.listdir(img_dir)):
                if file_name.endswith('.png'):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_name = file_name.replace('_leftImg8bit.png', mask_suffix)
                    self.targets.append(os.path.join(target_dir, target_name))

        if DEBUG_MODE:
            print("-" * 40)
            print(f"[DEBUG] Caricamento in modalità: {mode.upper()}")
            print(f"[DEBUG] Subset di 20 immagini attivo.")
            print("-" * 40)
            self.images = self.images[:20]
            self.targets = self.targets[:20]

        # Mapping ufficiale Cityscapes: 34 classi raw -> 19 classi training + 255 ignore
        self.mapping = {
            0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
            7: 0,   8: 1,   9: 255, 10: 255, 11: 2,  12: 3,  13: 4,
            14: 255, 15: 255, 16: 255, 17: 5,  18: 255, 19: 6,  20: 7,
            21: 8,  22: 9,  23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
        }

        # Lookup table vettorizzata per encode_target veloce (34 -> 19 classi)
        self._lut = np.full(256, 255, dtype=np.uint8)
        for k, v in self.mapping.items():
            if 0 <= k <= 255:
                self._lut[k] = v

    def encode_target(self, mask):
        return self._lut[mask]

    def _apply_augmentation(self, image, target):
        """
        Applica data augmentation con lo stesso seed su immagine e maschera.
        Le augmentation spaziali (flip, crop) vengono applicate a entrambe.
        Le augmentation di colore vengono applicate solo all'immagine.
        """
        # 1. Random Horizontal Flip (50%)
        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        # 2. Random Scale + Crop (scale tra 0.75 e 1.25 della dimensione target)
        scale = random.uniform(0.75, 1.25)
        new_h = int(IMG_HEIGHT * scale)
        new_w = int(IMG_WIDTH * scale)
        image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
        target = TF.resize(target, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

        # Crop o pad per riportare alle dimensioni originali
        if new_h >= IMG_HEIGHT and new_w >= IMG_WIDTH:
            i = random.randint(0, new_h - IMG_HEIGHT)
            j = random.randint(0, new_w - IMG_WIDTH)
            image = TF.crop(image, i, j, IMG_HEIGHT, IMG_WIDTH)
            target = TF.crop(target, i, j, IMG_HEIGHT, IMG_WIDTH)
        else:
            # Se la scala è < 1.0 faccio pad e poi crop centrato
            pad_h = max(IMG_HEIGHT - new_h, 0)
            pad_w = max(IMG_WIDTH - new_w, 0)
            image = TF.pad(image, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=0)
            target = TF.pad(target, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=255)
            image = TF.center_crop(image, (IMG_HEIGHT, IMG_WIDTH))
            target = TF.center_crop(target, (IMG_HEIGHT, IMG_WIDTH))

        # 3. Color Jitter (solo sull'immagine, non sulla maschera)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
        if random.random() > 0.3:
            image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))

        return image, target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.targets[idx])

        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        target = target.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)

        if self.augment:
            image, target = self._apply_augmentation(image, target)

        image = TF.to_tensor(image)
        target_np = self.encode_target(np.array(target, dtype=np.uint8))
        target_tensor = torch.from_numpy(target_np).long()

        # Normalizzazione ImageNet
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, target_tensor
