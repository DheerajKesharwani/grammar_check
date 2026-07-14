"""
Detection Dataset
=================
Dataset structure expected:
    dataset/vehicles/
        civilian/       ← images of normal cars, bikes, trucks
        police/         ← police vehicle images
        ambulance/      ← ambulance images
        fire_truck/     ← fire truck images

Each folder = one class. Images can be JPG/PNG.
Minimum recommended: 200+ images per class.

Public datasets to source images from:
  - Google Open Images (has emergency vehicle category)
  - COCO (has vehicle categories)
  - Roboflow Universe: search "emergency vehicle"
  - Custom: screenshot from dashcam footage
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import numpy as np


CLASSES = ['civilian', 'police', 'ambulance', 'fire_truck']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class VehicleDataset(Dataset):
    def __init__(self, root, split='train', image_size=224):
        root = Path(root)
        self.samples = []
        self.class_counts = []

        aug = [
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomRotation(10),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
        ] if split == 'train' else []

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            *aug,
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = root / split / cls_name
            if not cls_dir.exists():
                cls_dir = root / cls_name   # flat structure fallback
            if not cls_dir.exists():
                print(f"[Warning] Missing class folder: {cls_dir}")
                self.class_counts.append(0)
                continue
            imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            self.samples += [(str(p), cls_idx) for p in imgs]
            self.class_counts.append(len(imgs))
            print(f"  {cls_name:12s}: {len(imgs)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

    def get_sampler(self):
        """Weighted sampler to handle class imbalance."""
        total = sum(self.class_counts)
        weights_per_class = [total / max(c, 1) for c in self.class_counts]
        sample_weights = [weights_per_class[label] for _, label in self.samples]
        return WeightedRandomSampler(sample_weights, len(sample_weights))


def get_loaders(root, image_size=224, batch_size=32, num_workers=2):
    train_ds = VehicleDataset(root, 'train', image_size)
    val_ds   = VehicleDataset(root, 'val',   image_size)
    sampler  = train_ds.get_sampler()

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader
