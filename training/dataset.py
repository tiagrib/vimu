"""
Dataset for VIMU training.

Expects:
    data/
        frames/*.jpg
        labels.csv  →  frame, joint_1, ..., joint_N, base_roll, base_pitch

Base orientation columns may contain empty values for frames collected
on a flat surface (they'll be 0.0 from the sweep collector). Frames
with non-zero base values come from the tilted collection mode.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_synthetic_transforms(size: int = 224):
    """Transforms for synthetic data — aggressive augmentation to bridge sim-to-real gap."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


def get_val_transforms(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class VimuDataset(Dataset):
    BASE_COLS = ["base_roll", "base_pitch"]

    def __init__(self, data_dir: str, num_joints: int = 6, transform=None):
        self.frames_dir = os.path.join(data_dir, "frames")
        self.num_joints = num_joints
        self.transform = transform or get_val_transforms()

        self.df = pd.read_csv(os.path.join(data_dir, "labels.csv"))

        self.joint_cols = [f"joint_{i+1}" for i in range(num_joints)]
        for col in self.joint_cols:
            assert col in self.df.columns, f"Missing column: {col}"

        self.has_base = all(c in self.df.columns for c in self.BASE_COLS)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        fname = str(row["frame"])
        if not fname.endswith((".jpg", ".png", ".jpeg")):
            fname += ".jpg"
        img = Image.open(os.path.join(self.frames_dir, fname)).convert("RGB")
        img = self.transform(img)

        # Joint angles (always present)
        joints = torch.tensor([float(row[c]) for c in self.joint_cols], dtype=torch.float32)

        # Base state
        if self.has_base:
            base_vals = []
            base_mask = []
            for c in self.BASE_COLS:
                val = row[c]
                if pd.notna(val):
                    base_vals.append(float(val))
                    base_mask.append(True)
                else:
                    base_vals.append(0.0)
                    base_mask.append(False)
            base = torch.tensor(base_vals, dtype=torch.float32)
            base_mask = torch.tensor(base_mask, dtype=torch.bool)
        else:
            base = torch.zeros(2)
            base_mask = torch.zeros(2, dtype=torch.bool)

        return {
            "image": img,
            "joints": joints,
            "base": base,
            "base_mask": base_mask,
        }
