"""
VBAPS Dataset & DataLoader utilities.

Scans data/raw_level7/ for downloaded master tiles, builds a hex-to-class
mapping from the filenames, and serves random 224×224 crops that preserve
the native 10 m/px GSD (REQ-2.2.2).
"""

import os
import glob
import h3
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Discover tiles on disk and build the class map

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_level7")
TEST_DIR = os.path.join(BASE_DIR, "data", "test_2024")


def build_class_map(raw_dir: str = RAW_DIR):
    """
    Scan the raw tile directory and return:
      - file_paths : list[str]       – absolute paths to every .png
      - hex_ids    : list[str]       – matching H3 hex id per file
      - hex_to_idx : dict[str, int]  – hex id  → integer class label
      - idx_to_hex : dict[int, str]  – class label → hex id
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    hex_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    unique_hexes = sorted(set(hex_ids))
    hex_to_idx = {h: i for i, h in enumerate(unique_hexes)}
    idx_to_hex = {i: h for h, i in hex_to_idx.items()}
    return paths, hex_ids, hex_to_idx, idx_to_hex


def hex_center_coords(idx_to_hex: dict):
    """Return a (num_classes, 2) numpy array of [lat, lon] per class index."""
    num_classes = len(idx_to_hex)
    coords = np.zeros((num_classes, 2), dtype=np.float64)
    for idx, hex_id in idx_to_hex.items():
        lat, lon = h3.cell_to_latlng(hex_id)
        coords[idx] = [lat, lon]
    return coords


# Augmentation pipelines

def get_train_transform():
    """Light augmentations for training crops.

    Rotation is handled BEFORE this by the rotated-crop extraction in
    __getitem__, so we only add flip + colour here.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform():
    """Deterministic centre crop for validation."""
    return A.Compose([
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_test_transform():
    """Clean transform for 2024 test images (already 224x224)."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# Rotated-crop helper
#
# A 224×224 square rotated by any angle needs a bounding circle of radius
# 224*√2/2 ≈ 159 px. The crop centre must be at least that far from every
# edge of the 512×512 master tile to avoid black fill.
_CROP = 224
_HALF_DIAG = int(np.ceil(_CROP * np.sqrt(2) / 2))  # 159


def _random_rotated_crop(img, crop_size=_CROP, half_diag=_HALF_DIAG):
    """Extract a crop_size×crop_size patch at a random angle from img.

    The crop centre is sampled uniformly within the safe interior zone
    of the 512×512 tile, and the heading angle is uniform in [0, 360).
    Uses cv2.warpAffine — no resizing, preserves native GSD.
    """
    h, w = img.shape[:2]
    margin = half_diag
    # Random centre within the safe zone
    cx = np.random.randint(margin, w - margin)
    cy = np.random.randint(margin, h - margin)
    # Random heading angle (continuous, full 360°)
    angle = np.random.uniform(0, 360)

    # Build rotation matrix centred on (cx, cy)
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
    # Shift so the crop centre lands at (crop_size/2, crop_size/2)
    M[0, 2] += crop_size / 2 - cx
    M[1, 2] += crop_size / 2 - cy

    crop = cv2.warpAffine(
        img, M, (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return crop


# Datasets

class MasterTileDataset(Dataset):
    """Training / validation dataset.

    Each 512×512 master tile yields one 224×224 sample per epoch:
      - Training  → random position + random heading angle (rotated crop)
      - Validation → deterministic centre crop (0° heading)

    Over thousands of epochs the model sees thousands of unique
    perspectives of every hexagon, removing rotational invariance.
    """

    def __init__(self, file_paths, hex_ids, hex_to_idx, transform=None,
                 random_crop=True):
        self.file_paths = file_paths
        self.labels = [hex_to_idx[h] for h in hex_ids]
        self.transform = transform
        self.random_crop = random_crop

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        img = np.array(img)  # (512, 512, 3) uint8

        # 224×224 crop — NO resize, preserves native 10 m/px GSD
        if self.random_crop:
            img = _random_rotated_crop(img)
        # else: val transform applies CenterCrop(224,224) via albumentations

        if self.transform:
            img = self.transform(image=img)["image"]

        label = self.labels[idx]
        return img, label


class TestTileDataset(Dataset):
    """
    Test dataset (2024 temporal epoch).

    Images are already 224×224 as downloaded by fetch_test_set.py.
    A random rotation (0–360°) is applied to test rotational invariance,
    matching the random-heading crops used during training.
    """

    def __init__(self, test_dir, hex_to_idx, transform=None):
        self.paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))
        self.hex_ids = [os.path.splitext(os.path.basename(p))[0]
                        for p in self.paths]
        self.labels = [hex_to_idx[h] for h in self.hex_ids]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(img)

        # Random rotation around centre to test rotational invariance
        h, w = img.shape[:2]
        angle = np.random.uniform(0, 360)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        if self.transform:
            img = self.transform(image=img)["image"]

        label = self.labels[idx]
        return img, label


# DataLoader factory functions

def get_dataloaders(batch_size=32, val_ratio=0.2, seed=42, num_workers=0):
    """
    Return train_loader, val_loader, class metadata.

    Training: ALL tiles with random rotated crops (different each epoch).
    Validation: ALL tiles with deterministic centre crops.

    This is critical because each hexagon has exactly one master tile.
    A random split would make ~20% of classes unlearnable (the model
    would never see those hexagons during training).  Instead, val
    measures whether the model can recognise hexagons from a fixed,
    canonical perspective.  The 2024 test set is the true held-out eval.
    """
    paths, hex_ids, hex_to_idx, idx_to_hex = build_class_map()
    num_classes = len(hex_to_idx)
    print(f"[dataset] Found {len(paths)} tiles across {num_classes} classes")

    pin = torch.cuda.is_available()  # MPS does not support pin_memory

    train_ds = MasterTileDataset(
        paths, hex_ids, hex_to_idx,
        transform=get_train_transform(), random_crop=True,
    )
    val_ds = MasterTileDataset(
        paths, hex_ids, hex_to_idx,
        transform=get_val_transform(), random_crop=False,  # centre crop
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    meta = {
        "num_classes": num_classes,
        "hex_to_idx": hex_to_idx,
        "idx_to_hex": idx_to_hex,
        "coords": hex_center_coords(idx_to_hex),
    }

    return train_loader, val_loader, meta


def get_test_loader(hex_to_idx, batch_size=32, num_workers=0):
    """Return the 2024 temporal test loader (call after fetch_test_set.py)."""
    ds = TestTileDataset(TEST_DIR, hex_to_idx, transform=get_test_transform())
    print(f"[dataset] Test set: {len(ds)} images")
    pin = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin)
