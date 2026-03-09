import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import librosa
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class DataItem:
    path: str
    label: int  # 0=real, 1=fake


def build_items_from_binary_folders(root_dir: str, exts: set) -> List[DataItem]:
    """
    Expected layout:
    root_dir/
      real/...
      fake/...
    """
    items: List[DataItem] = []
    for folder_name, label in [("real", 0), ("fake", 1)]:
        folder = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder):
            continue
        for base, _, files in os.walk(folder):
            for name in files:
                ext = os.path.splitext(name.lower())[1]
                if ext in exts:
                    items.append(DataItem(path=os.path.join(base, name), label=label))
    return items


def train_val_split(items: List[DataItem], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[DataItem], List[DataItem]]:
    if not items:
        return [], []
    random.seed(seed)

    real = [x for x in items if x.label == 0]
    fake = [x for x in items if x.label == 1]
    random.shuffle(real)
    random.shuffle(fake)

    real_cut = max(1, int(len(real) * (1 - val_ratio))) if len(real) > 1 else len(real)
    fake_cut = max(1, int(len(fake) * (1 - val_ratio))) if len(fake) > 1 else len(fake)

    train_items = real[:real_cut] + fake[:fake_cut]
    val_items = real[real_cut:] + fake[fake_cut:]
    random.shuffle(train_items)
    random.shuffle(val_items)
    return train_items, val_items


class BinaryImageDataset(Dataset):
    def __init__(self, items: List[DataItem], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor([float(item.label)], dtype=torch.float32)
        return image, label


class BinaryAudioWaveDataset(Dataset):
    def __init__(self, items: List[DataItem], sample_rate: int = 16000, max_seconds: int = 30):
        self.items = items
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_seconds

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        y, sr = librosa.load(item.path, sr=self.sample_rate, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
        y = y[: self.max_samples]
        if len(y) < self.max_samples:
            pad_len = self.max_samples - len(y)
            y = librosa.util.fix_length(y, size=self.max_samples)
        wave = torch.tensor(y, dtype=torch.float32)
        label = torch.tensor([float(item.label)], dtype=torch.float32)
        return wave, label
