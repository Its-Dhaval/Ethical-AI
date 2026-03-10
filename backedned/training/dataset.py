import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


def infer_group_key(path: str, group_by: str = "stem") -> str:
    p = Path(path)
    stem = p.stem.lower()

    # Strip common trailing frame/index suffixes so near-duplicates stay in one split.
    stem = re.sub(r"(?:[_-]?(?:frame|img|image)?)?\d+$", "", stem)
    stem = re.sub(r"[_-]+$", "", stem) or p.stem.lower()

    if group_by == "none":
        return p.as_posix().lower()
    if group_by == "parent":
        return p.parent.name.lower()
    if group_by == "stem_parent":
        return f"{p.parent.name.lower()}::{stem}"
    return stem


def train_val_split(
    items: List[DataItem],
    val_ratio: float = 0.2,
    seed: int = 42,
    group_by: str = "stem",
) -> Tuple[List[DataItem], List[DataItem]]:
    if not items:
        return [], []

    rng = random.Random(seed)
    grouped: Dict[str, List[DataItem]] = defaultdict(list)
    for item in items:
        grouped[infer_group_key(item.path, group_by=group_by)].append(item)

    total_real = sum(1 for x in items if x.label == 0)
    total_fake = sum(1 for x in items if x.label == 1)
    target_val = {
        0: max(1, int(round(total_real * val_ratio))) if total_real > 1 else 0,
        1: max(1, int(round(total_fake * val_ratio))) if total_fake > 1 else 0,
    }

    pure_real_groups: List[str] = []
    pure_fake_groups: List[str] = []
    mixed_groups: List[str] = []
    group_counts: Dict[str, Dict[int, int]] = {}
    for key, gitems in grouped.items():
        c_real = sum(1 for x in gitems if x.label == 0)
        c_fake = sum(1 for x in gitems if x.label == 1)
        group_counts[key] = {0: c_real, 1: c_fake}
        if c_real > 0 and c_fake == 0:
            pure_real_groups.append(key)
        elif c_fake > 0 and c_real == 0:
            pure_fake_groups.append(key)
        else:
            mixed_groups.append(key)

    rng.shuffle(pure_real_groups)
    rng.shuffle(pure_fake_groups)
    rng.shuffle(mixed_groups)

    val_groups = set()
    val_counts = {0: 0, 1: 0}

    def take_groups(keys: List[str], label: int):
        for key in keys:
            if val_counts[label] >= target_val[label]:
                break
            val_groups.add(key)
            val_counts[0] += group_counts[key][0]
            val_counts[1] += group_counts[key][1]

    take_groups(pure_real_groups, label=0)
    take_groups(pure_fake_groups, label=1)

    # Fill remaining deficit using mixed groups if any.
    for key in sorted(mixed_groups, key=lambda k: len(grouped[k]), reverse=True):
        current_deficit = max(target_val[0] - val_counts[0], 0) + max(target_val[1] - val_counts[1], 0)
        if current_deficit <= 0:
            break
        new_deficit = max(target_val[0] - (val_counts[0] + group_counts[key][0]), 0) + max(
            target_val[1] - (val_counts[1] + group_counts[key][1]), 0
        )
        if new_deficit < current_deficit:
            val_groups.add(key)
            val_counts[0] += group_counts[key][0]
            val_counts[1] += group_counts[key][1]

    val_items: List[DataItem] = []
    train_items: List[DataItem] = []
    for key, gitems in grouped.items():
        if key in val_groups:
            val_items.extend(gitems)
        else:
            train_items.extend(gitems)

    # Safety fallback to avoid empty split on tiny datasets.
    if not val_items and train_items:
        val_items = train_items[: max(1, len(train_items) // 5)]
        train_items = train_items[len(val_items):]
    elif not train_items and val_items:
        train_items = val_items[: max(1, len(val_items) // 5)]
        val_items = val_items[len(train_items):]

    rng.shuffle(train_items)
    rng.shuffle(val_items)
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
            y = librosa.util.fix_length(y, size=self.max_samples)
        wave = torch.tensor(y, dtype=torch.float32)
        label = torch.tensor([float(item.label)], dtype=torch.float32)
        return wave, label
