import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class ImageBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


class AudioMLP(nn.Module):
    """
    Matches the architecture used in audio_detector.py.
    """

    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def extract_audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    try:
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
        valid_f0 = f0[np.isfinite(f0)]
        pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0
    except Exception:
        pitch_std = 0.0

    features = np.array(
        [
            float(np.var(rms)),
            float(np.var(zcr)),
            float(np.mean(flatness)),
            float(np.var(centroid) / max(sr, 1)),
            float(pitch_std / 100.0),
            float(np.var(mfcc) / 2000.0),
        ],
        dtype=np.float32,
    )
    return features
