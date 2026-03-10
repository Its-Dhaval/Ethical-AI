import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class ImageBinaryClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = self._build_backbone(backbone=backbone, pretrained=pretrained, dropout=dropout)

    def forward(self, x):
        return self.backbone(x)

    def _build_backbone(self, backbone: str, pretrained: bool, dropout: float) -> nn.Module:
        if backbone == "resnet18":
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.resnet18(weights=weights)
            except Exception:
                model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
            return model

        if backbone == "resnet50":
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                model = models.resnet50(weights=weights)
            except Exception:
                model = models.resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
            return model

        if backbone == "efficientnet_b0":
            try:
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.efficientnet_b0(weights=weights)
            except Exception:
                model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 1)
            return model

        if backbone == "efficientnet_v2_s":
            try:
                weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.efficientnet_v2_s(weights=weights)
            except Exception:
                model = models.efficientnet_v2_s(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 1)
            return model

        if backbone == "convnext_tiny":
            try:
                weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
                model = models.convnext_tiny(weights=weights)
            except Exception:
                model = models.convnext_tiny(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 1)
            return model

        raise ValueError(
            "Unsupported backbone. Use one of: "
            "resnet18, resnet50, efficientnet_b0, efficientnet_v2_s, convnext_tiny."
        )


def default_img_size_for_backbone(backbone: str) -> int:
    if backbone in {"resnet18", "resnet50", "efficientnet_b0"}:
        return 320
    if backbone in {"efficientnet_v2_s", "convnext_tiny"}:
        return 384
    return 320


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
