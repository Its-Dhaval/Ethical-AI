import os
import tempfile
from dataclasses import dataclass
from typing import List

import librosa
import numpy as np
import torch
import torch.nn as nn

from image_detector import ModelScore


@dataclass
class AudioPrediction:
    label: str
    fake_probability: float
    real_probability: float
    confidence: float
    method: str
    details: dict
    spectrogram_db: np.ndarray
    model_results: List[ModelScore]


class AudioMLP(nn.Module):
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


class AudioDeepfakeDetector:
    def __init__(self, model_path: str = "models/audio_detector.pth", device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.custom_model = None
        self.model_weights = {
            "custom_audio_mlp": 0.40,
            "spectral_artifact_model": 0.30,
            "mfcc_stability_model": 0.20,
            "harmonic_noise_model": 0.15,
            "segment_consistency_model": 0.35,
        }
        self._load_model()

    def _weight_for_model(self, model_name: str) -> float:
        return float(self.model_weights.get(model_name, 0.0))

    def _load_model(self):
        if os.path.exists(self.model_path):
            model = AudioMLP(input_dim=6).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            self.custom_model = model

    def _extract_features(self, y: np.ndarray, sr: int) -> np.ndarray:
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

        return np.array(
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

    def _spectral_model(self, features: np.ndarray) -> float:
        rms_var, zcr_var, flatness_mean, centroid_var, pitch_std_scaled, _mfcc_var = features
        low_energy_variation_fake = 1.0 - float(np.clip(rms_var * 120.0, 0.0, 1.0))
        low_zcr_variation_fake = 1.0 - float(np.clip(zcr_var * 250.0, 0.0, 1.0))
        flatness_fake = float(np.clip((flatness_mean - 0.05) / 0.30, 0.0, 1.0))
        low_pitch_variation_fake = 1.0 - float(np.clip(pitch_std_scaled, 0.0, 1.0))
        low_centroid_variation_fake = 1.0 - float(np.clip(centroid_var * 40.0, 0.0, 1.0))
        score = (
            0.25 * low_energy_variation_fake
            + 0.20 * low_zcr_variation_fake
            + 0.20 * flatness_fake
            + 0.20 * low_pitch_variation_fake
            + 0.15 * low_centroid_variation_fake
        )
        return float(np.clip(score, 0.0, 1.0))

    def _mfcc_stability_model(self, features: np.ndarray) -> float:
        mfcc_var = float(features[5])
        return float(1.0 - np.clip(mfcc_var * 5.0, 0.0, 1.0))

    def _harmonic_noise_model(self, y: np.ndarray) -> float:
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_energy = np.mean(np.abs(y_harmonic)) + 1e-8
            percussive_energy = np.mean(np.abs(y_percussive))
            ratio = percussive_energy / harmonic_energy
            return float(np.clip(ratio * 3.0, 0.0, 1.0))
        except Exception:
            return 0.5

    def _segment_scores(self, y: np.ndarray, sr: int) -> List[float]:
        window = int(3.0 * sr)
        hop = int(1.5 * sr)
        if len(y) <= window:
            return [self._spectral_model(self._extract_features(y, sr))]
        scores = []
        for start in range(0, max(1, len(y) - window + 1), hop):
            segment = y[start : start + window]
            if len(segment) < window:
                continue
            scores.append(self._spectral_model(self._extract_features(segment, sr)))
        return scores if scores else [0.5]

    def _segment_consistency_model(self, segment_scores: List[float]) -> float:
        mean_score = float(np.mean(segment_scores))
        std_score = float(np.std(segment_scores))
        consistency_bonus = 1.0 - min(std_score * 2.0, 1.0)
        return float(np.clip(0.7 * mean_score + 0.3 * consistency_bonus, 0.0, 1.0))

    def predict(self, audio_path: str) -> AudioPrediction:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
        y = y[: 30 * sr]

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features = self._extract_features(y, sr)

        model_results: List[ModelScore] = []
        if self.custom_model is not None:
            x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logit = self.custom_model(x).squeeze()
            model_results.append(ModelScore("custom_audio_mlp", round(float(torch.sigmoid(logit).item()), 4)))

        spectral_score = self._spectral_model(features)
        mfcc_score = self._mfcc_stability_model(features)
        hnr_score = self._harmonic_noise_model(y)
        segment_scores = self._segment_scores(y, sr)
        segment_consistency = self._segment_consistency_model(segment_scores)

        model_results.append(ModelScore("spectral_artifact_model", round(spectral_score, 4)))
        model_results.append(ModelScore("mfcc_stability_model", round(mfcc_score, 4)))
        model_results.append(ModelScore("harmonic_noise_model", round(hnr_score, 4)))
        model_results.append(ModelScore("segment_consistency_model", round(segment_consistency, 4)))

        weighted_sum = 0.0
        total_weight = 0.0
        for item in model_results:
            w = self._weight_for_model(item.model_name)
            if w <= 0:
                continue
            weighted_sum += item.fake_probability * w
            total_weight += w

        fake_prob = weighted_sum / total_weight if total_weight > 0 else float(
            np.mean([m.fake_probability for m in model_results])
        )
        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))

        real_prob = 1.0 - fake_prob
        label = "Fake" if fake_prob >= 0.5 else "Real"
        confidence = abs(fake_prob - 0.5) * 2.0

        details = {
            "num_models_used": len(model_results),
            "combination": "weighted_average",
            "weights_used": {m.model_name: self._weight_for_model(m.model_name) for m in model_results},
            "num_audio_segments": len(segment_scores),
        }

        return AudioPrediction(
            label=label,
            fake_probability=round(fake_prob, 4),
            real_probability=round(real_prob, 4),
            confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
            method="ensemble_weighted_average",
            details=details,
            spectrogram_db=mel_db,
            model_results=model_results,
        )


def save_uploaded_audio_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name
