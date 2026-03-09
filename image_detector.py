import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import InferenceClient
from PIL import Image
from torchvision import models, transforms


@dataclass
class ModelScore:
    model_name: str
    fake_probability: float


@dataclass
class ImagePrediction:
    label: str
    fake_probability: float
    real_probability: float
    confidence: float
    method: str
    details: dict
    model_results: List[ModelScore]


class BinaryResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


class ImageDeepfakeDetector:
    def __init__(self, model_path: str = "models/image_detector.pth", device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fake_threshold = 0.45

        self.hf_model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        self.hf_display_name = "Deep-Fake-Detector-v2-Model"

        self.model_weights = {
            "custom_resnet_binary": 0.30,
            "resnet_uncertainty": 0.15,
            "efficientnet_uncertainty": 0.15,
            "texture_artifact_score": 0.10,
            "frequency_artifact_score": 0.10,
            "face_region_score": 0.20,
            # Give higher influence to the HF deepfake model.
            "Deep-Fake-Detector-v2-Model": 0.55,
        }

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.custom_model = None
        self.resnet_model = None
        self.efficientnet_model = None
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.hf_client = None
        self.hf_ready = False
        self._load_models()
        self._init_hf_client()

    def _load_models(self):
        try:
            self.resnet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            self.resnet_model = models.resnet18(weights=None)
        self.resnet_model.to(self.device).eval()

        try:
            self.efficientnet_model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        except Exception:
            self.efficientnet_model = models.efficientnet_b0(weights=None)
        self.efficientnet_model.to(self.device).eval()

        if os.path.exists(self.model_path):
            model = BinaryResNet18().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            self.custom_model = model

    def _init_hf_client(self):
        try:
            hf_token = (
                os.getenv("HF_TOKEN")
                or os.getenv("HF_API_TOKEN")
                or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
            if hf_token:
                hf_token = hf_token.strip().strip("\"").strip("'")
            if not hf_token:
                raise KeyError("No HF token found")
            self.hf_client = InferenceClient(
                provider="hf-inference",
                api_key=hf_token,
            )
            self.hf_ready = True
        except Exception:
            self.hf_client = None
            self.hf_ready = False

    def _weight_for_model(self, model_name: str) -> float:
        base_key = model_name.split(":", 1)[0]
        return float(self.model_weights.get(base_key, 0.0))

    def _tensor_from_pil(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def _softmax_uncertainty_fake_score(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        max_entropy = np.log(probs.shape[1])
        return float((entropy / max_entropy).item())

    def _tta_uncertainty(self, model: nn.Module, image_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            logits_a = model(image_tensor)
            logits_b = model(torch.flip(image_tensor, dims=[3]))
        score_a = self._softmax_uncertainty_fake_score(logits_a)
        score_b = self._softmax_uncertainty_fake_score(logits_b)
        return float((score_a + score_b) / 2.0)

    def _texture_score(self, rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return 1.0 - float(np.clip(lap_var / 400.0, 0.0, 1.0))

    def _frequency_score(self, rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        freq = np.fft.fft2(gray)
        freq_shift = np.fft.fftshift(freq)
        mag = np.abs(freq_shift)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        low_r = min(h, w) // 8
        y, x = np.ogrid[:h, :w]
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        low_mask = dist_sq <= (low_r ** 2)
        high_mask = ~low_mask
        low_energy = np.mean(mag[low_mask]) + 1e-8
        high_energy = np.mean(mag[high_mask])
        ratio = high_energy / low_energy
        return float(np.clip(ratio / 3.0, 0.0, 1.0))

    def _detect_largest_face(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.15 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(rgb.shape[1], x + w + pad)
        y2 = min(rgb.shape[0], y + h + pad)
        return rgb[y1:y2, x1:x2]

    def _face_region_score(self, face_rgb: np.ndarray) -> float:
        face_img = Image.fromarray(face_rgb)
        face_tensor = self._tensor_from_pil(face_img)
        resnet_score = self._tta_uncertainty(self.resnet_model, face_tensor)
        eff_score = self._tta_uncertainty(self.efficientnet_model, face_tensor)
        texture_score = self._texture_score(face_rgb)
        frequency_score = self._frequency_score(face_rgb)
        return float(np.mean([resnet_score, eff_score, texture_score, frequency_score]))

    def _extract_hf_fake_probability(self, output) -> Optional[float]:
        if not output:
            return None
        fake_keys = ["fake", "deepfake", "ai", "synthetic", "spoof"]
        real_keys = ["real", "bonafide", "human", "authentic", "genuine"]
        fake_prob = 0.0
        real_prob = 0.0
        matched = False

        for item in output:
            if isinstance(item, dict):
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
            else:
                label = str(getattr(item, "label", "")).lower()
                score = float(getattr(item, "score", 0.0))

            if any(k in label for k in fake_keys):
                fake_prob += score
                matched = True
            elif any(k in label for k in real_keys):
                real_prob += score
                matched = True

        if not matched:
            return None
        if fake_prob > 0:
            return float(np.clip(fake_prob, 0.0, 1.0))
        return float(np.clip(1.0 - real_prob, 0.0, 1.0))

    def _hf_image_score(self, image: Image.Image):
        if not self.hf_ready or self.hf_client is None:
            return None, "HF client not initialized (set HF_TOKEN/HF_API_TOKEN/HUGGINGFACEHUB_API_TOKEN)"
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    image.save(tmp_file, format="JPEG")
                    tmp_path = tmp_file.name

                output = self.hf_client.image_classification(
                    image=tmp_path,
                    model=self.hf_model_id,
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            fake_prob = self._extract_hf_fake_probability(output)
            if fake_prob is None:
                return None, "HF model returned labels without fake/real mapping"
            return float(np.clip(fake_prob, 0.0, 1.0)), None
        except Exception as e:
            return None, str(e)

    def predict(self, image: Image.Image, include_hf: bool = True) -> ImagePrediction:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = self._tensor_from_pil(image)
        rgb = np.array(image)
        model_scores: List[ModelScore] = []

        if self.custom_model is not None:
            with torch.no_grad():
                logit = self.custom_model(image_tensor).squeeze()
                custom_fake = float(torch.sigmoid(logit).item())
            model_scores.append(ModelScore("custom_resnet_binary", round(custom_fake, 4)))

        model_scores.append(
            ModelScore("resnet_uncertainty", round(self._tta_uncertainty(self.resnet_model, image_tensor), 4))
        )
        model_scores.append(
            ModelScore(
                "efficientnet_uncertainty",
                round(self._tta_uncertainty(self.efficientnet_model, image_tensor), 4),
            )
        )
        model_scores.append(ModelScore("texture_artifact_score", round(self._texture_score(rgb), 4)))
        model_scores.append(ModelScore("frequency_artifact_score", round(self._frequency_score(rgb), 4)))

        face_crop = self._detect_largest_face(rgb)
        if face_crop is not None and face_crop.size > 0:
            model_scores.append(ModelScore("face_region_score", round(self._face_region_score(face_crop), 4)))

        hf_error = None
        hf_scored = False
        hf_score = None
        if include_hf:
            hf_score, hf_error = self._hf_image_score(image)
            if hf_score is not None:
                hf_scored = True
                model_scores.append(ModelScore(self.hf_display_name, round(float(hf_score), 4)))

        weighted_sum = 0.0
        total_weight = 0.0
        for item in model_scores:
            w = self._weight_for_model(item.model_name)
            if w <= 0:
                continue
            weighted_sum += item.fake_probability * w
            total_weight += w

        fake_prob = weighted_sum / total_weight if total_weight > 0 else float(
            np.mean([m.fake_probability for m in model_scores])
        )
        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))

        strong_fake = sum(1 for m in model_scores if m.fake_probability >= 0.60)
        very_strong_fake = any(m.fake_probability >= 0.75 for m in model_scores)
        if strong_fake >= 2:
            fake_prob = max(fake_prob, 0.55)
        if very_strong_fake:
            fake_prob = max(fake_prob, 0.62)

        real_prob = 1.0 - fake_prob
        label = "Fake" if fake_prob >= self.fake_threshold else "Real"
        confidence = abs(fake_prob - 0.5) * 2.0

        details = {
            "num_models_used": len(model_scores),
            "combination": "weighted_average",
            "weights_used": {m.model_name: self._weight_for_model(m.model_name) for m in model_scores},
            "face_detected": face_crop is not None,
            "decision_threshold": self.fake_threshold,
            "hf_model": self.hf_model_id,
            "hf_scored": hf_scored,
            "hf_error": hf_error,
            "hf_token_present": self.hf_ready,
        }

        return ImagePrediction(
            label=label,
            fake_probability=round(fake_prob, 4),
            real_probability=round(real_prob, 4),
            confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
            method="ensemble_weighted_average",
            details=details,
            model_results=model_scores,
        )
