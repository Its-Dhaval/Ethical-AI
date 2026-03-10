import json
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from huggingface_hub import InferenceClient
from PIL import Image
from torchvision import transforms

from training.models import ImageBinaryClassifier, default_img_size_for_backbone


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


class ImageDeepfakeDetector:
    def __init__(
        self,
        model_path: str = "models/image_detector.pth",
        ensemble_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.ensemble_path = ensemble_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Default values are replaced by checkpoint metadata/calibration files when present.
        self.fake_threshold = 0.5
        self.backbone = "efficientnet_b0"
        self.img_size = 320
        self.custom_model: Optional[torch.nn.Module] = None

        self.hf_model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        self.hf_display_name = "Deep-Fake-Detector-v2-Model"

        self.model_weights = {
            "custom_model_score": 0.55,
            "texture_artifact_score": 0.10,
            "frequency_artifact_score": 0.10,
            "face_region_score": 0.25,
            "Deep-Fake-Detector-v2-Model": 0.35,
        }

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.hf_client = None
        self.hf_ready = False

        self._load_custom_model()
        self._load_ensemble_calibration()
        self._init_hf_client()

    def _init_preprocess(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_custom_model(self):
        if not os.path.exists(self.model_path):
            return

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint
        meta = {}
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            meta = checkpoint.get("meta", {}) or {}

        backbone = str(meta.get("backbone", "")).strip()
        if not backbone:
            backbone = self._infer_backbone_from_state_dict(state_dict)

        img_size_meta = meta.get("img_size")
        if isinstance(img_size_meta, (int, float)):
            img_size = int(img_size_meta)
        else:
            img_size = default_img_size_for_backbone(backbone)
        decision_threshold = meta.get("decision_threshold")

        model = ImageBinaryClassifier(backbone=backbone, pretrained=False).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.custom_model = model
        self.backbone = backbone
        self.img_size = img_size
        if isinstance(decision_threshold, (int, float)) and 0.0 < float(decision_threshold) < 1.0:
            self.fake_threshold = float(decision_threshold)

        self._init_preprocess()

    def _infer_backbone_from_state_dict(self, state_dict) -> str:
        if not isinstance(state_dict, dict):
            return self.backbone
        keys = list(state_dict.keys())
        if any(k.startswith("backbone.features.") for k in keys):
            if any(".6." in k for k in keys):
                return "efficientnet_v2_s"
            return "efficientnet_b0"
        if any(k.startswith("backbone.layer4.2.") for k in keys):
            return "resnet50"
        if any(k.startswith("backbone.layer4.1.") for k in keys):
            return "resnet18"
        if any(k.startswith("backbone.stages.") for k in keys):
            return "convnext_tiny"
        return self.backbone

    def _load_ensemble_calibration(self):
        candidates = []
        if self.ensemble_path:
            candidates.append(self.ensemble_path)
        base_dir = os.path.dirname(self.model_path) or "."
        candidates.append(os.path.join(base_dir, "image_ensemble_weights.json"))
        candidates.append(os.path.join(base_dir, "ensemble_weights.json"))

        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                loaded_weights = payload.get("weights", {})
                if isinstance(loaded_weights, dict):
                    for k, v in loaded_weights.items():
                        if isinstance(v, (int, float)):
                            self.model_weights[str(k)] = float(v)
                threshold = payload.get("threshold")
                if isinstance(threshold, (int, float)) and 0.0 < float(threshold) < 1.0:
                    self.fake_threshold = float(threshold)
                break
            except Exception:
                continue

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

    def _custom_model_score(self, image: Image.Image) -> Optional[float]:
        if self.custom_model is None:
            return None
        x = self._tensor_from_pil(image)
        with torch.no_grad():
            prob = torch.sigmoid(self.custom_model(x)).item()
        return float(np.clip(prob, 0.0, 1.0))

    def _texture_score(self, rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(1.0 - np.clip(lap_var / 400.0, 0.0, 1.0))

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

    def _face_region_score(self, face_rgb: np.ndarray, tex_score: float, freq_score: float) -> float:
        if face_rgb is not None and face_rgb.size > 0 and self.custom_model is not None:
            face_img = Image.fromarray(face_rgb)
            face_score = self._custom_model_score(face_img)
            if face_score is not None:
                return float(face_score)
        return float((tex_score + freq_score) * 0.5)

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

        rgb = np.array(image)
        model_scores: List[ModelScore] = []

        custom_score = self._custom_model_score(image)
        if custom_score is not None:
            model_scores.append(ModelScore("custom_model_score", round(custom_score, 4)))

        tex_score = self._texture_score(rgb)
        freq_score = self._frequency_score(rgb)
        model_scores.append(ModelScore("texture_artifact_score", round(tex_score, 4)))
        model_scores.append(ModelScore("frequency_artifact_score", round(freq_score, 4)))

        face_crop = self._detect_largest_face(rgb)
        face_score = self._face_region_score(face_crop, tex_score, freq_score)
        model_scores.append(ModelScore("face_region_score", round(face_score, 4)))

        hf_error = None
        hf_scored = False
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

        if total_weight > 0:
            fake_prob = weighted_sum / total_weight
        else:
            fake_prob = float(np.mean([m.fake_probability for m in model_scores]))
        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))

        real_prob = 1.0 - fake_prob
        label = "Fake" if fake_prob >= self.fake_threshold else "Real"
        confidence = abs(fake_prob - self.fake_threshold) / max(self.fake_threshold, 1.0 - self.fake_threshold, 1e-6)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        details = {
            "num_models_used": len(model_scores),
            "combination": "weighted_average",
            "weights_used": {m.model_name: self._weight_for_model(m.model_name) for m in model_scores},
            "face_detected": face_crop is not None,
            "decision_threshold": self.fake_threshold,
            "backbone": self.backbone,
            "img_size": self.img_size,
            "hf_model": self.hf_model_id,
            "hf_scored": hf_scored,
            "hf_error": hf_error,
            "hf_token_present": self.hf_ready,
        }

        return ImagePrediction(
            label=label,
            fake_probability=round(fake_prob, 4),
            real_probability=round(real_prob, 4),
            confidence=round(confidence, 4),
            method="calibrated_weighted_ensemble",
            details=details,
            model_results=model_scores,
        )
