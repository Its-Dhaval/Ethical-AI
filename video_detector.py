import os
import tempfile
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image

from image_detector import ImageDeepfakeDetector, ModelScore


@dataclass
class VideoPrediction:
    label: str
    fake_probability: float
    real_probability: float
    confidence: float
    analyzed_frames: int
    frame_scores: list
    model_results: List[ModelScore]


class VideoDeepfakeDetector:
    def __init__(self, image_detector: ImageDeepfakeDetector, max_frames: int = 16):
        self.image_detector = image_detector
        self.max_frames = max_frames
        self.model_weights = {
            "frame_average_model": 0.45,
            "peak_frame_model": 0.20,
            "temporal_flicker_model": 0.15,
            "majority_vote_model": 0.20,
        }

    def _sample_indices(self, total_frames: int) -> np.ndarray:
        if total_frames <= 0:
            return np.array([], dtype=int)
        num = min(self.max_frames, total_frames)
        return np.linspace(0, total_frames - 1, num=num, dtype=int)

    def _temporal_flicker_score(self, frame_scores: List[float]) -> float:
        if len(frame_scores) < 2:
            return 0.5
        diffs = np.abs(np.diff(frame_scores))
        mean_diff = float(np.mean(diffs))
        return float(np.clip(mean_diff * 4.0, 0.0, 1.0))

    def predict(self, video_path: str) -> VideoPrediction:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_indices(total_frames)

        frame_scores = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pred = self.image_detector.predict(pil_image, include_hf=False)
            frame_scores.append(pred.fake_probability)

        cap.release()

        if not frame_scores:
            raise ValueError("Could not read frames from this video.")

        frame_avg = float(np.mean(frame_scores))
        peak_fake = float(np.max(frame_scores))
        temporal_flicker = self._temporal_flicker_score(frame_scores)
        majority_vote = float(np.mean([1.0 if s >= 0.5 else 0.0 for s in frame_scores]))

        model_results = [
            ModelScore("frame_average_model", round(frame_avg, 4)),
            ModelScore("peak_frame_model", round(peak_fake, 4)),
            ModelScore("temporal_flicker_model", round(temporal_flicker, 4)),
            ModelScore("majority_vote_model", round(majority_vote, 4)),
        ]

        weighted_sum = 0.0
        total_weight = 0.0
        for item in model_results:
            w = float(self.model_weights.get(item.model_name, 0.0))
            weighted_sum += item.fake_probability * w
            total_weight += w

        fake_prob = weighted_sum / total_weight if total_weight > 0 else float(
            np.mean([m.fake_probability for m in model_results])
        )
        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
        real_prob = 1.0 - fake_prob
        label = "Fake" if fake_prob >= 0.5 else "Real"

        std = float(np.std(frame_scores))
        base_conf = abs(fake_prob - 0.5) * 2.0
        consistency = 1.0 - min(std * 2.0, 1.0)
        confidence = 0.7 * base_conf + 0.3 * consistency

        return VideoPrediction(
            label=label,
            fake_probability=round(fake_prob, 4),
            real_probability=round(real_prob, 4),
            confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
            analyzed_frames=len(frame_scores),
            frame_scores=[round(s, 4) for s in frame_scores],
            model_results=model_results,
        )


def save_uploaded_video_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name
