import os
import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from audio_detector import AudioDeepfakeDetector
from image_detector import ImageDeepfakeDetector
from video_detector import VideoDeepfakeDetector


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
if not os.getenv("HF_TOKEN") and os.getenv("HF_API_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_API_TOKEN")

app = FastAPI(
    title="AI Deepfake Detector API",
    version="1.0.0",
    description="Detect real vs fake content for image, video, and audio files.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_detectors():
    image_detector = ImageDeepfakeDetector(model_path="models/image_detector.pth")
    video_detector = VideoDeepfakeDetector(image_detector=image_detector, max_frames=16)
    audio_detector = AudioDeepfakeDetector(model_path="models/audio_detector.pth")
    return image_detector, video_detector, audio_detector


def detect_media_type(filename: str, content_type: Optional[str]) -> str:
    mime = content_type or ""
    ext = os.path.splitext((filename or "").lower())[1]
    if mime.startswith("image/") or ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        return "image"
    if mime.startswith("video/") or ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return "video"
    if mime.startswith("audio/") or ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        return "audio"
    return "unknown"


def model_results_to_json(model_results):
    return [
        {
            "model_name": item.model_name,
            "fake_probability": float(item.fake_probability),
        }
        for item in model_results
    ]


def save_upload_to_temp(upload_file: UploadFile, default_suffix: str):
    suffix = os.path.splitext(upload_file.filename or "")[1] or default_suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(upload_file.file.read())
        return tmp_file.name


def image_response(pred):
    return {
        "media_type": "image",
        "label": pred.label,
        "fake_probability": float(pred.fake_probability),
        "real_probability": float(pred.real_probability),
        "confidence": float(pred.confidence),
        "method": pred.method,
        "details": pred.details,
        "model_results": model_results_to_json(pred.model_results),
    }


def video_response(pred):
    return {
        "media_type": "video",
        "label": pred.label,
        "fake_probability": float(pred.fake_probability),
        "real_probability": float(pred.real_probability),
        "confidence": float(pred.confidence),
        "analyzed_frames": int(pred.analyzed_frames),
        "frame_scores": [float(x) for x in pred.frame_scores],
        "model_results": model_results_to_json(pred.model_results),
    }


def audio_response(pred):
    return {
        "media_type": "audio",
        "label": pred.label,
        "fake_probability": float(pred.fake_probability),
        "real_probability": float(pred.real_probability),
        "confidence": float(pred.confidence),
        "method": pred.method,
        "details": pred.details,
        "model_results": model_results_to_json(pred.model_results),
        "spectrogram": {
            "shape": [int(x) for x in pred.spectrogram_db.shape],
            "min_db": float(pred.spectrogram_db.min()),
            "max_db": float(pred.spectrogram_db.max()),
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "hf_token_present": bool(
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        ),
    }


@app.post("/detect")
def detect(file: UploadFile = File(...), media_type: Optional[str] = Query(default=None)):
    image_detector, video_detector, audio_detector = get_detectors()
    inferred_type = media_type or detect_media_type(file.filename or "", file.content_type)

    if inferred_type not in {"image", "video", "audio"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if inferred_type == "image":
        try:
            image_bytes = file.file.read()
            image = Image.open(BytesIO(image_bytes))
            pred = image_detector.predict(image)
            return image_response(pred)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image detection failed: {e}")

    if inferred_type == "video":
        temp_path = None
        try:
            temp_path = save_upload_to_temp(file, ".mp4")
            pred = video_detector.predict(temp_path)
            return video_response(pred)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Video detection failed: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    temp_path = None
    try:
        temp_path = save_upload_to_temp(file, ".wav")
        pred = audio_detector.predict(temp_path)
        return audio_response(pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio detection failed: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/detect/image")
def detect_image(file: UploadFile = File(...)):
    return detect(file=file, media_type="image")


@app.post("/detect/video")
def detect_video(file: UploadFile = File(...)):
    return detect(file=file, media_type="video")


@app.post("/detect/audio")
def detect_audio(file: UploadFile = File(...)):
    return detect(file=file, media_type="audio")
