import os
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from audio_detector import AudioDeepfakeDetector, save_uploaded_audio_to_temp
from image_detector import ImageDeepfakeDetector
from video_detector import VideoDeepfakeDetector, save_uploaded_video_to_temp


st.set_page_config(page_title="AI Deepfake Detector", layout="centered")
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Backward compatibility for old variable name.
if not os.getenv("HF_TOKEN") and os.getenv("HF_API_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_API_TOKEN")


@st.cache_resource
def load_detectors(hf_token: str | None, cache_buster: str):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    image_detector = ImageDeepfakeDetector(model_path="models/image_detector.pth")
    video_detector = VideoDeepfakeDetector(image_detector=image_detector, max_frames=16)
    audio_detector = AudioDeepfakeDetector(model_path="models/audio_detector.pth")
    return image_detector, video_detector, audio_detector


def detect_media_type(uploaded_file) -> str:
    mime = uploaded_file.type or ""
    ext = os.path.splitext(uploaded_file.name.lower())[1]

    if mime.startswith("image/") or ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        return "image"
    if mime.startswith("video/") or ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return "video"
    if mime.startswith("audio/") or ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        return "audio"
    return "unknown"


def show_result(label: str, fake_prob: float, real_prob: float, confidence: float):
    st.subheader("Final Combined Result")
    if label == "Fake":
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")

    st.write(f"Combined Confidence: **{confidence * 100:.2f}%**")
    st.progress(float(confidence))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Fake Probability", f"{fake_prob * 100:.2f}%")
    with col2:
        st.metric("Average Real Probability", f"{real_prob * 100:.2f}%")


def _row_weight(weights, model_name):
    weights = weights or {}
    if model_name in weights:
        return float(weights.get(model_name, 0.0))
    base_name = model_name.split(":", 1)[0]
    return float(weights.get(base_name, 0.0))


def show_model_table(model_results, weights=None, extra_rows=None):
    st.subheader("Individual Model Results")
    rows = []
    for item in model_results:
        rows.append(
            {
                "Model": item.model_name,
                "Fake Probability": f"{item.fake_probability * 100:.2f}%",
                "Raw Score": f"{item.fake_probability:.4f}",
                "Weight": _row_weight(weights, item.model_name),
                "Note": "",
            }
        )

    for extra in extra_rows or []:
        model_name = extra.get("model_name", "unknown")
        rows.append(
            {
                "Model": model_name,
                "Fake Probability": extra.get("fake_probability", "N/A"),
                "Raw Score": "N/A",
                "Weight": _row_weight(weights, model_name),
                "Note": str(extra.get("raw_score", "N/A")),
            }
        )

    st.dataframe(rows, width="stretch")


def plot_spectrogram(mel_db):
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", cmap="magma", ax=ax)
    ax.set_title("Audio Spectrogram (Mel dB)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("AI Deepfake Detector")
    st.caption(
        "Upload image/video/audio, run multiple models, view each model score, and see the final combined weighted result."
    )

    uploaded_file = st.file_uploader(
        "Upload media",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "webp",
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
            "wav",
            "mp3",
            "flac",
            "ogg",
            "m4a",
        ],
    )

    if uploaded_file is None:
        st.info("Please upload an image, video, or audio file.")
        return

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    hf_token = hf_token.strip().strip("\"").strip("'") if hf_token else None

    media_type = detect_media_type(uploaded_file)
    st.write(f"Detected file type: **{media_type.upper()}**")
    st.caption(f"HF token detected: `{'Yes' if hf_token else 'No'}`")

    image_detector, video_detector, audio_detector = load_detectors(
        hf_token,
        "hf-path-fix-v2",
    )

    if media_type == "image":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")
        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing image with ensemble..."):
                pred = image_detector.predict(image)

            extra_rows = []
            if not pred.details.get("hf_scored", False):
                hf_name = image_detector.hf_display_name
                extra_rows.append(
                    {
                        "model_name": hf_name,
                        "fake_probability": "N/A",
                        "raw_score": pred.details.get("hf_error", "N/A"),
                    }
                )

            show_model_table(
                pred.model_results,
                getattr(image_detector, "model_weights", {}),
                extra_rows=extra_rows,
            )
            show_result(pred.label, pred.fake_probability, pred.real_probability, pred.confidence)
            st.caption(f"Combination Method: `{pred.method}`")
            st.caption(f"HF image model: `{pred.details.get('hf_model')}`")
            st.caption(f"HF scored: `{pred.details.get('hf_scored')}`")

    elif media_type == "video":
        st.video(uploaded_file)
        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing video with ensemble..."):
                temp_path = save_uploaded_video_to_temp(uploaded_file)
                try:
                    pred = video_detector.predict(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            show_model_table(pred.model_results, getattr(video_detector, "model_weights", {}))
            show_result(pred.label, pred.fake_probability, pred.real_probability, pred.confidence)
            st.write(f"Frames analyzed: **{pred.analyzed_frames}**")

    elif media_type == "audio":
        st.audio(uploaded_file)
        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing audio with ensemble..."):
                temp_path = save_uploaded_audio_to_temp(uploaded_file)
                try:
                    pred = audio_detector.predict(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            show_model_table(pred.model_results, getattr(audio_detector, "model_weights", {}))
            show_result(pred.label, pred.fake_probability, pred.real_probability, pred.confidence)
            st.caption(f"Combination Method: `{pred.method}`")
            st.subheader("Audio Spectrogram")
            plot_spectrogram(pred.spectrogram_db)

    else:
        st.error("Unsupported file type. Please upload a valid image, video, or audio file.")


if __name__ == "__main__":
    main()
