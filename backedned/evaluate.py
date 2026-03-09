import argparse
import os

import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from training.dataset import AUDIO_EXTS, IMAGE_EXTS, build_items_from_binary_folders
from training.models import AudioMLP, ImageBinaryClassifier, extract_audio_features


def binary_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def eval_image(model_path, data_dir, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageBinaryClassifier().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    items = build_items_from_binary_folders(data_dir, IMAGE_EXTS)
    y_true = []
    y_pred = []
    for item in items:
        img = Image.open(item.path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        y_true.append(item.label)
        y_pred.append(1 if prob >= threshold else 0)
    return binary_metrics(y_true, y_pred), len(items)


def eval_audio(model_path, data_dir, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioMLP(input_dim=6).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    items = build_items_from_binary_folders(data_dir, AUDIO_EXTS)
    y_true = []
    y_pred = []
    for item in items:
        y, sr = librosa.load(item.path, sr=16000, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
        feats = extract_audio_features(y, sr)
        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        y_true.append(item.label)
        y_pred.append(1 if prob >= threshold else 0)
    return binary_metrics(y_true, y_pred), len(items)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--modality", type=str, choices=["image", "audio"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    if args.modality == "image":
        metrics, n = eval_image(args.model_path, args.data_dir, args.threshold)
    else:
        metrics, n = eval_audio(args.model_path, args.data_dir, args.threshold)

    print(f"Evaluated samples: {n}")
    print(metrics)


if __name__ == "__main__":
    main()
