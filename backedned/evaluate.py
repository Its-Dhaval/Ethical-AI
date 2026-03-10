import argparse
import os

import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from training.dataset import AUDIO_EXTS, IMAGE_EXTS, build_items_from_binary_folders
from training.models import (
    AudioMLP,
    ImageBinaryClassifier,
    default_img_size_for_backbone,
    extract_audio_features,
)


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
        "balanced_acc": round((recall + (tn / max(tn + fp, 1))) / 2.0, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc(y_true, y_prob):
    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.5

    order = np.argsort(-y_prob)
    y = y_true[order]
    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return float(np.trapezoid(tpr, fpr))


def pr_auc(y_true, y_prob):
    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0

    order = np.argsort(-y_prob)
    y = y_true[order]
    tp = 0
    fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / positives)
        precisions.append(tp / max(tp + fp, 1))
    return float(np.trapezoid(precisions, recalls))


def eval_image(model_path, data_dir, threshold=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    def infer_backbone_from_state_dict(state_dict_obj):
        if not isinstance(state_dict_obj, dict):
            return "efficientnet_b0"
        keys = list(state_dict_obj.keys())
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
        return "efficientnet_b0"

    backbone = str(meta.get("backbone", "")).strip() or infer_backbone_from_state_dict(state_dict)
    img_size = int(meta.get("img_size", default_img_size_for_backbone(backbone)))
    decision_threshold = meta.get("decision_threshold", 0.5)
    if threshold is None:
        threshold = float(decision_threshold)

    model = ImageBinaryClassifier(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    items = build_items_from_binary_folders(data_dir, IMAGE_EXTS)
    y_true = []
    y_pred = []
    y_prob = []
    for item in items:
        img = Image.open(item.path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        y_true.append(item.label)
        y_prob.append(prob)
        y_pred.append(1 if prob >= threshold else 0)
    metrics = binary_metrics(y_true, y_pred)
    metrics["roc_auc"] = round(roc_auc(y_true, y_prob), 4)
    metrics["pr_auc"] = round(pr_auc(y_true, y_prob), 4)
    metrics["threshold"] = round(float(threshold), 4)
    metrics["backbone"] = backbone
    metrics["img_size"] = img_size
    return metrics, len(items)


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
    parser.add_argument("--threshold", type=float, default=None)
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
