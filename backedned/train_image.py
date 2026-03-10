import argparse
import copy
import io
import json
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from training.dataset import (
    IMAGE_EXTS,
    BinaryImageDataset,
    DataItem,
    build_items_from_binary_folders,
    train_val_split,
)
from training.models import ImageBinaryClassifier, default_img_size_for_backbone


class RandomJPEGCompression:
    def __init__(self, p: float = 0.35, quality_min: int = 45, quality_max: int = 95):
        self.p = p
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        quality = random.randint(self.quality_min, self.quality_max)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def binary_metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    tnr = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + tnr)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_acc": float(balanced_acc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
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


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
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


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    thresholds = np.unique(np.clip(y_prob, 0.0, 1.0))
    thresholds = np.concatenate(([0.1], thresholds, [0.9]))

    best_threshold = 0.5
    best_metrics = binary_metrics_from_preds(y_true, (y_prob >= 0.5).astype(np.int32))
    best_f1 = best_metrics["f1"]

    for thr in thresholds:
        preds = (y_prob >= thr).astype(np.int32)
        metrics = binary_metrics_from_preds(y_true, preds)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(thr)
            best_metrics = metrics

    return best_threshold, best_metrics


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total = 0
    all_probs: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            labels = y.squeeze(1).detach().cpu().numpy()
            all_probs.extend([float(v) for v in probs])
            all_labels.extend([int(v) for v in labels])

    y_true = np.array(all_labels, dtype=np.int32)
    y_prob = np.array(all_probs, dtype=np.float32)

    thr_05_metrics = binary_metrics_from_preds(y_true, (y_prob >= 0.5).astype(np.int32))
    best_threshold, best_metrics = find_best_threshold(y_true, y_prob)

    return {
        "loss": total_loss / max(total, 1),
        "y_true": y_true,
        "y_prob": y_prob,
        "roc_auc": roc_auc(y_true, y_prob),
        "pr_auc": pr_auc(y_true, y_prob),
        "thr_05": thr_05_metrics,
        "best_threshold": best_threshold,
        "best": best_metrics,
    }


def texture_score(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(1.0 - np.clip(lap_var / 400.0, 0.0, 1.0))


def frequency_score(rgb: np.ndarray) -> float:
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


def detect_largest_face(rgb: np.ndarray, face_detector) -> np.ndarray | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(
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


def calibrate_weighted_ensemble(
    rows: List[Dict],
    trials: int = 4000,
    seed: int = 42,
) -> Dict:
    if not rows:
        raise RuntimeError("No validation rows available for ensemble calibration.")

    rng = np.random.default_rng(seed)
    keys = sorted({k for row in rows for k in row["scores"].keys()})
    best = {"f1": -1.0, "threshold": 0.5, "weights": {}}

    def score_with(weights: Dict[str, float], threshold: float) -> float:
        y_true = []
        y_pred = []
        for row in rows:
            total_w = 0.0
            weighted_sum = 0.0
            for k, w in weights.items():
                if k not in row["scores"]:
                    continue
                weighted_sum += float(row["scores"][k]) * w
                total_w += w
            if total_w <= 0:
                continue
            prob = weighted_sum / total_w
            pred = 1 if prob >= threshold else 0
            y_true.append(int(row["label"]))
            y_pred.append(pred)
        if not y_true:
            return 0.0
        metrics = binary_metrics_from_preds(np.array(y_true, dtype=np.int32), np.array(y_pred, dtype=np.int32))
        return metrics["f1"]

    for _ in range(trials):
        vals = rng.random(len(keys))
        vals = vals / np.sum(vals)
        weights = {k: float(v) for k, v in zip(keys, vals)}
        threshold = float(rng.uniform(0.2, 0.8))
        f1 = score_with(weights=weights, threshold=threshold)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": threshold, "weights": weights}

    return best


def export_validation_rows(
    model: nn.Module,
    val_items: List[DataItem],
    device: str,
    val_tf,
    output_path: str,
) -> List[Dict]:
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    model.eval()
    rows: List[Dict] = []

    for item in val_items:
        image = Image.open(item.path).convert("RGB")
        rgb = np.array(image)
        x = val_tf(image).unsqueeze(0).to(device)
        with torch.no_grad():
            base_prob = float(torch.sigmoid(model(x)).item())

        tex = texture_score(rgb)
        freq = frequency_score(rgb)
        face_crop = detect_largest_face(rgb, face_detector)

        if face_crop is not None and face_crop.size > 0:
            face_img = Image.fromarray(face_crop)
            fx = val_tf(face_img).unsqueeze(0).to(device)
            with torch.no_grad():
                face_prob = float(torch.sigmoid(model(fx)).item())
        else:
            face_prob = float((tex + freq) * 0.5)

        rows.append(
            {
                "label": int(item.label),
                "scores": {
                    "custom_model_score": round(base_prob, 6),
                    "texture_artifact_score": round(tex, 6),
                    "frequency_artifact_score": round(freq, 6),
                    "face_region_score": round(face_prob, 6),
                },
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Train image deepfake binary classifier")
    parser.add_argument("--data_dir", type=str, default="data/image")
    parser.add_argument("--output", type=str, default="models/image_detector.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=0, help="0 = auto based on backbone")
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["resnet18", "resnet50", "efficientnet_b0", "efficientnet_v2_s", "convnext_tiny"],
    )
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--group_by", type=str, default="stem", choices=["none", "stem", "parent", "stem_parent"])
    parser.add_argument("--ensemble_trials", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dir = os.path.dirname(args.output)

    items = build_items_from_binary_folders(args.data_dir, IMAGE_EXTS)
    if len(items) < 4:
        raise RuntimeError("Not enough image files. Add data/image/real and data/image/fake samples.")

    train_items, val_items = train_val_split(
        items,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_by=args.group_by,
    )
    if not val_items:
        val_items = train_items[: max(1, len(train_items) // 5)]

    img_size = args.img_size if args.img_size > 0 else default_img_size_for_backbone(args.backbone)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.06)],
                p=0.6,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
            RandomJPEGCompression(p=0.35, quality_min=40, quality_max=95),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.01, 0.08), ratio=(0.3, 3.3), value="random"),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = BinaryImageDataset(train_items, transform=train_tf)
    val_ds = BinaryImageDataset(val_items, transform=val_tf)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageBinaryClassifier(backbone=args.backbone, pretrained=args.pretrained).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=max(args.lr * 0.05, 1e-6))

    train_pos = sum(1 for x in train_items if x.label == 1)
    train_neg = sum(1 for x in train_items if x.label == 0)
    if train_pos > 0 and train_neg > 0:
        pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    best_val_roc_auc = -1.0
    best_ckpt = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(seen, 1)
        report = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={report['loss']:.4f} "
            f"roc_auc={report['roc_auc']:.4f} pr_auc={report['pr_auc']:.4f} "
            f"f1@0.5={report['thr_05']['f1']:.4f} f1@best={report['best']['f1']:.4f} "
            f"best_thr={report['best_threshold']:.3f}"
        )

        if report["roc_auc"] > best_val_roc_auc:
            best_val_roc_auc = report["roc_auc"]
            best_ckpt = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "meta": {
                    "img_size": img_size,
                    "best_val_roc_auc": round(float(report["roc_auc"]), 6),
                    "best_val_pr_auc": round(float(report["pr_auc"]), 6),
                    "best_val_f1_at_05": round(float(report["thr_05"]["f1"]), 6),
                    "best_val_f1_at_calibrated_threshold": round(float(report["best"]["f1"]), 6),
                    "decision_threshold": round(float(report["best_threshold"]), 6),
                    "backbone": args.backbone,
                    "group_by": args.group_by,
                    "data_dir": args.data_dir,
                },
            }
            torch.save(
                best_ckpt,
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")

    if best_ckpt is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_ckpt["state_dict"], strict=False)

    scores_jsonl_path = os.path.join(output_dir, "image_validation_scores.jsonl")
    rows = export_validation_rows(
        model=model,
        val_items=val_items,
        device=device,
        val_tf=val_tf,
        output_path=scores_jsonl_path,
    )

    ensemble = calibrate_weighted_ensemble(rows, trials=args.ensemble_trials, seed=args.seed)
    ensemble_output = os.path.join(output_dir, "image_ensemble_weights.json")
    with open(ensemble_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": round(float(ensemble["threshold"]), 6),
                "best_f1": round(float(ensemble["f1"]), 6),
                "weights": ensemble["weights"],
                "source": "validation_random_search",
            },
            f,
            indent=2,
        )

    print(f"Saved validation scores to {scores_jsonl_path}")
    print(f"Saved calibrated ensemble weights to {ensemble_output}")
    print(f"Training complete. Best validation ROC-AUC: {best_val_roc_auc:.4f}")


if __name__ == "__main__":
    main()
