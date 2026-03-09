import argparse
import os

import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.dataset import AUDIO_EXTS, build_items_from_binary_folders, train_val_split
from training.models import AudioMLP, extract_audio_features


def build_feature_matrix(items, sample_rate=16000, max_seconds=30):
    xs = []
    ys = []
    max_samples = sample_rate * max_seconds

    for item in items:
        y, sr = librosa.load(item.path, sr=sample_rate, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
        y = y[:max_samples]
        feats = extract_audio_features(y, sr)
        xs.append(feats)
        ys.append([float(item.label)])

    x = np.array(xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)
    return x, y


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train audio deepfake binary classifier")
    parser.add_argument("--data_dir", type=str, default="data/audio")
    parser.add_argument("--output", type=str, default="models/audio_detector.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    items = build_items_from_binary_folders(args.data_dir, AUDIO_EXTS)
    if len(items) < 10:
        raise RuntimeError("Not enough audio files. Add data/audio/real and data/audio/fake samples.")

    train_items, val_items = train_val_split(items, val_ratio=args.val_ratio, seed=args.seed)
    if not val_items:
        val_items = train_items[: max(1, len(train_items) // 5)]

    x_train, y_train = build_feature_matrix(train_items)
    x_val, y_val = build_feature_matrix(val_items)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioMLP(input_dim=x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "meta": {
                        "best_val_acc": best_val_acc,
                        "feature_dim": int(x_train.shape[1]),
                        "data_dir": args.data_dir,
                    },
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
