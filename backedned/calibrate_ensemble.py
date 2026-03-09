import argparse
import json
import random

import numpy as np


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score_rows(rows, weights, threshold=0.5):
    y_true = []
    y_pred = []
    for row in rows:
        label = int(row["label"])
        scores = row["scores"]
        weighted_sum = 0.0
        total_w = 0.0
        for k, w in weights.items():
            if k not in scores:
                continue
            weighted_sum += float(scores[k]) * float(w)
            total_w += float(w)
        if total_w <= 0:
            continue
        prob = weighted_sum / total_w
        pred = 1 if prob >= threshold else 0
        y_true.append(label)
        y_pred.append(pred)
    return y_true, y_pred


def f1_score(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return (2 * precision * recall) / max(precision + recall, 1e-8)


def random_weights(keys):
    vals = np.random.rand(len(keys))
    vals = vals / np.sum(vals)
    return {k: float(v) for k, v in zip(keys, vals)}


def main():
    parser = argparse.ArgumentParser(description="Calibrate ensemble weights from validation score JSONL")
    parser.add_argument("--input", type=str, required=True, help="Path to JSONL rows with label and scores")
    parser.add_argument("--output", type=str, default="models/ensemble_weights.json")
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_jsonl(args.input)
    if not rows:
        raise RuntimeError("No rows found in input file.")

    all_keys = sorted({k for row in rows for k in row.get("scores", {}).keys()})
    if not all_keys:
        raise RuntimeError("No model score keys found in input rows.")

    best_f1 = -1.0
    best_weights = None
    for _ in range(args.trials):
        w = random_weights(all_keys)
        y_true, y_pred = score_rows(rows, w, threshold=args.threshold)
        if not y_true:
            continue
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = w

    if best_weights is None:
        raise RuntimeError("Could not find valid weights.")

    output_obj = {
        "threshold": args.threshold,
        "best_f1": round(best_f1, 6),
        "weights": best_weights,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print(f"Saved calibrated weights to {args.output}")
    print(output_obj)


if __name__ == "__main__":
    main()
